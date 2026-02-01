"""Core Agent class — orchestrates conversations with LLM providers."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import jinja2
from pydantic import BaseModel

from .memory import Memory
from .models import parse_structured_output, structured_output
from .provider import Provider, ProviderResponse, get_provider
from .tools import ToolDefinition, ToolRegistry
from .tracing import (
    get_tracer,
    llm_span,
    record_assistant_message,
    record_tool_call,
    record_tool_result,
    record_usage,
    record_user_message,
)

T = TypeVar("T", bound=BaseModel)


class Agent:
    """An LLM-powered agent that supports tools, structured output, memory, and tracing.

    Args:
        provider: Provider name ("anthropic" or "openai").
        model: Model name (e.g. "claude-sonnet-4-20250514", "gpt-4o").
        system: System prompt for the agent.
        tools: List of @tool-decorated functions.
        output_type: Pydantic model class for structured output.
        memory: A Memory instance for persistent storage.
        trace: Enable OpenTelemetry tracing.
        max_tokens: Max tokens for LLM responses.
        temperature: Temperature for LLM responses.
        max_iterations: Max tool-use loop iterations (prevents infinite loops).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        system: str = "You are a helpful assistant.",
        tools: Optional[List[Callable[..., Any]]] = None,
        output_type: Optional[Type[T]] = None,
        memory: Optional[Memory] = None,
        trace: bool = False,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_iterations: int = 10,
    ) -> None:
        self._provider: Provider = get_provider(provider, model)
        self._system = system
        self._trace = trace
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_iterations = max_iterations
        self._memory = memory
        self._output_type = output_type

        # Build tool registry from decorated functions
        self._registry = ToolRegistry()
        if tools:
            for func in tools:
                defn = getattr(func, "_tool_definition", None)
                if defn is not None:
                    self._registry._tools[defn.name] = defn
                else:
                    self._registry.register(func)

        # Structured output setup
        self._output_tool_schema: Optional[Dict[str, Any]] = None
        self._output_model: Optional[Type[BaseModel]] = None
        if output_type is not None:
            self._output_tool_schema, self._output_model = structured_output(output_type)

    def run(
        self,
        message: str,
        *,
        deps: Optional[BaseModel] = None,
        memory_id: Optional[str] = None,
        memory_update: bool = True,
    ) -> Union[str, T, Any]:
        """Run the agent with a user message.

        Args:
            message: The user's input message.
            deps: Optional Pydantic model whose fields are injected into the
                system prompt Jinja2 template as ``{{deps.field}}``.
            memory_id: Optional ID to load/store memory for this run.
                Requires a Memory instance on the agent.
            memory_update: If True (default) and memory_id is provided, a
                second LLM call updates memory after the main loop.

        Returns:
            - If output_type is set: a validated Pydantic model instance.
            - Otherwise: the plain text response string.
        """
        # --- Load memory ---
        current_memory = None
        if memory_id is not None:
            if self._memory is None:
                raise ValueError(
                    "memory_id was provided but no Memory instance is configured on this Agent."
                )
            current_memory = self._memory.get(memory_id)

        # --- Build system prompt ---
        rendered_system = self._render_system_prompt(deps, current_memory)

        with llm_span(
            provider_name=self._provider.provider_name,
            model=self._provider.model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ) as span:
            record_user_message(span, message)

            # Build tool schemas
            tool_schemas = self._registry.schemas()
            tool_choice: Optional[Any] = None

            if self._output_tool_schema is not None:
                tool_schemas.append(self._output_tool_schema)
                tool_choice = self._output_model.__name__

            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": message},
            ]

            total_input_tokens = 0
            total_output_tokens = 0

            for _ in range(self._max_iterations):
                response = self._provider.chat(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice=tool_choice if tool_schemas else None,
                    system=rendered_system,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )

                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

                # No tool calls — we're done
                if not response.tool_calls:
                    if response.text:
                        record_assistant_message(span, response.text)
                    record_usage(span, total_input_tokens, total_output_tokens)

                    main_result = response.text or ""
                    # --- Memory update ---
                    if memory_id is not None and memory_update and self._memory is not None:
                        self._update_memory(messages, memory_id, current_memory, main_result)
                    return main_result

                # Process tool calls
                # Build assistant message with tool use
                assistant_content = self._build_assistant_content(response)
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results_content: List[Dict[str, Any]] = []

                for tc in response.tool_calls:
                    # Check if this is the structured output tool
                    if self._output_model and tc.name == self._output_model.__name__:
                        record_usage(span, total_input_tokens, total_output_tokens)
                        parsed = parse_structured_output(self._output_model, tc.input)
                        # --- Memory update ---
                        if memory_id is not None and memory_update and self._memory is not None:
                            self._update_memory(messages, memory_id, current_memory, str(tc.input))
                        return parsed

                    # Execute registered tool
                    tool_def = self._registry.get(tc.name)
                    if tool_def is None:
                        result_str = f"Error: Unknown tool '{tc.name}'"
                    else:
                        record_tool_call(span, tc.name, json.dumps(tc.input))
                        try:
                            result = tool_def.execute(**tc.input)
                            result_str = str(result)
                        except Exception as e:
                            result_str = f"Error executing tool '{tc.name}': {e}"
                        record_tool_result(span, tc.name, result_str)

                    tool_results_content.append(
                        self._build_tool_result(tc.id, tc.name, result_str)
                    )

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results_content})

                # After processing tool calls, stop forcing a specific tool
                if tool_choice and tool_choice != "auto":
                    tool_choice = "auto"

            # Max iterations reached
            record_usage(span, total_input_tokens, total_output_tokens)
            main_result = response.text or ""
            if memory_id is not None and memory_update and self._memory is not None:
                self._update_memory(messages, memory_id, current_memory, main_result)
            return main_result

    def _render_system_prompt(
        self,
        deps: Optional[BaseModel],
        memory_data: Optional[BaseModel],
    ) -> str:
        """Render the system prompt with Jinja2 template and optional memory.

        Args:
            deps: Optional Pydantic model for template variable injection.
            memory_data: Optional loaded memory Pydantic instance.

        Returns:
            The fully rendered system prompt string.
        """
        deps_dict = deps.model_dump() if deps else {}
        template = jinja2.Template(self._system, undefined=jinja2.Undefined)
        rendered = template.render(deps=deps_dict)

        if memory_data is not None:
            memory_json = json.dumps(memory_data.model_dump(mode="json"), indent=2)
            rendered = f"<memory>\n{memory_json}\n</memory>\n\n{rendered}"

        return rendered

    def _update_memory(
        self,
        messages: List[Dict[str, Any]],
        memory_id: str,
        current_memory: Optional[BaseModel],
        assistant_response: str,
    ) -> None:
        """Make a second LLM call to extract updated memory from the conversation.

        Args:
            messages: The conversation messages from the main loop.
            memory_id: The memory item ID to store the update under.
            current_memory: The current memory data (may be None for new users).
            assistant_response: The final assistant response text.
        """
        assert self._memory is not None

        memory_schema = self._memory._schema
        memory_prompt = self._memory.memory_prompt or (
            "Based on the following conversation, extract and update the relevant information."
        )

        current_json = "{}"
        if current_memory is not None:
            current_json = json.dumps(current_memory.model_dump(mode="json"), indent=2)

        schema_json = json.dumps(memory_schema.model_json_schema(), indent=2)

        system_prompt = (
            f"{memory_prompt}\n\n"
            f"Current memory:\n{current_json}\n\n"
            f"Return the updated data matching this schema:\n{schema_json}"
        )

        # Use structured output to force the response into the memory schema
        tool_schema, _ = structured_output(memory_schema)

        # Build a condensed conversation for the memory update call
        update_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                update_messages.append(msg)
        # Include the final assistant response
        if assistant_response:
            update_messages.append({"role": "assistant", "content": assistant_response})

        # Add instruction as user message
        update_messages.append({
            "role": "user",
            "content": "Please update the memory based on this conversation.",
        })

        response = self._provider.chat(
            messages=update_messages,
            tools=[tool_schema],
            tool_choice=memory_schema.__name__,
            system=system_prompt,
            max_tokens=self._max_tokens,
            temperature=0,
        )

        # Parse the structured output from the tool call
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == memory_schema.__name__:
                    updated_data = memory_schema.model_validate(tc.input)
                    self._memory.put(memory_id, updated_data)
                    return

    def _build_assistant_content(self, response: ProviderResponse) -> Any:
        """Build assistant message content from a provider response.

        Returns format appropriate for the provider.
        """
        provider = self._provider.provider_name
        if provider == "anthropic":
            content: List[Dict[str, Any]] = []
            if response.text:
                content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            return content
        else:
            # OpenAI format — just return the raw message
            return response.raw.choices[0].message

    def _build_tool_result(self, tool_call_id: str, name: str, result: str) -> Dict[str, Any]:
        """Build a tool result message for the provider."""
        provider = self._provider.provider_name
        if provider == "anthropic":
            return {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result,
            }
        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
