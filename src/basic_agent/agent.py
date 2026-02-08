"""Core Agent class — orchestrates conversations with LLM providers."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import jinja2
from pydantic import BaseModel

from .models import parse_structured_output, structured_output
from .provider import Provider, ProviderResponse, Usage, get_provider
from .tools import ToolRegistry

T = TypeVar("T", bound=BaseModel)


@dataclass
class RunResult:
    """Stats from the most recent agent.run() call."""

    output: Any  # str or Pydantic model instance
    usage: Usage  # total tokens across all provider calls
    provider_calls: int  # number of chat() calls


class Agent:
    """An LLM-powered agent that supports tools and structured output.

    Args:
        provider: Provider name ("anthropic" or "openai").
        model: Model name (e.g. "claude-sonnet-4-20250514", "gpt-4o").
        system: System prompt for the agent.
        tools: List of @tool-decorated functions.
        output_type: Pydantic model class for structured output.
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
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_iterations: int = 10,
    ) -> None:
        self._provider: Provider = get_provider(provider, model)
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_iterations = max_iterations
        self._output_type = output_type
        self.last_run: RunResult | None = None

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
    ) -> Union[str, T, Any]:
        """Run the agent with a user message.

        Args:
            message: The user's input message.
            deps: Optional Pydantic model whose fields are injected into the
                system prompt Jinja2 template as ``{{deps.field}}``.

        Returns:
            - If output_type is set: a validated Pydantic model instance.
            - Otherwise: the plain text response string.
        """
        # --- Build system prompt ---
        rendered_system = self._render_system_prompt(deps)

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
        provider_calls = 0

        for _ in range(self._max_iterations):
            response = self._provider.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=tool_choice if tool_schemas else None,
                system=rendered_system,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            provider_calls += 1

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # No tool calls — we're done
            if not response.tool_calls:
                main_result = response.text or ""
                self.last_run = RunResult(
                    output=main_result,
                    usage=Usage(total_input_tokens, total_output_tokens),
                    provider_calls=provider_calls,
                )
                return main_result

            # Process tool calls
            # Build assistant message with tool use
            assistant_content = self._build_assistant_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for structured output tool first
            for tc in response.tool_calls:
                if self._output_model and tc.name == self._output_model.__name__:
                    parsed = parse_structured_output(self._output_model, tc.input)
                    self.last_run = RunResult(
                        output=parsed,
                        usage=Usage(total_input_tokens, total_output_tokens),
                        provider_calls=provider_calls,
                    )
                    return parsed

            # Execute all tool calls in parallel
            tool_results_content = self._execute_tool_calls_parallel(
                response.tool_calls
            )

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results_content})

            # After processing tool calls, stop forcing a specific tool
            if tool_choice and tool_choice != "auto":
                tool_choice = "auto"

        # Max iterations reached
        main_result = response.text or ""
        self.last_run = RunResult(
            output=main_result,
            usage=Usage(total_input_tokens, total_output_tokens),
            provider_calls=provider_calls,
        )
        return main_result

    def _render_system_prompt(
        self,
        deps: Optional[BaseModel],
    ) -> str:
        """Render the system prompt with Jinja2 template.

        Args:
            deps: Optional Pydantic model for template variable injection.

        Returns:
            The fully rendered system prompt string.
        """
        deps_dict = deps.model_dump() if deps else {}
        template = jinja2.Template(self._system, undefined=jinja2.StrictUndefined)
        return template.render(deps=deps_dict)

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

    def _execute_single_tool(self, tc: Any) -> Tuple[Any, str]:
        """Execute a single tool call and return (tool_call, result_str).

        Handles tool lookup, execution, and error handling. Thread-safe.
        """
        tool_def = self._registry.get(tc.name)
        if tool_def is None:
            return (tc, f"Error: Unknown tool '{tc.name}'")
        try:
            result = tool_def.execute(**tc.input)
            return (tc, str(result))
        except Exception as e:
            return (tc, f"Error executing tool '{tc.name}': {e}")

    def _execute_tool_calls_parallel(
        self, tool_calls: list
    ) -> List[Dict[str, Any]]:
        """Execute tool calls in parallel using a thread pool.

        Results are returned in the original tool call order.
        """
        if not tool_calls:
            return []

        max_workers = min(len(tool_calls), 10)
        results_by_id: Dict[str, Tuple[int, Any, str]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._execute_single_tool, tc): i
                for i, tc in enumerate(tool_calls)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                tc, result_str = future.result()
                results_by_id[tc.id] = (idx, tc, result_str)

        tool_results_content: List[Dict[str, Any]] = []
        for tc in tool_calls:
            _, _, result_str = results_by_id[tc.id]
            tool_results_content.append(
                self._build_tool_result(tc.id, tc.name, result_str)
            )

        return tool_results_content
