"""Provider abstraction for Anthropic and OpenAI APIs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    """A normalized tool call from the LLM response."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderResponse:
    """Normalized response from any provider."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _retryable_chat(call_fn: Any, max_retries: int = 3) -> Any:
    """Call call_fn() with automatic retries on transient errors.

    Retries on rate-limit (429), server errors (5xx), and connection errors
    with exponential backoff (1s, 2s, 4s). Non-retryable errors propagate
    immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return call_fn()
        except (anthropic.APIConnectionError, openai.APIConnectionError) as exc:
            last_exc = exc
        except (anthropic.APIStatusError, openai.APIStatusError) as exc:
            if exc.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


class Provider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> ProviderResponse: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...


def _to_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert provider-agnostic tool schemas to Anthropic format."""
    result = []
    for t in tools:
        result.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        })
    return result


def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert provider-agnostic tool schemas to OpenAI format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return result


def _to_anthropic_tool_choice(tool_choice: Any) -> Any:
    """Convert tool_choice to Anthropic format."""
    if tool_choice is None:
        return {"type": "auto"}
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "any":
            return {"type": "any"}
        # Specific tool name
        return {"type": "tool", "name": tool_choice}
    return tool_choice


def _to_openai_tool_choice(tool_choice: Any) -> Any:
    """Convert tool_choice to OpenAI format."""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        # Specific tool name
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


class AnthropicProvider:
    """Provider implementation for the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> ProviderResponse:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
            kwargs["tool_choice"] = _to_anthropic_tool_choice(tool_choice)
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = _retryable_chat(lambda: self._client.messages.create(**kwargs))

        text = None
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return ProviderResponse(text=text, tool_calls=tool_calls, usage=usage, raw=response)


class OpenAIProvider:
    """Provider implementation for the OpenAI API."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self._client = openai.OpenAI()
        self._model = model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> ProviderResponse:
        # OpenAI uses system message in the messages list
        oai_messages: List[Dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = _to_openai_tool_choice(tool_choice)
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = _retryable_chat(
            lambda: self._client.chat.completions.create(**kwargs)
        )

        message = response.choices[0].message
        text = message.content
        tool_calls: List[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )

        usage = Usage()
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens or 0,
            )

        return ProviderResponse(text=text, tool_calls=tool_calls, usage=usage, raw=response)


def get_provider(provider: str = "anthropic", model: Optional[str] = None) -> Provider:
    """Factory function to create a provider instance.

    Args:
        provider: "anthropic" or "openai"
        model: Model name override. Defaults to provider-specific default.
    """
    if provider == "anthropic":
        return AnthropicProvider(model=model or "claude-haiku-4-5-20251001")
    elif provider == "openai":
        return OpenAIProvider(model=model or "gpt-5.2")
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")
