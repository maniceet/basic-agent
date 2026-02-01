"""OpenTelemetry instrumentation for LLM calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

_tracer: Optional[trace.Tracer] = None


def setup_tracing(
    exporter: Optional[SpanExporter] = None,
    service_name: str = "basic-agent",
) -> trace.Tracer:
    """Configure OpenTelemetry tracing for the agent.

    Args:
        exporter: A span exporter. Defaults to ConsoleSpanExporter.
        service_name: The service name for the tracer.

    Returns:
        A configured Tracer instance.
    """
    global _tracer

    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is None:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _tracer = provider.get_tracer("basic-agent")
    return _tracer


def get_tracer() -> Optional[trace.Tracer]:
    """Get the current tracer, if tracing has been set up."""
    return _tracer


@contextmanager
def llm_span(
    provider_name: str,
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[trace.Span]:
    """Create a span around an LLM call following GenAI semantic conventions.

    Args:
        provider_name: "anthropic" or "openai"
        model: The model name
        temperature: Request temperature if set
        max_tokens: Request max tokens if set
    """
    tracer = _tracer
    if tracer is None:
        # No-op: yield a non-recording span
        yield trace.INVALID_SPAN
        return

    with tracer.start_as_current_span(f"llm.chat {provider_name}.{model}") as span:
        span.set_attribute("gen_ai.system", provider_name)
        span.set_attribute("gen_ai.request.model", model)
        if temperature is not None:
            span.set_attribute("gen_ai.request.temperature", temperature)
        if max_tokens is not None:
            span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        yield span


def record_user_message(span: trace.Span, content: str) -> None:
    """Record a user message as a span event."""
    if span.is_recording():
        span.add_event("gen_ai.user.message", attributes={"content": content})


def record_assistant_message(span: trace.Span, content: str) -> None:
    """Record an assistant message as a span event."""
    if span.is_recording():
        span.add_event("gen_ai.assistant.message", attributes={"content": content})


def record_tool_call(span: trace.Span, name: str, input_data: str) -> None:
    """Record a tool call as a span event."""
    if span.is_recording():
        span.add_event("gen_ai.tool.call", attributes={"name": name, "input": input_data})


def record_tool_result(span: trace.Span, name: str, output_data: str) -> None:
    """Record a tool result as a span event."""
    if span.is_recording():
        span.add_event("gen_ai.tool.result", attributes={"name": name, "output": output_data})


def record_usage(span: trace.Span, input_tokens: int, output_tokens: int) -> None:
    """Record token usage on the span."""
    if span.is_recording():
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
