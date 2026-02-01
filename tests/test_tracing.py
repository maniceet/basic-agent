"""Tests for OpenTelemetry tracing — uses in-memory span exporter."""

import threading
from typing import List, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from basic_agent.tracing import (
    llm_span,
    record_assistant_message,
    record_tool_call,
    record_tool_result,
    record_usage,
    record_user_message,
    setup_tracing,
)


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for testing."""

    def __init__(self) -> None:
        self._spans: List[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> List[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        self.clear()


def _setup_in_memory_tracing():
    """Set up tracing with an in-memory exporter for testing."""
    exporter = InMemorySpanExporter()
    setup_tracing(exporter=exporter, service_name="test-agent")
    return exporter


def test_setup_tracing_returns_tracer():
    exporter = InMemorySpanExporter()
    tracer = setup_tracing(exporter=exporter, service_name="test")
    assert tracer is not None


def test_llm_span_creates_span():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("anthropic", "claude-sonnet-4-20250514") as span:
        assert span.is_recording()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]
    assert "llm.chat" in span.name
    assert span.attributes["gen_ai.system"] == "anthropic"
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-20250514"


def test_llm_span_optional_attributes():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("openai", "gpt-4o", temperature=0.7, max_tokens=1000) as span:
        pass

    spans = exporter.get_finished_spans()
    span = spans[-1]
    assert span.attributes["gen_ai.request.temperature"] == 0.7
    assert span.attributes["gen_ai.request.max_tokens"] == 1000


def test_record_user_message():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("anthropic", "test-model") as span:
        record_user_message(span, "Hello there")

    spans = exporter.get_finished_spans()
    events = spans[-1].events
    assert any(e.name == "gen_ai.user.message" for e in events)
    user_event = next(e for e in events if e.name == "gen_ai.user.message")
    assert user_event.attributes["content"] == "Hello there"


def test_record_assistant_message():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("anthropic", "test-model") as span:
        record_assistant_message(span, "Hi back!")

    spans = exporter.get_finished_spans()
    events = spans[-1].events
    assert any(e.name == "gen_ai.assistant.message" for e in events)


def test_record_tool_call_and_result():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("anthropic", "test-model") as span:
        record_tool_call(span, "get_weather", '{"city": "London"}')
        record_tool_result(span, "get_weather", "Sunny, 58°F")

    spans = exporter.get_finished_spans()
    events = spans[-1].events
    tool_call_events = [e for e in events if e.name == "gen_ai.tool.call"]
    tool_result_events = [e for e in events if e.name == "gen_ai.tool.result"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].attributes["name"] == "get_weather"
    assert len(tool_result_events) == 1
    assert tool_result_events[0].attributes["output"] == "Sunny, 58°F"


def test_record_usage():
    exporter = _setup_in_memory_tracing()
    exporter.clear()

    with llm_span("openai", "gpt-4o") as span:
        record_usage(span, input_tokens=100, output_tokens=50)

    spans = exporter.get_finished_spans()
    span = spans[-1]
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["gen_ai.usage.output_tokens"] == 50
