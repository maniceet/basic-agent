"""Tests for the core Agent class — mocks the provider."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from basic_agent.agent import Agent
from basic_agent.memory import Memory
from basic_agent.provider import ProviderResponse, ToolCall, Usage
from basic_agent.tools import tool


class SentimentResult(BaseModel):
    """Structured output for sentiment analysis."""

    sentiment: str
    confidence: float


class UserContext(BaseModel):
    """Schema for memory tests."""

    name: str = ""
    language: str = "en"
    preferences: str = ""


class MyDeps(BaseModel):
    """Schema for deps injection tests."""

    role: str
    company: str


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@patch("basic_agent.agent.get_provider")
def test_agent_simple_text_response(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hello!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", system="Be helpful")
    result = agent.run("Hi")

    assert result == "Hello!"
    mock_provider.chat.assert_called_once()


@patch("basic_agent.agent.get_provider")
def test_agent_tool_use_loop(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # First call: model wants to use a tool
    first_response = ProviderResponse(
        text=None,
        tool_calls=[ToolCall(id="call_1", name="add_numbers", input={"a": 2, "b": 3})],
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    # Second call: model returns final text
    second_response = ProviderResponse(
        text="The sum is 5.",
        tool_calls=[],
        usage=Usage(input_tokens=30, output_tokens=15),
    )
    mock_provider.chat.side_effect = [first_response, second_response]
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", tools=[add_numbers])
    result = agent.run("What is 2 + 3?")

    assert result == "The sum is 5."
    assert mock_provider.chat.call_count == 2


@patch("basic_agent.agent.get_provider")
def test_agent_structured_output(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # Model responds with a tool call matching the output type
    mock_provider.chat.return_value = ProviderResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_so",
                name="SentimentResult",
                input={"sentiment": "positive", "confidence": 0.95},
            )
        ],
        usage=Usage(input_tokens=15, output_tokens=8),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", output_type=SentimentResult)
    result = agent.run("I love this product!")

    assert isinstance(result, SentimentResult)
    assert result.sentiment == "positive"
    assert result.confidence == 0.95


@patch("basic_agent.agent.get_provider")
def test_agent_unknown_tool(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # First response: calls a tool that doesn't exist
    first_response = ProviderResponse(
        text=None,
        tool_calls=[ToolCall(id="call_bad", name="nonexistent_tool", input={})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    # Second response: model gives up
    second_response = ProviderResponse(
        text="I couldn't find that tool.",
        tool_calls=[],
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    mock_provider.chat.side_effect = [first_response, second_response]
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic")
    result = agent.run("Do something")

    assert "couldn't find" in result.lower()


@patch("basic_agent.agent.get_provider")
def test_agent_max_iterations(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # Always return tool calls to trigger max iterations
    mock_provider.chat.return_value = ProviderResponse(
        text="Still working...",
        tool_calls=[ToolCall(id="call_loop", name="add_numbers", input={"a": 1, "b": 1})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", tools=[add_numbers], max_iterations=3)
    result = agent.run("Keep going")

    assert mock_provider.chat.call_count == 3


@patch("basic_agent.agent.get_provider")
def test_agent_empty_response(mock_get_provider):
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text=None,
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=0),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic")
    result = agent.run("Hi")

    assert result == ""


# ---- Jinja2 deps rendering tests ----


@patch("basic_agent.agent.get_provider")
def test_jinja2_deps_rendering(mock_get_provider):
    """Verify that {{deps.role}} and {{deps.company}} render correctly."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hello!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(
        provider="anthropic",
        system="You are a {{deps.role}} for {{deps.company}}.",
    )
    result = agent.run(
        "Hi",
        deps=MyDeps(role="support agent", company="Acme Corp"),
    )

    assert result == "Hello!"
    # Check the system prompt passed to provider.chat
    call_kwargs = mock_provider.chat.call_args
    assert call_kwargs.kwargs["system"] == "You are a support agent for Acme Corp."


@patch("basic_agent.agent.get_provider")
def test_plain_system_prompt_unchanged(mock_get_provider):
    """Verify that a plain system prompt (no Jinja2 variables) passes through unchanged."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hi!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", system="Be helpful and concise.")
    result = agent.run("Hello")

    call_kwargs = mock_provider.chat.call_args
    assert call_kwargs.kwargs["system"] == "Be helpful and concise."


@patch("basic_agent.agent.get_provider")
def test_run_no_kwargs_backward_compatible(mock_get_provider):
    """Verify that run() with only a message still works (backward compat)."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="All good!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic")
    result = agent.run("Hi")

    assert result == "All good!"


# ---- Memory ID / system prompt tests ----


@patch("basic_agent.agent.get_provider")
@patch("basic_agent.memory.psycopg")
def test_memory_id_loads_into_system_prompt(mock_psycopg, mock_get_provider):
    """Verify that memory data is prepended to the system prompt."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (
        {"name": "Alice", "language": "en", "preferences": "concise"},
    )
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hi Alice!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    memory = Memory(
        agent_id="test-bot",
        schema=UserContext,
        dsn="postgresql://test",
    )
    agent = Agent(provider="anthropic", system="Be helpful.", memory=memory)
    result = agent.run("Hi", memory_id="user-123", memory_update=False)

    assert result == "Hi Alice!"
    # Check that system prompt contains both memory and original prompt
    call_kwargs = mock_provider.chat.call_args
    system = call_kwargs.kwargs["system"]
    assert "<memory>" in system
    assert '"name": "Alice"' in system
    assert "Be helpful." in system


@patch("basic_agent.agent.get_provider")
@patch("basic_agent.memory.psycopg")
def test_memory_id_not_found_runs_without_memory(mock_psycopg, mock_get_provider):
    """Verify graceful handling when memory_id has no stored data."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None  # No memory found
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hello stranger!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    memory = Memory(agent_id="test-bot", schema=UserContext, dsn="postgresql://test")
    agent = Agent(provider="anthropic", system="Be helpful.", memory=memory)
    result = agent.run("Hi", memory_id="new-user", memory_update=False)

    assert result == "Hello stranger!"
    # System prompt should NOT contain <memory> section
    call_kwargs = mock_provider.chat.call_args
    system = call_kwargs.kwargs["system"]
    assert "<memory>" not in system
    assert system == "Be helpful."


@patch("basic_agent.agent.get_provider")
def test_memory_id_without_memory_raises(mock_get_provider):
    """Verify that passing memory_id without a Memory instance raises ValueError."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic")  # No memory configured
    with pytest.raises(ValueError, match="memory_id was provided but no Memory instance"):
        agent.run("Hi", memory_id="user-123")


# ---- Memory update tests ----


@patch("basic_agent.agent.get_provider")
@patch("basic_agent.memory.psycopg")
def test_memory_update_second_llm_call(mock_psycopg, mock_get_provider):
    """Verify that a second LLM call is made to update memory, and memory.put is called."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    # First call: memory.get returns existing data
    mock_cursor.fetchone.return_value = (
        {"name": "Alice", "language": "en", "preferences": ""},
    )
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # First call: main conversation response
    main_response = ProviderResponse(
        text="Got it, I'll keep things concise for you Alice!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=15),
    )
    # Second call: memory update structured output
    memory_update_response = ProviderResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="mem_call_1",
                name="UserContext",
                input={"name": "Alice", "language": "en", "preferences": "concise answers"},
            )
        ],
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    mock_provider.chat.side_effect = [main_response, memory_update_response]
    mock_get_provider.return_value = mock_provider

    memory = Memory(
        agent_id="test-bot",
        schema=UserContext,
        dsn="postgresql://test",
        memory_prompt="Update user preferences based on conversation.",
    )
    agent = Agent(provider="anthropic", system="Be helpful.", memory=memory)
    result = agent.run("I prefer concise answers.", memory_id="user-123", memory_update=True)

    assert result == "Got it, I'll keep things concise for you Alice!"
    # Two provider.chat calls: main loop + memory update
    assert mock_provider.chat.call_count == 2
    # Verify the memory update call used structured output (tool_choice = schema name)
    second_call_kwargs = mock_provider.chat.call_args_list[1]
    assert second_call_kwargs.kwargs["tool_choice"] == "UserContext"
    # Verify memory.put was called (via the DB cursor execute)
    # The put call triggers an INSERT/UPDATE SQL
    put_calls = [
        c for c in mock_cursor.execute.call_args_list
        if c.args and isinstance(c.args[0], str) and "INSERT INTO memory" in c.args[0]
    ]
    assert len(put_calls) >= 1


@patch("basic_agent.agent.get_provider")
@patch("basic_agent.memory.psycopg")
def test_memory_update_false_skips_update(mock_psycopg, mock_get_provider):
    """Verify that no second LLM call is made when memory_update=False."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (
        {"name": "Alice", "language": "en", "preferences": ""},
    )
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hello Alice!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    memory = Memory(agent_id="test-bot", schema=UserContext, dsn="postgresql://test")
    agent = Agent(provider="anthropic", system="Be helpful.", memory=memory)
    result = agent.run("Hi", memory_id="user-123", memory_update=False)

    assert result == "Hello Alice!"
    # Only one provider.chat call (no memory update call)
    assert mock_provider.chat.call_count == 1


# ---- Parallel tool execution tests ----


@patch("basic_agent.agent.get_provider")
def test_agent_parallel_tool_execution(mock_get_provider):
    """Verify that multiple tool calls in one response are executed and results returned in order."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    # First call: model requests two tools at once
    first_response = ProviderResponse(
        text=None,
        tool_calls=[
            ToolCall(id="call_add", name="add_numbers", input={"a": 2, "b": 3}),
            ToolCall(id="call_mul", name="multiply_numbers", input={"a": 4, "b": 5}),
        ],
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    # Second call: model returns final text using both results
    second_response = ProviderResponse(
        text="2+3=5 and 4*5=20",
        tool_calls=[],
        usage=Usage(input_tokens=40, output_tokens=15),
    )
    mock_provider.chat.side_effect = [first_response, second_response]
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", tools=[add_numbers, multiply_numbers])
    result = agent.run("Add 2+3 and multiply 4*5")

    assert result == "2+3=5 and 4*5=20"
    assert mock_provider.chat.call_count == 2

    # Verify tool results were sent back in the correct order (add first, then multiply)
    second_call_args = mock_provider.chat.call_args_list[1]
    messages = second_call_args.kwargs["messages"]
    # The last message should be the user message with tool results
    tool_results_msg = messages[-1]
    assert tool_results_msg["role"] == "user"
    tool_results = tool_results_msg["content"]
    assert len(tool_results) == 2
    assert tool_results[0]["tool_use_id"] == "call_add"
    assert tool_results[0]["content"] == "5"
    assert tool_results[1]["tool_use_id"] == "call_mul"
    assert tool_results[1]["content"] == "20"
