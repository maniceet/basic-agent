"""Tests for the core Agent class — mocks the provider."""

from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from basic_agent.agent import Agent, RunResult
from basic_agent.provider import ProviderResponse, ToolCall, Usage
from basic_agent.tools import tool


class SentimentResult(BaseModel):
    """Structured output for sentiment analysis."""

    sentiment: str
    confidence: float


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

    assert result.output == "Hello!"
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

    assert result.output == "The sum is 5."
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

    assert isinstance(result, RunResult)
    assert isinstance(result.output, SentimentResult)
    assert result.output.sentiment == "positive"
    assert result.output.confidence == 0.95


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

    assert "couldn't find" in result.output.lower()


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

    assert result.output == ""


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

    assert result.output == "Hello!"
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

    assert result.output == "All good!"


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

    assert result.output == "2+3=5 and 4*5=20"
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


# ---- RunResult tests ----


@patch("basic_agent.agent.get_provider")
def test_run_returns_run_result(mock_get_provider):
    """Verify run() returns a RunResult with output, usage, and provider_calls."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"
    mock_provider.chat.return_value = ProviderResponse(
        text="Hello!",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic")
    result = agent.run("Hi")

    assert isinstance(result, RunResult)
    assert result.output == "Hello!"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.provider_calls == 1


@patch("basic_agent.agent.get_provider")
def test_run_result_accumulates_usage(mock_get_provider):
    """Verify RunResult accumulates tokens across tool-use loop iterations."""
    mock_provider = MagicMock()
    mock_provider.provider_name = "anthropic"
    mock_provider.model_name = "claude-sonnet-4-20250514"

    first_response = ProviderResponse(
        text=None,
        tool_calls=[ToolCall(id="call_1", name="add_numbers", input={"a": 2, "b": 3})],
        usage=Usage(input_tokens=20, output_tokens=10),
    )
    second_response = ProviderResponse(
        text="The sum is 5.",
        tool_calls=[],
        usage=Usage(input_tokens=30, output_tokens=15),
    )
    mock_provider.chat.side_effect = [first_response, second_response]
    mock_get_provider.return_value = mock_provider

    agent = Agent(provider="anthropic", tools=[add_numbers])
    result = agent.run("What is 2 + 3?")

    assert result.output == "The sum is 5."
    assert result.usage.input_tokens == 50  # 20 + 30
    assert result.usage.output_tokens == 25  # 10 + 15
    assert result.provider_calls == 2
