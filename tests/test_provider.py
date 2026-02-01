"""Tests for provider abstraction — mocks both SDKs."""

from unittest.mock import MagicMock, patch

import pytest

from basic_agent.provider import (
    AnthropicProvider,
    OpenAIProvider,
    ProviderResponse,
    ToolCall,
    Usage,
    _to_anthropic_tool_choice,
    _to_anthropic_tools,
    _to_openai_tool_choice,
    _to_openai_tools,
    get_provider,
)


# --- Tool schema translation ---

def test_to_anthropic_tools():
    tools = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
    result = _to_anthropic_tools(tools)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["input_schema"] == {"type": "object"}
    assert "parameters" not in result[0]


def test_to_openai_tools():
    tools = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
    result = _to_openai_tools(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["parameters"] == {"type": "object"}


# --- Tool choice translation ---

def test_anthropic_tool_choice_auto():
    assert _to_anthropic_tool_choice("auto") == {"type": "auto"}
    assert _to_anthropic_tool_choice(None) == {"type": "auto"}


def test_anthropic_tool_choice_any():
    assert _to_anthropic_tool_choice("any") == {"type": "any"}


def test_anthropic_tool_choice_specific():
    result = _to_anthropic_tool_choice("get_weather")
    assert result == {"type": "tool", "name": "get_weather"}


def test_openai_tool_choice_auto():
    assert _to_openai_tool_choice("auto") == "auto"
    assert _to_openai_tool_choice(None) == "auto"


def test_openai_tool_choice_specific():
    result = _to_openai_tool_choice("get_weather")
    assert result == {"type": "function", "function": {"name": "get_weather"}}


# --- Factory ---

def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("invalid")


@patch("basic_agent.provider.anthropic")
def test_get_provider_anthropic(mock_anthropic):
    provider = get_provider("anthropic")
    assert provider.provider_name == "anthropic"
    assert provider.model_name == "claude-sonnet-4-20250514"


@patch("basic_agent.provider.openai")
def test_get_provider_openai(mock_openai):
    provider = get_provider("openai")
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-4o"


# --- AnthropicProvider.chat ---

@patch("basic_agent.provider.anthropic")
def test_anthropic_chat_text_response(mock_anthropic_module):
    mock_client = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    # Build mock response
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello, world!"

    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(model="claude-sonnet-4-20250514")
    result = provider.chat(messages=[{"role": "user", "content": "Hi"}], system="Be nice")

    assert isinstance(result, ProviderResponse)
    assert result.text == "Hello, world!"
    assert result.tool_calls == []
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@patch("basic_agent.provider.anthropic")
def test_anthropic_chat_tool_call(mock_anthropic_module):
    mock_client = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_123"
    tool_block.name = "get_weather"
    tool_block.input = {"city": "London"}

    mock_response = MagicMock()
    mock_response.content = [tool_block]
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 8
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider()
    result = provider.chat(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "London"}


# --- OpenAIProvider.chat ---

@patch("basic_agent.provider.openai")
def test_openai_chat_text_response(mock_openai_module):
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    mock_message = MagicMock()
    mock_message.content = "Hello from OpenAI!"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 12
    mock_response.usage.completion_tokens = 6
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider(model="gpt-4o")
    result = provider.chat(messages=[{"role": "user", "content": "Hi"}], system="Be nice")

    assert result.text == "Hello from OpenAI!"
    assert result.tool_calls == []
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 6


@patch("basic_agent.provider.openai")
def test_openai_chat_tool_call(mock_openai_module):
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    mock_tc = MagicMock()
    mock_tc.id = "call_456"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"city": "Tokyo"}'

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tc]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    result = provider.chat(
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "Tokyo"}
