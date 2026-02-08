# basic-agent

A reusable Python library for building LLM-powered agents. Install it as a dependency in any downstream project (e.g. an e-commerce chatbot, a support bot, an internal tool).

## What It Does

- **Dual-provider support** -- Connects to Anthropic or OpenAI with a single `provider="anthropic"` / `provider="openai"` flag. The provider layer normalizes requests and responses so the agent loop is provider-agnostic.
- **Tool calling** -- Register plain Python functions as LLM tools with a `@tool` decorator. JSON schemas are auto-generated from type hints. The agent executes tool calls in a loop (with parallel execution) until the LLM produces a final answer.
- **Structured output** -- Pass a Pydantic `BaseModel` as `output_type` and the agent returns a validated instance instead of raw text.
- **Persistent memory** -- Store and retrieve structured data in Redis, validated by a Pydantic schema.
- **Jinja2 system prompts** -- System prompts support `{{deps.field}}` template syntax, injected at runtime via a Pydantic `deps` model.

## Public API

Exported from `basic_agent`:

| Symbol      | Type  | Purpose |
|-------------|-------|---------|
| `Agent`     | class | Core agent -- orchestrates LLM conversations, tools, structured output |
| `RunResult` | dataclass | Stats from the most recent `agent.run()` call (output, usage, provider_calls) |
| `Memory`    | class | Redis-backed persistent memory with Pydantic validation |
| `tool`      | decorator | Marks a function as an agent tool |

## Quick Start

```python
from basic_agent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F, Sunny in {city}"

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are a helpful weather assistant.",
    tools=[get_weather],
)

response = agent.run("What's the weather like in London?")
print(response)
```

## Project Structure

```
basic-agent/
  pyproject.toml
  src/basic_agent/
    __init__.py        -- Public API re-exports
    agent.py           -- Core Agent class and agentic loop
    provider.py        -- Provider abstraction (Anthropic / OpenAI)
    tools.py           -- @tool decorator, schema generation, ToolRegistry
    models.py          -- Structured output (Pydantic model -> tool schema)
    memory.py          -- Redis-backed persistent memory
  examples/
    simple_chat.py
    tool_use.py
    structured_output.py
    jinja_template.py
    with_memory.py
  tests/
    test_agent.py
    test_provider.py
    test_tools.py
    test_models.py
    test_memory.py
```

## Dependencies

Runtime: `anthropic`, `openai`, `pydantic`, `python-dotenv`, `redis`, `jinja2`

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`

Managed with **uv**. Use `uv add <pkg>` / `uv add --dev <pkg>`.

## Environment Variables

See `.env.example`:

```
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
REDIS_URL=redis://localhost:6379
```

## Current Status

- Tool calls: fully implemented and tested
- Structured output: fully implemented and tested
- Memory (Redis): fully implemented and tested
- Jinja2 system prompt deps: fully implemented and tested
- OpenTelemetry tracing: removed (was previously implemented, intentionally stripped in commit d54413c)
- All 62 tests pass across 5 test modules
