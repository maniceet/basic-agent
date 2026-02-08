# basic-agent

[![codecov](https://codecov.io/gh/maniceet/basic-agent/graph/badge.svg)](https://codecov.io/gh/maniceet/basic-agent)

A reusable Python library for building LLM-powered agents. Connect to Anthropic or OpenAI with a single interface, register Python functions as tools, return structured outputs validated with Pydantic, and persist conversational memory in Redis.

```
pip install basic-agent
```

Requires Python >= 3.9. Dependencies managed with [uv](https://docs.astral.sh/uv/).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Provider Abstraction](#provider-abstraction)
- [Tool Registration](#tool-registration)
- [Structured Output](#structured-output)
- [Persistent Memory](#persistent-memory)
- [Jinja2 System Prompts](#jinja2-system-prompts)
- [Agent API Reference](#agent-api-reference)
- [Project Structure](#project-structure)
- [Development](#development)

---

## Quick Start

```python
from basic_agent import Agent

agent = Agent(
    provider="anthropic",               # or "openai"
    model="claude-sonnet-4-20250514",   # or "gpt-4o", etc.
    system="You are a friendly assistant. Keep responses concise.",
)

result = agent.run("What is the capital of France?")
print(result.output)         # "The capital of France is Paris."
print(result.usage)          # Usage(input_tokens=..., output_tokens=...)
print(result.provider_calls) # 1
```

Set your API key in the environment or a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
REDIS_URL=redis://localhost:6379
```

---

## Provider Abstraction

The library normalizes Anthropic and OpenAI behind a single `Provider` protocol. The agent loop works against one unified response shape — provider-specific quirks stay in the provider layer.

```python
# Anthropic (default)
agent = Agent(provider="anthropic", model="claude-sonnet-4-20250514")

# OpenAI
agent = Agent(provider="openai", model="gpt-4o")
```

Both providers produce a `ProviderResponse` containing:

| Field | Type | Description |
|---|---|---|
| `text` | `str \| None` | Plain text content |
| `tool_calls` | `list[ToolCall]` | Each with `id`, `name`, `input` |
| `usage` | `Usage` | `input_tokens`, `output_tokens` |
| `raw` | `Any` | Original SDK response for debugging |

Tool schemas, tool choice, and message formats are automatically translated to each SDK's expected format.

All provider calls include automatic retry logic with exponential backoff (1s, 2s, 4s) for rate limits (429) and server errors (5xx).

---

## Tool Registration

Decorate any typed Python function with `@tool` to make it available to the agent. The library auto-generates a JSON schema from the function's type hints and docstring.

```python
from basic_agent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "72F, Sunny",
        "london": "58F, Cloudy",
        "tokyo": "68F, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

agent = Agent(
    provider="anthropic",
    system="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = agent.run("What's the weather like in London?")
print(result.output)
```

The decorator extracts:
- Function name as the tool `name`
- Docstring as the tool `description`
- Type-hinted parameters as JSON Schema `parameters`

Supported types: `str`, `int`, `float`, `bool`, `list[X]`, `dict[str, X]`, `Optional[X]`, `Union[X, Y]`, `Literal["a", "b"]`, `enum.Enum` subclasses, and Pydantic `BaseModel` subclasses.

When the LLM returns multiple tool calls in a single response, all tools are executed in parallel using a thread pool (up to 10 concurrent workers).

---

## Structured Output

Force the agent to return a validated Pydantic model instead of free-form text. The library converts the model into a tool definition and forces the LLM to call it, then validates the response.

```python
from pydantic import BaseModel
from basic_agent import Agent

class MovieReview(BaseModel):
    """A structured movie review."""
    title: str
    year: int
    rating: float
    summary: str
    pros: list
    cons: list

agent = Agent(
    provider="anthropic",
    system="You are a movie critic. Provide structured reviews.",
    output_type=MovieReview,
)

result = agent.run("Review the movie 'Inception' (2010)")
review = result.output  # validated MovieReview instance
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Pros: {review.pros}")
```

The `result.output` is a fully validated Pydantic instance, not a dict or raw JSON.

---

## Persistent Memory

Store and retrieve structured data in Redis, validated through a Pydantic schema on every read and write. Memory is a standalone component — use it independently or alongside an agent.

```python
from pydantic import BaseModel
from basic_agent import Memory

class UserContext(BaseModel):
    name: str = ""
    language: str = "en"
    preferences: str = ""

memory = Memory(
    namespace="support-bot",
    schema=UserContext,
    url="redis://localhost:6379",   # optional, defaults to REDIS_URL env var
)

# Store (validates through Pydantic before writing)
memory.put("user-123", UserContext(name="Alice", language="fr", preferences="concise answers"))

# Retrieve — returns a validated Pydantic instance or None
item = memory.get("user-123")

# List all items for this namespace + schema
items = memory.list()

# Delete
memory.delete("user-123")

# Close connection
memory.close()
```

### How memory works

- **Key format**: `{namespace}:{schema_class_name}:{id}` (e.g. `support-bot:UserContext:user-123`)
- **Validation**: All writes validate through `schema.model_validate()` before storing. All reads deserialize and validate back through the model.
- **Lazy connection**: Redis connection is established on first operation, not at construction time.
- **URL resolution**: Constructor `url` param > `REDIS_URL` env var > `redis://localhost:6379`.

---

## Jinja2 System Prompts

System prompts support Jinja2 templates with runtime dependency injection. Define a Pydantic model for your dependencies and pass them at call time.

```python
from pydantic import BaseModel
from basic_agent import Agent

class MyDeps(BaseModel):
    role: str
    company: str

agent = Agent(
    provider="anthropic",
    system="You are a {{deps.role}} for {{deps.company}}. Be helpful and concise.",
)

# Same agent, different context at runtime
result = agent.run(
    "What can you help me with?",
    deps=MyDeps(role="support agent", company="Acme Corp"),
)

result = agent.run(
    "What can you help me with?",
    deps=MyDeps(role="sales representative", company="Globex Inc"),
)
```

---

## Agent API Reference

### Constructor

```python
Agent(
    provider="anthropic",              # "anthropic" or "openai"
    model=None,                        # Model override (uses provider default if None)
    system="You are a helpful assistant.",
    tools=None,                        # List of @tool-decorated functions
    output_type=None,                  # Pydantic BaseModel for structured output
    max_tokens=4096,                   # Max response tokens
    temperature=None,                  # LLM temperature
    max_iterations=10,                 # Max tool-use loop iterations
)
```

### `agent.run()`

```python
result = agent.run(
    message,                           # User message (str)
    deps=None,                         # Pydantic model for Jinja2 template injection
)
```

Returns a `RunResult` dataclass:

```python
@dataclass
class RunResult:
    output: Any           # str or Pydantic model instance
    usage: Usage          # total tokens (input_tokens, output_tokens)
    provider_calls: int   # number of chat() API calls made
```

```python
result = agent.run("Hello")
print(result.output)          # "Hi there!"
print(result.usage)           # Usage(input_tokens=10, output_tokens=5)
print(result.provider_calls)  # 1
```

### Execution flow

1. Render system prompt (Jinja2 templates with `deps` if provided)
2. Call provider with message and tool definitions
3. If tool calls returned: execute tools in parallel, append results, loop back to step 2
4. If `output_type` is set: validate response through the Pydantic model
5. Return `RunResult` with output, accumulated usage, and call count

---

## Project Structure

```
basic-agent/
├── pyproject.toml
├── uv.lock
├── src/
│   └── basic_agent/
│       ├── __init__.py         # Public API: Agent, RunResult, Memory, tool
│       ├── agent.py            # Core agent loop with tool execution
│       ├── provider.py         # Anthropic/OpenAI provider abstraction
│       ├── tools.py            # @tool decorator and schema generation
│       ├── models.py           # Structured output (Pydantic -> tool schema)
│       └── memory.py           # Redis-backed persistent memory
├── examples/
│   ├── simple_chat.py          # Minimal agent
│   ├── tool_use.py             # Agent with custom tools
│   ├── structured_output.py    # Pydantic-validated output
│   ├── jinja_template.py       # Jinja2 system prompt with deps
│   └── with_memory.py          # Persistent memory with Redis
├── docs/                       # Detailed documentation
│   ├── overview.md
│   ├── agent.md
│   ├── tools.md
│   ├── providers.md
│   ├── memory.md
│   ├── structured-output.md
│   └── examples.md
└── tests/
    ├── test_agent.py
    ├── test_provider.py
    ├── test_tools.py
    ├── test_models.py
    └── test_memory.py
```

---

## Development

### Setup

```bash
git clone <repo-url>
cd basic-agent
uv sync
```

### Run tests

```bash
uv run pytest
```

### Run tests with coverage

```bash
uv run pytest --cov=basic_agent --cov-report=term-missing
```
