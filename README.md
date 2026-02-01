# basic-agent

A reusable Python library for building LLM-powered agents. Connect to Anthropic or OpenAI with a single interface, register Python functions as tools, return structured outputs validated with Pydantic, persist conversational memory in PostgreSQL, and trace every LLM call with OpenTelemetry.

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
- [OpenTelemetry Tracing](#opentelemetry-tracing)
- [Agent API Reference](#agent-api-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Test Coverage](#test-coverage)

---

## Quick Start

```python
from basic_agent import Agent

agent = Agent(
    provider="anthropic",               # or "openai"
    model="claude-sonnet-4-20250514",   # or "gpt-4o", etc.
    system="You are a friendly assistant. Keep responses concise.",
)

response = agent.run("What is the capital of France?")
print(response)
```

Set your API key in the environment or a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
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

response = agent.run("What's the weather like in London?")
print(response)
```

The decorator extracts:
- Function name as the tool `name`
- Docstring as the tool `description`
- Type-hinted parameters as JSON Schema `parameters`

Supported types: `str`, `int`, `float`, `bool`, `list`, `dict`.

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

review = agent.run("Review the movie 'Inception' (2010)")
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Pros: {review.pros}")
```

The return value is a fully validated Pydantic instance, not a dict or raw JSON.

---

## Persistent Memory

Store and retrieve per-user context in PostgreSQL using JSONB. Memory is scoped by agent ID and validated through a Pydantic schema on every read and write.

```python
from pydantic import BaseModel
from basic_agent import Agent, Memory

class UserContext(BaseModel):
    name: str = ""
    language: str = "en"
    preferences: str = ""

memory = Memory(
    dsn="postgresql://user:pass@localhost:5432/basic_agent",
    agent_id="support-bot",
    schema=UserContext,
    memory_prompt="Based on the conversation, update the user's name, language, and preferences.",
)

agent = Agent(
    provider="anthropic",
    system="You are a helpful support assistant.",
    memory=memory,
)

# First run — memory is auto-extracted and stored after the conversation
result = agent.run(
    "Hi, I'm Alice. I prefer concise answers in French.",
    memory_id="user-123",
)

# Second run — memory is auto-loaded into the system prompt
result = agent.run(
    "What did I say my name was?",
    memory_id="user-123",
)

# Disable auto-update for read-only runs
result = agent.run(
    "Just a quick question.",
    memory_id="user-123",
    memory_update=False,
)

memory.close()
```

### How memory works

1. **On load**: When `memory_id` is passed to `agent.run()`, the agent retrieves the stored memory and prepends it to the system prompt as `<memory>JSON</memory>`.
2. **On update**: After the main conversation loop, a second LLM call extracts updated context from the conversation and writes it back using structured output forced to the memory schema.
3. **Opt-out**: Set `memory_update=False` to skip the extraction step.

### Memory API

```python
memory.put(id="user-123", data=UserContext(name="Alice", language="fr"))
item = memory.get(id="user-123")       # Returns UserContext or None
items = memory.list()                   # All items for this agent + schema
memory.delete(id="user-123")
memory.close()
```

The database table is auto-created on first use:

```sql
CREATE TABLE IF NOT EXISTS memory (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
```

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

## OpenTelemetry Tracing

Opt-in tracing instruments every LLM call with spans following the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

```python
from basic_agent import Agent, setup_tracing, tool

setup_tracing(service_name="my-app")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = Agent(
    provider="anthropic",
    system="You are a helpful math assistant.",
    tools=[calculate],
    trace=True,
)

response = agent.run("What is 42 * 17 + 3?")
```

### Recorded span data

**Attributes:**

| Attribute | Description |
|---|---|
| `gen_ai.system` | `"anthropic"` or `"openai"` |
| `gen_ai.request.model` | Model name |
| `gen_ai.request.temperature` | Temperature (if set) |
| `gen_ai.request.max_tokens` | Max tokens (if set) |
| `gen_ai.usage.input_tokens` | Total input tokens |
| `gen_ai.usage.output_tokens` | Total output tokens |

**Events:**

| Event | Description |
|---|---|
| `gen_ai.user.message` | User message content |
| `gen_ai.assistant.message` | Assistant response content |
| `gen_ai.tool.call` | Tool name and input |
| `gen_ai.tool.result` | Tool name and output |

By default, `setup_tracing()` uses a console exporter. Pass a custom `exporter` argument to send spans to any OTLP-compatible backend (Jaeger, Grafana, etc.).

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
    memory=None,                       # Memory instance
    trace=False,                       # Enable OpenTelemetry tracing
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
    memory_id=None,                    # Load/store memory for this ID
    memory_update=True,                # Auto-extract memory after run
)
```

Returns a `str` (plain text) or a validated Pydantic model instance if `output_type` is set.

### Execution flow

1. Load memory for `memory_id` (if provided)
2. Render system prompt (Jinja2 templates + memory prepend)
3. Call provider with message and tool definitions
4. If tool calls returned: execute tools in parallel, append results, loop back to step 3
5. If `output_type` is set: validate response through the Pydantic model
6. If `memory_update` is enabled: second LLM call extracts updated memory
7. Return result

---

## Project Structure

```
basic-agent/
├── pyproject.toml
├── uv.lock
├── src/
│   └── basic_agent/
│       ├── __init__.py         # Public API: Agent, Memory, tool, setup_tracing
│       ├── agent.py            # Core agent loop with tool execution and memory
│       ├── provider.py         # Anthropic/OpenAI provider abstraction
│       ├── tools.py            # @tool decorator and schema generation
│       ├── models.py           # Structured output (Pydantic -> tool schema)
│       ├── memory.py           # PostgreSQL JSONB memory layer
│       └── tracing.py          # OpenTelemetry instrumentation
├── examples/
│   ├── simple_chat.py          # Minimal agent
│   ├── tool_use.py             # Agent with custom tools
│   ├── structured_output.py    # Pydantic-validated output
│   ├── jinja_template.py       # Jinja2 system prompt with deps
│   ├── with_memory.py          # Persistent memory
│   └── with_tracing.py         # OpenTelemetry tracing
└── tests/
    ├── test_agent.py
    ├── test_provider.py
    ├── test_tools.py
    ├── test_models.py
    ├── test_memory.py
    └── test_tracing.py
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

---

## Test Coverage

58 tests across 6 test modules. Overall coverage: **96%** (459 statements, 17 missed). All external dependencies (Anthropic SDK, OpenAI SDK, PostgreSQL) are mocked — no live services required to run tests.

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/basic_agent/__init__.py       5      0   100%
src/basic_agent/agent.py        156      9    94%   74, 177, 190, 196, 312, 324, 341-342, 352
src/basic_agent/memory.py        62      0   100%
src/basic_agent/models.py        10      0   100%
src/basic_agent/provider.py     119      4    97%   101, 113, 151, 212
src/basic_agent/tools.py         59      2    97%   25, 38
src/basic_agent/tracing.py       48      2    96%   40, 49
-----------------------------------------------------------
TOTAL                           459     17    96%
```

### Breakdown by module

| Module | Tests | What is covered |
|---|---|---|
| `test_agent.py` | 15 | Text responses, tool-use loop, structured output, unknown tool handling, max iteration safety, Jinja2 template rendering, memory loading/updating, parallel tool execution |
| `test_provider.py` | 14 | Tool schema translation (Anthropic/OpenAI formats), tool choice translation, provider factory, `chat()` for text and tool call responses on both providers |
| `test_memory.py` | 9 | CRUD operations (put/get/list/delete), Pydantic validation on write, `memory_prompt` storage, connection lifecycle |
| `test_tracing.py` | 8 | Tracer setup, span creation with attributes, optional attributes (temperature, max_tokens), user/assistant message events, tool call/result events, token usage recording |
| `test_models.py` | 7 | Pydantic-to-tool-schema conversion, description extraction, validation round-trips, invalid/missing field rejection |
| `test_tools.py` | 5 | `@tool` decorator behavior, type-hint-to-JSON-schema conversion, required vs optional params, registry operations |

### Uncovered lines

The 17 uncovered lines are primarily:

- **OpenAI-specific branches** in `agent.py` (message formatting for OpenAI provider)
- **Memory update triggers** at structured output return and max-iteration exit paths in `agent.py`
- **Fallback paths** in tool schema generation and provider translation for edge-case type mappings
- **Conditional attribute setting** in tracing span creation
