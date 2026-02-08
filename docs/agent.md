# Agent

**File**: `src/basic_agent/agent.py`

The `Agent` class is the core orchestrator. It manages the conversation loop, tool execution, and structured output.

## Constructor

```python
Agent(
    provider="anthropic",          # "anthropic" or "openai"
    model=None,                    # model name (defaults: claude-haiku-4-5-20251001 / gpt-5.2)
    system="You are a helpful assistant.",
    tools=None,                    # list of @tool-decorated functions
    output_type=None,              # Pydantic BaseModel for structured output
    max_tokens=4096,
    temperature=None,
    max_iterations=10,             # max tool-use loop iterations
)
```

## `agent.run(message, *, deps=None)`

Runs the agent with a user message. Returns either a plain `str` or a validated Pydantic model instance (when `output_type` is set).

### How the loop works

1. Builds the message list (system + user message).
2. Calls the provider's `chat()` with tool schemas.
3. If the response has no tool calls, returns the text response.
4. If the response has tool calls:
   - Checks for structured output tool first (returns validated model if found).
   - Executes all other tool calls **in parallel** via `ThreadPoolExecutor`.
   - Appends tool results to messages and loops back to step 2.
5. Repeats up to `max_iterations` times (prevents infinite loops).

### Parallel tool execution

When the LLM returns multiple tool calls in one response, they are executed concurrently using a thread pool (up to 10 workers). Results are returned in the original tool call order.

### Error handling in tools

- Unknown tool name: returns `"Error: Unknown tool 'name'"` to the LLM instead of crashing.
- Tool execution exception: returns `"Error executing tool 'name': <exception>"` to the LLM.

### Deps injection (Jinja2 templates)

The `system` prompt supports Jinja2 template syntax. Pass a Pydantic model as `deps` at runtime:

```python
agent = Agent(system="You are a {{deps.role}} for {{deps.company}}.")
result = agent.run("Hello", deps=MyDeps(role="support agent", company="Acme"))
```

Uses `jinja2.StrictUndefined` -- missing variables raise errors.

## `agent.last_run`

After calling `run()`, `agent.last_run` is a `RunResult` dataclass:

```python
@dataclass
class RunResult:
    output: Any           # str or Pydantic model instance
    usage: Usage          # total tokens (input_tokens, output_tokens)
    provider_calls: int   # number of chat() API calls made
```

## Provider-specific message formatting

The agent handles the different message formats required by each provider:

- **Anthropic**: tool_use content blocks, tool_result content blocks
- **OpenAI**: raw message objects, tool role messages

This is handled internally by `_build_assistant_content()` and `_build_tool_result()`.
