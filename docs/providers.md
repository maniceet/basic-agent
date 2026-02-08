# Providers

**File**: `src/basic_agent/provider.py`

The provider layer abstracts Anthropic and OpenAI behind a common `Provider` protocol. The agent loop works against a normalized `ProviderResponse` -- provider-specific quirks stay in this layer.

## Factory

```python
from basic_agent.provider import get_provider

provider = get_provider("anthropic")              # default model: claude-haiku-4-5-20251001
provider = get_provider("anthropic", "claude-sonnet-4-20250514")
provider = get_provider("openai")                 # default model: gpt-5.2
provider = get_provider("openai", "gpt-4o")
```

## Provider Protocol

```python
class Provider(Protocol):
    def chat(messages, tools, tool_choice, system, max_tokens, temperature) -> ProviderResponse
    provider_name -> str    # "anthropic" or "openai"
    model_name -> str       # the model string
```

## Normalized Response

```python
@dataclass
class ProviderResponse:
    text: str | None                 # plain text content
    tool_calls: list[ToolCall]       # each has id, name, input
    usage: Usage                     # input_tokens, output_tokens
    raw: Any                         # original SDK response

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
```

## Retry Logic

All provider `chat()` calls go through `_retryable_chat()`:
- Retries on: 429 (rate limit), 500, 502, 503, 504 (server errors), connection errors
- Exponential backoff: 1s, 2s, 4s (3 attempts max)
- Non-retryable errors (e.g. 401 auth) propagate immediately

## Tool Choice Translation

| Input | Anthropic | OpenAI |
|---|---|---|
| `None` | `{"type": "auto"}` | `"auto"` |
| `"auto"` | `{"type": "auto"}` | `"auto"` |
| `"any"` | `{"type": "any"}` | -- |
| `"required"` / `"none"` | -- | `"required"` / `"none"` |
| `"ToolName"` | `{"type": "tool", "name": "ToolName"}` | `{"type": "function", "function": {"name": "ToolName"}}` |

## API Key Configuration

Both providers read API keys from environment variables (loaded via `python-dotenv`):
- Anthropic: `ANTHROPIC_API_KEY`
- OpenAI: `OPENAI_API_KEY`
