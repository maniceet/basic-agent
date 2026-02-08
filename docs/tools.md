# Tool System

**File**: `src/basic_agent/tools.py`

The tool system lets you register plain Python functions as LLM-callable tools. JSON schemas are auto-generated from type hints and docstrings.

## `@tool` Decorator

```python
from basic_agent import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"
```

What happens:
- Function name becomes the tool `name`
- Docstring becomes the tool `description`
- Type-hinted parameters become the JSON Schema `parameters`
- A `ToolDefinition` is attached to the function as `func._tool_definition`

Pass decorated functions to the `Agent`:

```python
agent = Agent(tools=[get_weather])
```

## Type Mapping

The schema generator converts Python types to JSON Schema:

| Python Type | JSON Schema |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list` | `{"type": "array"}` |
| `list[X]` | `{"type": "array", "items": <schema of X>}` |
| `dict` | `{"type": "object"}` |
| `dict[str, X]` | `{"type": "object", "additionalProperties": <schema of X>}` |
| `Optional[X]` / `X \| None` | schema of X (unwrapped) |
| `Union[X, Y]` | `{"anyOf": [<X>, <Y>]}` |
| `Literal["a", "b"]` | `{"type": "string", "enum": ["a", "b"]}` |
| `enum.Enum` subclass | `{"type": "string", "enum": [values]}` |
| Pydantic `BaseModel` | `model.model_json_schema()` |

Parameters without defaults are marked as `required`.

## Internals

### `ToolDefinition`

Holds a single tool's metadata and execution function:

```python
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema
    func: Callable

    def execute(**kwargs) -> Any   # calls the underlying function
    def to_schema() -> dict        # returns provider-agnostic schema
```

### `ToolRegistry`

Stores all registered tools. The agent creates one internally.

```python
registry = ToolRegistry()
registry.register(func)           # register a function
registry.get("name")              # lookup by name -> ToolDefinition | None
registry.list_tools()             # all ToolDefinition instances
registry.schemas()                # list of provider-agnostic schema dicts
```

### Provider-specific translation

The registry outputs **provider-agnostic** schemas:

```json
{"name": "...", "description": "...", "parameters": {...}}
```

The provider layer (`provider.py`) converts these to SDK-specific formats:
- **Anthropic**: `{"name", "description", "input_schema"}`
- **OpenAI**: `{"type": "function", "function": {"name", "description", "parameters"}}`
