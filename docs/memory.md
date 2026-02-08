# Memory

**File**: `src/basic_agent/memory.py`

Persistent key-value storage backed by Redis. Each item is validated against a Pydantic schema on both read and write.

## Usage

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
memory.put("user-123", UserContext(name="Alice", language="fr"))

# Retrieve (returns validated Pydantic instance or None)
item = memory.get("user-123")

# List all items for this namespace + schema
all_items = memory.list()

# Delete
memory.delete("user-123")

# Close connection
memory.close()
```

## How It Works

### Key format

Redis keys follow the pattern: `{namespace}:{schema_class_name}:{id}`

Example: `support-bot:UserContext:user-123`

### Validation

- **On write (`put`)**: data is validated through `schema.model_validate()` before being serialized to JSON and stored.
- **On read (`get`, `list`)**: raw JSON is deserialized and validated through `schema.model_validate()`.
- Invalid data raises Pydantic `ValidationError`.

### Connection

- **Lazy**: Redis connection is established on first operation, not at construction time.
- **URL resolution**: Constructor `url` param > `REDIS_URL` env var > `redis://localhost:6379`.
- **`close()`**: Closes the Redis connection and resets the client to `None`.

### `list()` implementation

Uses `scan_iter()` with the pattern `{namespace}:{schema_name}:*` to find all matching keys, then `mget()` to retrieve values in bulk.

## Constructor

```python
Memory(
    namespace: str,          # key prefix for scoping (e.g. "support-bot")
    schema: Type[BaseModel], # Pydantic model defining item shape
    url: str | None = None,  # Redis URL (optional)
)
```

## Requirements

- A running Redis instance
- `redis` Python package (included in project dependencies)
