"""Standalone Memory usage — store and retrieve structured data with Redis."""

from pydantic import BaseModel

from basic_agent import Memory


class UserContext(BaseModel):
    """Schema for user memory items."""

    name: str = ""
    language: str = "en"
    preferences: str = ""


# Create a memory instance scoped to a namespace
memory = Memory(
    url="redis://localhost:6379",
    namespace="support-bot",
    schema=UserContext,
)

# Store a memory item
memory.put("user-123", UserContext(name="Alice", language="fr", preferences="concise answers"))
print("Stored memory for user-123")

# Retrieve it back
item = memory.get("user-123")
if item:
    print(f"Retrieved: name={item.name}, language={item.language}, preferences={item.preferences}")

# List all items for this namespace + schema
all_items = memory.list()
print(f"Total items: {len(all_items)}")

# Delete the item
memory.delete("user-123")
print("Deleted user-123")

# Verify deletion
assert memory.get("user-123") is None
print("Confirmed deletion")

# Clean up
memory.close()
