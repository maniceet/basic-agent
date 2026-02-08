# Examples

All examples are in the `examples/` directory. Run with `uv run python examples/<file>.py`.

## simple_chat.py

Minimal agent with no tools. Sends a message, prints the response.

```python
agent = Agent(provider="anthropic", model="claude-sonnet-4-20250514", system="...")
result = agent.run("What is the capital of France?")
print(result.output)  # "The capital of France is Paris."
```

## tool_use.py

Registers a `get_weather(city: str)` tool and lets the agent call it during the conversation loop.

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return weather_data.get(city.lower(), "...")

agent = Agent(tools=[get_weather])
result = agent.run("What's the weather like in London?")
```

## structured_output.py

Defines a `MovieReview` Pydantic model. The agent returns a validated instance instead of raw text.

```python
class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    summary: str
    pros: list
    cons: list

agent = Agent(output_type=MovieReview)
result = agent.run("Review 'Inception'")  # result.output is a MovieReview instance
```

## jinja_template.py

System prompt with Jinja2 `{{deps.field}}` placeholders, injected at runtime via a Pydantic model.

```python
agent = Agent(system="You are a {{deps.role}} for {{deps.company}}.")
result = agent.run("Hello", deps=MyDeps(role="support agent", company="Acme"))
print(result.output)
```

The same agent can be reused with different deps for each call.

## with_memory.py

Standalone Memory usage -- store, retrieve, list, and delete structured data in Redis.

```python
memory = Memory(namespace="support-bot", schema=UserContext, url="redis://localhost:6379")
memory.put("user-123", UserContext(name="Alice", language="fr"))
item = memory.get("user-123")
all_items = memory.list()
memory.delete("user-123")
memory.close()
```
