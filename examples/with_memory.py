"""Agent with persistent memory — auto-loads into system prompt and auto-updates after each run."""

from pydantic import BaseModel

from basic_agent import Agent, Memory


class UserContext(BaseModel):
    """Schema for user memory items."""

    name: str = ""
    language: str = "en"
    preferences: str = ""


# Create a memory instance with a custom update prompt
memory = Memory(
    dsn="postgresql://user:pass@localhost:5432/basic_agent",
    agent_id="support-bot",
    schema=UserContext,
    memory_prompt="Based on the conversation, update the user's name, language, and preferences.",
)

# Create an agent with memory attached
agent = Agent(
    provider="anthropic",
    system="You are a helpful support assistant. Be concise.",
    memory=memory,
)

# First run — no existing memory for this user yet.
# After the run, memory is automatically extracted and stored.
result = agent.run(
    "Hi, I'm Alice. I prefer concise answers in French.",
    memory_id="user-123",
)
print(f"Agent: {result}")

# Verify memory was stored
stored = memory.get("user-123")
if stored:
    print(f"Stored memory: name={stored.name}, language={stored.language}, preferences={stored.preferences}")

# Second run — memory is auto-loaded into the system prompt
result = agent.run(
    "What did I say my name was?",
    memory_id="user-123",
)
print(f"Agent: {result}")

# You can also disable memory updates for read-only runs
result = agent.run(
    "Just a quick question, no need to remember this.",
    memory_id="user-123",
    memory_update=False,
)
print(f"Agent: {result}")

# Clean up
memory.close()
