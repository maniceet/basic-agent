"""Agent with Jinja2 system prompt and deps injection."""

from pydantic import BaseModel

from basic_agent import Agent


class MyDeps(BaseModel):
    """Dependencies injected into the system prompt at runtime."""

    role: str
    company: str


# The system prompt uses Jinja2 template syntax.
# {{deps.role}} and {{deps.company}} are replaced at runtime.
agent = Agent(
    provider="anthropic",
    system="You are a {{deps.role}} for {{deps.company}}. Be helpful and concise.",
)

# Pass deps at runtime — different callers can inject different values
result = agent.run(
    "What can you help me with?",
    deps=MyDeps(role="support agent", company="Acme Corp"),
)
print(f"Acme agent: {result.output}")

# Reuse the same agent with different deps
result = agent.run(
    "What can you help me with?",
    deps=MyDeps(role="sales representative", company="Globex Inc"),
)
print(f"Globex agent: {result.output}")
