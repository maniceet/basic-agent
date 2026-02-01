"""Minimal 'hello world' agent — no tools, just a chat."""

from basic_agent import Agent

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are a friendly assistant. Keep responses concise.",
)

response = agent.run("What is the capital of France?")
print(response)
