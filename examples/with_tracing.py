"""Agent with OpenTelemetry tracing enabled."""

from basic_agent import Agent, setup_tracing, tool

# Set up tracing with console exporter (prints spans to stdout)
setup_tracing(service_name="tracing-example")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are a helpful math assistant. Use the calculate tool for math.",
    tools=[calculate],
    trace=True,
)

response = agent.run("What is 42 * 17 + 3?")
print(f"\nAgent response: {response}")
