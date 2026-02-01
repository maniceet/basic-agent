"""Agent with a custom Python tool function."""

from basic_agent import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are a helpful weather assistant.",
    tools=[get_weather],
)

response = agent.run("What's the weather like in London?")
print(response)
