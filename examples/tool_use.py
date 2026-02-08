"""Agent with a custom Python tool function."""

from basic_agent import Agent, tool
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Output(BaseModel):
    out: List[str]
    temperature: float
    wind_speed: float

@tool
def get_temp(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")
    
@tool
def get_wind_speed(city: str) -> str:
    """Get the current weather detail for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "18 kmph",
        "tokyo": "68°F, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


agent = Agent(
    provider="anthropic",
    system="You are a helpful weather assistant.",
    tools=[get_temp, get_wind_speed],
    output_type=Output
)

result = agent.run("What's the weather like in London?")
print(result.output)
print(f"Usage: {result.usage}")
print(f"Provider calls: {result.provider_calls}")
