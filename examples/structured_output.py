"""Agent that returns a Pydantic-validated structured output."""

from pydantic import BaseModel

from basic_agent import Agent


class MovieReview(BaseModel):
    """A structured movie review."""

    title: str
    year: int
    rating: float
    summary: str
    pros: list
    cons: list


agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are a movie critic. Provide structured reviews.",
    output_type=MovieReview,
)

result = agent.run("Review the movie 'Inception' (2010)")
review = result.output
print(f"Title: {review.title}")
print(f"Year: {review.year}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
print(f"Pros: {review.pros}")
print(f"Cons: {review.cons}")
print(f"Usage: {result.usage}")
print(f"Provider calls: {result.provider_calls}")
