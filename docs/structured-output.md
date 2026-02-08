# Structured Output

**File**: `src/basic_agent/models.py`

Forces the LLM to return a response matching a Pydantic model, validated automatically.

## Usage

```python
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
    output_type=MovieReview,
)

review = agent.run("Review the movie 'Inception' (2010)")
# review is a validated MovieReview instance
print(review.title)   # "Inception"
print(review.rating)  # 8.5
```

## How It Works

1. `structured_output(model)` converts the Pydantic model into a tool schema:
   - Tool name = model class name (e.g. `"MovieReview"`)
   - Tool description = from JSON schema or fallback `"Structured output: MovieReview"`
   - Tool parameters = `model.model_json_schema()`

2. The agent forces the LLM to call this tool by setting `tool_choice` to the model name.

3. When the LLM responds with a tool call for the structured output tool, `parse_structured_output()` validates the raw dict through `model.model_validate(data)`.

4. After the first structured output response, `tool_choice` is relaxed to `"auto"` for any subsequent iterations.

## Functions

```python
structured_output(model: Type[BaseModel]) -> Tuple[dict, Type[BaseModel]]
# Returns (tool_schema_dict, model_class)

parse_structured_output(model: Type[BaseModel], data: dict) -> BaseModel
# Validates raw data and returns a model instance
```
