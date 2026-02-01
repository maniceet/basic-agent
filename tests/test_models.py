"""Tests for structured output model conversion and validation."""

import pytest
from pydantic import BaseModel, ValidationError

from basic_agent.models import parse_structured_output, structured_output


class MovieReview(BaseModel):
    """A movie review."""

    title: str
    rating: float
    summary: str


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def test_structured_output_schema():
    schema, model = structured_output(MovieReview)
    assert schema["name"] == "MovieReview"
    assert "parameters" in schema
    assert "properties" in schema["parameters"]
    props = schema["parameters"]["properties"]
    assert "title" in props
    assert "rating" in props
    assert "summary" in props
    assert model is MovieReview


def test_structured_output_description():
    schema, _ = structured_output(MovieReview)
    assert "movie review" in schema["description"].lower()


def test_structured_output_default_description():
    class NoDoc(BaseModel):
        x: int

    schema, _ = structured_output(NoDoc)
    assert "NoDoc" in schema["description"]


def test_parse_structured_output_valid():
    data = {"title": "Inception", "rating": 9.0, "summary": "Great movie"}
    result = parse_structured_output(MovieReview, data)
    assert result.title == "Inception"
    assert result.rating == 9.0
    assert result.summary == "Great movie"


def test_parse_structured_output_invalid():
    data = {"title": "Inception", "rating": "not a number"}
    with pytest.raises(ValidationError):
        parse_structured_output(MovieReview, data)


def test_parse_structured_output_missing_field():
    data = {"title": "Inception"}
    with pytest.raises(ValidationError):
        parse_structured_output(MovieReview, data)


def test_roundtrip_schema_and_parse():
    schema, model = structured_output(UserProfile)
    assert "name" in schema["parameters"]["properties"]
    assert "age" in schema["parameters"]["properties"]

    data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    result = parse_structured_output(model, data)
    assert result.name == "Alice"
    assert result.age == 30
