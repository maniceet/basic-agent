"""Structured output models — converts Pydantic models to tool definitions."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def structured_output(model: Type[T]) -> Tuple[Dict[str, Any], Type[T]]:
    """Convert a Pydantic model into a tool definition for structured output.

    Returns a tuple of (tool_schema, model_class) where:
    - tool_schema is a provider-agnostic tool definition dict
    - model_class is the original Pydantic model for validation

    The tool name is derived from the model class name, and the parameters
    are the model's JSON schema.
    """
    schema = model.model_json_schema()
    tool_schema = {
        "name": model.__name__,
        "description": schema.get("description", f"Structured output: {model.__name__}"),
        "parameters": schema,
    }
    return tool_schema, model


def parse_structured_output(model: Type[T], data: Dict[str, Any]) -> T:
    """Parse and validate raw data through a Pydantic model.

    Args:
        model: The Pydantic model class to validate against.
        data: The raw dict data from the LLM tool call.

    Returns:
        A validated Pydantic model instance.
    """
    return model.model_validate(data)
