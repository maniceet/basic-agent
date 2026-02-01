"""Tool registry and function-to-tool-schema converter."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, get_type_hints

_PYTHON_TYPE_TO_JSON: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: type) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type."""
    json_type = _PYTHON_TYPE_TO_JSON.get(tp)
    if json_type is not None:
        return {"type": json_type}
    # Fallback for unknown types
    return {"type": "string"}


def _build_parameters_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Build a JSON Schema 'parameters' object from a function's type hints."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        prop_schema = _python_type_to_json_schema(hints.get(name, str))
        properties[name] = prop_schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class ToolDefinition:
    """A provider-agnostic tool definition."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], func: Callable[..., Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def execute(self, **kwargs: Any) -> Any:
        """Execute the underlying function with the given arguments."""
        return self.func(**kwargs)

    def to_schema(self) -> Dict[str, Any]:
        """Return the provider-agnostic schema dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Stores registered tool definitions."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, func: Callable[..., Any]) -> ToolDefinition:
        """Register a function as a tool and return its definition."""
        name = func.__name__
        description = (func.__doc__ or "").strip()
        parameters = _build_parameters_schema(func)
        defn = ToolDefinition(name=name, description=description, parameters=parameters, func=func)
        self._tools[name] = defn
        return defn

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas as a list of dicts."""
        return [t.to_schema() for t in self._tools.values()]


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that marks a function as an agent tool.

    The function's name, docstring, and type hints are used to auto-generate
    a provider-agnostic JSON tool schema.

    Usage:
        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"Sunny in {city}"
    """
    name = func.__name__
    description = (func.__doc__ or "").strip()
    parameters = _build_parameters_schema(func)
    defn = ToolDefinition(name=name, description=description, parameters=parameters, func=func)
    # Attach the tool definition to the function for later retrieval
    func._tool_definition = defn  # type: ignore[attr-defined]
    return func
