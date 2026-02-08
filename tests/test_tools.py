"""Tests for tool registry and schema generation."""

import enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from basic_agent.tools import ToolRegistry, tool


def test_tool_decorator_attaches_definition():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    assert hasattr(greet, "_tool_definition")
    defn = greet._tool_definition
    assert defn.name == "greet"
    assert defn.description == "Say hello."


def test_tool_decorator_preserves_function():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add(2, 3) == 5


def test_schema_generation_types():
    @tool
    def process(name: str, count: int, rate: float, active: bool) -> str:
        """Process something."""
        return "done"

    schema = process._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["rate"]["type"] == "number"
    assert props["active"]["type"] == "boolean"


def test_schema_required_params():
    @tool
    def required_and_optional(name: str, greeting: str = "hello") -> str:
        """Test required vs optional."""
        return f"{greeting}, {name}"

    schema = required_and_optional._tool_definition.to_schema()
    assert "name" in schema["parameters"]["required"]
    assert "greeting" not in schema["parameters"].get("required", [])


def test_tool_registry():
    registry = ToolRegistry()

    def my_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    defn = registry.register(my_tool)
    assert defn.name == "my_tool"
    assert registry.get("my_tool") is defn
    assert len(registry.list_tools()) == 1
    assert len(registry.schemas()) == 1


def test_tool_execution():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    defn = multiply._tool_definition
    result = defn.execute(a=3, b=4)
    assert result == 12


# ---- Richer type mapping tests ----


def test_optional_str_parameter():
    @tool
    def greet(name: str, title: Optional[str] = None) -> str:
        """Greet someone."""
        return f"Hello, {title or ''} {name}"

    schema = greet._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    # Optional[str] should produce type "string"
    assert props["title"]["type"] == "string"
    # title has a default, so it should NOT be required
    assert "title" not in schema["parameters"].get("required", [])
    # name is required
    assert "name" in schema["parameters"]["required"]


def test_list_str_parameter():
    @tool
    def process(items: List[str]) -> str:
        """Process items."""
        return ", ".join(items)

    schema = process._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert props["items"]["items"] == {"type": "string"}


def test_dict_str_int_parameter():
    @tool
    def count(scores: Dict[str, int]) -> int:
        """Count scores."""
        return sum(scores.values())

    schema = count._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert props["scores"]["type"] == "object"
    assert props["scores"]["additionalProperties"] == {"type": "integer"}


def test_literal_parameter():
    @tool
    def set_mode(mode: Literal["fast", "slow"]) -> str:
        """Set mode."""
        return mode

    schema = set_mode._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert props["mode"]["type"] == "string"
    assert props["mode"]["enum"] == ["fast", "slow"]


def test_enum_parameter():
    class Color(enum.Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    @tool
    def paint(color: Color) -> str:
        """Paint."""
        return color.value

    schema = paint._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert props["color"]["type"] == "string"
    assert set(props["color"]["enum"]) == {"red", "green", "blue"}


def test_pydantic_model_parameter():
    class Address(BaseModel):
        street: str
        city: str

    @tool
    def ship(address: Address) -> str:
        """Ship to address."""
        return f"Shipped to {address.city}"

    schema = ship._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    addr_schema = props["address"]
    # Should have come from model_json_schema()
    assert addr_schema["type"] == "object"
    assert "street" in addr_schema["properties"]
    assert "city" in addr_schema["properties"]


def test_union_parameter():
    @tool
    def process(value: Union[str, int]) -> str:
        """Process a value."""
        return str(value)

    schema = process._tool_definition.to_schema()
    props = schema["parameters"]["properties"]
    assert "anyOf" in props["value"]
    types = [s["type"] for s in props["value"]["anyOf"]]
    assert "string" in types
    assert "integer" in types
