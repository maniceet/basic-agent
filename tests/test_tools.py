"""Tests for tool registry and schema generation."""

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
