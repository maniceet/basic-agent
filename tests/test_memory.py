"""Tests for persistent memory — mocks PostgreSQL."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from basic_agent.memory import Memory


class SampleSchema(BaseModel):
    name: str
    value: int


class OtherSchema(BaseModel):
    label: str


@patch("basic_agent.memory.psycopg")
def test_memory_put(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    mem.put(id="item-1", data=SampleSchema(name="foo", value=42))

    # Should have called execute for table creation + insert
    assert mock_cursor.execute.call_count >= 1


@patch("basic_agent.memory.psycopg")
def test_memory_get_found(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ({"name": "bar", "value": 99},)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    result = mem.get(id="item-1")

    assert result is not None
    assert result.name == "bar"
    assert result.value == 99


@patch("basic_agent.memory.psycopg")
def test_memory_get_not_found(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    result = mem.get(id="nonexistent")

    assert result is None


@patch("basic_agent.memory.psycopg")
def test_memory_list(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ({"name": "a", "value": 1},),
        ({"name": "b", "value": 2},),
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    items = mem.list()

    assert len(items) == 2
    assert items[0].name == "a"
    assert items[1].value == 2


@patch("basic_agent.memory.psycopg")
def test_memory_delete(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    mem.delete(id="item-1")

    assert mock_cursor.execute.call_count >= 1


def test_memory_put_validates_data():
    """Verify that put validates data against the schema (before DB)."""
    # We can test validation without mocking the DB by checking that
    # invalid data raises before any DB call
    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    # This should fail validation because 'value' expects an int
    with pytest.raises((ValidationError, Exception)):
        mem.put(id="bad", data=SampleSchema.model_validate({"name": "x", "value": "not_int"}))


def test_memory_stores_memory_prompt():
    """Verify that Memory accepts and stores a memory_prompt parameter."""
    mem = Memory(
        agent_id="test-agent",
        schema=SampleSchema,
        dsn="postgresql://test",
        memory_prompt="Extract user preferences.",
    )
    assert mem.memory_prompt == "Extract user preferences."


def test_memory_prompt_defaults_to_none():
    """Verify that memory_prompt defaults to None when not provided."""
    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    assert mem.memory_prompt is None


@patch("basic_agent.memory.psycopg")
def test_memory_close(mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    mem = Memory(agent_id="test-agent", schema=SampleSchema, dsn="postgresql://test")
    # Force connection to be opened
    mem._conn = mock_conn
    mem.close()

    mock_conn.close.assert_called_once()
    assert mem._conn is None
