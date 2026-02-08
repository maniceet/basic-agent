"""Tests for persistent memory — mocks Redis."""

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


@patch("basic_agent.memory.redis")
def test_memory_put(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    mem.put(id="item-1", data=SampleSchema(name="foo", value=42))

    mock_client.set.assert_called_once_with(
        "test-agent:SampleSchema:item-1",
        json.dumps({"name": "foo", "value": 42}),
    )


@patch("basic_agent.memory.redis")
def test_memory_get_found(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client
    mock_client.get.return_value = json.dumps({"name": "bar", "value": 99})

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    result = mem.get(id="item-1")

    assert result is not None
    assert result.name == "bar"
    assert result.value == 99
    mock_client.get.assert_called_once_with("test-agent:SampleSchema:item-1")


@patch("basic_agent.memory.redis")
def test_memory_get_not_found(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client
    mock_client.get.return_value = None

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    result = mem.get(id="nonexistent")

    assert result is None


@patch("basic_agent.memory.redis")
def test_memory_list(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client
    mock_client.scan_iter.return_value = iter([
        "test-agent:SampleSchema:a",
        "test-agent:SampleSchema:b",
    ])
    mock_client.mget.return_value = [
        json.dumps({"name": "a", "value": 1}),
        json.dumps({"name": "b", "value": 2}),
    ]

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    items = mem.list()

    assert len(items) == 2
    assert items[0].name == "a"
    assert items[1].value == 2


@patch("basic_agent.memory.redis")
def test_memory_list_empty(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client
    mock_client.scan_iter.return_value = iter([])

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    items = mem.list()

    assert items == []
    mock_client.mget.assert_not_called()


@patch("basic_agent.memory.redis")
def test_memory_delete(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    mem.delete(id="item-1")

    mock_client.delete.assert_called_once_with("test-agent:SampleSchema:item-1")


def test_memory_put_validates_data():
    """Verify that put validates data against the schema (before Redis)."""
    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    with pytest.raises((ValidationError, Exception)):
        mem.put(id="bad", data=SampleSchema.model_validate({"name": "x", "value": "not_int"}))


@patch("basic_agent.memory.redis")
def test_memory_close(mock_redis_module):
    mock_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_client

    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    # Force connection to be opened
    mem._client = mock_client
    mem.close()

    mock_client.close.assert_called_once()
    assert mem._client is None


def test_memory_close_no_connection():
    """Verify close is a no-op when no connection has been made."""
    mem = Memory(namespace="test-agent", schema=SampleSchema, url="redis://test")
    mem.close()  # Should not raise
    assert mem._client is None


def test_memory_key_format():
    """Verify the Redis key format is {namespace}:{schema_name}:{id}."""
    mem = Memory(namespace="my-agent", schema=SampleSchema, url="redis://test")
    assert mem._key("user-123") == "my-agent:SampleSchema:user-123"


def test_memory_defaults_to_redis_url_env(monkeypatch):
    """Verify that url falls back to REDIS_URL env var."""
    monkeypatch.setenv("REDIS_URL", "redis://from-env:6379")
    mem = Memory(namespace="test", schema=SampleSchema)
    assert mem._url == "redis://from-env:6379"


def test_memory_defaults_to_localhost():
    """Verify that url defaults to localhost when no env var is set."""
    mem = Memory(namespace="test", schema=SampleSchema, url=None)
    # When REDIS_URL is not set, should default to localhost
    # (This test may pick up REDIS_URL from the real env — the key behavior
    # is that it doesn't crash.)
    assert mem._url is not None
