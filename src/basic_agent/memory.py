"""Persistent memory layer backed by Redis."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Type, TypeVar

import redis
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Memory:
    """Persistent memory storage using Redis.

    Stores and retrieves structured memory items validated against a
    Pydantic model schema.

    Redis key format: ``{namespace}:{schema_name}:{id}``

    Args:
        namespace: Key prefix for scoping memory items.
        schema: A Pydantic BaseModel class defining the shape of memory items.
        url: Redis connection URL. Falls back to ``REDIS_URL`` env var.
    """

    def __init__(
        self,
        namespace: str,
        schema: Type[T],
        url: Optional[str] = None,
    ) -> None:
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._namespace = namespace
        self._schema: Type[T] = schema
        self._schema_name = schema.__name__
        self._client: Optional[redis.Redis] = None

    def _get_client(self) -> redis.Redis:
        """Lazily establish a Redis connection."""
        if self._client is None:
            self._client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    def _key(self, id: str) -> str:
        """Build the full Redis key for an item."""
        return f"{self._namespace}:{self._schema_name}:{id}"

    def put(self, id: str, data: T) -> None:
        """Store a memory item, validating through the Pydantic model.

        Args:
            id: Unique identifier for this memory item.
            data: A Pydantic model instance matching the schema.
        """
        validated = self._schema.model_validate(data.model_dump())
        json_str = json.dumps(validated.model_dump(mode="json"))
        client = self._get_client()
        client.set(self._key(id), json_str)

    def get(self, id: str) -> Optional[T]:
        """Retrieve a memory item by ID.

        Returns a validated Pydantic instance or None if not found.
        """
        client = self._get_client()
        raw = client.get(self._key(id))
        if raw is None:
            return None
        return self._schema.model_validate(json.loads(raw))

    def list(self) -> List[T]:
        """List all memory items for this namespace and schema."""
        client = self._get_client()
        pattern = f"{self._namespace}:{self._schema_name}:*"
        keys: List[str] = []
        for key in client.scan_iter(match=pattern):
            keys.append(key)
        if not keys:
            return []
        values = client.mget(keys)
        items: List[T] = []
        for v in values:
            if v is not None:
                items.append(self._schema.model_validate(json.loads(v)))
        return items

    def delete(self, id: str) -> None:
        """Delete a memory item by ID."""
        client = self._get_client()
        client.delete(self._key(id))

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
