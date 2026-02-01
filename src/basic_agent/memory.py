"""Persistent memory layer backed by PostgreSQL + JSONB."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type, TypeVar

import psycopg
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id);
"""


class Memory:
    """Persistent memory storage using PostgreSQL JSONB.

    Stores and retrieves structured memory items validated against a
    Pydantic model schema.

    Args:
        dsn: PostgreSQL connection string. Falls back to DATABASE_URL env var.
        agent_id: Scopes memory items per agent instance.
        schema: A Pydantic BaseModel class defining the shape of memory items.
        memory_prompt: Optional prompt used by the Agent to guide memory updates.
    """

    def __init__(
        self,
        agent_id: str,
        schema: Type[T],
        dsn: Optional[str] = None,
        memory_prompt: Optional[str] = None,
    ) -> None:
        self._dsn = dsn or os.environ.get("DATABASE_URL", "")
        self._agent_id = agent_id
        self._schema: Type[T] = schema
        self._schema_name = schema.__name__
        self._conn: Any = None
        self._initialized = False
        self.memory_prompt = memory_prompt

    def _get_conn(self) -> Any:
        """Lazily establish a database connection."""
        if self._conn is None:
            self._conn = psycopg.connect(self._dsn)
        if not self._initialized:
            self._ensure_table()
            self._initialized = True
        return self._conn

    def _ensure_table(self) -> None:
        """Create the memory table if it doesn't exist."""
        conn = self._conn
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
            cur.execute(_CREATE_INDEX_SQL)
        conn.commit()

    def put(self, id: str, data: T) -> None:
        """Store a memory item, validating through the Pydantic model.

        Args:
            id: Unique identifier for this memory item.
            data: A Pydantic model instance matching the schema.
        """
        # Validate the data matches the schema
        validated = self._schema.model_validate(data.model_dump())
        json_data = validated.model_dump(mode="json")

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory (id, agent_id, schema_name, data, updated_at)
                VALUES (%s, %s, %s, %s::jsonb, now())
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = now()
                """,
                (id, self._agent_id, self._schema_name, json.dumps(json_data)),
            )
        conn.commit()

    def get(self, id: str) -> Optional[T]:
        """Retrieve a memory item by ID.

        Returns a validated Pydantic instance or None if not found.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM memory WHERE id = %s AND agent_id = %s AND schema_name = %s",
                (id, self._agent_id, self._schema_name),
            )
            row = cur.fetchone()

        if row is None:
            return None
        return self._schema.model_validate(row[0])

    def list(self) -> List[T]:
        """List all memory items for this agent and schema."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM memory WHERE agent_id = %s AND schema_name = %s ORDER BY created_at",
                (self._agent_id, self._schema_name),
            )
            rows = cur.fetchall()

        return [self._schema.model_validate(row[0]) for row in rows]

    def delete(self, id: str) -> None:
        """Delete a memory item by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory WHERE id = %s AND agent_id = %s AND schema_name = %s",
                (id, self._agent_id, self._schema_name),
            )
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False
