"""
DocuMind Chat Memory
-----------------------
Redis-backed short-term conversation memory, keyed by chat_session_id.

Each session's turns are stored as a Redis list of JSON-encoded
{"role": "user"|"assistant", "content": "..."} objects, in chronological
order. The list is trimmed to the most recent N turns and refreshed
with a TTL on every write, so idle sessions expire automatically.

This is the "hot" cache used to build LLM prompt context. The durable,
permanent record of every message lives in Postgres (ChatSession /
Message tables) — Redis is not the source of truth, it's a fast recent
window.
"""
from __future__ import annotations

import json
import uuid
from typing import Dict, List

import redis

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_client: "redis.Redis | None" = None


def get_redis_client() -> "redis.Redis":
    """Return a process-wide Redis client (lazily created)."""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            decode_responses=True,
        )
    return _client


def _key(session_id: uuid.UUID) -> str:
    return f"chat:{session_id}:turns"


def append_turn(session_id: uuid.UUID, role: str, content: str) -> None:
    """Append one turn (user or assistant message) to a session's history."""
    client = get_redis_client()
    key = _key(session_id)
    entry = json.dumps({"role": role, "content": content})

    client.rpush(key, entry)
    # Keep only the most recent N turns (each Q+A pair = 2 entries)
    max_entries = settings.redis.redis_max_turns * 2
    client.ltrim(key, -max_entries, -1)
    client.expire(key, settings.redis.redis_chat_ttl_seconds)


def get_recent_history(session_id: uuid.UUID) -> List[Dict[str, str]]:
    """Return this session's recent turns, oldest first."""
    client = get_redis_client()
    raw_entries = client.lrange(_key(session_id), 0, -1)
    return [json.loads(entry) for entry in raw_entries]


def clear_session(session_id: uuid.UUID) -> None:
    """Delete a session's cached history (e.g. on explicit reset)."""
    client = get_redis_client()
    client.delete(_key(session_id))