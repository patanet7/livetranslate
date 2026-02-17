"""
Event queue abstraction for orchestration service.

Provides a minimal interface for publishing events to Redis Streams (or
other backends via future adapters). Publishing failures are logged but
never bubble up to API callers to keep the current synchronous workflow
intact.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from livetranslate_common.logging import get_logger

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - redis optional at import time
    redis = None  # type: ignore

logger = get_logger()


def _env(name: str, default: str) -> str:
    """Helper to fetch environment variables with defaults."""
    return os.getenv(name, default)


DEFAULT_STREAMS: dict[str, str] = {
    "audio_ingest": _env("EVENT_STREAM_AUDIO", "stream:audio-ingest"),
    "audio_results": _env("EVENT_STREAM_AUDIO_RESULTS", "stream:audio-results"),
    "config_sync": _env("EVENT_STREAM_CONFIG", "stream:config-sync"),
    "bot_control": _env("EVENT_STREAM_BOT", "stream:bot-control"),
    "monitoring": _env("EVENT_STREAM_MONITORING", "stream:monitoring"),
    "intelligence": _env("EVENT_STREAM_INTELLIGENCE", "stream:intelligence"),
}


@dataclass
class QueuePublishResult:
    """Return value from publish attempts for diagnostics."""

    succeeded: bool
    stream: str
    message_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EventPublisher:
    """Minimal event publisher with Redis Streams backend."""

    def __init__(
        self,
        redis_url: str | None,
        streams: dict[str, str] | None = None,
        *,
        enabled: bool = True,
        maxlen: int | None = None,
    ):
        self.redis_url = redis_url
        self.streams = streams or DEFAULT_STREAMS
        self.enabled = enabled and bool(redis_url) and redis is not None
        self.maxlen = maxlen or int(_env("EVENT_STREAM_MAXLEN", "1000"))
        self._client: Any = None

        if not self.enabled:
            logger.debug(
                "EventPublisher disabled (redis_url=%s, redis import=%s)",
                redis_url,
                redis is not None,
            )

    async def _get_client(self) -> Any:
        if not self.enabled:
            return None
        if self._client is None:
            assert redis is not None  # for type checkers
            if self.redis_url is None:
                return None
            try:
                self._client = redis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
            except Exception as exc:
                logger.warning("Failed to initialize Redis client: %s", exc)
                self.enabled = False
        return self._client

    def stream(self, alias: str, fallback: str | None = None) -> str:
        """Translate logical stream alias to real stream name."""
        return self.streams.get(alias, fallback or alias)

    async def publish(
        self,
        alias: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        source: str = "orchestration-api",
        stream_override: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueuePublishResult:
        """
        Publish an event to the configured stream.

        Returns QueuePublishResult but never raises exceptions to avoid
        impacting current synchronous flows.
        """
        stream_name = stream_override or self.stream(alias)

        if not self.enabled:
            return QueuePublishResult(
                succeeded=False,
                stream=stream_name,
                error="publisher_disabled",
            )

        client = await self._get_client()
        if client is None:
            return QueuePublishResult(
                succeeded=False,
                stream=stream_name,
                error="redis_unavailable",
            )

        envelope = {
            "schema_version": "v1",
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "source": source,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "payload": payload,
        }
        if metadata:
            envelope["metadata"] = metadata

        try:
            message_id = await client.xadd(
                stream_name,
                {"data": json.dumps(envelope)},
                maxlen=self.maxlen,
                approximate=True,
            )
            logger.debug("Published event %s to %s (%s)", event_type, stream_name, message_id)
            return QueuePublishResult(
                succeeded=True,
                stream=stream_name,
                message_id=message_id,
            )
        except Exception as exc:
            if not hasattr(self, "_publish_warned"):
                logger.warning("Failed to publish event %s to %s: %s", event_type, stream_name, exc)
                self._publish_warned = True
            else:
                logger.debug("Failed to publish event %s to %s: %s", event_type, stream_name, exc)
            return QueuePublishResult(
                succeeded=False,
                stream=stream_name,
                error=str(exc),
            )

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - best effort close
                logger.debug("Failed to close Redis client: %s", exc)
            finally:
                self._client = None
