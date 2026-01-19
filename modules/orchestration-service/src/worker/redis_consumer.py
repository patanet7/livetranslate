"""Minimal Redis Stream consumer skeleton for background workers.

This is a best-effort reference implementation. Production workers should
extend this class with domain-specific handlers and error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency at import time
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    """Configuration for stream consumption."""

    stream: str
    group: str
    consumer_name: str
    block_ms: int = 1000
    count: int = 10
    idle_ms: int = 60000


class RedisStreamConsumer:
    """Base consumer that reads Redis Streams and invokes a coroutine handler."""

    def __init__(
        self,
        config: ConsumerConfig,
        handler: Callable[[dict[str, str]], Awaitable[None]],
        *,
        redis_url: str | None = None,
    ) -> None:
        if redis is None:
            raise RuntimeError("redis-py not installed; cannot start consumer")

        self.config = config
        self.handler = handler
        resolved_url = redis_url or os.getenv("EVENT_BUS_REDIS_URL", "redis://localhost:6379/0")
        if not resolved_url:
            resolved_url = "redis://localhost:6379/0"
        elif "://" not in resolved_url:
            resolved_url = f"redis://{resolved_url}"

        self.redis_url = resolved_url
        self.client: redis.Redis | None = None  # type: ignore
        self._shutdown = asyncio.Event()

    async def setup(self) -> None:
        self.client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        try:
            await self.client.xgroup_create(
                self.config.stream, self.config.group, id="$", mkstream=True
            )
            logger.info("Created consumer group %s on %s", self.config.group, self.config.stream)
        except redis.ResponseError as exc:  # type: ignore[attr-defined]
            if "BUSYGROUP" in str(exc):
                logger.debug("Consumer group %s already exists", self.config.group)
            else:  # pragma: no cover - unexpected setup error
                raise

    async def run(self) -> None:
        if self.client is None:
            await self.setup()
        assert self.client is not None

        logger.info(
            "Starting Redis consumer %s on stream %s",
            self.config.consumer_name,
            self.config.stream,
        )
        while not self._shutdown.is_set():
            messages = await self.client.xreadgroup(
                self.config.group,
                self.config.consumer_name,
                streams={self.config.stream: ">"},
                count=self.config.count,
                block=self.config.block_ms,
            )
            if not messages:
                continue

            for stream_name, entries in messages:
                for message_id, data in entries:
                    payload = self._decode_payload(data)
                    try:
                        await self.handler(payload)
                        await self.client.xack(stream_name, self.config.group, message_id)
                    except Exception as exc:  # pragma: no cover - handler failure
                        logger.exception("Handler failure for %s: %s", message_id, exc)

    async def shutdown(self) -> None:
        self._shutdown.set()
        if self.client:
            await self.client.close()

    @staticmethod
    def _decode_payload(data: dict[str, str]) -> dict[str, str]:
        if "data" in data:
            try:
                return json.loads(data["data"])
            except json.JSONDecodeError:
                return {"raw": data["data"]}
        return data


async def example_handler(event: dict[str, str]) -> None:
    """Simple handler used for documentation/testing."""
    logger.info("Received event: %s", event)


async def main() -> None:
    """CLI entrypoint for quick testing."""
    consumer = RedisStreamConsumer(
        config=ConsumerConfig(
            stream=os.getenv("EVENT_STREAM_AUDIO", "stream:audio-ingest"),
            group=os.getenv("EVENT_CONSUMER_GROUP", "dev-audio"),
            consumer_name=os.getenv("EVENT_CONSUMER_NAME", "dev-audio-consumer"),
        ),
        handler=example_handler,
    )
    await consumer.run()


if __name__ == "__main__":  # pragma: no cover - CLI helper
    asyncio.run(main())
