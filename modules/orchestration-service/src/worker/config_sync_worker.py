"""Config synchronization worker consuming Redis Streams events."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict

from audio.config_sync import (
    ConfigSyncModes,
    get_config_sync_manager,
    get_config_sync_mode,
)
from infrastructure.queue import DEFAULT_STREAMS
from worker.redis_consumer import ConsumerConfig, RedisStreamConsumer

logger = logging.getLogger(__name__)


async def handle_config_event(event: Dict[str, str]) -> None:
    event_type = event.get("event_type")
    payload = event.get("payload", {})

    manager = await get_config_sync_manager()

    if event_type == "SystemSettingsUpdateRequested":
        updates = payload.get("settings", {})
        if updates:
            await manager.update_configuration("orchestration", updates)
            logger.info("Applied system settings update: %s", list(updates.keys()))
    elif event_type == "ServiceSettingsUpdateRequested":
        service_name = payload.get("service_name", "")
        updates = payload.get("settings", {})
        component = (
            "translation"
            if "translation" in service_name
            else "whisper"
            if "whisper" in service_name
            else "orchestration"
        )
        if updates:
            await manager.update_configuration(component, updates)
            logger.info(
                "Applied service settings update for %s (%s)", service_name, component
            )
    elif event_type == "UserSettingsUpdateRequested":
        logger.warning("User settings worker pathway not yet implemented: %s", payload)
    else:
        logger.debug("Unhandled config event type: %s", event_type)


async def main() -> None:
    if get_config_sync_mode() != ConfigSyncModes.WORKER:
        logger.error("CONFIG_SYNC_MODE is not 'worker'; aborting worker startup")
        return

    consumer = RedisStreamConsumer(
        config=ConsumerConfig(
            stream=os.getenv("EVENT_STREAM_CONFIG", DEFAULT_STREAMS["config_sync"]),
            group=os.getenv("EVENT_CONSUMER_GROUP", "config-sync"),
            consumer_name=os.getenv("EVENT_CONSUMER_NAME", "config-sync-worker"),
        ),
        handler=handle_config_event,
    )
    await consumer.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    asyncio.run(main())
