"""Background heartbeat monitor for orphaned meeting sessions.

Runs periodically (every 60 s by default) to check for active sessions that
haven't received audio for > heartbeat_timeout_s. Marks them as interrupted.

Intended to be started as an asyncio background task alongside the main
FastAPI application lifespan.
"""
from __future__ import annotations

import asyncio

from livetranslate_common.logging import get_logger

from meeting.session_manager import MeetingSessionManager

logger = get_logger()


async def run_heartbeat_monitor(
    session_manager: MeetingSessionManager,
    check_interval_s: int = 60,
) -> None:
    """Periodically detect and interrupt orphaned sessions.

    This coroutine loops indefinitely; cancel it (via the asyncio task) to stop.

    Args:
        session_manager: Manages session DB operations.
        check_interval_s: Seconds between each orphan-detection sweep.
    """
    logger.info("heartbeat_monitor_started", interval_s=check_interval_s)

    while True:
        try:
            orphans = await session_manager.detect_orphans()
            for orphan in orphans:
                await session_manager.mark_interrupted(orphan.id)
                logger.warning(
                    "orphan_session_interrupted",
                    session_id=str(orphan.id),
                    source_type=orphan.source_type,
                )
        except asyncio.CancelledError:
            logger.info("heartbeat_monitor_cancelled")
            raise
        except Exception:
            logger.exception("heartbeat_check_failed")

        await asyncio.sleep(check_interval_s)
