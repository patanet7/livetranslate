"""
Diarization Pipeline — database-backed async job queue.

Manages the lifecycle of offline diarization jobs with all state persisted to
the database via helpers in ``db.py``.  A background worker polls for queued
jobs and processes them.
"""

from __future__ import annotations

import asyncio
from typing import Any

from livetranslate_common.logging import get_logger

from services.diarization.db import (
    create_diarization_job,
    get_diarization_job,
    get_next_queued_job,
    list_diarization_jobs,
    update_job_status,
)

logger = get_logger()


class DiarizationPipeline:
    """Database-backed job queue for offline speaker diarization.

    All job state is persisted to the database.  The pipeline delegates CRUD
    operations to the DB helper functions and does not hold in-memory state.

    Args:
        session_factory: Async callable that returns an ``AsyncSession``
            (typically ``DatabaseManager.get_session``).
        vibevoice_url: Base URL for the VibeVoice-ASR HTTP API.
        max_concurrent: Maximum number of concurrent processing jobs
            (reserved for future enforcement).
    """

    def __init__(
        self,
        session_factory,
        vibevoice_url: str = "http://localhost:8000/v1",
        max_concurrent: int = 1,
    ) -> None:
        self.session_factory = session_factory
        self.vibevoice_url: str = vibevoice_url
        self.max_concurrent: int = max_concurrent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_job(
        self,
        meeting_id: str,
        triggered_by: str = "manual",
        hotwords: list[str] | None = None,
        rule_matched: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new diarization job in QUEUED state."""
        async with self.session_factory() as db:
            job = await create_diarization_job(
                db,
                meeting_id=meeting_id,
                triggered_by=triggered_by,
                rule_matched=rule_matched,
            )
        logger.info(
            "diarization_job_created",
            job_id=job["job_id"],
            meeting_id=meeting_id,
            triggered_by=triggered_by,
        )
        return job

    async def get_job(self, job_id: int) -> dict[str, Any] | None:
        """Return the job dict for *job_id*, or ``None`` if not found."""
        async with self.session_factory() as db:
            return await get_diarization_job(db, job_id)

    async def list_jobs(
        self, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return all jobs, optionally filtered by status."""
        async with self.session_factory() as db:
            return await list_diarization_jobs(db, status_filter=status, limit=limit)

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a QUEUED job.

        Returns ``True`` if the job was successfully cancelled, ``False`` otherwise.
        """
        async with self.session_factory() as db:
            job = await get_diarization_job(db, job_id)
            if job is None:
                logger.warning("diarization_cancel_not_found", job_id=job_id)
                return False
            if job["status"] != "queued":
                logger.warning(
                    "diarization_cancel_not_allowed",
                    job_id=job_id,
                    current_status=job["status"],
                )
                return False
            await update_job_status(db, job_id, "cancelled")
        return True

    async def update_status(
        self,
        job_id: int,
        status: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Update the status and optional metadata fields of a job."""
        async with self.session_factory() as db:
            return await update_job_status(db, job_id, status, **kwargs)


async def start_diarization_worker():
    """Background worker that processes queued diarization jobs."""
    from database import get_database_manager

    logger.info("diarization_worker_started")
    while True:
        try:
            db_manager = get_database_manager()
            async with db_manager.get_session() as db:
                job = await get_next_queued_job(db)
                if job:
                    logger.info("processing_diarization_job", job_id=job["job_id"])
                    await update_job_status(db, job["job_id"], "processing")
                    # TODO: actual diarization processing will be added when
                    # whisper service diarization endpoint is available
                    await update_job_status(db, job["job_id"], "completed")
        except Exception as e:
            logger.error("diarization_worker_error", error=str(e))
        await asyncio.sleep(10)  # Poll every 10 seconds
