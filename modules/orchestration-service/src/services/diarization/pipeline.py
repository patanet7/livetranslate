"""
Diarization Pipeline — in-process async job queue.

Manages the lifecycle of offline diarization jobs without requiring an external
queue broker.  All state is held in-memory; persistence is the responsibility of
the caller (e.g. a router that mirrors job state to the database).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from livetranslate_common.logging import get_logger

from models.diarization import DiarizationJobStatus

logger = get_logger()

# Terminal states after which a job cannot transition further.
_TERMINAL_STATES: frozenset[DiarizationJobStatus] = frozenset(
    {
        DiarizationJobStatus.completed,
        DiarizationJobStatus.failed,
        DiarizationJobStatus.cancelled,
    }
)

# Only queued jobs can be cancelled by the caller.
_CANCELLABLE_STATES: frozenset[DiarizationJobStatus] = frozenset(
    {DiarizationJobStatus.queued}
)


class DiarizationPipeline:
    """In-process job queue for offline speaker diarization.

    Jobs are stored in ``active_jobs`` keyed by a 12-character job ID derived
    from a UUID4.  The pipeline does *not* execute processing itself — it only
    manages job metadata and lifecycle transitions.  A separate worker (e.g. an
    asyncio task or Celery worker) is expected to call :meth:`update_status` as
    processing progresses.

    Args:
        vibevoice_url: Base URL for the VibeVoice-ASR HTTP API.
        max_concurrent: Maximum number of jobs that may be in a non-terminal,
            non-queued state simultaneously.  Reserved for future enforcement.
    """

    def __init__(
        self,
        vibevoice_url: str = "http://localhost:8000/v1",
        max_concurrent: int = 1,
    ) -> None:
        self.vibevoice_url: str = vibevoice_url
        self.max_concurrent: int = max_concurrent
        self.active_jobs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(
        self,
        meeting_id: int,
        triggered_by: str = "manual",
        hotwords: list[str] | None = None,
        rule_matched: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new diarization job in QUEUED state.

        Args:
            meeting_id: Database ID of the meeting to diarize.
            triggered_by: Identity of the actor triggering the job
                (e.g. ``"manual"``, ``"auto_rule"``).
            hotwords: Optional domain-specific hotwords to bias recognition.
            rule_matched: Optional auto-trigger rule configuration that caused
                this job to be created.

        Returns:
            The newly created job dict, also stored in :attr:`active_jobs`.
        """
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "job_id": job_id,
            "meeting_id": meeting_id,
            "triggered_by": triggered_by,
            "status": DiarizationJobStatus.queued,
            "hotwords": hotwords,
            "rule_matched": rule_matched,
            "created_at": datetime.now(timezone.utc),
            "completed_at": None,
            "error_message": None,
        }
        self.active_jobs[job_id] = job
        logger.info(
            "diarization_job_created",
            job_id=job_id,
            meeting_id=meeting_id,
            triggered_by=triggered_by,
        )
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return the job dict for *job_id*, or ``None`` if not found.

        Args:
            job_id: The 12-character job identifier.

        Returns:
            The job dict, or ``None``.
        """
        return self.active_jobs.get(job_id)

    def list_jobs(
        self, status: DiarizationJobStatus | None = None
    ) -> list[dict[str, Any]]:
        """Return all jobs, optionally filtered by status.

        Results are sorted by ``created_at`` descending (newest first).

        Args:
            status: When provided, only jobs with this status are returned.

        Returns:
            List of job dicts sorted by creation time descending.
        """
        jobs = list(self.active_jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j["status"] == status]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a QUEUED job.

        A job can only be cancelled if it exists and is in a cancellable state
        (currently only ``QUEUED``).

        Args:
            job_id: The 12-character job identifier.

        Returns:
            ``True`` if the job was successfully cancelled, ``False`` otherwise.
        """
        job = self.active_jobs.get(job_id)
        if job is None:
            logger.warning("diarization_cancel_not_found", job_id=job_id)
            return False
        if job["status"] not in _CANCELLABLE_STATES:
            logger.warning(
                "diarization_cancel_not_allowed",
                job_id=job_id,
                current_status=str(job["status"]),
            )
            return False
        self.update_status(job_id, DiarizationJobStatus.cancelled)
        return True

    def update_status(
        self,
        job_id: str,
        status: DiarizationJobStatus,
        **kwargs: Any,
    ) -> None:
        """Update the status and optional metadata fields of a job.

        Sets ``completed_at`` automatically when *status* is a terminal state
        and ``completed_at`` has not already been set.

        Args:
            job_id: The 12-character job identifier.
            status: The new lifecycle status.
            **kwargs: Additional fields to merge into the job dict
                (e.g. ``error_message``, ``num_speakers_detected``).

        Raises:
            KeyError: If *job_id* does not exist in :attr:`active_jobs`.
        """
        job = self.active_jobs[job_id]
        previous_status = job["status"]
        job["status"] = status
        job.update(kwargs)

        if status in _TERMINAL_STATES and job.get("completed_at") is None:
            job["completed_at"] = datetime.now(timezone.utc)

        logger.info(
            "diarization_job_status_updated",
            job_id=job_id,
            previous_status=str(previous_status),
            new_status=str(status),
        )
