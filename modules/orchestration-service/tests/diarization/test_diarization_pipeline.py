"""
Behavioral tests for DiarizationPipeline — database-backed async job queue.

All methods on DiarizationPipeline are async and persist state to PostgreSQL.
Tests use the real session-scoped testcontainer DB provided by the global conftest.
"""

import uuid

import pytest

from models.diarization import DiarizationJobStatus
from services.diarization.pipeline import DiarizationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_meeting(db_session_factory) -> str:
    """Insert a bare Meeting row and return its UUID string."""
    from database.models import Meeting

    meeting = Meeting(
        id=uuid.uuid4(),
        title="Test Meeting",
        participants=[],
        source="fireflies",
        status="live",
    )
    async with db_session_factory() as db:
        db.add(meeting)
        await db.commit()
        return str(meeting.id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_init(db_session_factory):
    p = DiarizationPipeline(
        session_factory=db_session_factory,
        vibevoice_url="http://localhost:8000/v1",
        max_concurrent=1,
    )
    assert p.max_concurrent == 1
    assert p.vibevoice_url == "http://localhost:8000/v1"


@pytest.mark.asyncio
async def test_create_job(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    assert job["meeting_id"] == meeting_id
    assert job["status"] == DiarizationJobStatus.queued
    assert job["job_id"] is not None


@pytest.mark.asyncio
async def test_create_job_with_rule_matched(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    rule = {"title_pattern": "standup"}
    job = await p.create_job(
        meeting_id=meeting_id, triggered_by="auto_rule", rule_matched=rule
    )
    assert job["rule_matched"] == rule


@pytest.mark.asyncio
async def test_get_job(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    fetched = await p.get_job(job["job_id"])
    assert fetched is not None
    assert fetched["job_id"] == job["job_id"]


@pytest.mark.asyncio
async def test_get_nonexistent(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    result = await p.get_job(999_999_999)
    assert result is None


@pytest.mark.asyncio
async def test_cancel_queued(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    cancelled = await p.cancel_job(job["job_id"])
    assert cancelled is True
    fetched = await p.get_job(job["job_id"])
    assert fetched["status"] == DiarizationJobStatus.cancelled


@pytest.mark.asyncio
async def test_cancel_nonexistent_job(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    result = await p.cancel_job(999_999_999)
    assert result is False


@pytest.mark.asyncio
async def test_cancel_non_queued_job(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    await p.update_status(job["job_id"], DiarizationJobStatus.processing)
    result = await p.cancel_job(job["job_id"])
    assert result is False


@pytest.mark.asyncio
async def test_list_jobs(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    m1 = await _make_meeting(db_session_factory)
    m2 = await _make_meeting(db_session_factory)
    await p.create_job(meeting_id=m1, triggered_by="manual")
    await p.create_job(meeting_id=m2, triggered_by="auto_rule")
    jobs = await p.list_jobs()
    # At least our two jobs are present (other tests may leave rows)
    assert len(jobs) >= 2


@pytest.mark.asyncio
async def test_list_jobs_by_status(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    m1 = await _make_meeting(db_session_factory)
    m2 = await _make_meeting(db_session_factory)
    j1 = await p.create_job(meeting_id=m1, triggered_by="manual")
    j2 = await p.create_job(meeting_id=m2, triggered_by="manual")
    await p.cancel_job(j2["job_id"])

    queued_jobs = await p.list_jobs(status=DiarizationJobStatus.queued)
    queued_ids = {j["job_id"] for j in queued_jobs}
    assert j1["job_id"] in queued_ids
    assert j2["job_id"] not in queued_ids


@pytest.mark.asyncio
async def test_create_job_has_created_at(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    assert job["created_at"] is not None


@pytest.mark.asyncio
async def test_update_status(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    await p.update_status(job["job_id"], DiarizationJobStatus.processing)
    fetched = await p.get_job(job["job_id"])
    assert fetched["status"] == DiarizationJobStatus.processing


@pytest.mark.asyncio
async def test_update_status_sets_completed_at_on_terminal(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    assert (await p.get_job(job["job_id"])).get("completed_at") is None
    await p.update_status(job["job_id"], DiarizationJobStatus.completed)
    fetched = await p.get_job(job["job_id"])
    assert fetched["completed_at"] is not None


@pytest.mark.asyncio
async def test_update_status_sets_completed_at_on_failed(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    await p.update_status(
        job["job_id"], DiarizationJobStatus.failed, error_message="boom"
    )
    fetched = await p.get_job(job["job_id"])
    assert fetched["completed_at"] is not None
    assert fetched["error_message"] == "boom"


@pytest.mark.asyncio
async def test_update_status_kwargs_stored(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    await p.update_status(
        job["job_id"], DiarizationJobStatus.completed, num_speakers_detected=3
    )
    fetched = await p.get_job(job["job_id"])
    assert fetched["num_speakers_detected"] == 3


@pytest.mark.asyncio
async def test_list_jobs_sorted_by_created_at_desc(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    m1 = await _make_meeting(db_session_factory)
    m2 = await _make_meeting(db_session_factory)
    j1 = await p.create_job(meeting_id=m1, triggered_by="manual")
    j2 = await p.create_job(meeting_id=m2, triggered_by="manual")
    jobs = await p.list_jobs()
    # Most recently created should appear first
    job_ids = [j["job_id"] for j in jobs]
    assert job_ids.index(j2["job_id"]) < job_ids.index(j1["job_id"])


@pytest.mark.asyncio
async def test_job_id_is_integer(db_session_factory):
    p = DiarizationPipeline(session_factory=db_session_factory)
    meeting_id = await _make_meeting(db_session_factory)
    job = await p.create_job(meeting_id=meeting_id, triggered_by="manual")
    assert isinstance(job["job_id"], int)
