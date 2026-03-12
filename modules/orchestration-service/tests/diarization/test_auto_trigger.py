"""Tests for diarization auto-trigger."""

import uuid

import pytest

from models.diarization import DiarizationRules
from services.diarization.auto_trigger import maybe_trigger_diarization
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline(db_session_factory):
    return DiarizationPipeline(session_factory=db_session_factory)


@pytest.fixture
def enabled_rules():
    return DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        title_patterns=["dev weekly*"],
        min_duration_minutes=5,
    )


@pytest.fixture
def disabled_rules():
    return DiarizationRules(enabled=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_on_participant_match(pipeline, enabled_rules, db_session_factory):
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["meeting_id"] == meeting_id
    assert job["triggered_by"] == "auto_rule"


@pytest.mark.asyncio
async def test_no_trigger_when_disabled(pipeline, disabled_rules, db_session_factory):
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 600,
    }
    job = await maybe_trigger_diarization(meeting, disabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_no_trigger_on_no_match(pipeline, enabled_rules, db_session_factory):
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "random chat",
        "participants": ["alice@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_trigger_on_title_match(pipeline, enabled_rules, db_session_factory):
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "Dev Weekly Sync - March",
        "participants": ["alice@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["triggered_by"] == "auto_rule"


@pytest.mark.asyncio
async def test_job_contains_rule_match_info(pipeline, enabled_rules, db_session_factory):
    """Verify rule_matched metadata is stored on the created job."""
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "standup",
        "participants": ["eric@acme.org"],
        "duration": 900,
        "sentence_count": 20,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["rule_matched"] is not None
    assert job["rule_matched"]["match_type"] == "participant"
    assert job["rule_matched"]["matched_value"] == "eric@acme.org"


@pytest.mark.asyncio
async def test_no_trigger_when_duration_too_short(pipeline, enabled_rules, db_session_factory):
    """Meetings shorter than min_duration_minutes are skipped even if patterns match."""
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 60,  # 1 minute — below the 5-minute threshold
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_job_persisted_in_db(pipeline, enabled_rules, db_session_factory):
    """Created job is retrievable from the DB via the pipeline."""
    meeting_id = await _make_meeting(db_session_factory)
    meeting = {
        "id": meeting_id,
        "title": "Dev Weekly Planning",
        "participants": ["bob@company.com"],
        "duration": 1800,
        "sentence_count": 50,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    fetched = await pipeline.get_job(job["job_id"])
    assert fetched is not None
    assert fetched["job_id"] == job["job_id"]
