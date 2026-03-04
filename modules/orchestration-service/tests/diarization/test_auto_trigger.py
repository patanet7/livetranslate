"""Tests for diarization auto-trigger."""

import pytest

from models.diarization import DiarizationRules
from services.diarization.auto_trigger import maybe_trigger_diarization
from services.diarization.pipeline import DiarizationPipeline


@pytest.fixture
def pipeline():
    return DiarizationPipeline()


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


@pytest.mark.asyncio
async def test_trigger_on_participant_match(pipeline, enabled_rules):
    meeting = {
        "id": 42,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["meeting_id"] == 42
    assert job["triggered_by"] == "auto_rule"


@pytest.mark.asyncio
async def test_no_trigger_when_disabled(pipeline, disabled_rules):
    meeting = {
        "id": 42,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 600,
    }
    job = await maybe_trigger_diarization(meeting, disabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_no_trigger_on_no_match(pipeline, enabled_rules):
    meeting = {
        "id": 42,
        "title": "random chat",
        "participants": ["alice@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_trigger_on_title_match(pipeline, enabled_rules):
    meeting = {
        "id": 42,
        "title": "Dev Weekly Sync - March",
        "participants": ["alice@company.com"],
        "duration": 600,
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["triggered_by"] == "auto_rule"


@pytest.mark.asyncio
async def test_job_contains_rule_match_info(pipeline, enabled_rules):
    """Verify rule_matched metadata is stored on the created job."""
    meeting = {
        "id": 99,
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
async def test_no_trigger_when_duration_too_short(pipeline, enabled_rules):
    """Meetings shorter than min_duration_minutes are skipped even if patterns match."""
    meeting = {
        "id": 5,
        "title": "standup",
        "participants": ["eric@company.com"],
        "duration": 60,  # 1 minute — below the 5-minute threshold
        "sentence_count": 10,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is None


@pytest.mark.asyncio
async def test_job_stored_in_pipeline(pipeline, enabled_rules):
    """Created job is accessible from the pipeline's active_jobs dict."""
    meeting = {
        "id": 77,
        "title": "Dev Weekly Planning",
        "participants": ["bob@company.com"],
        "duration": 1800,
        "sentence_count": 50,
    }
    job = await maybe_trigger_diarization(meeting, enabled_rules, pipeline)
    assert job is not None
    assert job["job_id"] in pipeline.active_jobs
