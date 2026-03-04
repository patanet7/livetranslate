"""
Behavioral tests for DiarizationPipeline — in-process async job queue.

No mocks. Tests exercise real pipeline logic only.
"""

import pytest

from models.diarization import DiarizationJobStatus
from services.diarization.pipeline import DiarizationPipeline


def test_pipeline_init():
    p = DiarizationPipeline(vibevoice_url="http://localhost:8000/v1", max_concurrent=1)
    assert p.max_concurrent == 1
    assert p.active_jobs == {}


def test_create_job():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=42, triggered_by="manual")
    assert job["meeting_id"] == 42
    assert job["status"] == DiarizationJobStatus.queued
    assert "job_id" in job


def test_create_job_with_hotwords():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=42, triggered_by="auto_rule", hotwords=["sprint"])
    assert job["hotwords"] == ["sprint"]


def test_get_job():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=42, triggered_by="manual")
    assert p.get_job(job["job_id"]) is not None


def test_get_nonexistent():
    p = DiarizationPipeline()
    assert p.get_job("nonexistent") is None


def test_cancel_queued():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=42, triggered_by="manual")
    assert p.cancel_job(job["job_id"]) is True
    assert p.get_job(job["job_id"])["status"] == DiarizationJobStatus.cancelled


def test_list_jobs():
    p = DiarizationPipeline()
    p.create_job(meeting_id=1, triggered_by="manual")
    p.create_job(meeting_id=2, triggered_by="auto_rule")
    assert len(p.list_jobs()) == 2


def test_list_jobs_by_status():
    p = DiarizationPipeline()
    p.create_job(meeting_id=1, triggered_by="manual")
    j2 = p.create_job(meeting_id=2, triggered_by="manual")
    p.cancel_job(j2["job_id"])
    assert len(p.list_jobs(status=DiarizationJobStatus.queued)) == 1


def test_create_job_has_created_at():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=10, triggered_by="manual")
    assert job["created_at"] is not None


def test_create_job_no_hotwords():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=10, triggered_by="manual")
    assert job["hotwords"] is None


def test_create_job_with_rule_matched():
    p = DiarizationPipeline()
    rule = {"title_pattern": "standup"}
    job = p.create_job(meeting_id=10, triggered_by="auto_rule", rule_matched=rule)
    assert job["rule_matched"] == rule


def test_cancel_nonexistent_job():
    p = DiarizationPipeline()
    assert p.cancel_job("does-not-exist") is False


def test_cancel_non_queued_job():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=5, triggered_by="manual")
    p.update_status(job["job_id"], DiarizationJobStatus.processing)
    assert p.cancel_job(job["job_id"]) is False


def test_update_status():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=7, triggered_by="manual")
    p.update_status(job["job_id"], DiarizationJobStatus.processing)
    assert p.get_job(job["job_id"])["status"] == DiarizationJobStatus.processing


def test_update_status_sets_completed_at_on_terminal():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=7, triggered_by="manual")
    assert p.get_job(job["job_id"]).get("completed_at") is None
    p.update_status(job["job_id"], DiarizationJobStatus.completed)
    assert p.get_job(job["job_id"])["completed_at"] is not None


def test_update_status_sets_completed_at_on_failed():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=8, triggered_by="manual")
    p.update_status(job["job_id"], DiarizationJobStatus.failed, error_message="boom")
    retrieved = p.get_job(job["job_id"])
    assert retrieved["completed_at"] is not None
    assert retrieved["error_message"] == "boom"


def test_update_status_kwargs_stored():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=9, triggered_by="manual")
    p.update_status(job["job_id"], DiarizationJobStatus.completed, num_speakers_detected=3)
    assert p.get_job(job["job_id"])["num_speakers_detected"] == 3


def test_list_jobs_sorted_by_created_at_desc():
    p = DiarizationPipeline()
    j1 = p.create_job(meeting_id=1, triggered_by="manual")
    j2 = p.create_job(meeting_id=2, triggered_by="manual")
    jobs = p.list_jobs()
    # Most recently created should appear first
    assert jobs[0]["job_id"] == j2["job_id"]
    assert jobs[1]["job_id"] == j1["job_id"]


def test_job_id_is_12_chars():
    p = DiarizationPipeline()
    job = p.create_job(meeting_id=1, triggered_by="manual")
    assert len(job["job_id"]) == 12


def test_active_jobs_stores_all_created():
    p = DiarizationPipeline()
    j1 = p.create_job(meeting_id=1, triggered_by="manual")
    j2 = p.create_job(meeting_id=2, triggered_by="manual")
    assert j1["job_id"] in p.active_jobs
    assert j2["job_id"] in p.active_jobs


def test_list_jobs_empty():
    p = DiarizationPipeline()
    assert p.list_jobs() == []
