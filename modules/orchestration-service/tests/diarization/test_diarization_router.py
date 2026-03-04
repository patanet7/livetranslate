"""Tests for diarization router endpoints."""

import pytest
import routers.diarization as diarization_module
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.diarization import router


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset the module-level pipeline singleton between tests."""
    diarization_module._pipeline = None
    yield
    diarization_module._pipeline = None


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/diarization")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_create_job(client):
    resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    assert resp.status_code == 201
    data = resp.json()
    assert data["meeting_id"] == 42
    assert data["status"] == "queued"
    assert "job_id" in data


def test_list_jobs(client):
    client.post("/api/diarization/jobs", json={"meeting_id": 1})
    client.post("/api/diarization/jobs", json={"meeting_id": 2})
    resp = client.get("/api/diarization/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_job(client):
    create_resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    job_id = create_resp.json()["job_id"]
    resp = client.get(f"/api/diarization/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["meeting_id"] == 42


def test_get_nonexistent_job(client):
    resp = client.get("/api/diarization/jobs/nonexistent")
    assert resp.status_code == 404


def test_cancel_job(client):
    create_resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    job_id = create_resp.json()["job_id"]
    resp = client.post(f"/api/diarization/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_get_rules(client):
    resp = client.get("/api/diarization/rules")
    assert resp.status_code == 200
    assert "enabled" in resp.json()


def test_update_rules(client):
    resp = client.put("/api/diarization/rules", json={
        "enabled": True,
        "participant_patterns": ["eric@*"],
        "title_patterns": ["dev weekly*"],
    })
    assert resp.status_code == 200
    assert resp.json()["participant_patterns"] == ["eric@*"]


def test_list_speakers(client):
    resp = client.get("/api/diarization/speakers")
    assert resp.status_code == 200


def test_create_speaker(client):
    resp = client.post("/api/diarization/speakers", json={"name": "Eric Chen", "email": "eric@co.com"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "Eric Chen"
