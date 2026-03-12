"""Tests for diarization router endpoints.

Uses real testcontainer PostgreSQL via the global conftest's db_session_factory fixture.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from database import get_db_session
from database.models import Meeting
from routers.diarization import router


@pytest.fixture
def app(db_session_factory):
    """FastAPI app with get_db_session overridden to use test DB."""
    app = FastAPI()
    app.include_router(router, prefix="/api/diarization")

    async def _override():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def meeting_id(db_session_factory):
    """Create a real Meeting row and return its UUID string."""
    async with db_session_factory() as session:
        meeting = Meeting(title="Test Meeting for Diarization")
        session.add(meeting)
        await session.commit()
        await session.refresh(meeting)
        return str(meeting.id)


@pytest.mark.asyncio
async def test_create_job(client, meeting_id):
    resp = await client.post("/api/diarization/jobs", json={"meeting_id": meeting_id})
    assert resp.status_code == 201
    data = resp.json()
    assert data["meeting_id"] == meeting_id
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_list_jobs(client, meeting_id):
    await client.post("/api/diarization/jobs", json={"meeting_id": meeting_id})
    await client.post("/api/diarization/jobs", json={"meeting_id": meeting_id})
    resp = await client.get("/api/diarization/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


@pytest.mark.asyncio
async def test_get_job(client, meeting_id):
    create_resp = await client.post("/api/diarization/jobs", json={"meeting_id": meeting_id})
    job_id = create_resp.json()["job_id"]
    resp = await client.get(f"/api/diarization/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["meeting_id"] == meeting_id


@pytest.mark.asyncio
async def test_get_nonexistent_job(client):
    resp = await client.get("/api/diarization/jobs/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job(client, meeting_id):
    create_resp = await client.post("/api/diarization/jobs", json={"meeting_id": meeting_id})
    job_id = create_resp.json()["job_id"]
    resp = await client.post(f"/api/diarization/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_get_rules(client):
    resp = await client.get("/api/diarization/rules")
    assert resp.status_code == 200
    assert "enabled" in resp.json()


@pytest.mark.asyncio
async def test_update_rules(client):
    resp = await client.put("/api/diarization/rules", json={
        "enabled": True,
        "participant_patterns": ["eric@*"],
        "title_patterns": ["dev weekly*"],
    })
    assert resp.status_code == 200
    assert resp.json()["participant_patterns"] == ["eric@*"]


@pytest.mark.asyncio
async def test_list_speakers(client):
    resp = await client.get("/api/diarization/speakers")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_speaker(client):
    resp = await client.post(
        "/api/diarization/speakers",
        json={"name": "Eric Chen", "email": "eric@co.com"},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "Eric Chen"
