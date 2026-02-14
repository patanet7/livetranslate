"""
Glossary Router Tests

Tests for the glossary API endpoints using real PostgreSQL database.
"""

import sys
from pathlib import Path
from uuid import uuid4

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(orchestration_root))

from httpx import ASGITransport, AsyncClient
from routers.glossary import (
    BulkImportRequest,
    CreateEntryRequest,
    CreateGlossaryRequest,
)

# =============================================================================
# Request Model Tests (No database needed)
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_create_glossary_request_minimal(self):
        """Test minimal valid request."""
        request = CreateGlossaryRequest(
            name="Test",
            target_languages=["es"],
        )
        assert request.name == "Test"
        assert request.target_languages == ["es"]
        assert request.source_language == "en"  # default
        assert request.is_default is False  # default

    def test_create_glossary_request_full(self):
        """Test full request with all fields."""
        request = CreateGlossaryRequest(
            name="Medical Terms",
            description="Medical terminology glossary",
            domain="medical",
            source_language="en",
            target_languages=["es", "fr", "de"],
            is_default=True,
        )
        assert request.name == "Medical Terms"
        assert request.domain == "medical"
        assert len(request.target_languages) == 3

    def test_create_entry_request(self):
        """Test entry request model."""
        request = CreateEntryRequest(
            source_term="API",
            translations={"es": "API", "fr": "API"},
            context="Application Programming Interface",
            priority=10,
        )
        assert request.source_term == "API"
        assert request.translations["es"] == "API"
        assert request.priority == 10
        assert request.match_whole_word is True  # default

    def test_bulk_import_request(self):
        """Test bulk import request."""
        request = BulkImportRequest(
            entries=[
                CreateEntryRequest(
                    source_term="term1",
                    translations={"es": "término1"},
                ),
                CreateEntryRequest(
                    source_term="term2",
                    translations={"es": "término2"},
                ),
            ]
        )
        assert len(request.entries) == 2


# =============================================================================
# Integration Tests (Using real database)
# =============================================================================


@pytest.fixture
async def async_client(db_session_factory):
    """Create async test client with real PostgreSQL-backed glossary service.

    Uses the shared db_session_factory from fireflies conftest.py to provide
    a real PostgreSQL database. Overrides get_db_session so the glossary router
    uses our async session.
    """
    from fastapi import FastAPI
    from routers.glossary import get_db_session, router

    app = FastAPI()
    app.include_router(router, prefix="/api")

    async def _override_db_session():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGlossaryEndpointsIntegration:
    """Integration tests for glossary CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_create_and_get_glossary(self, async_client):
        """Test creating and retrieving a glossary."""
        # Create
        create_response = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"Test Glossary {uuid4().hex[:8]}",
                "description": "Test description",
                "domain": "tech",
                "target_languages": ["es", "fr"],
            },
        )

        assert create_response.status_code == 201
        data = create_response.json()
        assert "glossary_id" in data
        glossary_id = data["glossary_id"]

        # Get
        get_response = await async_client.get(f"/api/glossaries/{glossary_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == data["name"]

        # Cleanup
        await async_client.delete(f"/api/glossaries/{glossary_id}")

    @pytest.mark.asyncio
    async def test_list_glossaries(self, async_client):
        """Test listing glossaries."""
        # Create two glossaries
        g1 = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"List Test 1 {uuid4().hex[:8]}",
                "target_languages": ["es"],
                "domain": "tech",
            },
        )
        g2 = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"List Test 2 {uuid4().hex[:8]}",
                "target_languages": ["fr"],
                "domain": "medical",
            },
        )

        # List all
        list_response = await async_client.get("/api/glossaries")
        assert list_response.status_code == 200
        all_glossaries = list_response.json()
        assert len(all_glossaries) >= 2

        # Filter by domain
        tech_response = await async_client.get("/api/glossaries?domain=tech")
        assert tech_response.status_code == 200
        tech_glossaries = tech_response.json()
        for g in tech_glossaries:
            assert g["domain"] == "tech"

        # Cleanup
        await async_client.delete(f"/api/glossaries/{g1.json()['glossary_id']}")
        await async_client.delete(f"/api/glossaries/{g2.json()['glossary_id']}")

    @pytest.mark.asyncio
    async def test_update_glossary(self, async_client):
        """Test updating a glossary."""
        # Create
        create_response = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"Update Test {uuid4().hex[:8]}",
                "target_languages": ["es"],
            },
        )
        glossary_id = create_response.json()["glossary_id"]

        # Update
        update_response = await async_client.patch(
            f"/api/glossaries/{glossary_id}",
            json={"name": "Updated Name", "domain": "legal"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Name"
        assert update_response.json()["domain"] == "legal"

        # Cleanup
        await async_client.delete(f"/api/glossaries/{glossary_id}")

    @pytest.mark.asyncio
    async def test_delete_glossary(self, async_client):
        """Test deleting a glossary."""
        # Create
        create_response = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"Delete Test {uuid4().hex[:8]}",
                "target_languages": ["es"],
            },
        )
        glossary_id = create_response.json()["glossary_id"]

        # Delete
        delete_response = await async_client.delete(f"/api/glossaries/{glossary_id}")
        assert delete_response.status_code == 204

        # Verify deleted
        get_response = await async_client.get(f"/api/glossaries/{glossary_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_glossary_not_found(self, async_client):
        """Test 404 for non-existent glossary."""
        fake_id = str(uuid4())
        response = await async_client.get(f"/api/glossaries/{fake_id}")
        assert response.status_code == 404


class TestEntryEndpointsIntegration:
    """Integration tests for glossary entry endpoints."""

    @pytest.mark.asyncio
    async def test_add_and_list_entries(self, async_client):
        """Test adding and listing entries."""
        # Create glossary
        glossary = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"Entry Test {uuid4().hex[:8]}",
                "target_languages": ["es"],
            },
        )
        glossary_id = glossary.json()["glossary_id"]

        # Add entry
        entry_response = await async_client.post(
            f"/api/glossaries/{glossary_id}/entries",
            json={
                "source_term": "database",
                "translations": {"es": "base de datos"},
                "priority": 5,
            },
        )
        assert entry_response.status_code == 201
        assert entry_response.json()["source_term"] == "database"

        # List entries
        list_response = await async_client.get(f"/api/glossaries/{glossary_id}/entries")
        assert list_response.status_code == 200
        entries = list_response.json()
        assert len(entries) >= 1

        # Cleanup
        await async_client.delete(f"/api/glossaries/{glossary_id}")


class TestBulkImportIntegration:
    """Integration tests for bulk import."""

    @pytest.mark.asyncio
    async def test_bulk_import(self, async_client):
        """Test bulk importing entries."""
        # Create glossary
        glossary = await async_client.post(
            "/api/glossaries",
            json={
                "name": f"Bulk Import Test {uuid4().hex[:8]}",
                "target_languages": ["es"],
            },
        )
        glossary_id = glossary.json()["glossary_id"]

        # Bulk import
        import_response = await async_client.post(
            f"/api/glossaries/{glossary_id}/import",
            json={
                "entries": [
                    {"source_term": "API", "translations": {"es": "API"}},
                    {"source_term": "database", "translations": {"es": "base de datos"}},
                    {"source_term": "server", "translations": {"es": "servidor"}},
                ]
            },
        )
        assert import_response.status_code == 200
        data = import_response.json()
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert data["total"] == 3

        # Cleanup
        await async_client.delete(f"/api/glossaries/{glossary_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
