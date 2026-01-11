"""
Glossary Router Tests

Tests for the glossary API endpoints.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(orchestration_root))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.glossary import (
    router,
    CreateGlossaryRequest,
    UpdateGlossaryRequest,
    CreateEntryRequest,
    BulkImportRequest,
    TermLookupRequest,
    GlossaryResponse,
    EntryResponse,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockGlossary:
    """Mock glossary for testing."""

    def __init__(
        self,
        glossary_id=None,
        name="Test Glossary",
        description="Test description",
        domain="tech",
        source_language="en",
        target_languages=None,
        is_active=True,
        is_default=False,
        entry_count=0,
    ):
        self.glossary_id = glossary_id or uuid4()
        self.name = name
        self.description = description
        self.domain = domain
        self.source_language = source_language
        self.target_languages = target_languages or ["es", "fr"]
        self.is_active = is_active
        self.is_default = is_default
        self.entry_count = entry_count
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class MockEntry:
    """Mock glossary entry for testing."""

    def __init__(
        self,
        entry_id=None,
        glossary_id=None,
        source_term="API",
        translations=None,
        context=None,
        notes=None,
        case_sensitive=False,
        match_whole_word=True,
        priority=0,
    ):
        self.entry_id = entry_id or uuid4()
        self.glossary_id = glossary_id or uuid4()
        self.source_term = source_term
        self.translations = translations or {"es": "API", "fr": "API"}
        self.context = context
        self.notes = notes
        self.case_sensitive = case_sensitive
        self.match_whole_word = match_whole_word
        self.priority = priority
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class MockGlossaryService:
    """Mock GlossaryService for testing."""

    def __init__(self):
        self.glossaries = {}
        self.entries = {}

    def create_glossary(self, **kwargs):
        glossary = MockGlossary(**kwargs)
        self.glossaries[glossary.glossary_id] = glossary
        return glossary

    def get_glossary(self, glossary_id):
        return self.glossaries.get(glossary_id)

    def list_glossaries(self, domain=None, source_language=None, active_only=True):
        result = list(self.glossaries.values())
        if domain:
            result = [g for g in result if g.domain == domain]
        if source_language:
            result = [g for g in result if g.source_language == source_language]
        if active_only:
            result = [g for g in result if g.is_active]
        return result

    def update_glossary(self, glossary_id, **kwargs):
        glossary = self.glossaries.get(glossary_id)
        if not glossary:
            return None
        for key, value in kwargs.items():
            if value is not None:
                setattr(glossary, key, value)
        return glossary

    def delete_glossary(self, glossary_id):
        if glossary_id in self.glossaries:
            del self.glossaries[glossary_id]
            return True
        return False

    def add_entry(self, glossary_id, **kwargs):
        if glossary_id not in self.glossaries:
            return None
        entry = MockEntry(glossary_id=glossary_id, **kwargs)
        self.entries[entry.entry_id] = entry
        self.glossaries[glossary_id].entry_count += 1
        return entry

    def get_entry(self, entry_id):
        return self.entries.get(entry_id)

    def list_entries(self, glossary_id, target_language=None):
        result = [e for e in self.entries.values() if e.glossary_id == glossary_id]
        if target_language:
            result = [
                e for e in result if target_language in (e.translations or {})
            ]
        return result

    def update_entry(self, entry_id, **kwargs):
        entry = self.entries.get(entry_id)
        if not entry:
            return None
        for key, value in kwargs.items():
            if value is not None:
                setattr(entry, key, value)
        return entry

    def delete_entry(self, entry_id):
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False

    def import_entries(self, glossary_id, entries):
        if glossary_id not in self.glossaries:
            return (0, len(entries))
        successful = 0
        failed = 0
        for entry_data in entries:
            if entry_data.get("source_term") and entry_data.get("translations"):
                self.add_entry(glossary_id, **entry_data)
                successful += 1
            else:
                failed += 1
        return (successful, failed)

    def get_glossary_stats(self, glossary_id):
        glossary = self.glossaries.get(glossary_id)
        if not glossary:
            return None
        return {
            "glossary_id": str(glossary.glossary_id),
            "name": glossary.name,
            "entry_count": glossary.entry_count,
        }

    def find_matching_terms(self, text, glossary_id, target_language, domain=None):
        # Simple mock implementation
        matches = []
        for entry in self.entries.values():
            if entry.source_term.lower() in text.lower():
                translation = entry.translations.get(target_language)
                if translation:
                    start = text.lower().find(entry.source_term.lower())
                    matches.append((
                        entry.source_term,
                        translation,
                        start,
                        start + len(entry.source_term),
                    ))
        return matches

    def get_glossary_terms(
        self, glossary_id, target_language, domain=None, include_default=True
    ):
        terms = {}
        for entry in self.entries.values():
            if glossary_id and entry.glossary_id != glossary_id:
                continue
            translation = entry.translations.get(target_language)
            if translation:
                terms[entry.source_term] = translation
        return terms


@pytest.fixture
def mock_service():
    """Create a mock GlossaryService."""
    return MockGlossaryService()


@pytest.fixture
def test_client(mock_service):
    """Create test client with mocked dependencies."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    # Override the dependency
    from routers.glossary import get_glossary_service

    app.dependency_overrides[get_glossary_service] = lambda: mock_service

    return TestClient(app)


# =============================================================================
# Request Model Tests
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
# Glossary Endpoint Tests
# =============================================================================


class TestGlossaryEndpoints:
    """Test glossary CRUD endpoints."""

    def test_create_glossary(self, test_client, mock_service):
        """Test POST /api/glossaries."""
        response = test_client.post(
            "/api/glossaries",
            json={
                "name": "Test Glossary",
                "description": "Test description",
                "domain": "tech",
                "target_languages": ["es", "fr"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Glossary"
        assert data["domain"] == "tech"
        assert "glossary_id" in data

    def test_list_glossaries_empty(self, test_client, mock_service):
        """Test GET /api/glossaries with no glossaries."""
        response = test_client.get("/api/glossaries")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_glossaries(self, test_client, mock_service):
        """Test GET /api/glossaries."""
        # Create glossaries
        mock_service.create_glossary(name="Glossary 1", target_languages=["es"])
        mock_service.create_glossary(name="Glossary 2", target_languages=["fr"])

        response = test_client.get("/api/glossaries")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_glossaries_filter_domain(self, test_client, mock_service):
        """Test GET /api/glossaries with domain filter."""
        mock_service.create_glossary(
            name="Medical", domain="medical", target_languages=["es"]
        )
        mock_service.create_glossary(
            name="Tech", domain="tech", target_languages=["es"]
        )

        response = test_client.get("/api/glossaries?domain=medical")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["domain"] == "medical"

    def test_get_glossary(self, test_client, mock_service):
        """Test GET /api/glossaries/{id}."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )

        response = test_client.get(f"/api/glossaries/{glossary.glossary_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test"

    def test_get_glossary_not_found(self, test_client, mock_service):
        """Test GET /api/glossaries/{id} with invalid ID."""
        fake_id = uuid4()
        response = test_client.get(f"/api/glossaries/{fake_id}")

        assert response.status_code == 404

    def test_update_glossary(self, test_client, mock_service):
        """Test PATCH /api/glossaries/{id}."""
        glossary = mock_service.create_glossary(
            name="Original", target_languages=["es"]
        )

        response = test_client.patch(
            f"/api/glossaries/{glossary.glossary_id}",
            json={"name": "Updated", "domain": "legal"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"
        assert data["domain"] == "legal"

    def test_delete_glossary(self, test_client, mock_service):
        """Test DELETE /api/glossaries/{id}."""
        glossary = mock_service.create_glossary(
            name="ToDelete", target_languages=["es"]
        )

        response = test_client.delete(f"/api/glossaries/{glossary.glossary_id}")

        assert response.status_code == 204

        # Verify deleted
        assert mock_service.get_glossary(glossary.glossary_id) is None

    def test_get_glossary_stats(self, test_client, mock_service):
        """Test GET /api/glossaries/{id}/stats."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "API"},
        )

        response = test_client.get(f"/api/glossaries/{glossary.glossary_id}/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["entry_count"] == 1


# =============================================================================
# Entry Endpoint Tests
# =============================================================================


class TestEntryEndpoints:
    """Test glossary entry endpoints."""

    def test_create_entry(self, test_client, mock_service):
        """Test POST /api/glossaries/{id}/entries."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )

        response = test_client.post(
            f"/api/glossaries/{glossary.glossary_id}/entries",
            json={
                "source_term": "database",
                "translations": {"es": "base de datos"},
                "priority": 5,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["source_term"] == "database"
        assert data["translations"]["es"] == "base de datos"
        assert data["priority"] == 5

    def test_list_entries(self, test_client, mock_service):
        """Test GET /api/glossaries/{id}/entries."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "API"},
        )
        mock_service.add_entry(
            glossary.glossary_id,
            source_term="database",
            translations={"es": "base de datos"},
        )

        response = test_client.get(f"/api/glossaries/{glossary.glossary_id}/entries")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_update_entry(self, test_client, mock_service):
        """Test PATCH /api/glossaries/{id}/entries/{entry_id}."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        entry = mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "API"},
        )

        response = test_client.patch(
            f"/api/glossaries/{glossary.glossary_id}/entries/{entry.entry_id}",
            json={"translations": {"es": "Interfaz de Programación"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["translations"]["es"] == "Interfaz de Programación"

    def test_delete_entry(self, test_client, mock_service):
        """Test DELETE /api/glossaries/{id}/entries/{entry_id}."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        entry = mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "API"},
        )

        response = test_client.delete(
            f"/api/glossaries/{glossary.glossary_id}/entries/{entry.entry_id}"
        )

        assert response.status_code == 204


# =============================================================================
# Bulk Import Tests
# =============================================================================


class TestBulkImport:
    """Test bulk import functionality."""

    def test_bulk_import(self, test_client, mock_service):
        """Test POST /api/glossaries/{id}/import."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )

        response = test_client.post(
            f"/api/glossaries/{glossary.glossary_id}/import",
            json={
                "entries": [
                    {"source_term": "API", "translations": {"es": "API"}},
                    {"source_term": "database", "translations": {"es": "base de datos"}},
                    {"source_term": "server", "translations": {"es": "servidor"}},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert data["total"] == 3


# =============================================================================
# Term Lookup Tests
# =============================================================================


class TestTermLookup:
    """Test term lookup functionality."""

    def test_lookup_terms(self, test_client, mock_service):
        """Test POST /api/glossaries/{id}/lookup."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "Interfaz de Programación"},
        )

        response = test_client.post(
            f"/api/glossaries/{glossary.glossary_id}/lookup",
            json={
                "text": "The API is used for integration.",
                "target_language": "es",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["match_count"] >= 1

    def test_get_terms_for_translation(self, test_client, mock_service):
        """Test GET /api/glossaries/terms/{target_language}."""
        glossary = mock_service.create_glossary(
            name="Test", target_languages=["es"]
        )
        mock_service.add_entry(
            glossary.glossary_id,
            source_term="API",
            translations={"es": "API"},
        )

        response = test_client.get(
            f"/api/glossaries/terms/es?glossary_id={glossary.glossary_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["target_language"] == "es"
        assert "terms" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
