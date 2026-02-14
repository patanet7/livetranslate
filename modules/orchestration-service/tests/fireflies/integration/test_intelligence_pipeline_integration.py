#!/usr/bin/env python3
"""
Meeting Intelligence Pipeline Integration Tests

Tests the full integration of MeetingIntelligenceService with:
- Real database I/O via PostgreSQL
- HTTP endpoint tests using httpx.AsyncClient with ASGITransport
- Pipeline coordinator with real MeetingIntelligenceService
- Dashboard HTML smoke tests

NO MOCKS. Real services, real data flow, real database.
All test output goes to tests/output/ with timestamp format.

Shared DB fixtures (db_session_factory, bot_session_id, intelligence_service)
are provided by tests/fireflies/conftest.py using real PostgreSQL.
"""

import importlib.util
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

# Output directory
OUTPUT_DIR = orchestration_root / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_output_path(test_name: str) -> Path:
    """Get timestamped output file path."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_test_{test_name}_results.log"


def write_output(path: Path, content: str):
    """Write test output to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# Fixtures specific to integration tests
# =============================================================================


@pytest.fixture
async def seeded_service(intelligence_service):
    """Return intelligence service with templates already seeded."""
    await intelligence_service.load_default_templates()
    return intelligence_service


# =============================================================================
# Test FastAPI App Fixture (lightweight, for HTTP endpoint testing)
# =============================================================================


@pytest.fixture
def test_app(intelligence_service):
    """Create a minimal FastAPI app with the insights router for HTTP tests.

    This avoids importing the full main_fastapi.py which triggers heavy
    dependency initialization. Instead we mount just the insights router
    with dependency overrides pointing to our real DB-backed service.
    """
    from fastapi import FastAPI
    from routers.insights import get_intelligence_service, router

    app = FastAPI()
    app.include_router(router, prefix="/api/intelligence")

    # Override the dependency to return our real DB-backed service
    app.dependency_overrides[get_intelligence_service] = lambda: intelligence_service

    return app


@pytest.fixture
async def client(test_app):
    """Create an httpx async client against the test FastAPI app."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# =============================================================================
# Pipeline Auto-Notes Integration (Real Service, No Mocks)
# =============================================================================


class TestAutoNotesPipelineIntegration:
    """Test auto-notes integration with real MeetingIntelligenceService."""

    def test_auto_note_buffer_with_real_service(self, intelligence_service):
        """Test that coordinator auto-note buffer works with real service."""
        from services.meeting_intelligence import MeetingIntelligenceService
        from services.pipeline.adapters.base import ChunkAdapter, TranscriptChunk
        from services.pipeline.config import PipelineConfig
        from services.pipeline.coordinator import TranscriptionPipelineCoordinator

        class DummyAdapter(ChunkAdapter):
            source_type = "test"

            def adapt(self, raw):
                return TranscriptChunk(
                    text=raw.get("text", ""),
                    chunk_id=raw.get("chunk_id", ""),
                    speaker_name=raw.get("speaker_name", "Unknown"),
                )

            def extract_speaker(self, raw_chunk):
                return raw_chunk.get("speaker_name") if isinstance(raw_chunk, dict) else None

        config = PipelineConfig(
            session_id="test-session",
            source_type="test",
            enable_auto_notes=True,
            auto_notes_interval=5,
        )

        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=DummyAdapter(),
            meeting_intelligence=intelligence_service,
        )

        # Verify the service is real (not a mock)
        assert isinstance(coordinator.meeting_intelligence, MeetingIntelligenceService)
        assert coordinator._auto_note_buffer == []
        assert coordinator.config.enable_auto_notes is True
        assert coordinator.config.auto_notes_interval == 5

        output = get_output_path("auto_note_buffer_real_service")
        write_output(
            output,
            "Auto-note buffer integration with real service:\n"
            f"  service type: {type(coordinator.meeting_intelligence).__name__}\n"
            f"  enable_auto_notes: {config.enable_auto_notes}\n"
            f"  auto_notes_interval: {config.auto_notes_interval}\n",
        )

    def test_pipeline_config_auto_notes_disabled_by_default(self):
        """Test that auto-notes are disabled by default in PipelineConfig."""
        from services.pipeline.config import PipelineConfig

        config = PipelineConfig(session_id="test", source_type="test")
        assert config.enable_auto_notes is False

    def test_pipeline_config_with_auto_notes_enabled(self):
        """Test PipelineConfig with auto-notes enabled."""
        from services.pipeline.config import PipelineConfig

        config = PipelineConfig(
            session_id="test",
            source_type="fireflies",
            enable_auto_notes=True,
            auto_notes_interval=10,
            auto_notes_template="auto_note",
            intelligence_llm_backend="ollama",
        )
        assert config.enable_auto_notes is True
        assert config.auto_notes_interval == 10
        assert config.auto_notes_template == "auto_note"
        assert config.intelligence_llm_backend == "ollama"


# =============================================================================
# HTTP Endpoint Tests (Real DB, Real FastAPI Router)
# =============================================================================


class TestNotesHTTPEndpoints:
    """Test notes HTTP endpoints with real DB via httpx."""

    @pytest.mark.asyncio
    async def test_create_manual_note_via_http(self, client, bot_session_id):
        """Test POST /api/intelligence/sessions/{id}/notes creates a note in real DB."""
        response = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/notes",
            json={"content": "HTTP test note", "speaker_name": "TestSpeaker"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["note_type"] == "manual"
        assert data["content"] == "HTTP test note"
        assert data["speaker_name"] == "TestSpeaker"
        assert data["session_id"] == bot_session_id
        assert data["note_id"] is not None

        output = get_output_path("http_create_note")
        write_output(
            output,
            f"HTTP POST note created:\n"
            f"  status: {response.status_code}\n"
            f"  note_id: {data['note_id']}\n"
            f"  content: {data['content']}\n",
        )

    @pytest.mark.asyncio
    async def test_list_notes_via_http(self, client, bot_session_id):
        """Test GET /api/intelligence/sessions/{id}/notes lists notes from real DB."""
        # Create two notes
        await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/notes",
            json={"content": "Note A"},
        )
        await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/notes",
            json={"content": "Note B"},
        )

        response = await client.get(f"/api/intelligence/sessions/{bot_session_id}/notes")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["notes"]) == 2

        contents = [n["content"] for n in data["notes"]]
        assert "Note A" in contents
        assert "Note B" in contents

        output = get_output_path("http_list_notes")
        write_output(
            output,
            f"HTTP GET notes:\n"
            f"  status: {response.status_code}\n"
            f"  count: {data['count']}\n"
            f"  contents: {contents}\n",
        )

    @pytest.mark.asyncio
    async def test_delete_note_via_http(self, client, bot_session_id):
        """Test DELETE /api/intelligence/notes/{id} removes a note from real DB."""
        # Create a note
        create_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/notes",
            json={"content": "To be deleted"},
        )
        note_id = create_resp.json()["note_id"]

        # Delete it
        delete_resp = await client.delete(f"/api/intelligence/notes/{note_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["success"] is True

        # Verify it is gone
        list_resp = await client.get(f"/api/intelligence/sessions/{bot_session_id}/notes")
        assert list_resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_note_returns_404(self, client):
        """Test DELETE /api/intelligence/notes/{id} returns 404 for missing note."""
        fake_id = str(uuid.uuid4())
        response = await client.delete(f"/api/intelligence/notes/{fake_id}")
        assert response.status_code == 404


class TestTemplatesHTTPEndpoints:
    """Test template HTTP endpoints with real DB via httpx."""

    @pytest.mark.asyncio
    async def test_list_templates_after_seeding(self, client, seeded_service):
        """Test GET /api/intelligence/templates lists seeded templates."""
        response = await client.get("/api/intelligence/templates")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 6
        assert len(data["templates"]) >= 6

        names = {t["name"] for t in data["templates"]}
        assert "meeting_summary" in names
        assert "action_items" in names
        assert "auto_note" in names

        output = get_output_path("http_list_templates")
        write_output(
            output,
            f"HTTP GET templates:\n"
            f"  status: {response.status_code}\n"
            f"  count: {data['count']}\n"
            f"  names: {sorted(names)}\n",
        )

    @pytest.mark.asyncio
    async def test_create_custom_template_via_http(self, client):
        """Test POST /api/intelligence/templates creates a custom template."""
        response = await client.post(
            "/api/intelligence/templates",
            json={
                "name": "http_test_template",
                "prompt_template": "Test: $transcript",
                "category": "custom",
                "description": "Created via HTTP test",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "http_test_template"
        assert data["is_builtin"] is False
        assert data["is_active"] is True

        output = get_output_path("http_create_template")
        write_output(
            output,
            f"HTTP POST template created:\n"
            f"  status: {response.status_code}\n"
            f"  name: {data['name']}\n"
            f"  template_id: {data['template_id']}\n",
        )

    @pytest.mark.asyncio
    async def test_get_template_by_name_via_http(self, client, seeded_service):
        """Test GET /api/intelligence/templates/{name} retrieves a template."""
        response = await client.get("/api/intelligence/templates/meeting_summary")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "meeting_summary"
        assert data["category"] == "summary"
        assert data["is_builtin"] is True

    @pytest.mark.asyncio
    async def test_delete_builtin_template_returns_404(self, client, seeded_service):
        """Test DELETE /api/intelligence/templates/{id} fails for builtin templates."""
        # Get the builtin template's ID
        get_resp = await client.get("/api/intelligence/templates/meeting_summary")
        template_id = get_resp.json()["template_id"]

        # Try to delete it - should fail
        del_resp = await client.delete(f"/api/intelligence/templates/{template_id}")
        assert del_resp.status_code == 404  # Not found or is builtin

    @pytest.mark.asyncio
    async def test_delete_custom_template_via_http(self, client):
        """Test DELETE /api/intelligence/templates/{id} deletes a custom template."""
        # Create a custom template
        create_resp = await client.post(
            "/api/intelligence/templates",
            json={
                "name": "to_delete_template",
                "prompt_template": "Delete: $transcript",
                "category": "custom",
            },
        )
        template_id = create_resp.json()["template_id"]

        # Delete it
        del_resp = await client.delete(f"/api/intelligence/templates/{template_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        # Verify it is gone
        get_resp = await client.get(f"/api/intelligence/templates/{template_id}")
        assert get_resp.status_code == 404


class TestAgentHTTPEndpoints:
    """Test agent conversation HTTP endpoints with real DB."""

    @pytest.mark.asyncio
    async def test_create_conversation_via_http(self, client, bot_session_id):
        """Test POST /api/intelligence/sessions/{id}/agent/conversations."""
        response = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={"title": "HTTP Test Conversation"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == bot_session_id
        assert data["title"] == "HTTP Test Conversation"
        assert data["status"] == "active"

        output = get_output_path("http_create_conversation")
        write_output(
            output,
            f"HTTP POST conversation created:\n"
            f"  status: {response.status_code}\n"
            f"  conversation_id: {data['conversation_id']}\n"
            f"  title: {data['title']}\n",
        )

    @pytest.mark.asyncio
    async def test_send_message_via_http(self, client, bot_session_id):
        """Test POST /api/intelligence/agent/conversations/{id}/messages."""
        # Create conversation first
        conv_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={"title": "Message Test"},
        )
        conv_id = conv_resp.json()["conversation_id"]

        # Send a message
        msg_resp = await client.post(
            f"/api/intelligence/agent/conversations/{conv_id}/messages",
            json={"content": "What decisions were made?"},
        )

        assert msg_resp.status_code == 201
        data = msg_resp.json()
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["suggested_queries"] is not None

    @pytest.mark.asyncio
    async def test_send_message_stream_via_http(self, client, bot_session_id):
        """Test POST /api/intelligence/agent/conversations/{id}/messages/stream returns SSE."""
        # Create conversation first
        conv_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={"title": "Stream Test"},
        )
        conv_id = conv_resp.json()["conversation_id"]

        # Send a streaming message
        stream_resp = await client.post(
            f"/api/intelligence/agent/conversations/{conv_id}/messages/stream",
            json={"content": "Summarize the meeting"},
        )

        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")

        # Parse SSE chunks
        body = stream_resp.text
        chunks = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(chunks) >= 1  # At least a final done chunk

        # Last chunk should have done=true
        import json

        last_data = json.loads(chunks[-1][6:])
        assert last_data.get("done") is True

        output = get_output_path("http_streaming_endpoint")
        write_output(
            output,
            f"Streaming SSE endpoint test:\n"
            f"  status: {stream_resp.status_code}\n"
            f"  content-type: {stream_resp.headers.get('content-type')}\n"
            f"  chunks: {len(chunks)}\n"
            f"  last_chunk_done: {last_data.get('done')}\n",
        )

    @pytest.mark.asyncio
    async def test_get_conversation_with_history_via_http(self, client, bot_session_id):
        """Test GET /api/intelligence/agent/conversations/{id} with messages."""
        # Create conversation and send a message
        conv_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={"title": "History Test"},
        )
        conv_id = conv_resp.json()["conversation_id"]

        await client.post(
            f"/api/intelligence/agent/conversations/{conv_id}/messages",
            json={"content": "Summarize the meeting"},
        )

        # Get conversation with history
        history_resp = await client.get(f"/api/intelligence/agent/conversations/{conv_id}")

        assert history_resp.status_code == 200
        data = history_resp.json()
        assert len(data["messages"]) == 2  # user + assistant
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

        output = get_output_path("http_conversation_history")
        write_output(
            output,
            f"HTTP GET conversation history:\n"
            f"  status: {history_resp.status_code}\n"
            f"  messages: {len(data['messages'])}\n"
            f"  user: {data['messages'][0]['content']}\n"
            f"  assistant: {data['messages'][1]['content'][:60]}...\n",
        )

    @pytest.mark.asyncio
    async def test_get_suggestions_via_http(self, client, bot_session_id):
        """Test GET /api/intelligence/sessions/{id}/agent/suggestions."""
        response = await client.get(
            f"/api/intelligence/sessions/{bot_session_id}/agent/suggestions"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["queries"]) >= 5
        assert any("action items" in q.lower() for q in data["queries"])

        output = get_output_path("http_suggestions")
        write_output(
            output,
            f"HTTP GET suggestions:\n"
            f"  status: {response.status_code}\n"
            f"  queries: {data['queries']}\n",
        )


# =============================================================================
# Insight Router Integration (Structural)
# =============================================================================


class TestInsightRouterIntegration:
    """Test that the insights router structure is correct."""

    def test_router_imports(self):
        """Test that the insights router can be imported."""
        from routers.insights import router

        assert router is not None
        assert hasattr(router, "routes")

        output = get_output_path("insight_router_imports")
        route_count = len(router.routes)
        write_output(output, f"Insights router imported successfully with {route_count} routes")

    def test_router_has_expected_endpoints(self):
        """Test that the router has all expected endpoint paths."""
        from routers.insights import router

        paths = set()
        for route in router.routes:
            path = getattr(route, "path", "")
            if path:
                paths.add(path)

        expected_paths = {
            "/sessions/{session_id}/notes",
            "/sessions/{session_id}/notes/analyze",
            "/notes/{note_id}",
            "/sessions/{session_id}/insights/generate",
            "/sessions/{session_id}/insights",
            "/insights/{insight_id}",
            "/templates",
            "/templates/{template_id}",
            "/sessions/{session_id}/agent/conversations",
            "/agent/conversations/{conversation_id}",
            "/agent/conversations/{conversation_id}/messages",
            "/agent/conversations/{conversation_id}/messages/stream",
            "/sessions/{session_id}/agent/suggestions",
        }

        missing = expected_paths - paths
        assert not missing, f"Missing endpoints: {missing}"

        output = get_output_path("insight_router_endpoints")
        write_output(
            output,
            f"All {len(expected_paths)} expected endpoints found:\n"
            + "\n".join(f"  - {p}" for p in sorted(paths)),
        )


# =============================================================================
# Alembic Migration Validation
# =============================================================================


class TestAlembicMigration:
    """Test that the Alembic migration file is valid."""

    def test_migration_file_exists(self):
        """Test that the meeting intelligence migration exists."""
        migration_path = (
            orchestration_root / "alembic" / "versions" / "004_add_meeting_intelligence_tables.py"
        )
        assert migration_path.exists(), f"Migration file not found: {migration_path}"

    def test_migration_has_upgrade_and_downgrade(self):
        """Test migration has both upgrade() and downgrade() functions."""
        migration_path = (
            orchestration_root / "alembic" / "versions" / "004_add_meeting_intelligence_tables.py"
        )
        spec = importlib.util.spec_from_file_location("migration", migration_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "upgrade"), "Migration missing upgrade()"
        assert hasattr(module, "downgrade"), "Migration missing downgrade()"
        assert module.revision == "004_meeting_intelligence"
        assert module.down_revision == "003_consolidate_glossaries"

        output = get_output_path("alembic_migration_integration")
        write_output(
            output,
            "Alembic migration validated:\n"
            f"  revision: {module.revision}\n"
            f"  down_revision: {module.down_revision}\n"
            "  Has upgrade(): True\n"
            "  Has downgrade(): True\n",
        )


# =============================================================================
# Dashboard Integration
# =============================================================================


class TestDashboardIntegration:
    """Test dashboard has Intelligence tab."""

    def test_dashboard_has_intelligence_tab(self):
        """Test that the dashboard HTML includes the Intelligence tab."""
        dashboard_path = orchestration_root / "static" / "fireflies-dashboard.html"
        assert dashboard_path.exists()

        content = dashboard_path.read_text(encoding="utf-8")

        assert "showTab('intelligence')" in content, "Intelligence tab button not found"
        assert 'id="tab-intelligence"' in content, "Intelligence tab content not found"
        assert "Meeting Notes" in content, "Notes section not found"
        assert "Post-Meeting Insights" in content, "Insights section not found"
        assert "Meeting Q&A Agent" in content, "Agent chat section not found"
        assert "addManualNote()" in content, "addManualNote function reference not found"
        assert "generateInsight()" in content, "generateInsight function reference not found"
        assert "sendAgentMessage()" in content, "sendAgentMessage function reference not found"

        output = get_output_path("dashboard_intelligence_tab")
        write_output(
            output,
            "Dashboard Intelligence tab validated:\n"
            "  - Intelligence tab button present\n"
            "  - Tab content div present\n"
            "  - Notes section present\n"
            "  - Insights section present\n"
            "  - Agent chat section present\n"
            "  - All JS functions referenced\n",
        )
