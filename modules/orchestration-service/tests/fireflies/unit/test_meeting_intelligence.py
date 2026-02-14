#!/usr/bin/env python3
"""
Meeting Intelligence Unit Tests

Tests the MeetingIntelligenceService core logic with REAL database I/O:
- Template YAML validation and structure
- Pydantic model validation
- SQLAlchemy model column checks
- Config defaults
- Real DB: template seeding, CRUD, notes, conversations
- Pipeline config/stats behavior

Uses real PostgreSQL database (no SQLite, no mocks).
All test output goes to tests/output/ with timestamp format.

Shared DB fixtures (db_session_factory, bot_session_id, intelligence_service)
are provided by tests/fireflies/conftest.py using real PostgreSQL.
"""

import importlib.util
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

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
# Template YAML Loading Tests
# =============================================================================


class TestInsightTemplatesConfig:
    """Test that the default templates YAML is valid and complete."""

    def test_templates_yaml_exists(self):
        """Test that the insight_templates.yaml file exists."""
        import yaml

        templates_path = orchestration_root / "config" / "insight_templates.yaml"
        assert templates_path.exists(), f"Templates YAML not found at {templates_path}"

        with open(templates_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "templates" in data
        assert len(data["templates"]) >= 6

        output = get_output_path("templates_yaml_exists")
        write_output(
            output,
            f"Templates YAML loaded: {len(data['templates'])} templates\n"
            + "\n".join(f"  - {t['name']} ({t['category']})" for t in data["templates"]),
        )

    def test_templates_have_required_fields(self):
        """Test that each template has all required fields."""
        import yaml

        templates_path = orchestration_root / "config" / "insight_templates.yaml"
        with open(templates_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        required_fields = {"name", "category", "prompt_template"}
        results = []

        for tmpl in data["templates"]:
            missing = required_fields - set(tmpl.keys())
            assert not missing, f"Template '{tmpl.get('name', '?')}' missing fields: {missing}"
            results.append(f"  OK: {tmpl['name']} has all required fields")

        output = get_output_path("templates_required_fields")
        write_output(output, "All templates have required fields:\n" + "\n".join(results))

    def test_templates_have_transcript_variable(self):
        """Test that each template uses the transcript variable."""
        import yaml

        templates_path = orchestration_root / "config" / "insight_templates.yaml"
        with open(templates_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for tmpl in data["templates"]:
            assert (
                "$transcript" in tmpl["prompt_template"]
            ), f"Template '{tmpl['name']}' missing $transcript variable"

    def test_auto_note_template_exists(self):
        """Test that the auto_note template is present."""
        import yaml

        templates_path = orchestration_root / "config" / "insight_templates.yaml"
        with open(templates_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        auto_note = [t for t in data["templates"] if t["name"] == "auto_note"]
        assert len(auto_note) == 1, "auto_note template not found"
        assert auto_note[0]["category"] == "summary"


# =============================================================================
# Config Tests
# =============================================================================


class TestMeetingIntelligenceConfig:
    """Test the MeetingIntelligenceSettings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        from config import MeetingIntelligenceSettings

        settings = MeetingIntelligenceSettings()

        assert settings.enabled is True
        assert settings.auto_notes_enabled is True
        assert settings.auto_notes_interval_sentences == 10
        assert settings.auto_notes_template == "auto_note"
        assert settings.default_llm_backend == "ollama"
        assert settings.default_temperature == 0.3
        assert settings.default_max_tokens == 1024
        assert settings.max_transcript_chars_for_insight == 100000

        output = get_output_path("intelligence_config_defaults")
        write_output(
            output,
            f"Config defaults valid:\n"
            f"  enabled={settings.enabled}\n"
            f"  auto_notes_enabled={settings.auto_notes_enabled}\n"
            f"  auto_notes_interval={settings.auto_notes_interval_sentences}\n"
            f"  default_backend={settings.default_llm_backend}\n",
        )

    def test_settings_in_main_config(self):
        """Test that intelligence settings are accessible from main Settings."""
        from config import Settings

        settings = Settings()
        assert hasattr(settings, "intelligence")
        assert settings.intelligence.enabled is True


# =============================================================================
# Pipeline Config Tests
# =============================================================================


class TestPipelineConfigAutoNotes:
    """Test auto-notes fields in PipelineConfig."""

    def test_auto_notes_config_defaults(self):
        """Test default auto-notes config in PipelineConfig."""
        from services.pipeline.config import PipelineConfig

        config = PipelineConfig(session_id="test", source_type="test")
        assert config.enable_auto_notes is False
        assert config.auto_notes_interval == 10
        assert config.auto_notes_template == "auto_note"
        assert config.intelligence_llm_backend == ""

    def test_pipeline_stats_auto_notes(self):
        """Test auto_notes_generated in PipelineStats."""
        from services.pipeline.config import PipelineStats

        stats = PipelineStats()
        assert stats.auto_notes_generated == 0

        stats.auto_notes_generated += 1
        d = stats.to_dict()
        assert d["auto_notes_generated"] == 1


# =============================================================================
# Pydantic Models Tests
# =============================================================================


class TestInsightPydanticModels:
    """Test Pydantic request/response models."""

    def test_note_create_request(self):
        """Test NoteCreateRequest validation."""
        from models.insights import NoteCreateRequest

        req = NoteCreateRequest(content="Test note")
        assert req.content == "Test note"
        assert req.speaker_name is None

    def test_note_create_request_with_speaker(self):
        """Test NoteCreateRequest with speaker."""
        from models.insights import NoteCreateRequest

        req = NoteCreateRequest(content="Test note", speaker_name="Alice")
        assert req.speaker_name == "Alice"

    def test_note_create_request_empty_content_fails(self):
        """Test that empty content raises validation error."""
        from models.insights import NoteCreateRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NoteCreateRequest(content="")

    def test_insight_generate_request(self):
        """Test InsightGenerateRequest validation."""
        from models.insights import InsightGenerateRequest

        req = InsightGenerateRequest(template_names=["meeting_summary"])
        assert req.template_names == ["meeting_summary"]
        assert req.custom_instructions is None

    def test_template_create_request(self):
        """Test TemplateCreateRequest validation."""
        from models.insights import TemplateCreateRequest

        req = TemplateCreateRequest(
            name="test_template",
            prompt_template="Analyze: $transcript",
            category="custom",
        )
        assert req.name == "test_template"
        assert req.default_temperature == 0.3
        assert req.default_max_tokens == 1024

    def test_agent_message_request(self):
        """Test AgentMessageRequest validation."""
        from models.insights import AgentMessageRequest

        req = AgentMessageRequest(content="What were the action items?")
        assert req.content == "What were the action items?"

    def test_note_response_serialization(self):
        """Test NoteResponse serialization."""
        from models.insights import NoteResponse

        resp = NoteResponse(
            note_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            note_type="manual",
            content="Test note content",
            created_at=datetime.now(UTC),
        )
        assert resp.note_type == "manual"
        assert resp.content == "Test note content"

        output = get_output_path("pydantic_models_validation")
        write_output(output, "All Pydantic models validated successfully")


# =============================================================================
# SQLAlchemy Model Tests
# =============================================================================


class TestDatabaseModels:
    """Test SQLAlchemy model definitions."""

    def test_meeting_note_model(self):
        """Test MeetingNote model has all required columns."""
        from database.models import MeetingNote

        assert MeetingNote.__tablename__ == "meeting_notes"
        columns = {c.name for c in MeetingNote.__table__.columns}
        expected = {
            "note_id",
            "session_id",
            "note_type",
            "content",
            "prompt_used",
            "context_sentences",
            "speaker_name",
            "llm_backend",
            "llm_model",
            "processing_time_ms",
            "created_at",
            "updated_at",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

    def test_meeting_insight_model(self):
        """Test MeetingInsight model has all required columns."""
        from database.models import MeetingInsight

        assert MeetingInsight.__tablename__ == "meeting_insights"
        columns = {c.name for c in MeetingInsight.__table__.columns}
        expected = {
            "insight_id",
            "session_id",
            "template_id",
            "insight_type",
            "title",
            "content",
            "prompt_used",
            "transcript_length",
            "llm_backend",
            "llm_model",
            "processing_time_ms",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

    def test_insight_prompt_template_model(self):
        """Test InsightPromptTemplate model."""
        from database.models import InsightPromptTemplate

        assert InsightPromptTemplate.__tablename__ == "insight_prompt_templates"
        columns = {c.name for c in InsightPromptTemplate.__table__.columns}
        expected = {
            "template_id",
            "name",
            "category",
            "prompt_template",
            "system_prompt",
            "expected_output_format",
            "default_temperature",
            "default_max_tokens",
            "is_builtin",
            "is_active",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

    def test_agent_conversation_model(self):
        """Test AgentConversation model."""
        from database.models import AgentConversation

        assert AgentConversation.__tablename__ == "agent_conversations"
        columns = {c.name for c in AgentConversation.__table__.columns}
        expected = {"conversation_id", "session_id", "title", "status", "system_context"}
        assert expected.issubset(columns)

    def test_agent_message_model(self):
        """Test AgentMessage model."""
        from database.models import AgentMessage

        assert AgentMessage.__tablename__ == "agent_messages"
        columns = {c.name for c in AgentMessage.__table__.columns}
        expected = {
            "message_id",
            "conversation_id",
            "role",
            "content",
            "llm_backend",
            "llm_model",
            "suggested_queries",
        }
        assert expected.issubset(columns)

        output = get_output_path("database_models")
        write_output(
            output,
            "All 5 intelligence database models validated:\n"
            "  - meeting_notes\n"
            "  - meeting_insights\n"
            "  - insight_prompt_templates\n"
            "  - agent_conversations\n"
            "  - agent_messages\n",
        )


# =============================================================================
# Real DB: Template Seeding Tests
# =============================================================================


class TestTemplateSeedingRealDB:
    """Test template seeding against a real database."""

    @pytest.mark.asyncio
    async def test_load_default_templates_seeds_all(self, intelligence_service, db_session_factory):
        """Test that load_default_templates seeds templates into the real DB."""
        seeded_count = await intelligence_service.load_default_templates()
        assert seeded_count >= 6, f"Expected >= 6 templates seeded, got {seeded_count}"

        # Verify templates persist in DB
        templates = await intelligence_service.get_templates()
        assert len(templates) >= 6

        template_names = {t["name"] for t in templates}
        expected_names = {
            "meeting_summary",
            "action_items",
            "key_decisions",
            "speaker_analysis",
            "follow_up_questions",
            "auto_note",
        }
        assert expected_names.issubset(
            template_names
        ), f"Missing templates: {expected_names - template_names}"

        # Verify each template has the right structure
        for tmpl in templates:
            assert tmpl["is_builtin"] is True
            assert tmpl["is_active"] is True
            assert "$transcript" in tmpl["prompt_template"]

        output = get_output_path("template_seeding_real_db")
        write_output(
            output,
            f"Seeded {seeded_count} templates into real DB.\n"
            f"Template names: {sorted(template_names)}\n"
            f"All templates are builtin and active.\n",
        )

    @pytest.mark.asyncio
    async def test_load_default_templates_idempotent(
        self, intelligence_service, db_session_factory
    ):
        """Test that calling load_default_templates twice does not duplicate."""
        first_count = await intelligence_service.load_default_templates()
        assert first_count >= 6

        # Call again - should seed 0 new templates
        second_count = await intelligence_service.load_default_templates()
        assert second_count == 0, f"Expected 0 on second call, got {second_count}"

        # Verify total count is unchanged
        templates = await intelligence_service.get_templates()
        assert len(templates) == first_count

        output = get_output_path("template_seeding_idempotent")
        write_output(
            output,
            f"First seed: {first_count}, Second seed: {second_count}\n"
            "Idempotency confirmed - no duplicates.\n",
        )

    @pytest.mark.asyncio
    async def test_get_template_by_name(self, intelligence_service, db_session_factory):
        """Test retrieving a template by name from real DB."""
        await intelligence_service.load_default_templates()

        tmpl = await intelligence_service.get_template("meeting_summary")
        assert tmpl is not None
        assert tmpl["name"] == "meeting_summary"
        assert tmpl["category"] == "summary"

    @pytest.mark.asyncio
    async def test_get_template_by_uuid(self, intelligence_service, db_session_factory):
        """Test retrieving a template by UUID from real DB."""
        await intelligence_service.load_default_templates()

        # Get the template by name first to get its UUID
        tmpl = await intelligence_service.get_template("action_items")
        assert tmpl is not None

        # Now get it by UUID
        tmpl_by_id = await intelligence_service.get_template(tmpl["template_id"])
        assert tmpl_by_id is not None
        assert tmpl_by_id["name"] == "action_items"

    @pytest.mark.asyncio
    async def test_get_templates_by_category(self, intelligence_service, db_session_factory):
        """Test filtering templates by category from real DB."""
        await intelligence_service.load_default_templates()

        analysis_templates = await intelligence_service.get_templates(category="analysis")
        assert len(analysis_templates) >= 2
        for t in analysis_templates:
            assert t["category"] == "analysis"


# =============================================================================
# Real DB: Custom Template CRUD Tests
# =============================================================================


class TestCustomTemplateCRUDRealDB:
    """Test custom template creation, update, and deletion against real DB."""

    @pytest.mark.asyncio
    async def test_create_custom_template(self, intelligence_service, db_session_factory):
        """Test creating a custom template in real DB."""
        tmpl = await intelligence_service.create_template(
            {
                "name": "custom_risk_analysis",
                "description": "Analyze risks discussed in the meeting",
                "category": "custom",
                "prompt_template": "Identify risks from:\n$transcript",
                "default_temperature": 0.5,
                "default_max_tokens": 2048,
            }
        )

        assert tmpl["name"] == "custom_risk_analysis"
        assert tmpl["is_builtin"] is False
        assert tmpl["is_active"] is True
        assert tmpl["default_temperature"] == 0.5
        assert tmpl["default_max_tokens"] == 2048

        # Verify it persists
        fetched = await intelligence_service.get_template("custom_risk_analysis")
        assert fetched is not None
        assert fetched["template_id"] == tmpl["template_id"]

        output = get_output_path("custom_template_create")
        write_output(
            output,
            f"Custom template created: {tmpl['name']}\n"
            f"  template_id: {tmpl['template_id']}\n"
            f"  is_builtin: {tmpl['is_builtin']}\n",
        )

    @pytest.mark.asyncio
    async def test_update_custom_template(self, intelligence_service, db_session_factory):
        """Test updating a custom template in real DB."""
        tmpl = await intelligence_service.create_template(
            {
                "name": "updatable_template",
                "prompt_template": "Original: $transcript",
                "category": "custom",
            }
        )

        updated = await intelligence_service.update_template(
            tmpl["template_id"],
            {"description": "Updated description", "default_temperature": 0.8},
        )

        assert updated is not None
        assert updated["description"] == "Updated description"
        assert updated["default_temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_delete_custom_template(self, intelligence_service, db_session_factory):
        """Test deleting a custom template from real DB."""
        tmpl = await intelligence_service.create_template(
            {
                "name": "deletable_template",
                "prompt_template": "Delete me: $transcript",
                "category": "custom",
            }
        )

        deleted = await intelligence_service.delete_template(tmpl["template_id"])
        assert deleted is True

        # Verify it no longer exists
        fetched = await intelligence_service.get_template(tmpl["template_id"])
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_builtin_template_fails(self, intelligence_service, db_session_factory):
        """Test that builtin templates cannot be deleted."""
        await intelligence_service.load_default_templates()

        tmpl = await intelligence_service.get_template("meeting_summary")
        assert tmpl is not None

        deleted = await intelligence_service.delete_template(tmpl["template_id"])
        assert deleted is False  # Should fail for builtin

        # Verify it still exists
        still_exists = await intelligence_service.get_template("meeting_summary")
        assert still_exists is not None


# =============================================================================
# Real DB: Manual Notes CRUD Tests
# =============================================================================


class TestManualNotesCRUDRealDB:
    """Test manual note operations against real DB."""

    @pytest.mark.asyncio
    async def test_create_manual_note(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test creating a manual note in real DB."""
        note = await intelligence_service.create_manual_note(
            session_id=bot_session_id,
            content="This is a key discussion point about the API design.",
            speaker_name="Alice",
        )

        assert note["note_type"] == "manual"
        assert note["content"] == "This is a key discussion point about the API design."
        assert note["speaker_name"] == "Alice"
        assert note["session_id"] == bot_session_id
        assert note["note_id"] is not None

        output = get_output_path("manual_note_create")
        write_output(
            output,
            f"Manual note created in real DB:\n"
            f"  note_id: {note['note_id']}\n"
            f"  session_id: {note['session_id']}\n"
            f"  note_type: {note['note_type']}\n"
            f"  content: {note['content']}\n"
            f"  speaker: {note['speaker_name']}\n",
        )

    @pytest.mark.asyncio
    async def test_get_notes_for_session(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test querying notes for a session from real DB."""
        await intelligence_service.create_manual_note(
            session_id=bot_session_id,
            content="Note 1: Project timeline",
        )
        await intelligence_service.create_manual_note(
            session_id=bot_session_id,
            content="Note 2: Budget discussion",
            speaker_name="Bob",
        )

        notes = await intelligence_service.get_notes(bot_session_id)
        assert len(notes) == 2

        contents = [n["content"] for n in notes]
        assert "Note 2: Budget discussion" in contents
        assert "Note 1: Project timeline" in contents

    @pytest.mark.asyncio
    async def test_get_notes_by_type(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test filtering notes by type from real DB."""
        await intelligence_service.create_manual_note(
            session_id=bot_session_id,
            content="Manual note",
        )

        manual_notes = await intelligence_service.get_notes(bot_session_id, note_type="manual")
        assert len(manual_notes) == 1
        assert manual_notes[0]["note_type"] == "manual"

        # No auto notes exist
        auto_notes = await intelligence_service.get_notes(bot_session_id, note_type="auto")
        assert len(auto_notes) == 0

    @pytest.mark.asyncio
    async def test_delete_note(self, intelligence_service, db_session_factory, bot_session_id):
        """Test deleting a note from real DB."""
        note = await intelligence_service.create_manual_note(
            session_id=bot_session_id,
            content="Note to delete",
        )

        deleted = await intelligence_service.delete_note(note["note_id"])
        assert deleted is True

        notes = await intelligence_service.get_notes(bot_session_id)
        assert len(notes) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_note(self, intelligence_service, db_session_factory):
        """Test deleting a note that does not exist."""
        deleted = await intelligence_service.delete_note(str(uuid.uuid4()))
        assert deleted is False


# =============================================================================
# Real DB: Agent Conversation Tests
# =============================================================================


class TestAgentConversationRealDB:
    """Test agent conversation operations against real DB."""

    @pytest.mark.asyncio
    async def test_create_conversation(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test creating a conversation in real DB."""
        conv = await intelligence_service.create_conversation(
            session_id=bot_session_id,
            title="Test Q&A Session",
        )

        assert conv["session_id"] == bot_session_id
        assert conv["title"] == "Test Q&A Session"
        assert conv["status"] == "active"
        assert conv["conversation_id"] is not None

        output = get_output_path("conversation_create")
        write_output(
            output,
            f"Conversation created in real DB:\n"
            f"  conversation_id: {conv['conversation_id']}\n"
            f"  session_id: {conv['session_id']}\n"
            f"  title: {conv['title']}\n"
            f"  status: {conv['status']}\n",
        )

    @pytest.mark.asyncio
    async def test_create_conversation_with_transcript_context(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test conversation creation with transcript context."""
        conv = await intelligence_service.create_conversation(
            session_id=bot_session_id,
            title="Contextualized Q&A",
            transcript_text="Alice: Let us discuss the API.\nBob: Sounds good.",
        )

        assert conv["system_context"] is not None
        assert "meeting transcript" in conv["system_context"].lower()
        assert "Alice: Let us discuss the API." in conv["system_context"]

    @pytest.mark.asyncio
    async def test_send_message_and_get_history(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test sending a message and retrieving conversation history from real DB."""
        conv = await intelligence_service.create_conversation(
            session_id=bot_session_id,
            title="Message Test",
        )

        reply = await intelligence_service.send_message(
            conversation_id=conv["conversation_id"],
            content="What were the action items?",
        )

        assert reply["role"] == "assistant"
        assert reply["content"] is not None
        assert len(reply["content"]) > 0
        assert reply["suggested_queries"] is not None
        assert len(reply["suggested_queries"]) > 0
        # No LLM client configured, so message should indicate that
        assert "No LLM backend" in reply["content"] or reply.get("llm_backend") is not None

        history = await intelligence_service.get_conversation_history(conv["conversation_id"])

        assert history is not None
        assert len(history["messages"]) == 2  # user + assistant
        assert history["messages"][0]["role"] == "user"
        assert history["messages"][0]["content"] == "What were the action items?"
        assert history["messages"][1]["role"] == "assistant"

        output = get_output_path("conversation_messages")
        write_output(
            output,
            f"Conversation with messages in real DB:\n"
            f"  conversation_id: {conv['conversation_id']}\n"
            f"  messages: {len(history['messages'])}\n"
            f"  user msg: {history['messages'][0]['content']}\n"
            f"  assistant msg: {history['messages'][1]['content'][:80]}...\n",
        )

    @pytest.mark.asyncio
    async def test_suggested_queries(self, intelligence_service):
        """Test suggested queries are returned."""
        queries = await intelligence_service.get_suggested_queries("test-session")
        assert len(queries) >= 5
        assert any("action items" in q.lower() for q in queries)


# =============================================================================
# Coordinator Auto-Notes Integration Tests
# =============================================================================


class TestCoordinatorAutoNotes:
    """Test auto-notes integration in the pipeline coordinator."""

    def test_coordinator_accepts_meeting_intelligence_param(self):
        """Test that coordinator accepts the meeting_intelligence parameter."""
        from services.pipeline.adapters.base import ChunkAdapter, TranscriptChunk
        from services.pipeline.config import PipelineConfig
        from services.pipeline.coordinator import TranscriptionPipelineCoordinator

        class DummyAdapter(ChunkAdapter):
            source_type = "test"

            def adapt(self, raw):
                return TranscriptChunk(text="", chunk_id="", speaker_name="")

            def extract_speaker(self, raw_chunk):
                return None

        config = PipelineConfig(
            session_id="test",
            source_type="test",
            enable_auto_notes=True,
            auto_notes_interval=5,
        )

        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=DummyAdapter(),
            meeting_intelligence="sentinel_service",
        )
        assert coordinator.meeting_intelligence == "sentinel_service"
        assert coordinator.config.enable_auto_notes is True
        assert coordinator._auto_note_buffer == []

    def test_coordinator_with_real_service(self, intelligence_service):
        """Test coordinator with a real MeetingIntelligenceService."""
        from services.meeting_intelligence import MeetingIntelligenceService
        from services.pipeline.adapters.base import ChunkAdapter, TranscriptChunk
        from services.pipeline.config import PipelineConfig
        from services.pipeline.coordinator import TranscriptionPipelineCoordinator

        class DummyAdapter(ChunkAdapter):
            source_type = "test"

            def adapt(self, raw):
                return TranscriptChunk(text="", chunk_id="", speaker_name="")

            def extract_speaker(self, raw_chunk):
                return None

        config = PipelineConfig(
            session_id="test",
            source_type="test",
            enable_auto_notes=True,
            auto_notes_interval=5,
        )

        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=DummyAdapter(),
            meeting_intelligence=intelligence_service,
        )

        assert isinstance(coordinator.meeting_intelligence, MeetingIntelligenceService)

    def test_coordinator_stats_include_auto_notes(self):
        """Test that coordinator stats include auto_notes_generated."""
        from services.pipeline.config import PipelineStats

        stats = PipelineStats()
        d = stats.to_dict()
        assert "auto_notes_generated" in d
        assert d["auto_notes_generated"] == 0

        output = get_output_path("coordinator_auto_notes")
        write_output(output, "Coordinator auto-notes integration tests passed")


# =============================================================================
# Real DB: Service Constructor Tests
# =============================================================================


class TestServiceConstructorRealDB:
    """Test MeetingIntelligenceService construction with real DB."""

    @pytest.mark.asyncio
    async def test_service_templates_loaded_flag(self, intelligence_service, db_session_factory):
        """Test that _templates_loaded flag updates after seeding."""
        assert intelligence_service._templates_loaded is False

        await intelligence_service.load_default_templates()

        assert intelligence_service._templates_loaded is True

    @pytest.mark.asyncio
    async def test_service_create_analyzed_note_requires_client(
        self, intelligence_service, db_session_factory, bot_session_id
    ):
        """Test that analyzed note creation fails without translation client."""
        with pytest.raises(RuntimeError, match="Translation client not configured"):
            await intelligence_service.create_analyzed_note(
                session_id=bot_session_id,
                prompt="Analyze this meeting",
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

        output = get_output_path("alembic_migration")
        write_output(
            output,
            "Alembic migration validated:\n"
            f"  revision: {module.revision}\n"
            f"  down_revision: {module.down_revision}\n"
            "  Has upgrade(): True\n"
            "  Has downgrade(): True\n",
        )
