"""Initial schema and glossary tables for Fireflies integration

Revision ID: 001_initial
Revises:
Create Date: 2025-01-08

Creates all base tables for the orchestration service including:
- bot_sessions: Core session tracking
- audio_files: Audio file storage metadata
- transcripts: Transcription storage (supports Fireflies source)
- translations: Translation storage
- correlations: Time correlation data
- participants: Meeting participant tracking
- session_events: Session event logging
- session_statistics: Aggregated session stats
- glossaries: Translation glossary collections (for Fireflies)
- glossary_entries: Individual glossary term mappings
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ===========================================================================
    # Core Tables
    # ===========================================================================

    # bot_sessions - Main session tracking
    op.create_table(
        "bot_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("bot_id", sa.String(255), nullable=False, index=True),
        sa.Column("meeting_id", sa.String(255), nullable=False, index=True),
        sa.Column("meeting_title", sa.String(500), nullable=True),
        sa.Column("meeting_uri", sa.Text(), nullable=True),
        sa.Column(
            "bot_type", sa.String(50), nullable=False, server_default="google_meet"
        ),
        # Session lifecycle
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        # Configuration
        sa.Column("target_languages", postgresql.JSON(), nullable=True),
        sa.Column(
            "enable_translation", sa.Boolean(), nullable=False, server_default="true"
        ),
        sa.Column(
            "enable_transcription", sa.Boolean(), nullable=False, server_default="true"
        ),
        sa.Column(
            "enable_virtual_webcam",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "audio_storage_enabled", sa.Boolean(), nullable=False, server_default="true"
        ),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("idx_bot_sessions_status", "bot_sessions", ["status"])
    op.create_index("idx_bot_sessions_created_at", "bot_sessions", ["created_at"])

    # audio_files - Audio storage metadata
    op.create_table(
        "audio_files",
        sa.Column("file_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        # File information
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        # Audio properties
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("sample_rate", sa.Integer(), nullable=True),
        sa.Column("channels", sa.Integer(), nullable=True),
        sa.Column("bit_depth", sa.Integer(), nullable=True),
        # Timestamps
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
    )
    op.create_index("idx_audio_files_session_id", "audio_files", ["session_id"])
    op.create_index("idx_audio_files_created_at", "audio_files", ["created_at"])
    op.create_index("idx_audio_files_start_time", "audio_files", ["start_time"])
    op.create_index("idx_audio_files_hash", "audio_files", ["file_hash"])

    # transcripts - Transcription storage (supports multiple sources including Fireflies)
    op.create_table(
        "transcripts",
        sa.Column("transcript_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        # Transcript content
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        # Source information (google_meet, whisper, fireflies, manual)
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column(
            "audio_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audio_files.file_id"),
            nullable=True,
        ),
        # Speaker information
        sa.Column("speaker_id", sa.String(100), nullable=True),
        sa.Column("speaker_name", sa.String(255), nullable=True),
        # Timing information
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=False),
        sa.Column("start_offset", sa.Float(), nullable=True),
        sa.Column("end_offset", sa.Float(), nullable=True),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
    )
    op.create_index("idx_transcripts_session_id", "transcripts", ["session_id"])
    op.create_index("idx_transcripts_start_time", "transcripts", ["start_time"])
    op.create_index("idx_transcripts_speaker_id", "transcripts", ["speaker_id"])
    op.create_index("idx_transcripts_language", "transcripts", ["language"])
    op.create_index("idx_transcripts_source", "transcripts", ["source"])

    # translations - Translation storage
    op.create_table(
        "translations",
        sa.Column("translation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        sa.Column(
            "transcript_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("transcripts.transcript_id"),
            nullable=True,
        ),
        # Translation content
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("source_language", sa.String(10), nullable=False),
        sa.Column("target_language", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        # Speaker information
        sa.Column("speaker_id", sa.String(100), nullable=True),
        sa.Column("speaker_name", sa.String(255), nullable=True),
        # Timing information
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=False),
        # Quality metrics
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("character_count", sa.Integer(), nullable=True),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
    )
    op.create_index("idx_translations_session_id", "translations", ["session_id"])
    op.create_index("idx_translations_start_time", "translations", ["start_time"])
    op.create_index("idx_translations_speaker_id", "translations", ["speaker_id"])
    op.create_index(
        "idx_translations_source_language", "translations", ["source_language"]
    )
    op.create_index(
        "idx_translations_target_language", "translations", ["target_language"]
    )

    # correlations - Time synchronization between sources
    op.create_table(
        "correlations",
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        # External source (e.g., Google Meet captions)
        sa.Column("external_source", sa.String(50), nullable=False),
        sa.Column("external_text", sa.Text(), nullable=False),
        sa.Column("external_timestamp", sa.DateTime(), nullable=False),
        # Internal source (e.g., Whisper transcription)
        sa.Column("internal_source", sa.String(50), nullable=False),
        sa.Column("internal_text", sa.Text(), nullable=False),
        sa.Column("internal_timestamp", sa.DateTime(), nullable=False),
        # Correlation metrics
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("time_offset", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
    )
    op.create_index("idx_correlations_session_id", "correlations", ["session_id"])
    op.create_index(
        "idx_correlations_external_timestamp", "correlations", ["external_timestamp"]
    )
    op.create_index(
        "idx_correlations_internal_timestamp", "correlations", ["internal_timestamp"]
    )
    op.create_index(
        "idx_correlations_similarity_score", "correlations", ["similarity_score"]
    )

    # participants - Meeting participant tracking
    op.create_table(
        "participants",
        sa.Column("participant_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        # Participant information
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        # Speaker mapping
        sa.Column("speaker_id", sa.String(100), nullable=True, index=True),
        sa.Column("speaker_embedding", postgresql.JSON(), nullable=True),
        # Participation metrics
        sa.Column("joined_at", sa.DateTime(), nullable=True),
        sa.Column("left_at", sa.DateTime(), nullable=True),
        sa.Column("speaking_time", sa.Float(), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        # Metadata
        sa.Column("session_metadata", postgresql.JSON(), nullable=True),
    )
    op.create_index("idx_participants_session_id", "participants", ["session_id"])
    op.create_index("idx_participants_external_id", "participants", ["external_id"])
    op.create_index("idx_participants_joined_at", "participants", ["joined_at"])

    # session_events - Event logging
    op.create_table(
        "session_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        # Event information
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("event_name", sa.String(255), nullable=False),
        sa.Column("event_data", postgresql.JSON(), nullable=True),
        # Timing
        sa.Column(
            "timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        # Severity
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
        # Source
        sa.Column("source", sa.String(100), nullable=False),
    )
    op.create_index("idx_session_events_session_id", "session_events", ["session_id"])
    op.create_index("idx_session_events_timestamp", "session_events", ["timestamp"])
    op.create_index("idx_session_events_event_type", "session_events", ["event_type"])
    op.create_index("idx_session_events_severity", "session_events", ["severity"])

    # session_statistics - Aggregated stats
    op.create_table(
        "session_statistics",
        sa.Column("statistics_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
            unique=True,
        ),
        # Basic metrics
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column(
            "total_participants", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_audio_files", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_transcripts", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_translations", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_correlations", sa.Integer(), nullable=False, server_default="0"
        ),
        # Audio metrics
        sa.Column("total_audio_duration", sa.Float(), nullable=True),
        sa.Column("total_audio_size", sa.Integer(), nullable=True),
        sa.Column("audio_quality_score", sa.Float(), nullable=True),
        # Transcription metrics
        sa.Column("total_words", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_characters", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("average_confidence", sa.Float(), nullable=True),
        sa.Column("languages_detected", postgresql.JSON(), nullable=True),
        # Translation metrics
        sa.Column("translation_quality_score", sa.Float(), nullable=True),
        sa.Column("translation_coverage", sa.Float(), nullable=True),
        # Speaker metrics
        sa.Column("unique_speakers", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("speaker_distribution", postgresql.JSON(), nullable=True),
        # Timing metrics
        sa.Column("processing_time", sa.Float(), nullable=True),
        sa.Column("correlation_accuracy", sa.Float(), nullable=True),
        # Timestamps
        sa.Column(
            "calculated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "idx_session_statistics_session_id", "session_statistics", ["session_id"]
    )
    op.create_index(
        "idx_session_statistics_calculated_at", "session_statistics", ["calculated_at"]
    )

    # ===========================================================================
    # Glossary Tables (for Fireflies Integration)
    # ===========================================================================

    # glossaries - Glossary collections
    op.create_table(
        "glossaries",
        sa.Column("glossary_id", postgresql.UUID(as_uuid=True), primary_key=True),
        # Glossary identification
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Domain categorization
        sa.Column("domain", sa.String(100), nullable=True),
        # Language settings
        sa.Column(
            "source_language", sa.String(10), nullable=False, server_default="en"
        ),
        sa.Column("target_languages", postgresql.JSON(), nullable=False),
        # Status
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
        # Metadata
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("created_by", sa.String(255), nullable=True),
        # Entry count (denormalized)
        sa.Column("entry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_glossaries_name", "glossaries", ["name"])
    op.create_index("idx_glossaries_domain", "glossaries", ["domain"])
    op.create_index("idx_glossaries_is_active", "glossaries", ["is_active"])
    op.create_index("idx_glossaries_source_language", "glossaries", ["source_language"])

    # glossary_entries - Individual term mappings
    op.create_table(
        "glossary_entries",
        sa.Column("entry_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "glossary_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("glossaries.glossary_id"),
            nullable=False,
        ),
        # Source term
        sa.Column("source_term", sa.String(500), nullable=False),
        sa.Column("source_term_normalized", sa.String(500), nullable=False),
        # Target language translations (JSON: {"es": "término", "fr": "terme"})
        sa.Column("translations", postgresql.JSON(), nullable=False),
        # Context and notes
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        # Matching settings
        sa.Column(
            "case_sensitive", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column(
            "match_whole_word", sa.Boolean(), nullable=False, server_default="true"
        ),
        # Priority
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        # Timestamps
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "idx_glossary_entries_glossary_id", "glossary_entries", ["glossary_id"]
    )
    op.create_index(
        "idx_glossary_entries_source_term",
        "glossary_entries",
        ["source_term_normalized"],
    )
    op.create_index("idx_glossary_entries_priority", "glossary_entries", ["priority"])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table("glossary_entries")
    op.drop_table("glossaries")
    op.drop_table("session_statistics")
    op.drop_table("session_events")
    op.drop_table("participants")
    op.drop_table("correlations")
    op.drop_table("translations")
    op.drop_table("transcripts")
    op.drop_table("audio_files")
    op.drop_table("bot_sessions")
