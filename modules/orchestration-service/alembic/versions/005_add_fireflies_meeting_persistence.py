"""Add Fireflies meeting persistence tables

Revision ID: 005_fireflies_meeting_persistence
Revises: 004_meeting_intelligence
Create Date: 2026-02-20

Adds 6 new tables for Fireflies real-time meeting persistence:
1. meetings             — Core meeting record (one row per Fireflies session)
2. meeting_chunks       — Deduplicated raw transcript chunks
3. meeting_sentences    — Aggregated sentences assembled from chunks
4. meeting_translations — Per-sentence translations into target languages
5. meeting_data_insights — AI-generated insights as JSONB (named meeting_data_insights
                           to avoid collision with the bot-sessions meeting_insights
                           table managed by migration 004)
6. meeting_speakers     — Per-meeting speaker analytics

Note: The insights table is named meeting_data_insights (not meeting_insights) to
avoid collision with the existing meeting_insights table from migration 004 which
is linked to bot_sessions via session_id and stores LLM-generated structured text.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "005_fireflies_persistence"
down_revision = "004_meeting_intelligence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # =========================================================================
    # 1. meetings — core meeting record
    # =========================================================================
    op.create_table(
        "meetings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("fireflies_transcript_id", sa.Text, nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("meeting_link", sa.Text, nullable=True),
        sa.Column("organizer_email", sa.Text, nullable=True),
        sa.Column(
            "participants",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration", sa.Integer, nullable=True),
        sa.Column("source", sa.Text, nullable=False, server_default="fireflies"),
        sa.Column("status", sa.Text, nullable=False, server_default="live"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_meetings_ff_id", "meetings", ["fireflies_transcript_id"])
    op.create_index("idx_meetings_status", "meetings", ["status"])
    op.create_index("idx_meetings_source", "meetings", ["source"])
    op.create_index("idx_meetings_created_at", "meetings", ["created_at"])

    # =========================================================================
    # 2. meeting_chunks — deduplicated raw transcript chunks
    # =========================================================================
    op.create_table(
        "meeting_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "meeting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("speaker_name", sa.Text, nullable=True),
        sa.Column("start_time", sa.Float, nullable=True),
        sa.Column("end_time", sa.Float, nullable=True),
        sa.Column("is_command", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("meeting_id", "chunk_id", name="uq_meeting_chunks_meeting_chunk"),
    )
    op.create_index("idx_chunks_meeting", "meeting_chunks", ["meeting_id"])
    # GIN full-text search index on chunk text
    op.execute(
        "CREATE INDEX idx_chunks_text_search ON meeting_chunks "
        "USING gin(to_tsvector('english', text))"
    )

    # =========================================================================
    # 3. meeting_sentences — aggregated sentences from chunks
    # =========================================================================
    op.create_table(
        "meeting_sentences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "meeting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("speaker_name", sa.Text, nullable=True),
        sa.Column("start_time", sa.Float, nullable=True),
        sa.Column("end_time", sa.Float, nullable=True),
        sa.Column("boundary_type", sa.Text, nullable=True),
        sa.Column(
            "chunk_ids",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_sentences_meeting", "meeting_sentences", ["meeting_id"])
    op.create_index("idx_sentences_speaker", "meeting_sentences", ["speaker_name"])
    op.create_index("idx_sentences_start_time", "meeting_sentences", ["start_time"])
    # GIN full-text search index on sentence text
    op.execute(
        "CREATE INDEX idx_sentences_text_search ON meeting_sentences "
        "USING gin(to_tsvector('english', text))"
    )

    # =========================================================================
    # 4. meeting_translations — per-sentence translations
    # =========================================================================
    op.create_table(
        "meeting_translations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "sentence_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sentences.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("translated_text", sa.Text, nullable=False),
        sa.Column("target_language", sa.Text, nullable=False),
        sa.Column("source_language", sa.Text, nullable=False, server_default="en"),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("translation_time_ms", sa.Float, nullable=True),
        sa.Column("model_used", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_mtrans_sentence", "meeting_translations", ["sentence_id"])
    op.create_index("idx_mtrans_target_language", "meeting_translations", ["target_language"])

    # =========================================================================
    # 5. meeting_data_insights — AI-generated insights (JSONB, Fireflies)
    #    Named meeting_data_insights to avoid collision with migration 004's
    #    meeting_insights table (which is linked to bot_sessions via session_id).
    # =========================================================================
    op.create_table(
        "meeting_data_insights",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "meeting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("insight_type", sa.Text, nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("source", sa.Text, nullable=False, server_default="fireflies"),
        sa.Column("model_used", sa.Text, nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_data_insights_meeting", "meeting_data_insights", ["meeting_id"])
    op.create_index("idx_data_insights_type", "meeting_data_insights", ["insight_type"])
    op.create_index("idx_data_insights_created_at", "meeting_data_insights", ["created_at"])

    # =========================================================================
    # 6. meeting_speakers — speaker analytics per meeting
    # =========================================================================
    op.create_table(
        "meeting_speakers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "meeting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meetings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("speaker_name", sa.Text, nullable=False),
        sa.Column("email", sa.Text, nullable=True),
        sa.Column("talk_time_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("word_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("sentiment_score", sa.Float, nullable=True),
        sa.Column("analytics", postgresql.JSONB, nullable=True),
        sa.UniqueConstraint(
            "meeting_id", "speaker_name", name="uq_meeting_speakers_meeting_speaker"
        ),
    )
    op.create_index("idx_speakers_meeting", "meeting_speakers", ["meeting_id"])
    op.create_index("idx_speakers_name", "meeting_speakers", ["speaker_name"])

    # =========================================================================
    # Trigger: auto-update updated_at on meetings table
    # =========================================================================
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_meetings_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trigger_meetings_updated_at ON meetings")
    op.execute(
        """
        CREATE TRIGGER trigger_meetings_updated_at
            BEFORE UPDATE ON meetings
            FOR EACH ROW
            EXECUTE FUNCTION update_meetings_updated_at()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trigger_meetings_updated_at ON meetings")
    op.execute("DROP FUNCTION IF EXISTS update_meetings_updated_at()")
    op.drop_table("meeting_speakers")
    op.drop_table("meeting_data_insights")
    op.drop_table("meeting_translations")
    op.drop_table("meeting_sentences")
    op.drop_table("meeting_chunks")
    op.drop_table("meetings")
