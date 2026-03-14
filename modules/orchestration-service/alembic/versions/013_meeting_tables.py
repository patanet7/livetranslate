"""Create meeting_sessions, meeting_transcripts, session_translations tables.

Additive migration: creates new tables alongside existing bot_sessions.
Does NOT drop or modify bot_sessions.

Revision ID: 013_meeting_tables
Revises: 012_ai_connections
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "013_meeting_tables"
down_revision = "012_ai_connections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meeting_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="ephemeral"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_languages", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("target_languages", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("recording_path", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_meeting_sessions_status", "meeting_sessions", ["status"])
    op.create_index(
        "ix_meeting_sessions_started_at", "meeting_sessions", ["started_at"]
    )

    op.create_table(
        "meeting_transcripts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id"),
            nullable=False,
        ),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("speaker_id", sa.Text(), nullable=True),
        sa.Column("speaker_name", sa.Text(), nullable=True),
        sa.Column("source_language", sa.Text(), nullable=True),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_final", sa.Boolean(), server_default="false"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_meeting_transcripts_session", "meeting_transcripts", ["session_id"]
    )
    op.create_index(
        "ix_meeting_transcripts_ts", "meeting_transcripts", ["timestamp_ms"]
    )

    op.create_table(
        "session_translations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "transcript_id",
            sa.BigInteger(),
            sa.ForeignKey("meeting_transcripts.id"),
            nullable=False,
        ),
        sa.Column("target_language", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("model_used", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_session_translations_transcript", "session_translations", ["transcript_id"]
    )


def downgrade() -> None:
    op.drop_table("session_translations")
    op.drop_table("meeting_transcripts")
    op.drop_table("meeting_sessions")
