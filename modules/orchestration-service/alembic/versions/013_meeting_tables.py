"""Extend existing meeting tables for unified pipeline support.

Adds columns needed by the loopback/gmeet pipeline to the existing
meetings, meeting_chunks, and meeting_translations tables.  Does NOT
create new tables and does NOT drop any existing columns.

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
    # -----------------------------------------------------------------------
    # meetings — add pipeline columns
    # -----------------------------------------------------------------------
    op.add_column(
        "meetings",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column(
            "source_languages", postgresql.ARRAY(sa.Text()), nullable=True
        ),
    )
    op.add_column(
        "meetings",
        sa.Column(
            "target_languages", postgresql.ARRAY(sa.Text()), nullable=True
        ),
    )
    op.add_column(
        "meetings",
        sa.Column("recording_path", sa.Text(), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_meetings_last_activity_at", "meetings", ["last_activity_at"]
    )

    # -----------------------------------------------------------------------
    # meeting_chunks — add real-time pipeline columns
    # -----------------------------------------------------------------------
    op.add_column(
        "meeting_chunks",
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "meeting_chunks",
        sa.Column("is_final", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "meeting_chunks",
        sa.Column("confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "meeting_chunks",
        sa.Column("source_language", sa.Text(), nullable=True),
    )
    op.add_column(
        "meeting_chunks",
        sa.Column("source_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "meeting_chunks",
        sa.Column("speaker_id", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_chunks_timestamp_ms", "meeting_chunks", ["timestamp_ms"]
    )
    op.create_index(
        "ix_chunks_is_final", "meeting_chunks", ["is_final"]
    )

    # -----------------------------------------------------------------------
    # meeting_translations — make sentence_id nullable so translations can
    # attach to chunks directly (loopback/gmeet pipeline path).
    # Add chunk_id FK for the direct-to-chunk translation path.
    # -----------------------------------------------------------------------
    op.alter_column("meeting_translations", "sentence_id", nullable=True)
    op.add_column(
        "meeting_translations",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_chunks.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_mtrans_chunk", "meeting_translations", ["chunk_id"]
    )


def downgrade() -> None:
    # meeting_translations
    op.drop_index("ix_mtrans_chunk", table_name="meeting_translations")
    op.drop_column("meeting_translations", "chunk_id")
    op.alter_column("meeting_translations", "sentence_id", nullable=False)

    # meeting_chunks
    op.drop_index("ix_chunks_is_final", table_name="meeting_chunks")
    op.drop_index("ix_chunks_timestamp_ms", table_name="meeting_chunks")
    op.drop_column("meeting_chunks", "speaker_id")
    op.drop_column("meeting_chunks", "source_id")
    op.drop_column("meeting_chunks", "source_language")
    op.drop_column("meeting_chunks", "confidence")
    op.drop_column("meeting_chunks", "is_final")
    op.drop_column("meeting_chunks", "timestamp_ms")

    # meetings
    op.drop_index("ix_meetings_last_activity_at", table_name="meetings")
    op.drop_column("meetings", "last_activity_at")
    op.drop_column("meetings", "metadata")
    op.drop_column("meetings", "recording_path")
    op.drop_column("meetings", "target_languages")
    op.drop_column("meetings", "source_languages")
    op.drop_column("meetings", "ended_at")
    op.drop_column("meetings", "started_at")
