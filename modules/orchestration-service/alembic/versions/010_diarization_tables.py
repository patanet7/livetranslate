"""Add diarization jobs and speaker profiles tables

Revision ID: 010_diarization_tables
Revises: 009_meeting_retry_cols
Create Date: 2026-03-04
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "010_diarization_tables"
down_revision = "009_meeting_retry_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "diarization_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("meeting_id", UUID(as_uuid=True), sa.ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("triggered_by", sa.String(20), nullable=False, server_default="manual"),
        sa.Column("rule_matched", JSONB, nullable=True),
        sa.Column("audio_url", sa.Text, nullable=True),
        sa.Column("audio_size_bytes", sa.BigInteger, nullable=True),
        sa.Column("raw_segments", JSONB, nullable=True),
        sa.Column("detected_language", sa.String(10), nullable=True),
        sa.Column("num_speakers_detected", sa.Integer, nullable=True),
        sa.Column("processing_time_seconds", sa.Float, nullable=True),
        sa.Column("speaker_map", JSONB, nullable=True),
        sa.Column("unmapped_speakers", JSONB, nullable=True),
        sa.Column("merge_applied", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("merge_applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_diarization_jobs_meeting", "diarization_jobs", ["meeting_id"])
    op.create_index("ix_diarization_jobs_status", "diarization_jobs", ["status"])

    op.create_table(
        "speaker_profiles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("embedding", JSONB, nullable=True),
        sa.Column("enrollment_source", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("sample_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_speaker_profiles_email", "speaker_profiles", ["email"])
    op.create_index("ix_speaker_profiles_name", "speaker_profiles", ["name"])


def downgrade() -> None:
    op.drop_index("ix_speaker_profiles_name", table_name="speaker_profiles")
    op.drop_index("ix_speaker_profiles_email", table_name="speaker_profiles")
    op.drop_table("speaker_profiles")
    op.drop_index("ix_diarization_jobs_status", table_name="diarization_jobs")
    op.drop_index("ix_diarization_jobs_meeting", table_name="diarization_jobs")
    op.drop_table("diarization_jobs")
