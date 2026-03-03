"""Add sync status and media URL columns to meetings table

Revision ID: 006_meeting_sync_media
Revises: 005_fireflies_persistence
Create Date: 2026-03-02

Adds columns to the meetings table for sync status tracking and media URLs:
- audio_url, video_url, transcript_url — media URLs from Fireflies
- sync_status — lifecycle state (none → live → syncing → synced → failed)
- sync_error — error message when sync_status = 'failed'
- synced_at — timestamp of last successful sync
- Index on sync_status for efficient filtering
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "006_meeting_sync_media"
down_revision = "005_fireflies_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meetings", sa.Column("audio_url", sa.Text, nullable=True))
    op.add_column("meetings", sa.Column("video_url", sa.Text, nullable=True))
    op.add_column("meetings", sa.Column("transcript_url", sa.Text, nullable=True))
    op.add_column(
        "meetings",
        sa.Column("sync_status", sa.Text, nullable=False, server_default="none"),
    )
    op.add_column("meetings", sa.Column("sync_error", sa.Text, nullable=True))
    op.add_column(
        "meetings",
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_meetings_sync_status", "meetings", ["sync_status"])


def downgrade() -> None:
    op.drop_index("ix_meetings_sync_status", table_name="meetings")
    op.drop_column("meetings", "synced_at")
    op.drop_column("meetings", "sync_error")
    op.drop_column("meetings", "sync_status")
    op.drop_column("meetings", "transcript_url")
    op.drop_column("meetings", "video_url")
    op.drop_column("meetings", "audio_url")
