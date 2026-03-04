"""Add system_config table and unique constraint on fireflies_transcript_id

Revision ID: 007_system_config_unique_ff
Revises: 006_meeting_sync_media
Create Date: 2026-03-03

Adds:
- system_config table for storing boot sync timestamps and other settings
- Unique constraint on meetings.fireflies_transcript_id to prevent duplicates
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "007_system_config_unique_ff"
down_revision = "006_meeting_sync_media"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # System config key-value table for boot sync timestamps, etc.
    op.create_table(
        "system_config",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )

    # Prevent duplicate Fireflies transcript IDs.
    # NULLs are allowed (non-Fireflies meetings) — unique only applies to non-NULL values.
    op.create_index(
        "ix_meetings_fireflies_transcript_id_unique",
        "meetings",
        ["fireflies_transcript_id"],
        unique=True,
        postgresql_where=sa.text("fireflies_transcript_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_meetings_fireflies_transcript_id_unique",
        table_name="meetings",
    )
    op.drop_table("system_config")
