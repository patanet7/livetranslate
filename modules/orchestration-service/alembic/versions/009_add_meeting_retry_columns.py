"""Add retry tracking columns to meetings table

Revision ID: 009_meeting_retry_cols
Revises: 008_chat_msg_speaker_cols
Create Date: 2026-03-04

Adds columns for automatic retry of failed Fireflies sync:
- retry_count — number of retry attempts (0 = first attempt)
- last_retry_at — timestamp of most recent retry attempt
- next_retry_at — computed next eligible retry time (exponential backoff)
- Index on (sync_status, next_retry_at) for efficient retry queries
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "009_meeting_retry_cols"
down_revision = "008_chat_msg_speaker_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "meetings",
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "meetings",
        sa.Column("last_retry_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_meetings_retry_eligible",
        "meetings",
        ["sync_status", "next_retry_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_meetings_retry_eligible", table_name="meetings")
    op.drop_column("meetings", "next_retry_at")
    op.drop_column("meetings", "last_retry_at")
    op.drop_column("meetings", "retry_count")
