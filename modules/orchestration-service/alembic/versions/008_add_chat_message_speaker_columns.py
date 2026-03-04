"""Add speaker_id and speaker_name columns to chat_messages table.

Revision ID: 008_add_chat_message_speaker_columns
Revises: 007_system_config_and_unique_ff_id
"""

from alembic import op
import sqlalchemy as sa

revision = "008_chat_msg_speaker_cols"
down_revision = "007_system_config_unique_ff"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("speaker_id", sa.String(255), nullable=True))
    op.add_column("chat_messages", sa.Column("speaker_name", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "speaker_name")
    op.drop_column("chat_messages", "speaker_id")
