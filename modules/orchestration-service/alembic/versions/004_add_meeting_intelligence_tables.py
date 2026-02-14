"""Add meeting intelligence tables

Revision ID: 004_meeting_intelligence
Revises: 003_consolidate_glossaries
Create Date: 2026-02-13

Adds 5 new tables for the Meeting Intelligence system:
1. insight_prompt_templates - Configurable prompt templates
2. meeting_notes - Real-time notes (auto + manual)
3. meeting_insights - Post-meeting generated insights
4. agent_conversations - Chat sessions about a meeting
5. agent_messages - Individual messages in agent chat
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "004_meeting_intelligence"
down_revision = "003_consolidate_glossaries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # =========================================================================
    # 1. insight_prompt_templates (must come before meeting_insights FK)
    # =========================================================================
    op.create_table(
        "insight_prompt_templates",
        sa.Column("template_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("category", sa.String(100), nullable=False, server_default="custom"),
        sa.Column("prompt_template", sa.Text, nullable=False),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column(
            "expected_output_format", sa.String(50), nullable=False, server_default="markdown"
        ),
        sa.Column("default_llm_backend", sa.String(100), nullable=True),
        sa.Column("default_temperature", sa.Float, nullable=False, server_default="0.3"),
        sa.Column("default_max_tokens", sa.Integer, nullable=False, server_default="1024"),
        sa.Column("is_builtin", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("template_metadata", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_insight_templates_name", "insight_prompt_templates", ["name"])
    op.create_index("idx_insight_templates_category", "insight_prompt_templates", ["category"])
    op.create_index("idx_insight_templates_is_active", "insight_prompt_templates", ["is_active"])

    # =========================================================================
    # 2. meeting_notes
    # =========================================================================
    op.create_table(
        "meeting_notes",
        sa.Column("note_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        sa.Column("note_type", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("prompt_used", sa.Text, nullable=True),
        sa.Column("context_sentences", postgresql.JSON, nullable=True),
        sa.Column("speaker_name", sa.String(255), nullable=True),
        sa.Column("transcript_range_start", sa.Float, nullable=True),
        sa.Column("transcript_range_end", sa.Float, nullable=True),
        sa.Column("llm_backend", sa.String(100), nullable=True),
        sa.Column("llm_model", sa.String(255), nullable=True),
        sa.Column("processing_time_ms", sa.Float, nullable=True),
        sa.Column("note_metadata", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_meeting_notes_session_id", "meeting_notes", ["session_id"])
    op.create_index("idx_meeting_notes_note_type", "meeting_notes", ["note_type"])
    op.create_index("idx_meeting_notes_created_at", "meeting_notes", ["created_at"])

    # =========================================================================
    # 3. meeting_insights
    # =========================================================================
    op.create_table(
        "meeting_insights",
        sa.Column("insight_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        sa.Column(
            "template_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("insight_prompt_templates.template_id"),
            nullable=True,
        ),
        sa.Column("insight_type", sa.String(50), nullable=False, server_default="summary"),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("prompt_used", sa.Text, nullable=True),
        sa.Column("transcript_length", sa.Integer, nullable=True),
        sa.Column("llm_backend", sa.String(100), nullable=True),
        sa.Column("llm_model", sa.String(255), nullable=True),
        sa.Column("processing_time_ms", sa.Float, nullable=True),
        sa.Column("insight_metadata", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_meeting_insights_session_id", "meeting_insights", ["session_id"])
    op.create_index("idx_meeting_insights_insight_type", "meeting_insights", ["insight_type"])
    op.create_index("idx_meeting_insights_created_at", "meeting_insights", ["created_at"])

    # =========================================================================
    # 4. agent_conversations
    # =========================================================================
    op.create_table(
        "agent_conversations",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("bot_sessions.session_id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("system_context", sa.Text, nullable=True),
        sa.Column("conversation_metadata", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_agent_conversations_session_id", "agent_conversations", ["session_id"])
    op.create_index("idx_agent_conversations_status", "agent_conversations", ["status"])
    op.create_index("idx_agent_conversations_created_at", "agent_conversations", ["created_at"])

    # =========================================================================
    # 5. agent_messages
    # =========================================================================
    op.create_table(
        "agent_messages",
        sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_conversations.conversation_id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("llm_backend", sa.String(100), nullable=True),
        sa.Column("llm_model", sa.String(255), nullable=True),
        sa.Column("processing_time_ms", sa.Float, nullable=True),
        sa.Column("suggested_queries", postgresql.JSON, nullable=True),
        sa.Column("message_metadata", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_agent_messages_conversation_id", "agent_messages", ["conversation_id"])
    op.create_index("idx_agent_messages_role", "agent_messages", ["role"])
    op.create_index("idx_agent_messages_created_at", "agent_messages", ["created_at"])


def downgrade() -> None:
    op.drop_table("agent_messages")
    op.drop_table("agent_conversations")
    op.drop_table("meeting_insights")
    op.drop_table("meeting_notes")
    op.drop_table("insight_prompt_templates")
