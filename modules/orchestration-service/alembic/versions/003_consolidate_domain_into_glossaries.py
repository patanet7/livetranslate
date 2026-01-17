"""Consolidate domain terminology into glossaries

Revision ID: 003_consolidate_glossaries
Revises: 002_session_id_nullable
Create Date: 2025-01-17

This migration consolidates the domain_terminology system into the glossaries system:
1. Adds phonetic and common_context columns to glossary_entries
2. Drops all unused domain_* tables (all empty, 0 rows)

The glossary system already supports:
- domain field for categorization
- priority for term importance
- translations (JSON) for multi-language support
- case_sensitive and match_whole_word for matching behavior

We add:
- phonetic: pronunciation hints for Whisper prompting
- common_context: example usage context from domain_terminology

Tables dropped (in FK order - children first, then parent):
- domain_usage_logs (FK to domain_categories, domain_prompts)
- user_domain_preferences (FK to domain_categories)
- domain_prompts (FK to domain_categories)
- domain_terminology (FK to domain_categories)
- domain_categories (parent table, dropped last)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "003_consolidate_glossaries"
down_revision: Union[str, None] = "002_session_id_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = connection.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = :table_name
            )
        """),
        {"table_name": table_name}
    )
    return result.scalar()


def column_exists(connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    result = connection.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = :table_name
                AND column_name = :column_name
            )
        """),
        {"table_name": table_name, "column_name": column_name}
    )
    return result.scalar()


def index_exists(connection, index_name: str) -> bool:
    """Check if an index exists."""
    result = connection.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE indexname = :index_name
            )
        """),
        {"index_name": index_name}
    )
    return result.scalar()


def upgrade() -> None:
    """Add columns to glossary_entries and drop unused domain_* tables.

    This migration is idempotent - safe to run multiple times.
    """
    connection = op.get_bind()

    # =========================================================================
    # Step 1: Add new columns to glossary_entries
    # =========================================================================

    # Add phonetic column if it doesn't exist
    if not column_exists(connection, 'glossary_entries', 'phonetic'):
        op.add_column(
            'glossary_entries',
            sa.Column('phonetic', sa.String(255), nullable=True)
        )
        print("Added column: glossary_entries.phonetic")
    else:
        print("Column glossary_entries.phonetic already exists, skipping...")

    # Add common_context column if it doesn't exist
    if not column_exists(connection, 'glossary_entries', 'common_context'):
        op.add_column(
            'glossary_entries',
            sa.Column('common_context', sa.Text(), nullable=True)
        )
        print("Added column: glossary_entries.common_context")
    else:
        print("Column glossary_entries.common_context already exists, skipping...")

    # =========================================================================
    # Step 2: Drop domain_* tables in FK dependency order
    # =========================================================================

    # Drop domain_usage_logs first (has FK to domain_categories and domain_prompts)
    if table_exists(connection, 'domain_usage_logs'):
        # Drop indexes first
        for idx in ['ix_domain_usage_session', 'ix_domain_usage_logs_created_at',
                    'ix_domain_usage_domain_date', 'ix_domain_usage_domain',
                    'ix_domain_usage_created']:
            if index_exists(connection, idx):
                op.drop_index(idx, table_name='domain_usage_logs')
        op.drop_table('domain_usage_logs')
        print("Dropped table: domain_usage_logs")
    else:
        print("Table domain_usage_logs does not exist, skipping...")

    # Drop user_domain_preferences (has FK to domain_categories)
    if table_exists(connection, 'user_domain_preferences'):
        # Drop indexes first
        for idx in ['ix_user_domain_prefs_user_domain', 'ix_user_domain_prefs_user',
                    'ix_user_domain_prefs_domain']:
            if index_exists(connection, idx):
                op.drop_index(idx, table_name='user_domain_preferences')
        op.drop_table('user_domain_preferences')
        print("Dropped table: user_domain_preferences")
    else:
        print("Table user_domain_preferences does not exist, skipping...")

    # Drop domain_prompts (has FK to domain_categories)
    if table_exists(connection, 'domain_prompts'):
        # Drop indexes first
        for idx in ['ix_domain_prompts_usage', 'ix_domain_prompts_domain',
                    'ix_domain_prompts_default']:
            if index_exists(connection, idx):
                op.drop_index(idx, table_name='domain_prompts')
        op.drop_table('domain_prompts')
        print("Dropped table: domain_prompts")
    else:
        print("Table domain_prompts does not exist, skipping...")

    # Drop domain_terminology (has FK to domain_categories)
    if table_exists(connection, 'domain_terminology'):
        # Drop indexes first
        for idx in ['ix_domain_terminology_term', 'ix_domain_terminology_normalized',
                    'ix_domain_terminology_importance', 'ix_domain_terminology_domain_importance']:
            if index_exists(connection, idx):
                op.drop_index(idx, table_name='domain_terminology')
        op.drop_table('domain_terminology')
        print("Dropped table: domain_terminology")
    else:
        print("Table domain_terminology does not exist, skipping...")

    # Drop domain_categories last (parent table)
    if table_exists(connection, 'domain_categories'):
        # Drop indexes first
        for idx in ['ix_domain_categories_usage', 'ix_domain_categories_name',
                    'ix_domain_categories_active']:
            if index_exists(connection, idx):
                op.drop_index(idx, table_name='domain_categories')
        op.drop_table('domain_categories')
        print("Dropped table: domain_categories")
    else:
        print("Table domain_categories does not exist, skipping...")

    print("\nMigration 003_consolidate_glossaries completed successfully.")


def downgrade() -> None:
    """Recreate domain_* tables and remove added columns from glossary_entries.

    Note: This recreates empty tables. Original data cannot be recovered.
    """
    connection = op.get_bind()

    # =========================================================================
    # Step 1: Recreate domain_categories (parent table first)
    # =========================================================================

    if not table_exists(connection, 'domain_categories'):
        op.create_table(
            'domain_categories',
            sa.Column('domain_id', sa.UUID(), nullable=False),
            sa.Column('name', sa.String(length=100), nullable=False),
            sa.Column('display_name', sa.String(length=255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=True),
            sa.Column('usage_count', sa.Integer(), nullable=False),
            sa.Column('success_rate', sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint('domain_id')
        )
        op.create_index('ix_domain_categories_active', 'domain_categories',
                        ['is_active'], unique=False)
        op.create_index(op.f('ix_domain_categories_name'), 'domain_categories',
                        ['name'], unique=True)
        op.create_index('ix_domain_categories_usage', 'domain_categories',
                        ['usage_count'], unique=False)
        print("Recreated table: domain_categories")

    # =========================================================================
    # Step 2: Recreate domain_terminology
    # =========================================================================

    if not table_exists(connection, 'domain_terminology'):
        op.create_table(
            'domain_terminology',
            sa.Column('term_id', sa.UUID(), nullable=False),
            sa.Column('domain_id', sa.UUID(), nullable=False),
            sa.Column('term', sa.String(length=255), nullable=False),
            sa.Column('normalized_term', sa.String(length=255), nullable=False),
            sa.Column('phonetic', sa.String(length=255), nullable=True),
            sa.Column('common_context', sa.Text(), nullable=True),
            sa.Column('alternatives', sa.ARRAY(sa.String()), nullable=True),
            sa.Column('importance', sa.Integer(), nullable=False),
            sa.Column('frequency', sa.Integer(), nullable=False),
            sa.Column('accuracy_improvement', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['domain_id'], ['domain_categories.domain_id']),
            sa.PrimaryKeyConstraint('term_id')
        )
        op.create_index('ix_domain_terminology_domain_importance', 'domain_terminology',
                        ['domain_id', 'importance'], unique=False)
        op.create_index('ix_domain_terminology_importance', 'domain_terminology',
                        ['importance'], unique=False)
        op.create_index('ix_domain_terminology_normalized', 'domain_terminology',
                        ['normalized_term'], unique=False)
        op.create_index('ix_domain_terminology_term', 'domain_terminology',
                        ['term'], unique=False)
        print("Recreated table: domain_terminology")

    # =========================================================================
    # Step 3: Recreate domain_prompts
    # =========================================================================

    if not table_exists(connection, 'domain_prompts'):
        op.create_table(
            'domain_prompts',
            sa.Column('prompt_id', sa.UUID(), nullable=False),
            sa.Column('domain_id', sa.UUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('template', sa.Text(), nullable=False),
            sa.Column('is_default', sa.Boolean(), nullable=False),
            sa.Column('max_tokens', sa.Integer(), nullable=False),
            sa.Column('usage_count', sa.Integer(), nullable=False),
            sa.Column('average_quality_score', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['domain_id'], ['domain_categories.domain_id']),
            sa.PrimaryKeyConstraint('prompt_id')
        )
        op.create_index('ix_domain_prompts_default', 'domain_prompts',
                        ['is_default'], unique=False)
        op.create_index('ix_domain_prompts_domain', 'domain_prompts',
                        ['domain_id'], unique=False)
        op.create_index('ix_domain_prompts_usage', 'domain_prompts',
                        ['usage_count'], unique=False)
        print("Recreated table: domain_prompts")

    # =========================================================================
    # Step 4: Recreate user_domain_preferences
    # =========================================================================

    if not table_exists(connection, 'user_domain_preferences'):
        op.create_table(
            'user_domain_preferences',
            sa.Column('preference_id', sa.UUID(), nullable=False),
            sa.Column('user_id', sa.String(length=255), nullable=False),
            sa.Column('domain_id', sa.UUID(), nullable=False),
            sa.Column('custom_terminology', postgresql.JSONB(astext_type=sa.Text()),
                      nullable=True),
            sa.Column('custom_prompt_template', sa.Text(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False),
            sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('usage_count', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['domain_id'], ['domain_categories.domain_id']),
            sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
            sa.PrimaryKeyConstraint('preference_id')
        )
        op.create_index('ix_user_domain_prefs_domain', 'user_domain_preferences',
                        ['domain_id'], unique=False)
        op.create_index('ix_user_domain_prefs_user', 'user_domain_preferences',
                        ['user_id'], unique=False)
        op.create_index('ix_user_domain_prefs_user_domain', 'user_domain_preferences',
                        ['user_id', 'domain_id'], unique=True)
        print("Recreated table: user_domain_preferences")

    # =========================================================================
    # Step 5: Recreate domain_usage_logs
    # =========================================================================

    if not table_exists(connection, 'domain_usage_logs'):
        op.create_table(
            'domain_usage_logs',
            sa.Column('log_id', sa.UUID(), nullable=False),
            sa.Column('session_id', sa.UUID(), nullable=True),
            sa.Column('domain_id', sa.UUID(), nullable=False),
            sa.Column('prompt_id', sa.UUID(), nullable=True),
            sa.Column('user_id', sa.String(length=255), nullable=True),
            sa.Column('model_used', sa.String(length=100), nullable=False),
            sa.Column('transcription_quality', sa.Integer(), nullable=True),
            sa.Column('error_count', sa.Integer(), nullable=False),
            sa.Column('terminology_matches', sa.Integer(), nullable=False),
            sa.Column('processing_time_ms', sa.Integer(), nullable=True),
            sa.Column('prompt_tokens_used', sa.Integer(), nullable=True),
            sa.Column('context_tokens_used', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True),
                      server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['domain_id'], ['domain_categories.domain_id']),
            sa.ForeignKeyConstraint(['prompt_id'], ['domain_prompts.prompt_id']),
            sa.ForeignKeyConstraint(['session_id'], ['conversation_sessions.session_id']),
            sa.PrimaryKeyConstraint('log_id')
        )
        op.create_index('ix_domain_usage_created', 'domain_usage_logs',
                        ['created_at'], unique=False)
        op.create_index('ix_domain_usage_domain', 'domain_usage_logs',
                        ['domain_id'], unique=False)
        op.create_index('ix_domain_usage_domain_date', 'domain_usage_logs',
                        ['domain_id', 'created_at'], unique=False)
        op.create_index(op.f('ix_domain_usage_logs_created_at'), 'domain_usage_logs',
                        ['created_at'], unique=False)
        op.create_index('ix_domain_usage_session', 'domain_usage_logs',
                        ['session_id'], unique=False)
        print("Recreated table: domain_usage_logs")

    # =========================================================================
    # Step 6: Remove added columns from glossary_entries
    # =========================================================================

    if column_exists(connection, 'glossary_entries', 'common_context'):
        op.drop_column('glossary_entries', 'common_context')
        print("Dropped column: glossary_entries.common_context")

    if column_exists(connection, 'glossary_entries', 'phonetic'):
        op.drop_column('glossary_entries', 'phonetic')
        print("Dropped column: glossary_entries.phonetic")

    print("\nDowngrade of 003_consolidate_glossaries completed.")
