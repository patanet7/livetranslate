"""Make session_events.session_id nullable for dashboard settings

Revision ID: 002_session_id_nullable
Revises: 5f3bcf8a26da
Create Date: 2025-01-17

This migration makes the session_id column in session_events nullable to support
storing dashboard settings (translation model preference, prompt templates, etc.)
that are not tied to any specific bot session.

These settings use event_type='dashboard_setting' and do not require a session_id.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_session_id_nullable"
down_revision: str | None = "5f3bcf8a26da"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Make session_id nullable in session_events table.

    This migration is idempotent - it checks the current state before modifying.
    Safe to run multiple times.
    """
    # Use a connection to check current column state
    connection = op.get_bind()

    # Check if column is already nullable (PostgreSQL specific query)
    result = connection.execute(
        sa.text("""
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_name = 'session_events'
            AND column_name = 'session_id'
        """)
    )
    row = result.fetchone()

    if row and row[0] == "YES":
        # Column is already nullable, nothing to do
        print("session_events.session_id is already nullable, skipping...")
        return

    # Make the column nullable by dropping the NOT NULL constraint
    # This also requires temporarily dropping and recreating the foreign key

    # Step 1: Drop the foreign key constraint if it exists
    # PostgreSQL stores constraint names - we need to find the FK name
    fk_result = connection.execute(
        sa.text("""
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = 'session_events'
            AND kcu.column_name = 'session_id'
        """)
    )
    fk_row = fk_result.fetchone()

    if fk_row:
        fk_name = fk_row[0]
        op.drop_constraint(fk_name, "session_events", type_="foreignkey")

    # Step 2: Alter column to be nullable
    op.alter_column(
        "session_events",
        "session_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
        existing_nullable=False,
    )

    # Step 3: Recreate the foreign key constraint (allowing NULL values)
    op.create_foreign_key(
        "fk_session_events_session_id",
        "session_events",
        "bot_sessions",
        ["session_id"],
        ["session_id"],
        ondelete="CASCADE",
    )

    print("Successfully made session_events.session_id nullable")


def downgrade() -> None:
    """Revert session_id to NOT NULL.

    WARNING: This will fail if there are any rows with NULL session_id.
    You must first delete or update those rows.
    """
    connection = op.get_bind()

    # Check if there are any NULL session_id values
    result = connection.execute(
        sa.text("""
            SELECT COUNT(*)
            FROM session_events
            WHERE session_id IS NULL
        """)
    )
    null_count = result.scalar()

    if null_count and null_count > 0:
        raise RuntimeError(
            f"Cannot downgrade: {null_count} rows have NULL session_id. "
            "Delete or update these rows before running downgrade."
        )

    # Drop the foreign key constraint
    op.drop_constraint("fk_session_events_session_id", "session_events", type_="foreignkey")

    # Alter column back to NOT NULL
    op.alter_column(
        "session_events",
        "session_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
        existing_nullable=True,
    )

    # Recreate the foreign key constraint
    op.create_foreign_key(
        "fk_session_events_session_id",
        "session_events",
        "bot_sessions",
        ["session_id"],
        ["session_id"],
        ondelete="CASCADE",
    )

    print("Successfully reverted session_events.session_id to NOT NULL")
