"""Tests for the 013_meeting_tables Alembic migration.

Verifies that:
- upgrade() creates meeting_sessions, meeting_transcripts, session_translations
- downgrade() drops those tables cleanly

Uses the testcontainers PostgreSQL fixture from conftest.py.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text


@pytest.mark.integration
class TestMeetingMigration:
    """Migration round-trip test against a real PostgreSQL testcontainer."""

    def _get_tables(self, engine) -> set[str]:
        """Return the set of table names in the public schema."""
        inspector = inspect(engine)
        return set(inspector.get_table_names(schema="public"))

    def test_upgrade_creates_meeting_tables(self, database_url, run_migrations):
        """After alembic upgrade head the three meeting tables must exist."""
        # run_migrations has already run upgrade head — just verify the result.
        engine = create_engine(database_url)
        try:
            tables = self._get_tables(engine)
            assert "meeting_sessions" in tables, "meeting_sessions table missing after upgrade"
            assert "meeting_transcripts" in tables, "meeting_transcripts table missing after upgrade"
            assert "session_translations" in tables, "session_translations table missing after upgrade"
        finally:
            engine.dispose()

    def test_meeting_sessions_columns(self, database_url, run_migrations):
        """meeting_sessions must have the columns specified in the migration."""
        engine = create_engine(database_url)
        try:
            inspector = inspect(engine)
            cols = {c["name"] for c in inspector.get_columns("meeting_sessions")}
            expected = {
                "id",
                "source_type",
                "status",
                "started_at",
                "ended_at",
                "source_languages",
                "target_languages",
                "recording_path",
                "metadata",
                "last_activity_at",
            }
            assert expected <= cols, f"Missing columns: {expected - cols}"
        finally:
            engine.dispose()

    def test_meeting_transcripts_columns(self, database_url, run_migrations):
        """meeting_transcripts must have the columns specified in the migration."""
        engine = create_engine(database_url)
        try:
            inspector = inspect(engine)
            cols = {c["name"] for c in inspector.get_columns("meeting_transcripts")}
            expected = {
                "id",
                "session_id",
                "timestamp_ms",
                "speaker_id",
                "speaker_name",
                "source_language",
                "source_id",
                "text",
                "confidence",
                "is_final",
                "created_at",
            }
            assert expected <= cols, f"Missing columns: {expected - cols}"
        finally:
            engine.dispose()

    def test_session_translations_columns(self, database_url, run_migrations):
        """session_translations must have the columns specified in the migration."""
        engine = create_engine(database_url)
        try:
            inspector = inspect(engine)
            cols = {c["name"] for c in inspector.get_columns("session_translations")}
            expected = {
                "id",
                "transcript_id",
                "target_language",
                "translated_text",
                "model_used",
                "created_at",
            }
            assert expected <= cols, f"Missing columns: {expected - cols}"
        finally:
            engine.dispose()

    def test_downgrade_removes_meeting_tables(self, database_url, run_migrations):
        """Downgrade -1 must drop the three meeting tables, leaving others intact."""
        from alembic import command
        from alembic.config import Config

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        cfg = Config(str(alembic_ini))

        async_url = database_url
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        cfg.set_main_option("sqlalchemy.url", async_url)

        # Downgrade one step (removes 013_meeting_tables)
        command.downgrade(cfg, "-1")

        engine = create_engine(database_url)
        try:
            tables = self._get_tables(engine)
            assert "meeting_sessions" not in tables, "meeting_sessions still exists after downgrade"
            assert "meeting_transcripts" not in tables, "meeting_transcripts still exists after downgrade"
            assert "session_translations" not in tables, "session_translations still exists after downgrade"
            # The previous migration's tables must still be present
            assert "alembic_version" in tables
        finally:
            engine.dispose()

            # Re-run upgrade to restore state for subsequent tests in the session
            command.upgrade(cfg, "head")
