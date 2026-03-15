"""Tests for the 013_meeting_tables Alembic migration.

Verifies that:
- upgrade() adds the pipeline columns to meetings, meeting_chunks,
  and meeting_translations (no new tables are created)
- downgrade() removes those columns cleanly

Uses the testcontainers PostgreSQL fixture from conftest.py.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect


@pytest.mark.integration
class TestMeetingMigration:
    """Migration round-trip test against a real PostgreSQL testcontainer."""

    def _get_tables(self, engine) -> set[str]:
        """Return the set of table names in the public schema."""
        inspector = inspect(engine)
        return set(inspector.get_table_names(schema="public"))

    def _get_columns(self, engine, table: str) -> set[str]:
        inspector = inspect(engine)
        return {c["name"] for c in inspector.get_columns(table)}

    def test_upgrade_does_not_create_duplicate_tables(self, database_url, run_migrations):
        """After upgrade head the duplicate tables must NOT exist."""
        engine = create_engine(database_url)
        try:
            tables = self._get_tables(engine)
            assert "meeting_sessions" not in tables, (
                "meeting_sessions duplicate table must not exist — use meetings instead"
            )
            assert "meeting_transcripts" not in tables, (
                "meeting_transcripts duplicate table must not exist — use meeting_chunks instead"
            )
            assert "session_translations" not in tables, (
                "session_translations duplicate table must not exist — use meeting_translations instead"
            )
        finally:
            engine.dispose()

    def test_upgrade_extends_meetings_table(self, database_url, run_migrations):
        """After upgrade, meetings must have all pipeline columns."""
        engine = create_engine(database_url)
        try:
            cols = self._get_columns(engine, "meetings")
            expected = {
                "id",
                "source",
                "status",
                "started_at",
                "ended_at",
                "source_languages",
                "target_languages",
                "recording_path",
                "metadata",
                "last_activity_at",
            }
            assert expected <= cols, f"Missing columns in meetings: {expected - cols}"
        finally:
            engine.dispose()

    def test_upgrade_extends_meeting_chunks_table(self, database_url, run_migrations):
        """After upgrade, meeting_chunks must have all pipeline columns."""
        engine = create_engine(database_url)
        try:
            cols = self._get_columns(engine, "meeting_chunks")
            expected = {
                "id",
                "meeting_id",
                "chunk_id",
                "text",
                "speaker_name",
                "timestamp_ms",
                "is_final",
                "confidence",
                "source_language",
                "source_id",
                "speaker_id",
                "created_at",
            }
            assert expected <= cols, f"Missing columns in meeting_chunks: {expected - cols}"
        finally:
            engine.dispose()

    def test_upgrade_extends_meeting_translations_table(self, database_url, run_migrations):
        """After upgrade, meeting_translations must have chunk_id column."""
        engine = create_engine(database_url)
        try:
            cols = self._get_columns(engine, "meeting_translations")
            expected = {
                "id",
                "sentence_id",
                "chunk_id",
                "target_language",
                "translated_text",
                "model_used",
                "created_at",
            }
            assert expected <= cols, f"Missing columns in meeting_translations: {expected - cols}"
        finally:
            engine.dispose()

    def test_downgrade_removes_pipeline_columns(self, database_url, run_migrations):
        """Downgrade -1 must remove the added columns, leaving base schema intact."""
        from alembic import command
        from alembic.config import Config

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        cfg = Config(str(alembic_ini))

        async_url = database_url
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        cfg.set_main_option("sqlalchemy.url", async_url)

        # Downgrade one step (removes 013_meeting_tables additions)
        command.downgrade(cfg, "-1")

        engine = create_engine(database_url)
        try:
            cols = self._get_columns(engine, "meetings")
            assert "last_activity_at" not in cols, "last_activity_at still present after downgrade"
            assert "recording_path" not in cols, "recording_path still present after downgrade"
            assert "started_at" not in cols, "started_at still present after downgrade"
            # Base columns must remain
            assert "id" in cols
            assert "source" in cols
            assert "status" in cols

            chunk_cols = self._get_columns(engine, "meeting_chunks")
            assert "timestamp_ms" not in chunk_cols, "timestamp_ms still present after downgrade"
            assert "is_final" not in chunk_cols, "is_final still present after downgrade"

            trans_cols = self._get_columns(engine, "meeting_translations")
            assert "chunk_id" not in trans_cols, "chunk_id still present after downgrade"
            # sentence_id must be back and still present
            assert "sentence_id" in trans_cols
        finally:
            engine.dispose()

            # Re-run upgrade to restore state for subsequent tests in the session
            command.upgrade(cfg, "head")
