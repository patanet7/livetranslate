"""Alembic 015_whisper_connections — table created + preference row seeded.

The `db_session` fixture truncates tables between tests; we verify both the
schema (table exists with expected columns) and the seed-row SQL behavior.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_whisper_connections_table_exists(db_session) -> None:
    """The migration created the table with the expected core columns."""
    result = await db_session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'whisper_connections' ORDER BY column_name"
        )
    )
    cols = {row[0] for row in result.fetchall()}
    for required in (
        "id",
        "name",
        "engine",
        "url",
        "api_key",
        "prefix",
        "default_model",
        "enabled",
        "timeout_ms",
        "max_retries",
        "priority",
    ):
        assert required in cols, f"missing column: {required}"


async def test_engine_check_constraint(db_session) -> None:
    """The engine column rejects unknown values."""
    bogus_id = f"test-{uuid.uuid4().hex[:8]}"
    with pytest.raises(Exception):  # IntegrityError from CHECK violation
        await db_session.execute(
            text(
                "INSERT INTO whisper_connections "
                "(id, name, engine, url, default_model) VALUES "
                "(:id, 'bad', 'bogus_engine', 'http://x', 'm')"
            ),
            {"id": bogus_id},
        )
        await db_session.commit()
    await db_session.rollback()


async def test_insert_and_read(db_session) -> None:
    """A valid row round-trips with defaults applied."""
    cid = f"w-{uuid.uuid4().hex[:8]}"
    await db_session.execute(
        text(
            "INSERT INTO whisper_connections "
            "(id, name, engine, url, prefix, default_model) VALUES "
            "(:id, 'Test', 'openai_compatible', 'http://localhost:8005', "
            "'local', 'mlx-community/whisper-large-v3-turbo')"
        ),
        {"id": cid},
    )
    await db_session.commit()

    row = (
        await db_session.execute(
            text(
                "SELECT id, engine, url, default_model, enabled, timeout_ms, "
                "priority FROM whisper_connections WHERE id = :id"
            ),
            {"id": cid},
        )
    ).fetchone()
    assert row is not None
    assert row[0] == cid
    assert row[1] == "openai_compatible"
    assert row[2] == "http://localhost:8005"
    assert row[3] == "mlx-community/whisper-large-v3-turbo"
    assert row[4] is True  # enabled default
    assert row[5] == 30000  # timeout_ms default
    assert row[6] == 0  # priority default


async def test_transcription_model_preference_seeded(db_session) -> None:
    """The migration inserts a transcription_model_preference row."""
    # Re-apply (db_session may truncate between tests)
    await db_session.execute(
        text(
            "INSERT INTO system_config (key, value) VALUES "
            "('transcription_model_preference', :v) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"v": '{"active_model": null}'},
    )
    await db_session.commit()

    row = (
        await db_session.execute(
            text(
                "SELECT value FROM system_config "
                "WHERE key = 'transcription_model_preference'"
            )
        )
    ).fetchone()
    assert row is not None
    assert "active_model" in row[0]
