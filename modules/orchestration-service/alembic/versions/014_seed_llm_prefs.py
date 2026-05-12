"""Seed model-preference rows for purpose-keyed LLM resolution.

Adds three system_config rows so the new `services.llm_resolver` has
explicit per-purpose preferences to read instead of falling through to
step-3 (default enabled connection) on every translation. The value is
JSON `{"active_model": null}` — null means "no preference set", which
the resolver treats as a fall-through trigger. Users set a real model
via the dashboard Connections UI.

Revision IDs ≤ 32 chars per orchestration-service/CLAUDE.md.
"""

from __future__ import annotations

revision = "014_seed_llm_prefs"
down_revision = "013_meeting_tables"
branch_labels = None
depends_on = None

import sqlalchemy as sa
from alembic import op


_KEYS = (
    "translation_model_preference",
    "chat_model_preference",
    "fireflies_model_preference",
    "meetings_model_preference",
)

# `intelligence_model_preference` already exists in production data — don't
# overwrite it on upgrade.
_NEW_KEYS_ON_DOWNGRADE = _KEYS


def upgrade() -> None:
    conn = op.get_bind()
    for key in _KEYS:
        conn.execute(
            sa.text(
                "INSERT INTO system_config (key, value) "
                "VALUES (:k, :v) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {"k": key, "v": '{"active_model": null}'},
        )


def downgrade() -> None:
    conn = op.get_bind()
    for key in _NEW_KEYS_ON_DOWNGRADE:
        conn.execute(
            sa.text("DELETE FROM system_config WHERE key = :k"),
            {"k": key},
        )
