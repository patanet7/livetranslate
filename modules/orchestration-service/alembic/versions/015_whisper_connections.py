"""Add whisper_connections table + transcription_model_preference.

Mirror of `012_ai_connections` for the Whisper inference side. Stores
"which Whisper backend to use" the same way ai_connections stores "which
LLM backend to use". Read by `services.whisper_resolver`, written by the
dashboard /config/connections UI (Whisper section).

Schema fields parallel ai_connections except:
  - `default_model` column carries the preferred Whisper model name
    (e.g. "mlx-community/whisper-large-v3-turbo"). The LLM side stores
    model under preference JSON; the Whisper side promotes it to a column
    because a single Whisper "connection" almost always implies a single
    Whisper model, while an LLM connection (Ollama, OpenAI) routinely
    serves many models from one URL.
  - engine check constraint restricted to Whisper-relevant engines.

Revision IDs ≤ 32 chars per orchestration-service/CLAUDE.md.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "015_whisper_connections"
down_revision = "014_seed_llm_prefs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "whisper_connections",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column(
            "engine",
            sa.Text,
            sa.CheckConstraint(
                "engine IN ('openai_compatible', 'mlx_local', 'faster_whisper_local')",
                name="ck_whisper_connections_engine",
            ),
            nullable=False,
        ),
        sa.Column("url", sa.Text, nullable=False, server_default=""),
        sa.Column("api_key", sa.Text, nullable=False, server_default=""),
        sa.Column("prefix", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "default_model",
            sa.Text,
            nullable=False,
            server_default="mlx-community/whisper-large-v3-turbo",
        ),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("timeout_ms", sa.Integer, nullable=False, server_default="30000"),
        sa.Column("max_retries", sa.Integer, nullable=False, server_default="1"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Seed the transcription model preference row (active_model: null means
    # fall-through to default-enabled connection in the resolver).
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "INSERT INTO system_config (key, value) VALUES "
            "('transcription_model_preference', :v) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"v": '{"active_model": null}'},
    )


def downgrade() -> None:
    op.drop_table("whisper_connections")
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "DELETE FROM system_config WHERE key = 'transcription_model_preference'"
        )
    )
