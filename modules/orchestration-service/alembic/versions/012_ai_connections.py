"""Add ai_connections table and migrate connection data.

Revision ID: 012_ai_connections
Revises: 011_chat_tables
Create Date: 2026-03-12
"""

import json
from pathlib import Path

from alembic import op
import sqlalchemy as sa

revision = "012_ai_connections"
down_revision = "011_chat_tables"
branch_labels = None
depends_on = None

# Path to old translation config (relative to where alembic runs)
_TRANSLATION_CONFIG = Path("config/translation.json")


def upgrade():
    # 1. Create ai_connections table
    op.create_table(
        "ai_connections",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column(
            "engine",
            sa.Text,
            sa.CheckConstraint(
                "engine IN ('ollama', 'openai', 'anthropic', 'openai_compatible')",
                name="ck_ai_connections_engine",
            ),
            nullable=False,
        ),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("api_key", sa.Text, nullable=False, server_default=""),
        sa.Column("prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("context_length", sa.Integer, nullable=True),
        sa.Column("timeout_ms", sa.Integer, nullable=False, server_default="30000"),
        sa.Column("max_retries", sa.Integer, nullable=False, server_default="3"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 2. Migrate existing data (best-effort, inside the same transaction)
    conn = op.get_bind()
    migrated_ids = set()

    # 2a. Migrate from system_config.chat_settings
    row = conn.execute(
        sa.text("SELECT value FROM system_config WHERE key = 'chat_settings'")
    ).fetchone()
    if row:
        try:
            chat_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            provider = chat_data.get("provider", "ollama")
            base_url = chat_data.get("base_url", "http://localhost:11434")
            api_key = chat_data.get("api_key", "")

            conn_id = f"chat-{provider}"
            engine = provider if provider in ("ollama", "openai", "anthropic") else "openai_compatible"

            # Set URL based on provider
            if provider == "openai":
                url = "https://api.openai.com"
            elif provider == "anthropic":
                url = "https://api.anthropic.com"
            else:
                url = base_url or "http://localhost:11434"

            conn.execute(
                sa.text(
                    "INSERT INTO ai_connections (id, name, engine, url, api_key, prefix, enabled) "
                    "VALUES (:id, :name, :engine, :url, :api_key, :prefix, true) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {
                    "id": conn_id,
                    "name": f"Chat {provider.title()}",
                    "engine": engine,
                    "url": url,
                    "api_key": api_key or "",
                    "prefix": provider,
                },
            )
            migrated_ids.add(conn_id)

            # Write chat model preference
            model = chat_data.get("model", "")
            pref = json.dumps({
                "active_model": f"{provider}/{model}" if model else "",
                "temperature": chat_data.get("temperature", 0.7),
                "max_tokens": chat_data.get("max_tokens", 4096),
            })
            conn.execute(
                sa.text(
                    "INSERT INTO system_config (key, value) VALUES ('chat_model_preference', :val) "
                    "ON CONFLICT (key) DO UPDATE SET value = :val"
                ),
                {"val": pref},
            )
        except Exception:
            pass  # Best-effort migration

    # 2b. Migrate from config/translation.json
    if _TRANSLATION_CONFIG.exists():
        try:
            translation_data = json.loads(_TRANSLATION_CONFIG.read_text())
            connections = translation_data.get("connections", [])
            for i, tc in enumerate(connections):
                tc_id = tc.get("id", f"translation-{i}")
                engine = tc.get("engine", "openai_compatible")
                # Map vllm/triton to openai_compatible
                if engine in ("vllm", "triton"):
                    engine = "openai_compatible"

                if tc_id not in migrated_ids:
                    conn.execute(
                        sa.text(
                            "INSERT INTO ai_connections "
                            "(id, name, engine, url, api_key, prefix, enabled, timeout_ms, max_retries) "
                            "VALUES (:id, :name, :engine, :url, :api_key, :prefix, :enabled, :timeout_ms, :max_retries) "
                            "ON CONFLICT (id) DO NOTHING"
                        ),
                        {
                            "id": tc_id,
                            "name": tc.get("name", f"Translation {i}"),
                            "engine": engine,
                            "url": tc.get("url", "http://localhost:5003"),
                            "api_key": tc.get("api_key", ""),
                            "prefix": tc.get("prefix", ""),
                            "enabled": tc.get("enabled", True),
                            "timeout_ms": tc.get("timeout_ms", 30000),
                            "max_retries": tc.get("max_retries", 3),
                        },
                    )
                    migrated_ids.add(tc_id)

            # Write translation model preference
            active_model = translation_data.get("active_model", "")
            fallback_model = translation_data.get("fallback_model", "")
            pref = json.dumps({
                "active_model": active_model,
                "fallback_model": fallback_model,
                "temperature": translation_data.get("quality", {}).get("temperature", 0.3),
                "max_tokens": translation_data.get("quality", {}).get("max_tokens", 512),
            })
            conn.execute(
                sa.text(
                    "INSERT INTO system_config (key, value) VALUES ('translation_model_preference', :val) "
                    "ON CONFLICT (key) DO UPDATE SET value = :val"
                ),
                {"val": pref},
            )

            # Migrate translation sub-configs to system_config rows
            for sub_key in ("languages", "quality", "service", "caching", "realtime"):
                sub_val = translation_data.get(sub_key)
                if sub_val:
                    config_key = f"translation_{sub_key}"
                    conn.execute(
                        sa.text(
                            "INSERT INTO system_config (key, value) VALUES (:key, :val) "
                            "ON CONFLICT (key) DO UPDATE SET value = :val"
                        ),
                        {"key": config_key, "val": json.dumps(sub_val)},
                    )
        except Exception:
            pass  # Best-effort migration

    # 2c. Write intelligence model preference (defaults)
    intel_pref = json.dumps({
        "active_model": "",
        "temperature": 0.3,
        "max_tokens": 1024,
    })
    conn.execute(
        sa.text(
            "INSERT INTO system_config (key, value) VALUES ('intelligence_model_preference', :val) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"val": intel_pref},
    )

    # 2d. Seed default if no connections migrated
    if not migrated_ids:
        conn.execute(
            sa.text(
                "INSERT INTO ai_connections (id, name, engine, url, prefix, enabled) "
                "VALUES ('local', 'Local Ollama', 'ollama', 'http://localhost:11434', 'local', true) "
                "ON CONFLICT (id) DO NOTHING"
            )
        )


def downgrade():
    op.drop_table("ai_connections")
    conn = op.get_bind()
    for key in (
        "chat_model_preference",
        "translation_model_preference",
        "intelligence_model_preference",
        "translation_languages",
        "translation_quality",
        "translation_service",
        "translation_caching",
        "translation_realtime",
    ):
        conn.execute(sa.text("DELETE FROM system_config WHERE key = :key"), {"key": key})
