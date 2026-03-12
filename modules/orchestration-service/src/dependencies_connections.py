"""Async dependency helpers for resolving LLM clients from ai_connections."""

import json

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from clients.llm_client import create_llm_client
from clients.protocol import LLMClientProtocol
from database import get_db_session
from livetranslate_common.logging import get_logger

logger = get_logger()


async def resolve_intelligence_llm_client(
    db: AsyncSession = Depends(get_db_session),
) -> LLMClientProtocol | None:
    """Resolve the intelligence LLM client from ai_connections at request time.

    Returns None if no preference is configured, allowing fallback behavior.
    """
    try:
        result = await db.execute(
            text("SELECT value FROM system_config WHERE key = 'intelligence_model_preference'")
        )
        row = result.fetchone()
        if not row or not row[0]:
            return None

        pref = json.loads(row[0])
        active_model = pref.get("active_model", "")
        if not active_model or "/" not in active_model:
            return None

        prefix, model_name = active_model.split("/", 1)
        conn_result = await db.execute(
            text(
                "SELECT url, api_key, engine FROM ai_connections "
                "WHERE prefix = :prefix AND enabled = true"
            ),
            {"prefix": prefix},
        )
        conn_row = conn_result.fetchone()
        if not conn_row:
            return None

        base_url = conn_row[0].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        return create_llm_client(
            base_url=base_url,
            api_key=conn_row[1] or "",
            model=model_name,
            max_tokens=pref.get("max_tokens", 1024),
            temperature=pref.get("temperature", 0.3),
            proxy_mode=False,
        )
    except Exception as e:
        logger.warning("intelligence_llm_resolve_failed", error=str(e))
        return None
