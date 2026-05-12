"""Compatibility shim — superseded by `services.llm_resolver`.

The old `resolve_intelligence_llm_client` is preserved as a thin wrapper
around `services.llm_resolver.resolve_llm_client` so existing FastAPI
`Depends(...)` wiring (e.g. `routers/insights.py`) keeps working.

New code MUST import from `services.llm_resolver` directly.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db_session
from livetranslate_common.logging import get_logger
from services.llm_resolver import resolve_llm_client

logger = get_logger()


async def resolve_intelligence_llm_client(
    db: AsyncSession = Depends(get_db_session),
):
    """FastAPI-friendly resolver for the 'intelligence' purpose.

    Returns the merged LLMClient. Always returns a client (the new resolver
    has a hard-coded last-resort default) — never None. Callers that branched
    on None should treat 'always present' as the new contract.
    """
    try:
        return await resolve_llm_client("intelligence", db)
    except Exception as e:
        logger.warning("intelligence_llm_resolve_failed", error=str(e))
        return None
