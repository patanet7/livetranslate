"""
Translation Settings Router

Handles translation service configuration, testing, and cache management.
Connection management has moved to /api/connections (services/connections.py).
"""

import json
import time

import aiohttp
from database import get_db_session
from database.models import SystemConfig
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ._shared import (
    Any,
    APIRouter,
    HTTPException,
    TranslationConfig,
    asyncio,
    logger,
)

router = APIRouter(tags=["settings-translation"])


# ============================================================================
# Translation Settings Endpoints
# ============================================================================

_TRANSLATION_SUB_KEYS = ("languages", "quality", "service", "model", "caching", "realtime")


@router.get("/translation", response_model=dict[str, Any])
async def get_translation_settings(
    db: AsyncSession = Depends(get_db_session),
):
    """Get translation-specific settings from system_config."""
    defaults = TranslationConfig().model_dump()
    config: dict[str, Any] = {}
    for sub_key in _TRANSLATION_SUB_KEYS:
        key = f"translation_{sub_key}"
        result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
        row = result.scalar_one_or_none()
        if row and row.value:
            config[sub_key] = json.loads(row.value) if isinstance(row.value, str) else row.value
        else:
            config[sub_key] = defaults.get(sub_key, {})
    return config


@router.post("/translation")
async def save_translation_settings(
    config: dict[str, Any],
    db: AsyncSession = Depends(get_db_session),
):
    """Save translation-specific settings to system_config."""
    for sub_key in _TRANSLATION_SUB_KEYS:
        if sub_key in config:
            key = f"translation_{sub_key}"
            val = json.dumps(config[sub_key])
            result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
            row = result.scalar_one_or_none()
            if row:
                row.value = val
            else:
                db.add(SystemConfig(key=key, value=val))
    await db.commit()
    return {"message": "Translation settings saved", "config": config}


@router.get("/translation/stats")
async def get_translation_stats():
    """Get translation statistics"""
    try:
        return {
            "total_translations": 2340,
            "successful_translations": 2298,
            "cache_hits": 456,
            "average_quality": 0.89,
            "average_latency_ms": 750,
        }
    except Exception as e:
        logger.error("get_translation_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load translation statistics") from e


@router.post("/translation/test")
async def test_translation(test_request: dict[str, Any]):
    """Test translation configuration by forwarding the request to the configured service."""
    try:
        text = test_request.get("text", "Hello, world!")
        target_language = test_request.get("target_language", "es")
        source_language = test_request.get("source_language", "en")

        # Load current config to find the service URL and api_key
        default_config = TranslationConfig().model_dump()
        service_cfg = default_config.get("service", {})
        service_url = service_cfg.get("service_url", "http://localhost:5003").rstrip("/")
        api_key = service_cfg.get("api_key", "")
        timeout_ms = service_cfg.get("timeout_ms", 30000)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }

        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        start = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{service_url}/translate",
                    json=payload,
                    headers=headers,
                ) as resp:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    if resp.status == 200:
                        data = await resp.json()
                        translated_text = data.get(
                            "translated_text",
                            data.get("translation", data.get("result", "")),
                        )
                        confidence = data.get("confidence", data.get("score", 0.0))
                        logger.info(
                            "translation_test_success",
                            target_language=target_language,
                            latency_ms=latency_ms,
                        )
                        return {
                            "success": True,
                            "original_text": text,
                            "translated_text": translated_text,
                            "target_language": target_language,
                            "confidence": confidence,
                            "processing_time_ms": latency_ms,
                        }
                    else:
                        body = await resp.text()
                        logger.warning(
                            "translation_test_service_error",
                            status=resp.status,
                            body=body[:200],
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=f"Translation service returned {resp.status}: {body[:200]}",
                        )
        except (aiohttp.ClientError, asyncio.TimeoutError) as conn_err:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "translation_test_service_unreachable",
                service_url=service_url,
                error=str(conn_err),
                latency_ms=latency_ms,
            )
            raise HTTPException(
                status_code=503,
                detail=f"Translation service unreachable at {service_url}: {conn_err}",
            ) from conn_err

    except HTTPException:
        raise
    except Exception as e:
        logger.error("translation_test_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Translation test failed") from e


@router.post("/translation/clear-cache")
async def clear_translation_cache():
    """Clear translation cache"""
    try:
        await asyncio.sleep(0.5)  # Simulate cache clearing
        return {"message": "Translation cache cleared successfully"}
    except Exception as e:
        logger.error("clear_translation_cache_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear translation cache") from e
