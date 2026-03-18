"""Meetings API router for history, search, upload, and insight generation."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from livetranslate_common.logging import get_logger
from dependencies import get_database_manager, get_translation_service_client
from meeting.translation_recovery import (
    get_translation_recovery_metrics,
    recover_pending_translations_with_fresh_session,
)
from services.meeting_store import MeetingStore

logger = get_logger()

router = APIRouter(prefix="/meetings", tags=["meetings"])

# Lazy-initialized singleton
_meeting_store: MeetingStore | None = None


async def _get_store() -> MeetingStore:
    """Get or create MeetingStore singleton."""
    global _meeting_store
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    if _meeting_store is None:
        _meeting_store = MeetingStore(db_url)
        await _meeting_store.initialize()
    return _meeting_store


# --- Pydantic Models ---


class MeetingResponse(BaseModel):
    """Meeting details response."""

    id: str
    title: str | None = None
    status: str = "live"
    source: str = "fireflies"
    created_at: str | None = None
    chunk_count: int = 0
    sentence_count: int = 0


class InsightGenerateRequest(BaseModel):
    """Request to generate AI insights."""

    insight_types: list[str] = Field(
        default=["summary", "action_items", "keywords"],
        description="Types of insights to generate",
    )


# --- Endpoints ---


@router.get("/")
async def list_meetings(
    limit: int = Query(default=50, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
    min_sentences: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List all meetings with pagination.

    Args:
        min_sentences: Minimum transcript unit count to include.
            Uses the larger of sentence_count and chunk_count.
    """
    store = await _get_store()
    result = await store.list_meetings(limit=limit, offset=offset, min_sentences=min_sentences)
    return {
        "meetings": result["meetings"],
        "total": result["total"],
        "limit": limit,
        "offset": offset,
    }


@router.get("/search")
async def search_meetings(
    q: str = Query(description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Full-text search across all meeting transcripts."""
    store = await _get_store()
    results = await store.search_meetings(query=q, limit=limit)
    return {"results": results, "query": q, "count": len(results)}


@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str) -> dict[str, Any]:
    """Get meeting details with stats."""
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    backlog = await store.get_meeting_translation_backlog(meeting_id)
    meeting["translation_backlog"] = backlog
    return {"meeting": meeting}


@router.get("/backlog")
async def get_translation_backlog(
    limit: int = Query(default=50, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
    only_pending: bool = Query(default=True),
) -> dict[str, Any]:
    """List meetings with translation backlog and fleet-level backlog totals."""
    store = await _get_store()
    backlog = await store.list_translation_backlog(
        limit=limit,
        offset=offset,
        only_pending=only_pending,
    )
    counters = await get_translation_recovery_metrics()
    return {
        "meetings": backlog["meetings"],
        "summary": backlog["summary"],
        "total": backlog["total"],
        "limit": limit,
        "offset": offset,
        "recovery_counters": counters,
    }


@router.get("/{meeting_id}/translation-status")
async def get_meeting_translation_status(meeting_id: str) -> dict[str, Any]:
    """Return per-meeting translation backlog and persisted translation counts."""
    store = await _get_store()
    backlog = await store.get_meeting_translation_backlog(meeting_id)
    if not backlog:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"meeting_id": meeting_id, "translation_status": backlog}


@router.post("/{meeting_id}/translations/recover")
async def recover_meeting_translations(
    meeting_id: str,
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict[str, Any]:
    """Re-run shared translation recovery for a single meeting."""
    store = await _get_store()
    backlog_before = await store.get_meeting_translation_backlog(meeting_id)
    if not backlog_before:
        raise HTTPException(status_code=404, detail="Meeting not found")

    db_session = await get_database_manager().get_session_direct()
    try:
        stats = await recover_pending_translations_with_fresh_session(
            db_session,
            Path(os.getenv("RECORDING_BASE_PATH", str(Path.home() / ".livetranslate" / "recordings"))),
            get_translation_service_client(),
            meeting_ids=[uuid.UUID(meeting_id)],
            limit=limit,
        )
    finally:
        await db_session.close()

    backlog_after = await store.get_meeting_translation_backlog(meeting_id)
    return {
        "meeting_id": meeting_id,
        "before": backlog_before,
        "recovery": stats,
        "after": backlog_after,
    }


@router.get("/{meeting_id}/insights")
async def get_meeting_insights(meeting_id: str) -> dict[str, Any]:
    """Get all AI insights for a meeting."""
    store = await _get_store()
    # Verify meeting exists
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    insights = await store.get_meeting_insights(meeting_id)
    return {"meeting_id": meeting_id, "insights": insights, "count": len(insights)}


@router.get("/{meeting_id}/speakers")
async def get_meeting_speakers(meeting_id: str) -> dict[str, Any]:
    """Get all speakers for a meeting with analytics."""
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    speakers = await store.get_meeting_speakers(meeting_id)
    return {"meeting_id": meeting_id, "speakers": speakers, "count": len(speakers)}


@router.post("/{meeting_id}/sync")
async def sync_meeting(
    meeting_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Trigger a re-sync of Fireflies intelligence data for a meeting.

    Looks up the meeting's fireflies_transcript_id and triggers a full
    data download in the background.
    """
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    ff_id = meeting.get("fireflies_transcript_id")
    if not ff_id:
        raise HTTPException(
            status_code=400,
            detail="Not a Fireflies meeting — no transcript ID to sync",
        )

    # Mark as syncing immediately
    await store.update_sync_status(meeting_id, "syncing")

    # Trigger background download (pass known_meeting_id for accurate error handling)
    from routers.fireflies import _download_meeting_data

    background_tasks.add_task(_download_meeting_data, ff_id, meeting_id)

    logger.info("meeting_sync_triggered", meeting_id=meeting_id, ff_id=ff_id)
    return {"success": True, "message": "Sync started"}


@router.get("/{meeting_id}/transcript")
async def get_meeting_transcript(meeting_id: str) -> dict[str, Any]:
    """Get full transcript (sentences + translations) for a meeting."""
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Get sentences and their translations
    pool = store._pool
    sentences = await pool.fetch(
        """
        SELECT ms.*,
               COALESCE(
                   json_agg(
                       json_build_object(
                           'translated_text', mt.translated_text,
                           'target_language', mt.target_language,
                           'confidence', mt.confidence,
                           'model_used', mt.model_used
                       )
                   ) FILTER (WHERE mt.id IS NOT NULL),
                   '[]'::json
               ) as translations
        FROM meeting_sentences ms
        LEFT JOIN meeting_translations mt ON mt.sentence_id = ms.id
        WHERE ms.meeting_id = $1::uuid
        GROUP BY ms.id
        ORDER BY ms.start_time, ms.created_at
        """,
        meeting_id,
    )

    # Fall back to raw chunks if no sentences exist yet (e.g. live session still in progress)
    if not sentences:
        chunks = await pool.fetch(
            """
            SELECT mc.*,
                   COALESCE(
                       json_agg(
                           json_build_object(
                               'translated_text', mt.translated_text,
                               'target_language', mt.target_language,
                               'confidence', mt.confidence,
                               'model_used', mt.model_used
                           )
                       ) FILTER (WHERE mt.id IS NOT NULL),
                       '[]'::json
                   ) as translations
            FROM meeting_chunks mc
            LEFT JOIN meeting_translations mt ON mt.chunk_id = mc.id
            WHERE mc.meeting_id = $1::uuid
            GROUP BY mc.id
            ORDER BY mc.start_time, mc.created_at
            """,
            meeting_id,
        )
        return {
            "meeting_id": meeting_id,
            "sentences": [dict(c) for c in chunks],
            "count": len(chunks),
            "source": "chunks",
        }

    return {
        "meeting_id": meeting_id,
        "sentences": [dict(s) for s in sentences],
        "count": len(sentences),
    }


@router.post("/upload")
async def upload_transcript(
    file: UploadFile = File(description="Transcript file (JSON, TXT, or SRT)"),
    title: str | None = Query(default=None, description="Optional meeting title"),
) -> dict[str, Any]:
    """Upload a transcript file from another source."""
    store = await _get_store()

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    # Create meeting record
    meeting_id = await store.create_meeting(
        title=title or file.filename or "Uploaded Transcript",
        source="upload",
        status="completed",
    )

    # Parse and store based on file type
    filename = (file.filename or "").lower()

    if filename.endswith(".json"):
        import json

        try:
            data = json.loads(text)
            # Handle common JSON transcript formats
            sentences = (
                data
                if isinstance(data, list)
                else data.get("sentences", data.get("transcript", []))
            )
            for item in sentences:
                if isinstance(item, dict):
                    await store.store_sentence(
                        meeting_id=meeting_id,
                        text=item.get("text", item.get("content", "")),
                        speaker_name=item.get(
                            "speaker_name", item.get("speaker", "Unknown")
                        ),
                        start_time=float(item.get("start_time", 0)),
                        end_time=float(item.get("end_time", 0)),
                        boundary_type="upload",
                    )
                elif isinstance(item, str):
                    await store.store_sentence(
                        meeting_id=meeting_id,
                        text=item,
                        speaker_name="Unknown",
                        boundary_type="upload",
                    )
        except json.JSONDecodeError:
            # Fall back to treating as plain text
            await store.store_sentence(
                meeting_id=meeting_id,
                text=text,
                speaker_name="Unknown",
                boundary_type="upload",
            )
    else:
        # Plain text or SRT -- store as single or line-separated sentences
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter out SRT timestamp lines and index numbers
        content_lines = [
            line for line in lines if not line.isdigit() and "-->" not in line
        ]

        for line in content_lines:
            await store.store_sentence(
                meeting_id=meeting_id,
                text=line,
                speaker_name="Unknown",
                boundary_type="upload",
            )

    logger.info(
        "transcript_uploaded",
        meeting_id=meeting_id,
        filename=file.filename,
        source="upload",
    )

    return {
        "success": True,
        "meeting_id": meeting_id,
        "filename": file.filename,
    }


@router.post("/{meeting_id}/insights/generate")
async def generate_insights(
    meeting_id: str, request: InsightGenerateRequest | None = None
) -> dict[str, Any]:
    """Generate Ollama insights for a meeting.

    Reads the meeting's sentences and generates AI insights using the configured LLM.
    """
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Get all sentences for context
    pool = store._pool
    sentences = await pool.fetch(
        "SELECT text, speaker_name FROM meeting_sentences WHERE meeting_id = $1::uuid ORDER BY start_time, created_at",
        meeting_id,
    )

    if not sentences:
        raise HTTPException(
            status_code=400, detail="No transcript data available for this meeting"
        )

    # Build transcript text
    transcript_text = "\n".join(
        f"{s['speaker_name'] or 'Unknown'}: {s['text']}" for s in sentences
    )

    # Generate insights using Ollama
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    insight_types = (
        request.insight_types
        if request
        else ["summary", "action_items", "keywords"]
    )
    generated: list[dict[str, Any]] = []

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            for insight_type in insight_types:
                prompt = _build_insight_prompt(insight_type, transcript_text)

                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )

                if resp.status_code == 200:
                    result = resp.json()
                    insight_content = {"text": result.get("response", "")}

                    await store.store_insight(
                        meeting_id=meeting_id,
                        insight_type=insight_type,
                        content=insight_content,
                        source="ollama",
                        model_used=model,
                    )
                    generated.append(
                        {"type": insight_type, "content": insight_content}
                    )
                    logger.info(
                        "insight_generated",
                        meeting_id=meeting_id,
                        insight_type=insight_type,
                        model=model,
                    )
    except Exception as e:
        logger.error(
            "insight_generation_failed", meeting_id=meeting_id, error=str(e)
        )
        raise HTTPException(
            status_code=502, detail=f"Insight generation failed: {e}"
        ) from e

    return {
        "meeting_id": meeting_id,
        "generated": generated,
        "count": len(generated),
    }


def _build_insight_prompt(insight_type: str, transcript: str) -> str:
    """Build a prompt for insight generation based on type."""
    prompts = {
        "summary": (
            f"Provide a concise summary of this meeting transcript:\n\n"
            f"{transcript}\n\nSummary:"
        ),
        "action_items": (
            f"Extract all action items from this meeting transcript. "
            f"List each as a bullet point:\n\n{transcript}\n\nAction Items:"
        ),
        "keywords": (
            f"Extract the key topics and keywords from this meeting transcript. "
            f"List them:\n\n{transcript}\n\nKeywords:"
        ),
        "decisions": (
            f"What decisions were made in this meeting? List each:\n\n"
            f"{transcript}\n\nDecisions:"
        ),
        "questions": (
            f"What questions were asked in this meeting? List each:\n\n"
            f"{transcript}\n\nQuestions:"
        ),
    }
    return prompts.get(
        insight_type,
        f"Analyze this meeting transcript for '{insight_type}':\n\n{transcript}\n\nAnalysis:",
    )
