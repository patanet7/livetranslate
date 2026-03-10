"""Diarization Router — FastAPI endpoints for offline diarization management.

Endpoints:
  POST /jobs              - Submit meeting for diarization
  GET  /jobs              - List diarization jobs
  GET  /jobs/{job_id}     - Get job detail
  POST /jobs/{job_id}/cancel - Cancel a queued job

  GET  /rules             - Get auto-trigger rules
  PUT  /rules             - Update auto-trigger rules

  GET  /speakers          - List speaker profiles
  POST /speakers          - Create speaker profile
  PUT  /speakers/{id}     - Update speaker profile
  POST /speakers/merge    - Merge two speaker profiles
  DELETE /speakers/{id}   - Delete speaker profile

  GET  /meetings/{id}/compare - Side-by-side transcript comparison
  POST /meetings/{id}/apply   - Apply diarization to meeting
"""

from typing import Any

from database import get_database_manager, get_db_session
from database.models import MeetingSentence
from fastapi import APIRouter, Depends, HTTPException, Query, status
from livetranslate_common.logging import get_logger
from models.diarization import (
    DiarizationJobCreate,
    DiarizationJobResponse,
    DiarizationRules,
    SpeakerMergeRequest,
    SpeakerProfileCreate,
    SpeakerProfileResponse,
)
from services.diarization.db import (
    create_diarization_job,
    create_speaker,
    delete_speaker,
    get_diarization_job,
    get_diarization_rules,
    get_meeting_sentences_for_compare,
    list_diarization_jobs,
    list_speakers,
    merge_speakers,
    save_diarization_rules,
    update_job_status,
    update_speaker,
)
from services.diarization.pipeline import DiarizationPipeline
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger()

router = APIRouter(tags=["diarization"])

# Module-level pipeline instance (initialized on first use or at app startup)
_pipeline: DiarizationPipeline | None = None


def get_pipeline() -> DiarizationPipeline:
    global _pipeline
    if _pipeline is None:
        from config import DiarizationSettings

        settings = DiarizationSettings()
        db_manager = get_database_manager()
        _pipeline = DiarizationPipeline(
            session_factory=db_manager.get_session,
            vibevoice_url=settings.vibevoice_url,
            max_concurrent=settings.max_concurrent_jobs,
        )
    return _pipeline


# --- Job endpoints ---


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job_endpoint(
    req: DiarizationJobCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Submit a meeting for offline diarization."""
    job = await create_diarization_job(
        db,
        meeting_id=req.meeting_id,
        triggered_by="manual",
        rule_matched=None,
    )
    return job


@router.get("/jobs")
async def list_jobs_endpoint(
    status_filter: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """List diarization jobs, optionally filtered by status."""
    return await list_diarization_jobs(db, status_filter=status_filter)


@router.get("/jobs/{job_id}")
async def get_job_endpoint(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    job = await get_diarization_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_job_endpoint(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    job = await get_diarization_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "queued":
        raise HTTPException(status_code=400, detail="Job is not cancellable")
    await update_job_status(db, job_id, "cancelled")
    return {"status": "cancelled"}


# --- Speaker endpoints ---


@router.get("/speakers")
async def list_speakers_endpoint(
    db: AsyncSession = Depends(get_db_session),
    name: str | None = Query(None, description="Filter speakers by name (case-insensitive)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """List all known speaker profiles."""
    return await list_speakers(db, name_filter=name, limit=limit, offset=offset)


@router.post("/speakers", status_code=status.HTTP_201_CREATED)
async def create_speaker_endpoint(
    req: SpeakerProfileCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Create a new speaker profile."""
    return await create_speaker(db, name=req.name, email=req.email)


@router.put("/speakers/{speaker_id}")
async def update_speaker_endpoint(
    speaker_id: int,
    req: SpeakerProfileCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Update speaker profile name/email."""
    result = await update_speaker(db, speaker_id, req.model_dump(exclude_unset=True))
    if result is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return result


@router.post("/speakers/merge")
async def merge_speakers_endpoint(
    req: SpeakerMergeRequest,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Merge source speaker profile into target."""
    result = await merge_speakers(db, source_id=req.source_id, target_id=req.target_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Source or target speaker not found")
    return result


@router.delete("/speakers/{speaker_id}")
async def delete_speaker_endpoint(
    speaker_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Delete a speaker profile."""
    deleted = await delete_speaker(db, speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "deleted"}


# --- Rules endpoints ---


@router.get("/rules")
async def get_rules(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get current auto-trigger rules."""
    return await get_diarization_rules(db)


@router.put("/rules")
async def update_rules(
    request: DiarizationRules,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Update auto-trigger rules."""
    return await save_diarization_rules(db, request.model_dump())


# --- Comparison endpoints ---


@router.get("/meetings/{meeting_id}/compare")
async def compare_transcripts(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Side-by-side comparison of Fireflies vs VibeVoice transcript."""
    from services.diarization.transcript_merge import merge_transcripts

    ff_sentences, vv_segments, speaker_map = await get_meeting_sentences_for_compare(
        db, meeting_id
    )
    if not ff_sentences:
        raise HTTPException(
            status_code=404, detail=f"No sentences found for meeting {meeting_id}"
        )

    merged = (
        merge_transcripts(ff_sentences, vv_segments, speaker_map)
        if vv_segments
        else ff_sentences
    )
    return {
        "meeting_id": meeting_id,
        "fireflies_sentences": ff_sentences,
        "vibevoice_segments": vv_segments,
        "speaker_map": speaker_map,
        "merged": merged,
    }


@router.post("/meetings/{meeting_id}/apply")
async def apply_diarization(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Apply diarization results to meeting sentences."""
    from services.diarization.transcript_merge import merge_transcripts

    ff_sentences, vv_segments, speaker_map = await get_meeting_sentences_for_compare(
        db, meeting_id
    )
    if not vv_segments:
        raise HTTPException(
            status_code=400, detail="No diarization results available to apply"
        )

    merged = merge_transcripts(ff_sentences, vv_segments, speaker_map)

    updated_count = 0
    for item in merged:
        if item.get("diarization_source"):
            sentence_id = item.get("id")
            if sentence_id:
                result = await db.execute(
                    select(MeetingSentence).where(
                        MeetingSentence.id == sentence_id
                    )
                )
                sentence = result.scalar_one_or_none()
                if sentence:
                    sentence.speaker_name = item["speaker_name"]
                    updated_count += 1

    await db.commit()
    return {"meeting_id": meeting_id, "updated_sentences": updated_count}
