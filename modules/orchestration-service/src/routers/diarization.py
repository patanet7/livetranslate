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

from config import DiarizationSettings
from fastapi import APIRouter, HTTPException, status
from livetranslate_common.logging import get_logger
from models.diarization import (
    DiarizationJobCreate,
    DiarizationJobResponse,
    DiarizationRules,
    SpeakerMergeRequest,
    SpeakerProfileCreate,
    SpeakerProfileResponse,
)
from services.diarization.pipeline import DiarizationPipeline

logger = get_logger()

router = APIRouter(tags=["diarization"])

# Module-level pipeline instance (initialized on first use or at app startup)
_pipeline: DiarizationPipeline | None = None


def get_pipeline() -> DiarizationPipeline:
    global _pipeline
    if _pipeline is None:
        settings = DiarizationSettings()
        _pipeline = DiarizationPipeline(
            vibevoice_url=settings.vibevoice_url,
            max_concurrent=settings.max_concurrent_jobs,
        )
    return _pipeline


# --- Job endpoints ---


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_diarization_job(req: DiarizationJobCreate) -> dict[str, Any]:
    """Submit a meeting for offline diarization."""
    pipeline = get_pipeline()
    job = pipeline.create_job(
        meeting_id=req.meeting_id,
        triggered_by="manual",
        hotwords=req.hotwords,
    )
    return job


@router.get("/jobs")
async def list_diarization_jobs(status_filter: str | None = None) -> list[dict[str, Any]]:
    """List diarization jobs, optionally filtered by status."""
    pipeline = get_pipeline()
    return pipeline.list_jobs(status=status_filter)


@router.get("/jobs/{job_id}")
async def get_diarization_job(job_id: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_diarization_job(job_id: str) -> dict[str, str]:
    pipeline = get_pipeline()
    if not pipeline.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Job not found or not cancellable")
    return {"status": "cancelled"}


# --- Speaker endpoints ---


@router.get("/speakers")
async def list_speakers() -> list[dict[str, Any]]:
    """List all known speaker profiles."""
    # TODO: Query speaker_profiles table
    return []


@router.post("/speakers", status_code=status.HTTP_201_CREATED)
async def create_speaker(req: SpeakerProfileCreate) -> dict[str, Any]:
    """Create a new speaker profile."""
    # TODO: Insert into speaker_profiles table
    return {"id": 0, "name": req.name, "email": req.email, "enrollment_source": "manual", "sample_count": 0}


@router.put("/speakers/{speaker_id}")
async def update_speaker(speaker_id: int, req: SpeakerProfileCreate) -> dict[str, Any]:
    """Update speaker profile name/email."""
    # TODO: Update speaker_profiles table
    return {"id": speaker_id, "name": req.name, "email": req.email}


@router.post("/speakers/merge")
async def merge_speakers(req: SpeakerMergeRequest) -> dict[str, str]:
    """Merge source speaker profile into target."""
    # TODO: Merge in speaker_profiles table + update diarization_jobs speaker_maps
    return {"status": "merged", "kept": str(req.target_id), "removed": str(req.source_id)}


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int) -> dict[str, str]:
    """Delete a speaker profile."""
    # TODO: Delete from speaker_profiles table
    return {"status": "deleted"}


# --- Rules endpoints ---


@router.get("/rules")
async def get_rules() -> dict[str, Any]:
    """Get current auto-trigger rules."""
    # TODO: Read from system_config table
    return DiarizationRules().model_dump()


@router.put("/rules")
async def update_rules(rules: DiarizationRules) -> dict[str, Any]:
    """Update auto-trigger rules."""
    # TODO: Write to system_config table
    return rules.model_dump()


# --- Comparison endpoints ---


@router.get("/meetings/{meeting_id}/compare")
async def compare_transcripts(meeting_id: int) -> dict[str, Any]:
    """Side-by-side comparison of Fireflies vs VibeVoice transcript."""
    # TODO: Fetch both transcripts from DB
    return {"meeting_id": meeting_id, "fireflies_sentences": [], "vibevoice_segments": []}


@router.post("/meetings/{meeting_id}/apply")
async def apply_diarization(meeting_id: int) -> dict[str, str]:
    """Apply diarization results to the meeting transcript."""
    # TODO: Run transcript merge and update meeting_sentences
    return {"status": "applied", "meeting_id": str(meeting_id)}
