"""Database operations for diarization."""

import json
from datetime import UTC, datetime

from database.models import (
    DiarizationJob,
    MeetingSentence,
    SpeakerProfile,
    SystemConfig,
)
from livetranslate_common.logging import get_logger
from models.diarization import DiarizationRules
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger()


async def list_speakers(
    db: AsyncSession,
    name_filter: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List speaker profiles with optional name filter."""
    query = select(SpeakerProfile).order_by(SpeakerProfile.name)
    if name_filter:
        query = query.where(SpeakerProfile.name.ilike(f"%{name_filter}%"))
    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    rows = result.scalars().all()
    return [_speaker_to_dict(r) for r in rows]


async def create_speaker(db: AsyncSession, name: str, email: str | None = None) -> dict:
    """Create a new speaker profile."""
    profile = SpeakerProfile(name=name, email=email, enrollment_source="manual")
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return _speaker_to_dict(profile)


async def update_speaker(db: AsyncSession, speaker_id: int, data: dict) -> dict | None:
    """Update a speaker profile."""
    result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == speaker_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return None
    for key, value in data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    profile.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(profile)
    return _speaker_to_dict(profile)


async def merge_speakers(db: AsyncSession, source_id: int, target_id: int) -> dict | None:
    """Merge source speaker into target. Returns updated target."""
    source_result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == source_id))
    target_result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == target_id))
    source = source_result.scalar_one_or_none()
    target = target_result.scalar_one_or_none()
    if not source or not target:
        return None

    target.sample_count += source.sample_count
    if source.email and not target.email:
        target.email = source.email
    target.updated_at = datetime.now(UTC)

    await db.delete(source)
    await db.commit()
    await db.refresh(target)
    return _speaker_to_dict(target)


async def delete_speaker(db: AsyncSession, speaker_id: int) -> bool:
    """Delete a speaker profile."""
    result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == speaker_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return False
    await db.delete(profile)
    await db.commit()
    return True


async def get_diarization_rules(db: AsyncSession) -> dict:
    """Read diarization rules from system_config."""
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "diarization_rules")
    )
    row = result.scalar_one_or_none()
    if not row or not row.value:
        return DiarizationRules().model_dump()
    try:
        return json.loads(row.value)
    except (json.JSONDecodeError, TypeError):
        return DiarizationRules().model_dump()


async def save_diarization_rules(db: AsyncSession, rules: dict) -> dict:
    """Save diarization rules to system_config."""
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "diarization_rules")
    )
    row = result.scalar_one_or_none()
    value_str = json.dumps(rules)
    if row:
        row.value = value_str
        row.updated_at = datetime.now(UTC)
    else:
        row = SystemConfig(key="diarization_rules", value=value_str)
        db.add(row)
    await db.commit()
    return rules


async def get_meeting_sentences_for_compare(
    db: AsyncSession, meeting_id: str
) -> tuple[list[dict], list[dict], dict]:
    """Fetch Fireflies sentences and any VibeVoice segments for comparison."""
    result = await db.execute(
        select(MeetingSentence)
        .where(MeetingSentence.meeting_id == meeting_id)
        .order_by(MeetingSentence.start_time)
    )
    sentences = result.scalars().all()

    fireflies_sentences = []
    for s in sentences:
        fireflies_sentences.append({
            "id": str(s.id),
            "text": s.text,
            "speaker_name": s.speaker_name,
            "start_time": s.start_time,
            "end_time": s.end_time,
        })

    # Get VibeVoice segments from the latest completed diarization job
    job_result = await db.execute(
        select(DiarizationJob)
        .where(
            DiarizationJob.meeting_id == meeting_id,
            DiarizationJob.status == "completed",
        )
        .order_by(DiarizationJob.completed_at.desc())
        .limit(1)
    )
    job = job_result.scalar_one_or_none()
    vibevoice_segments = job.raw_segments if job and job.raw_segments else []
    speaker_map = job.speaker_map if job and job.speaker_map else {}

    return fireflies_sentences, vibevoice_segments, speaker_map


async def create_diarization_job(
    db: AsyncSession,
    meeting_id: str,
    triggered_by: str = "manual",
    rule_matched: dict | None = None,
) -> dict:
    """Create a diarization job in the database."""
    job = DiarizationJob(
        meeting_id=meeting_id,
        status="queued",
        triggered_by=triggered_by,
        rule_matched=rule_matched,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


async def get_diarization_job(db: AsyncSession, job_id: int) -> dict | None:
    """Get a diarization job by ID."""
    result = await db.execute(select(DiarizationJob).where(DiarizationJob.id == job_id))
    job = result.scalar_one_or_none()
    return _job_to_dict(job) if job else None


async def list_diarization_jobs(
    db: AsyncSession, status_filter: str | None = None, limit: int = 50
) -> list[dict]:
    """List diarization jobs."""
    query = select(DiarizationJob).order_by(DiarizationJob.created_at.desc()).limit(limit)
    if status_filter:
        query = query.where(DiarizationJob.status == status_filter)
    result = await db.execute(query)
    return [_job_to_dict(j) for j in result.scalars().all()]


async def update_job_status(
    db: AsyncSession,
    job_id: int,
    status: str,
    error_message: str | None = None,
    **extra_fields,
) -> dict | None:
    """Update a diarization job status."""
    result = await db.execute(select(DiarizationJob).where(DiarizationJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        return None
    job.status = status
    job.updated_at = datetime.now(UTC)
    if error_message:
        job.error_message = error_message
    if status in ("completed", "failed", "cancelled"):
        job.completed_at = datetime.now(UTC)
    for key, value in extra_fields.items():
        if hasattr(job, key):
            setattr(job, key, value)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


async def get_next_queued_job(db: AsyncSession) -> dict | None:
    """Get the oldest queued job for processing."""
    result = await db.execute(
        select(DiarizationJob)
        .where(DiarizationJob.status == "queued")
        .order_by(DiarizationJob.created_at)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return _job_to_dict(job) if job else None


def _job_to_dict(job: DiarizationJob) -> dict:
    """Convert DiarizationJob ORM object to dict."""
    return {
        "job_id": job.id,
        "meeting_id": str(job.meeting_id),
        "status": job.status,
        "triggered_by": job.triggered_by,
        "rule_matched": job.rule_matched,
        "audio_url": job.audio_url,
        "raw_segments": job.raw_segments,
        "detected_language": job.detected_language,
        "num_speakers_detected": job.num_speakers_detected,
        "processing_time_seconds": job.processing_time_seconds,
        "speaker_map": job.speaker_map,
        "unmapped_speakers": job.unmapped_speakers,
        "merge_applied": job.merge_applied,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _speaker_to_dict(profile: SpeakerProfile) -> dict:
    """Convert SpeakerProfile ORM object to dict."""
    return {
        "id": profile.id,
        "name": profile.name,
        "email": profile.email,
        "embedding": profile.embedding,
        "enrollment_source": profile.enrollment_source,
        "sample_count": profile.sample_count,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }
