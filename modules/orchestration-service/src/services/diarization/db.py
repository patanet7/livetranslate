"""Database operations for diarization."""

from datetime import UTC, datetime

from database.models import SpeakerProfile
from livetranslate_common.logging import get_logger
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
