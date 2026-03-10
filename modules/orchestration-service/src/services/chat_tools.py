"""Business insight tools for the chat system.

Each tool queries the database to provide business analytics data
to the LLM during conversations.
"""

from typing import Any

from database.models import (
    DiarizationJob,
    Meeting,
    MeetingSentence,
    MeetingSpeaker,
    MeetingTranslation,
    SpeakerProfile,
)
from livetranslate_common.logging import get_logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger()


# ── Tool implementations ──────────────────────────────────────────


async def query_meetings(
    db: AsyncSession,
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query meetings with optional filters."""
    query = (
        select(Meeting)
        .order_by(Meeting.start_time.desc().nullslast())
        .limit(min(limit, 100))
    )
    if date_from:
        query = query.where(Meeting.start_time >= date_from)
    if date_to:
        query = query.where(Meeting.start_time <= date_to)
    if participant:
        query = query.where(
            Meeting.title.ilike(f"%{participant}%")
            | Meeting.organizer_email.ilike(f"%{participant}%")
        )
    result = await db.execute(query)
    meetings = result.scalars().all()
    return {
        "count": len(meetings),
        "meetings": [
            {
                "id": str(m.id),
                "title": m.title,
                "start_time": m.start_time.isoformat() if m.start_time else None,
                "duration": m.duration,
                "organizer_email": m.organizer_email,
                "num_participants": len(m.participants) if m.participants else 0,
                "source": m.source,
                "status": m.status,
            }
            for m in meetings
        ],
    }


async def get_meeting_details(db: AsyncSession, meeting_id: str) -> dict[str, Any]:
    """Get detailed information about a specific meeting."""
    result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = result.scalar_one_or_none()
    if not meeting:
        return {"error": f"Meeting {meeting_id} not found"}

    sent_count = await db.execute(
        select(func.count())
        .select_from(MeetingSentence)
        .where(MeetingSentence.meeting_id == meeting_id)
    )
    trans_count = await db.execute(
        select(func.count())
        .select_from(MeetingTranslation)
        .join(MeetingSentence)
        .where(MeetingSentence.meeting_id == meeting_id)
    )

    return {
        "id": str(meeting.id),
        "title": meeting.title,
        "start_time": meeting.start_time.isoformat() if meeting.start_time else None,
        "end_time": meeting.end_time.isoformat() if meeting.end_time else None,
        "duration": meeting.duration,
        "organizer_email": meeting.organizer_email,
        "participants": meeting.participants,
        "source": meeting.source,
        "status": meeting.status,
        "sentence_count": sent_count.scalar() or 0,
        "translation_count": trans_count.scalar() or 0,
    }


async def get_translation_stats(
    db: AsyncSession,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Get aggregate translation statistics."""
    query = (
        select(
            MeetingTranslation.target_language,
            func.count().label("count"),
        )
        .join(MeetingSentence)
        .join(Meeting)
        .group_by(MeetingTranslation.target_language)
    )
    if date_from:
        query = query.where(Meeting.start_time >= date_from)
    if date_to:
        query = query.where(Meeting.start_time <= date_to)
    result = await db.execute(query)
    rows = result.all()
    return {
        "total_translations": sum(r.count for r in rows),
        "by_language": {r.target_language: r.count for r in rows},
    }


async def get_speaker_analytics(
    db: AsyncSession,
    date_from: str | None = None,
    date_to: str | None = None,
    speaker_name: str | None = None,
) -> dict[str, Any]:
    """Get speaker participation analytics."""
    query = (
        select(
            MeetingSpeaker.speaker_name,
            func.sum(MeetingSpeaker.word_count).label("total_words"),
            func.sum(MeetingSpeaker.talk_time_seconds).label("total_talk_time"),
            func.count().label("meeting_count"),
        )
        .join(Meeting)
        .group_by(MeetingSpeaker.speaker_name)
        .order_by(func.sum(MeetingSpeaker.talk_time_seconds).desc())
        .limit(50)
    )
    if date_from:
        query = query.where(Meeting.start_time >= date_from)
    if date_to:
        query = query.where(Meeting.start_time <= date_to)
    if speaker_name:
        query = query.where(MeetingSpeaker.speaker_name.ilike(f"%{speaker_name}%"))
    result = await db.execute(query)
    rows = result.all()
    return {
        "speakers": [
            {
                "name": r.speaker_name,
                "total_words": r.total_words,
                "total_talk_time_seconds": float(r.total_talk_time) if r.total_talk_time else 0,
                "meeting_count": r.meeting_count,
            }
            for r in rows
        ],
    }


async def get_language_distribution(
    db: AsyncSession,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Get language pair usage distribution."""
    query = (
        select(
            MeetingTranslation.source_language,
            MeetingTranslation.target_language,
            func.count().label("count"),
        )
        .join(MeetingSentence)
        .join(Meeting)
        .group_by(MeetingTranslation.source_language, MeetingTranslation.target_language)
        .order_by(func.count().desc())
    )
    if date_from:
        query = query.where(Meeting.start_time >= date_from)
    if date_to:
        query = query.where(Meeting.start_time <= date_to)
    result = await db.execute(query)
    rows = result.all()
    return {
        "languages": [
            {
                "source_language": r.source_language,
                "target_language": r.target_language,
                "count": r.count,
            }
            for r in rows
        ],
    }


async def get_diarization_stats(db: AsyncSession) -> dict[str, Any]:
    """Get diarization job statistics."""
    status_query = select(
        DiarizationJob.status, func.count().label("count")
    ).group_by(DiarizationJob.status)
    result = await db.execute(status_query)
    status_counts = {r.status: r.count for r in result.all()}

    profile_count = await db.execute(select(func.count()).select_from(SpeakerProfile))

    return {
        "job_counts_by_status": status_counts,
        "total_jobs": sum(status_counts.values()),
        "speaker_profile_count": profile_count.scalar() or 0,
    }


async def search_transcripts(
    db: AsyncSession,
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Full-text search across meeting transcripts."""
    stmt = (
        select(MeetingSentence, Meeting.title)
        .join(Meeting)
        .where(MeetingSentence.text.ilike(f"%{query}%"))
        .order_by(MeetingSentence.created_at.desc())
        .limit(min(limit, 50))
    )
    if date_from:
        stmt = stmt.where(Meeting.start_time >= date_from)
    if date_to:
        stmt = stmt.where(Meeting.start_time <= date_to)
    result = await db.execute(stmt)
    rows = result.all()
    return {
        "count": len(rows),
        "results": [
            {
                "meeting_id": str(s.meeting_id),
                "meeting_title": title,
                "speaker": s.speaker_name,
                "text": s.text,
                "start_time": s.start_time,
            }
            for s, title in rows
        ],
    }


async def get_usage_trends(
    db: AsyncSession,
    date_from: str | None = None,
    date_to: str | None = None,
    metric: str = "meetings",
    interval: str = "day",
) -> dict[str, Any]:
    """Get time-series usage data."""
    query = (
        select(
            func.date(Meeting.start_time).label("day"),
            func.count().label("count"),
        )
        .where(Meeting.start_time.isnot(None))
        .group_by(func.date(Meeting.start_time))
        .order_by(func.date(Meeting.start_time))
    )
    if date_from:
        query = query.where(Meeting.start_time >= date_from)
    if date_to:
        query = query.where(Meeting.start_time <= date_to)
    result = await db.execute(query)
    rows = result.all()
    return {
        "metric": metric,
        "interval": interval,
        "data": [{"date": str(r.day), "count": r.count} for r in rows],
    }


# ── Tool definitions (JSON Schema) ───────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_meetings",
        "description": "Search and list meetings with optional date range and participant filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "participant": {
                    "type": "string",
                    "description": "Participant name or email to filter by",
                },
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
        },
        "handler": query_meetings,
    },
    {
        "name": "get_meeting_details",
        "description": (
            "Get detailed information about a specific meeting including "
            "sentence and translation counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string", "description": "Meeting UUID"},
            },
            "required": ["meeting_id"],
        },
        "handler": get_meeting_details,
    },
    {
        "name": "get_translation_stats",
        "description": "Get aggregate translation statistics grouped by target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            },
        },
        "handler": get_translation_stats,
    },
    {
        "name": "get_speaker_analytics",
        "description": (
            "Get speaker participation analytics — who speaks most, "
            "word counts, talk time, and meeting counts per speaker."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "speaker_name": {
                    "type": "string",
                    "description": "Filter by speaker name (partial match)",
                },
            },
        },
        "handler": get_speaker_analytics,
    },
    {
        "name": "get_language_distribution",
        "description": "Get distribution of translation source/target language pairs.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            },
        },
        "handler": get_language_distribution,
    },
    {
        "name": "get_diarization_stats",
        "description": "Get diarization job statistics and speaker profile count.",
        "parameters": {"type": "object", "properties": {}},
        "handler": get_diarization_stats,
    },
    {
        "name": "search_transcripts",
        "description": "Full-text search across all meeting transcripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
            "required": ["query"],
        },
        "handler": search_transcripts,
    },
    {
        "name": "get_usage_trends",
        "description": "Get time-series usage data (meeting counts by date).",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "metric": {
                    "type": "string",
                    "description": "Metric name (default: meetings)",
                },
                "interval": {
                    "type": "string",
                    "description": "Time interval: day, week, month (default: day)",
                },
            },
        },
        "handler": get_usage_trends,
    },
]


def register_all_tools(executor: Any) -> None:
    """Register all business insight tools with a ToolExecutor."""
    for tool_def in TOOL_DEFINITIONS:
        executor.register_tool(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            handler=tool_def["handler"],
        )
    logger.info("chat_tools_registered", count=len(TOOL_DEFINITIONS))
