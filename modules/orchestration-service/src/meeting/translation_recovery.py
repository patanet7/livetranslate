"""Crash-safe recovery for meeting translations.

Reuses the shared TranslationService singleton and replays any finalized
meeting content that is still missing persisted meeting_translations rows.
Supports both chunk-backed loopback meetings and sentence-backed Fireflies
meetings/imports.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from database.models import Meeting, MeetingChunk, MeetingSentence
from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationRequest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from meeting.session_manager import MeetingSessionManager
from translation.context_store import DirectionalContextStore
from translation.service import TranslationService

logger = get_logger()


_recovery_metrics_lock = asyncio.Lock()
_recovery_metrics: dict[str, object] = {
    "runs": 0,
    "total_pending_seen": 0,
    "total_recovered": 0,
    "total_failed": 0,
    "last_run_started_at": None,
    "last_run_completed_at": None,
    "last_run_scope": "all",
    "last_run_pending": 0,
    "last_run_recovered": 0,
    "last_run_failed": 0,
}

@dataclass(slots=True)
class PendingMeetingTranslation:
    meeting_id: uuid.UUID
    lineage_kind: str
    lineage_id: uuid.UUID
    text: str
    speaker_name: str | None
    source_language: str
    target_language: str


def _resolved_targets(meeting: Meeting, source_language: str) -> list[str]:
    targets = list(meeting.target_languages or [])
    if not targets:
        targets = [os.getenv("DEFAULT_TARGET_LANGUAGE", "en")]
    return [target for target in targets if target and target != source_language]


async def collect_pending_translations(
    db: AsyncSession,
    *,
    meeting_ids: list[uuid.UUID] | None = None,
    limit: int = 200,
) -> list[PendingMeetingTranslation]:
    """Collect missing translations for both chunk and sentence meeting paths."""
    meeting_filters = [Meeting.status != "ephemeral"]
    if meeting_ids:
        meeting_filters.append(Meeting.id.in_(meeting_ids))

    chunk_rows = (
        await db.execute(
            select(MeetingChunk)
            .join(MeetingChunk.meeting)
            .options(
                selectinload(MeetingChunk.meeting),
                selectinload(MeetingChunk.direct_translations),
            )
            .where(
                MeetingChunk.is_final.is_(True),
                *meeting_filters,
            )
            .order_by(MeetingChunk.created_at.asc())
            .limit(limit)
        )
    ).scalars().all()

    sentence_rows = (
        await db.execute(
            select(MeetingSentence)
            .join(MeetingSentence.meeting)
            .options(
                selectinload(MeetingSentence.meeting),
                selectinload(MeetingSentence.translations),
            )
            .where(*meeting_filters)
            .order_by(MeetingSentence.created_at.asc())
            .limit(limit)
        )
    ).scalars().all()

    pending: list[PendingMeetingTranslation] = []

    for chunk in chunk_rows:
        source_language = chunk.source_language or (chunk.meeting.source_languages or ["en"])[0]
        translated_targets = {translation.target_language for translation in chunk.direct_translations}
        for target_language in _resolved_targets(chunk.meeting, source_language):
            if target_language in translated_targets or not chunk.text.strip():
                continue
            pending.append(
                PendingMeetingTranslation(
                    meeting_id=chunk.meeting_id,
                    lineage_kind="chunk",
                    lineage_id=chunk.id,
                    text=chunk.text,
                    speaker_name=chunk.speaker_name or chunk.speaker_id,
                    source_language=source_language,
                    target_language=target_language,
                )
            )

    for sentence in sentence_rows:
        source_language = (sentence.meeting.source_languages or ["en"])[0]
        translated_targets = {translation.target_language for translation in sentence.translations}
        for target_language in _resolved_targets(sentence.meeting, source_language):
            if target_language in translated_targets or not sentence.text.strip():
                continue
            pending.append(
                PendingMeetingTranslation(
                    meeting_id=sentence.meeting_id,
                    lineage_kind="sentence",
                    lineage_id=sentence.id,
                    text=sentence.text,
                    speaker_name=sentence.speaker_name,
                    source_language=source_language,
                    target_language=target_language,
                )
            )

    return pending[:limit]


async def recover_pending_translations(
    session_manager: MeetingSessionManager,
    translation_service: TranslationService,
    *,
    meeting_ids: list[uuid.UUID] | None = None,
    limit: int = 200,
) -> dict[str, int]:
    """Translate and persist any missing meeting translations.

    Uses lightweight per-meeting directional context stores so recovery benefits
    from local context while keeping the shared singleton's global context clean.
    """
    pending = await collect_pending_translations(
        session_manager.db,
        meeting_ids=meeting_ids,
        limit=limit,
    )
    started_at = datetime.now(UTC)
    async with _recovery_metrics_lock:
        _recovery_metrics["runs"] = int(_recovery_metrics["runs"]) + 1
        _recovery_metrics["last_run_started_at"] = started_at
        _recovery_metrics["last_run_scope"] = (
            "all" if not meeting_ids else ",".join(str(meeting_id) for meeting_id in meeting_ids)
        )
        _recovery_metrics["last_run_pending"] = len(pending)

    if not pending:
        async with _recovery_metrics_lock:
            _recovery_metrics["last_run_completed_at"] = datetime.now(UTC)
            _recovery_metrics["last_run_recovered"] = 0
            _recovery_metrics["last_run_failed"] = 0
        return {"pending": 0, "recovered": 0, "failed": 0}

    stats = {"pending": len(pending), "recovered": 0, "failed": 0}
    context_stores: dict[tuple[uuid.UUID, str, str], DirectionalContextStore] = {}

    for item in pending:
        store_key = (item.meeting_id, item.source_language, item.target_language)
        context_store = context_stores.get(store_key)
        if context_store is None:
            context_store = DirectionalContextStore(
                max_entries=translation_service.config.context_window_size,
                max_tokens=translation_service.config.max_context_tokens,
                cross_direction_max_tokens=translation_service.config.cross_direction_max_tokens,
            )
            context_stores[store_key] = context_store

        request = TranslationRequest(
            text=item.text,
            source_language=item.source_language,
            target_language=item.target_language,
            speaker_name=item.speaker_name,
        )

        try:
            response = await translation_service.translate(
                request,
                context_store=context_store,
            )
            kwargs = {"chunk_id": item.lineage_id} if item.lineage_kind == "chunk" else {
                "sentence_id": item.lineage_id
            }
            await session_manager.save_translation(
                **kwargs,
                translated_text=response.translated_text,
                source_language=response.source_language,
                target_language=response.target_language,
                model_used=response.model_used,
                translation_time_ms=response.latency_ms,
            )
            stats["recovered"] += 1
        except Exception as exc:
            stats["failed"] += 1
            logger.warning(
                "meeting_translation_recovery_failed",
                meeting_id=str(item.meeting_id),
                lineage_kind=item.lineage_kind,
                lineage_id=str(item.lineage_id),
                target_language=item.target_language,
                error=str(exc),
            )

    logger.info("meeting_translation_recovery_complete", **stats)
    async with _recovery_metrics_lock:
        _recovery_metrics["total_pending_seen"] = int(_recovery_metrics["total_pending_seen"]) + stats["pending"]
        _recovery_metrics["total_recovered"] = int(_recovery_metrics["total_recovered"]) + stats["recovered"]
        _recovery_metrics["total_failed"] = int(_recovery_metrics["total_failed"]) + stats["failed"]
        _recovery_metrics["last_run_completed_at"] = datetime.now(UTC)
        _recovery_metrics["last_run_recovered"] = stats["recovered"]
        _recovery_metrics["last_run_failed"] = stats["failed"]
    return stats


async def recover_pending_translations_with_fresh_session(
    db: AsyncSession,
    recording_base_path: Path,
    translation_service: TranslationService,
    *,
    meeting_ids: list[uuid.UUID] | None = None,
    limit: int = 200,
) -> dict[str, int]:
    """Helper for startup/background jobs that only have a DB session."""
    session_manager = MeetingSessionManager(
        db=db,
        recording_base_path=recording_base_path,
    )
    return await recover_pending_translations(
        session_manager,
        translation_service,
        meeting_ids=meeting_ids,
        limit=limit,
    )


async def get_translation_recovery_metrics() -> dict[str, object]:
    """Return cumulative and last-run translation recovery counters."""
    async with _recovery_metrics_lock:
        return dict(_recovery_metrics)
