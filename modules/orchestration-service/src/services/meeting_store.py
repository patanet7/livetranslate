"""Meeting persistence service for storing Fireflies transcript data.

Handles all PostgreSQL operations for meeting data including CRUD for
meetings, chunks, sentences, translations, insights, and speakers.
Uses asyncpg connection pooling following the pattern established in
database/bot_session_manager.py.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()


class MeetingStore:
    """Handles all PostgreSQL operations for meeting data.

    Uses asyncpg connection pooling. All methods are async.
    Follows the pattern established in database/bot_session_manager.py.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self._pool: Any = None  # asyncpg.Pool, typed as Any to avoid import at module level

    async def initialize(self) -> None:
        """Create asyncpg connection pool."""
        import asyncpg

        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
            logger.info("meeting_store_initialized", db_url=self.db_url[:20] + "...")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("meeting_store_closed")

    async def _ensure_pool(self) -> None:
        """Ensure connection pool exists."""
        if self._pool is None:
            await self.initialize()

    @asynccontextmanager
    async def transaction(self):
        """Yield an asyncpg connection inside a transaction.

        Usage::

            async with store.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("INSERT ...")
                # auto-committed on exit, rolled back on exception
        """
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    # ------------------------------------------------------------------ #
    # Retry helpers
    # ------------------------------------------------------------------ #

    # Backoff schedule: base delay × 3^retry_count
    _BACKOFF_BASE_SECS = 300  # 5 minutes
    _BACKOFF_RATE_LIMIT_BASE_SECS = 3600  # 1 hour for rate-limit errors
    _MAX_RETRIES = 3

    @staticmethod
    def compute_next_retry_at(
        retry_count: int, is_rate_limit: bool = False
    ) -> datetime | None:
        """Compute next retry time using exponential backoff.

        Normal errors:     5min → 15min → 45min  (base × 3^n)
        Rate-limit errors: 1hr  → 4hr  → 12hr    (base × 4^n but capped)

        Returns None if retry_count >= MAX_RETRIES (no more retries).
        """
        if retry_count >= MeetingStore._MAX_RETRIES:
            return None  # exhausted
        if is_rate_limit:
            delay = MeetingStore._BACKOFF_RATE_LIMIT_BASE_SECS * (4 ** retry_count)
        else:
            delay = MeetingStore._BACKOFF_BASE_SECS * (3 ** retry_count)
        return datetime.now(UTC) + timedelta(seconds=delay)

    async def mark_sync_failed(
        self,
        meeting_id: str,
        error_msg: str,
        is_rate_limit: bool = False,
    ) -> None:
        """Mark a meeting sync as failed with retry tracking.

        Increments retry_count, sets sync_error, and computes next_retry_at
        using exponential backoff. If retries exhausted, next_retry_at is NULL.
        """
        await self._ensure_pool()
        row = await self._pool.fetchrow(
            "SELECT retry_count FROM meetings WHERE id = $1::uuid",
            meeting_id,
        )
        current_count = row["retry_count"] if row else 0
        new_count = current_count + 1
        next_retry = self.compute_next_retry_at(new_count, is_rate_limit)

        await self._pool.execute(
            """UPDATE meetings
               SET sync_status = 'failed',
                   sync_error = $2,
                   retry_count = $3,
                   last_retry_at = NOW(),
                   next_retry_at = $4
               WHERE id = $1::uuid""",
            meeting_id,
            error_msg[:500],  # truncate long errors
            new_count,
            next_retry,
        )
        logger.info(
            "meeting_sync_marked_failed",
            meeting_id=meeting_id,
            retry_count=new_count,
            next_retry_at=next_retry.isoformat() if next_retry else "exhausted",
            is_rate_limit=is_rate_limit,
        )

    async def get_retryable_meetings(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get meetings eligible for retry (failed, under max retries, past backoff)."""
        await self._ensure_pool()
        rows = await self._pool.fetch(
            """SELECT id, fireflies_transcript_id, title, retry_count,
                      sync_error, next_retry_at
               FROM meetings
               WHERE sync_status = 'failed'
                 AND retry_count < $1
                 AND (next_retry_at IS NULL OR next_retry_at <= NOW())
               ORDER BY next_retry_at ASC NULLS FIRST
               LIMIT $2""",
            self._MAX_RETRIES,
            limit,
        )
        return [dict(r) for r in rows]

    async def reset_meeting_for_resync(self, meeting_id: str) -> bool:
        """Clear all synced data for a meeting and reset to 'none' for re-sync.

        Deletes sentences, insights, and speakers, then resets sync columns.
        Returns True if the meeting existed.
        """
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT id FROM meetings WHERE id = $1::uuid", meeting_id
                )
                if not row:
                    return False
                # CASCADE would handle this, but explicit is clearer
                await conn.execute(
                    "DELETE FROM meeting_sentences WHERE meeting_id = $1::uuid",
                    meeting_id,
                )
                await conn.execute(
                    "DELETE FROM meeting_data_insights WHERE meeting_id = $1::uuid",
                    meeting_id,
                )
                await conn.execute(
                    "DELETE FROM meeting_speakers WHERE meeting_id = $1::uuid",
                    meeting_id,
                )
                await conn.execute(
                    """UPDATE meetings
                       SET sync_status = 'none', sync_error = NULL,
                           retry_count = 0, last_retry_at = NULL,
                           next_retry_at = NULL, synced_at = NULL
                       WHERE id = $1::uuid""",
                    meeting_id,
                )
        logger.info("meeting_reset_for_resync", meeting_id=meeting_id)
        return True

    async def cleanup_stuck_syncing(self, older_than_minutes: int = 30) -> int:
        """Mark meetings stuck in 'syncing' for longer than threshold as 'failed'.

        Returns the number of meetings cleaned up.
        """
        await self._ensure_pool()
        result = await self._pool.execute(
            """UPDATE meetings
               SET sync_status = 'failed',
                   sync_error = 'Stuck in syncing state (auto-cleanup)'
               WHERE sync_status = 'syncing'
                 AND updated_at < NOW() - ($1 || ' minutes')::interval""",
            str(older_than_minutes),
        )
        count = int(result.split()[-1]) if result else 0
        if count:
            logger.info("stuck_syncing_cleaned_up", count=count, threshold_minutes=older_than_minutes)
        return count

    # ------------------------------------------------------------------ #
    # Meeting CRUD
    # ------------------------------------------------------------------ #

    async def create_meeting(
        self,
        fireflies_transcript_id: str | None = None,
        title: str | None = None,
        meeting_link: str | None = None,
        organizer_email: str | None = None,
        participants: list[str] | None = None,
        source_languages: list[str] | None = None,
        target_languages: list[str] | None = None,
        meeting_metadata: dict[str, Any] | None = None,
        source: str = "fireflies",
        status: str = "live",
    ) -> str:
        """Create a new meeting record. Returns meeting UUID.

        If a meeting with the same fireflies_transcript_id already exists
        (unique constraint), returns the existing meeting's ID instead of
        crashing — handles race conditions between concurrent sync tasks.
        """
        await self._ensure_pool()
        meeting_id = str(uuid.uuid4())
        participants_json = json.dumps(participants or [])

        row = await self._pool.fetchrow(
            """
            INSERT INTO meetings (id, fireflies_transcript_id, title, meeting_link,
                                  organizer_email, participants, source_languages,
                                  target_languages, metadata, source, status, start_time)
            VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7::text[], $8::text[],
                    $9::jsonb, $10, $11, $12)
            ON CONFLICT (fireflies_transcript_id)
                WHERE fireflies_transcript_id IS NOT NULL
            DO UPDATE SET title = COALESCE(EXCLUDED.title, meetings.title),
                         meeting_link = COALESCE(EXCLUDED.meeting_link, meetings.meeting_link),
                         organizer_email = COALESCE(EXCLUDED.organizer_email, meetings.organizer_email),
                         source_languages = COALESCE(EXCLUDED.source_languages, meetings.source_languages),
                         target_languages = COALESCE(EXCLUDED.target_languages, meetings.target_languages),
                         metadata = COALESCE(EXCLUDED.metadata, meetings.metadata)
            RETURNING id
            """,
            meeting_id,
            fireflies_transcript_id,
            title,
            meeting_link,
            organizer_email,
            participants_json,
            source_languages,
            target_languages,
            json.dumps(meeting_metadata) if meeting_metadata is not None else None,
            source,
            status,
            datetime.now(UTC),
        )
        actual_id = str(row["id"])
        if actual_id != meeting_id:
            logger.info("meeting_create_conflict_resolved", existing_id=actual_id, source=source)
        else:
            logger.info("meeting_created", meeting_id=actual_id, source=source)
        return actual_id

    async def update_meeting(
        self,
        meeting_id: str,
        title: str | None = None,
        meeting_link: str | None = None,
        organizer_email: str | None = None,
        participants: list[str] | None = None,
        source_languages: list[str] | None = None,
        target_languages: list[str] | None = None,
        meeting_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update meeting metadata (title, link, organizer, participants)."""
        await self._ensure_pool()
        updates: list[str] = []
        params: list[Any] = []
        idx = 1

        if title is not None:
            idx += 1
            updates.append(f"title = ${idx}")
            params.append(title)
        if meeting_link is not None:
            idx += 1
            updates.append(f"meeting_link = ${idx}")
            params.append(meeting_link)
        if organizer_email is not None:
            idx += 1
            updates.append(f"organizer_email = ${idx}")
            params.append(organizer_email)
        if participants is not None:
            idx += 1
            updates.append(f"participants = ${idx}::jsonb")
            params.append(json.dumps(participants))
        if source_languages is not None:
            idx += 1
            updates.append(f"source_languages = ${idx}::text[]")
            params.append(source_languages)
        if target_languages is not None:
            idx += 1
            updates.append(f"target_languages = ${idx}::text[]")
            params.append(target_languages)
        if meeting_metadata is not None:
            idx += 1
            updates.append(f"metadata = ${idx}::jsonb")
            params.append(json.dumps(meeting_metadata))

        if not updates:
            return

        query = f"UPDATE meetings SET {', '.join(updates)} WHERE id = $1::uuid"
        await self._pool.execute(query, meeting_id, *params)
        logger.info("meeting_updated", meeting_id=meeting_id, fields=list(updates))

    async def get_meeting_by_ff_id(self, fireflies_transcript_id: str) -> dict[str, Any] | None:
        """Find meeting by Fireflies transcript ID (includes insight_count)."""
        await self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT m.*,
                   (SELECT COUNT(*) FROM meeting_data_insights WHERE meeting_id = m.id) as insight_count
            FROM meetings m WHERE m.fireflies_transcript_id = $1
            """,
            fireflies_transcript_id,
        )
        return dict(row) if row else None

    async def complete_meeting(self, meeting_id: str) -> None:
        """Mark meeting as completed with end time."""
        await self._ensure_pool()
        await self._pool.execute(
            """
            UPDATE meetings SET status = 'completed', end_time = $2,
                   duration = EXTRACT(EPOCH FROM ($2 - start_time))::INTEGER
            WHERE id = $1::uuid
            """,
            meeting_id,
            datetime.now(UTC),
        )
        logger.info("meeting_completed", meeting_id=meeting_id)

    async def update_sync_status(
        self,
        meeting_id: str,
        sync_status: str,
        sync_error: str | None = None,
        audio_url: str | None = None,
        video_url: str | None = None,
        transcript_url: str | None = None,
    ) -> None:
        """Update meeting sync status and media URLs."""
        await self._ensure_pool()
        updates: list[str] = ["sync_status = $2"]
        params: list[Any] = [meeting_id, sync_status]
        idx = 2

        if sync_status == "synced":
            idx += 1
            updates.append(f"synced_at = ${idx}")
            params.append(datetime.now(UTC))

        if sync_error is not None:
            idx += 1
            updates.append(f"sync_error = ${idx}")
            params.append(sync_error)
        elif sync_status == "synced":
            updates.append("sync_error = NULL")

        for col, val in [
            ("audio_url", audio_url),
            ("video_url", video_url),
            ("transcript_url", transcript_url),
        ]:
            if val is not None:
                idx += 1
                updates.append(f"{col} = ${idx}")
                params.append(val)

        query = f"UPDATE meetings SET {', '.join(updates)} WHERE id = $1::uuid"
        await self._pool.execute(query, *params)
        logger.info("meeting_sync_updated", meeting_id=meeting_id, sync_status=sync_status)

    async def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        """Get meeting by ID with basic stats."""
        await self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT m.*,
                   (SELECT COUNT(*) FROM meeting_chunks WHERE meeting_id = m.id) as chunk_count,
                   (SELECT COUNT(*) FROM meeting_sentences WHERE meeting_id = m.id) as sentence_count,
                   (SELECT COUNT(*) FROM meeting_translations mt
                    LEFT JOIN meeting_sentences ms ON mt.sentence_id = ms.id
                    LEFT JOIN meeting_chunks mc ON mt.chunk_id = mc.id
                    WHERE ms.meeting_id = m.id OR mc.meeting_id = m.id) as translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_chunks mc
                    LEFT JOIN meeting_translations mt ON mt.chunk_id = mc.id
                    WHERE mc.meeting_id = m.id
                      AND mc.is_final = TRUE
                      AND mt.id IS NULL) as pending_chunk_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_sentences ms
                    LEFT JOIN meeting_translations mt ON mt.sentence_id = ms.id
                    WHERE ms.meeting_id = m.id
                      AND mt.id IS NULL) as pending_sentence_translation_count,
                   (SELECT COUNT(*) FROM meeting_data_insights WHERE meeting_id = m.id) as insight_count
            FROM meetings m WHERE m.id = $1::uuid
            """,
            meeting_id,
        )
        return dict(row) if row else None

    async def list_meetings(
        self, limit: int = 50, offset: int = 0, min_sentences: int = 1
    ) -> dict[str, Any]:
        """List meetings with pagination, newest first.

        Args:
            limit: Max meetings per page.
            offset: Pagination offset.
            min_sentences: Minimum sentence count to include (0 = show all).

        Returns:
            Dict with ``meetings`` list and ``total`` count for pagination.
        """
        await self._ensure_pool()

        base_query = """
            SELECT m.*,
                   (SELECT COUNT(*) FROM meeting_chunks WHERE meeting_id = m.id) as chunk_count,
                   (SELECT COUNT(*) FROM meeting_sentences WHERE meeting_id = m.id) as sentence_count,
                   (SELECT COUNT(*) FROM meeting_translations mt
                    LEFT JOIN meeting_sentences ms ON mt.sentence_id = ms.id
                    LEFT JOIN meeting_chunks mc ON mt.chunk_id = mc.id
                    WHERE ms.meeting_id = m.id OR mc.meeting_id = m.id) as translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_chunks mc
                    LEFT JOIN meeting_translations mt ON mt.chunk_id = mc.id
                    WHERE mc.meeting_id = m.id
                      AND mc.is_final = TRUE
                      AND mt.id IS NULL) as pending_chunk_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_sentences ms
                    LEFT JOIN meeting_translations mt ON mt.sentence_id = ms.id
                    WHERE ms.meeting_id = m.id
                      AND mt.id IS NULL) as pending_sentence_translation_count
            FROM meetings m
        """

        if min_sentences > 0:
            # Wrap to filter on computed column
            count_row = await self._pool.fetchrow(
                f"""SELECT COUNT(*) as cnt
                    FROM ({base_query}) sub
                    WHERE GREATEST(sub.sentence_count, sub.chunk_count) >= $1""",
                min_sentences,
            )
            total = count_row["cnt"] if count_row else 0

            rows = await self._pool.fetch(
                f"""SELECT * FROM ({base_query}) sub
                    WHERE GREATEST(sub.sentence_count, sub.chunk_count) >= $1
                    ORDER BY sub.created_at DESC
                    LIMIT $2 OFFSET $3""",
                min_sentences,
                limit,
                offset,
            )
        else:
            count_row = await self._pool.fetchrow(
                f"SELECT COUNT(*) as cnt FROM ({base_query}) sub"
            )
            total = count_row["cnt"] if count_row else 0

            rows = await self._pool.fetch(
                f"""{base_query}
                    ORDER BY m.created_at DESC
                    LIMIT $1 OFFSET $2""",
                limit,
                offset,
            )

        return {"meetings": [dict(r) for r in rows], "total": total}

    async def get_meeting_translation_backlog(self, meeting_id: str) -> dict[str, Any] | None:
        """Get translation backlog counts and metadata for a single meeting."""
        await self._ensure_pool()
        row = await self._pool.fetchrow(
            """
            SELECT m.id,
                   m.title,
                   m.source,
                   m.status,
                   m.target_languages,
                   (SELECT COUNT(*)
                    FROM meeting_chunks mc
                    LEFT JOIN meeting_translations mt ON mt.chunk_id = mc.id
                    WHERE mc.meeting_id = m.id
                      AND mc.is_final = TRUE
                      AND mt.id IS NULL) as pending_chunk_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_sentences ms
                    LEFT JOIN meeting_translations mt ON mt.sentence_id = ms.id
                    WHERE ms.meeting_id = m.id
                      AND mt.id IS NULL) as pending_sentence_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_translations mt
                    LEFT JOIN meeting_sentences ms ON mt.sentence_id = ms.id
                    LEFT JOIN meeting_chunks mc ON mt.chunk_id = mc.id
                    WHERE ms.meeting_id = m.id OR mc.meeting_id = m.id) as translation_count
            FROM meetings m
            WHERE m.id = $1::uuid
            """,
            meeting_id,
        )
        if row is None:
            return None

        data = dict(row)
        data["pending_translation_count"] = (
            data["pending_chunk_translation_count"] + data["pending_sentence_translation_count"]
        )
        return data

    async def list_translation_backlog(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        only_pending: bool = True,
    ) -> dict[str, Any]:
        """List meetings with translation backlog counts plus fleet totals."""
        await self._ensure_pool()

        base_query = """
            SELECT m.id,
                   m.title,
                   m.source,
                   m.status,
                   m.created_at,
                   m.target_languages,
                   (SELECT COUNT(*)
                    FROM meeting_chunks mc
                    LEFT JOIN meeting_translations mt ON mt.chunk_id = mc.id
                    WHERE mc.meeting_id = m.id
                      AND mc.is_final = TRUE
                      AND mt.id IS NULL) as pending_chunk_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_sentences ms
                    LEFT JOIN meeting_translations mt ON mt.sentence_id = ms.id
                    WHERE ms.meeting_id = m.id
                      AND mt.id IS NULL) as pending_sentence_translation_count,
                   (SELECT COUNT(*)
                    FROM meeting_translations mt
                    LEFT JOIN meeting_sentences ms ON mt.sentence_id = ms.id
                    LEFT JOIN meeting_chunks mc ON mt.chunk_id = mc.id
                    WHERE ms.meeting_id = m.id OR mc.meeting_id = m.id) as translation_count
            FROM meetings m
        """

        where_clause = """
            WHERE (
                sub.pending_chunk_translation_count + sub.pending_sentence_translation_count
            ) > 0
        """ if only_pending else ""

        count_row = await self._pool.fetchrow(
            f"SELECT COUNT(*) as cnt FROM ({base_query}) sub {where_clause}"
        )
        total = count_row["cnt"] if count_row else 0

        rows = await self._pool.fetch(
            f"""
            SELECT *,
                   (
                       sub.pending_chunk_translation_count
                       + sub.pending_sentence_translation_count
                   ) as pending_translation_count
            FROM ({base_query}) sub
            {where_clause}
            ORDER BY pending_translation_count DESC, created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

        summary_row = await self._pool.fetchrow(
            f"""
            SELECT
                COALESCE(SUM(sub.pending_chunk_translation_count), 0) as pending_chunk_translation_count,
                COALESCE(SUM(sub.pending_sentence_translation_count), 0) as pending_sentence_translation_count,
                COALESCE(SUM(sub.translation_count), 0) as translation_count
            FROM ({base_query}) sub
            {where_clause}
            """
        )

        summary = dict(summary_row) if summary_row else {
            "pending_chunk_translation_count": 0,
            "pending_sentence_translation_count": 0,
            "translation_count": 0,
        }
        summary["pending_translation_count"] = (
            summary["pending_chunk_translation_count"] + summary["pending_sentence_translation_count"]
        )

        return {
            "meetings": [dict(r) for r in rows],
            "total": total,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    # Chunk Storage
    # ------------------------------------------------------------------ #

    async def store_chunk(
        self,
        meeting_id: str,
        chunk_id: str,
        text: str,
        speaker_name: str | None = None,
        start_time: float = 0.0,
        end_time: float = 0.0,
        is_command: bool = False,
    ) -> None:
        """Store a deduplicated chunk. Uses UPSERT on (meeting_id, chunk_id)."""
        await self._ensure_pool()
        await self._pool.execute(
            """
            INSERT INTO meeting_chunks (id, meeting_id, chunk_id, text, speaker_name,
                                        start_time, end_time, is_command)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (meeting_id, chunk_id) DO UPDATE SET text = EXCLUDED.text
            """,
            str(uuid.uuid4()),
            meeting_id,
            chunk_id,
            text,
            speaker_name,
            start_time,
            end_time,
            is_command,
        )

    # ------------------------------------------------------------------ #
    # Sentence Storage
    # ------------------------------------------------------------------ #

    async def store_sentence(
        self,
        meeting_id: str,
        text: str,
        speaker_name: str | None = None,
        start_time: float = 0.0,
        end_time: float = 0.0,
        boundary_type: str | None = None,
        chunk_ids: list[str] | None = None,
    ) -> str:
        """Store an aggregated sentence. Returns sentence UUID."""
        await self._ensure_pool()
        sentence_id = str(uuid.uuid4())
        chunk_ids_json = json.dumps(chunk_ids or [])

        await self._pool.execute(
            """
            INSERT INTO meeting_sentences (id, meeting_id, text, speaker_name,
                                           start_time, end_time, boundary_type, chunk_ids)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            sentence_id,
            meeting_id,
            text,
            speaker_name,
            start_time,
            end_time,
            boundary_type,
            chunk_ids_json,
        )
        return sentence_id

    # ------------------------------------------------------------------ #
    # Translation Storage
    # ------------------------------------------------------------------ #

    async def store_translation(
        self,
        sentence_id: str,
        translated_text: str,
        target_language: str,
        source_language: str = "en",
        confidence: float = 1.0,
        translation_time_ms: float = 0.0,
        model_used: str | None = None,
    ) -> None:
        """Store a translation for a sentence."""
        await self._ensure_pool()
        await self._pool.execute(
            """
            INSERT INTO meeting_translations (id, sentence_id, translated_text,
                                              target_language, source_language,
                                              confidence, translation_time_ms, model_used)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            sentence_id,
            translated_text,
            target_language,
            source_language,
            confidence,
            translation_time_ms,
            model_used,
        )

    # ------------------------------------------------------------------ #
    # Insight Storage
    # ------------------------------------------------------------------ #

    async def store_insight(
        self,
        meeting_id: str,
        insight_type: str,
        content: dict[str, Any],
        source: str = "fireflies",
        model_used: str | None = None,
    ) -> None:
        """Store an AI insight (summary, action items, etc.)."""
        await self._ensure_pool()
        content_json = json.dumps(content)

        await self._pool.execute(
            """
            INSERT INTO meeting_data_insights (id, meeting_id, insight_type, content,
                                               source, model_used)
            VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5, $6)
            """,
            str(uuid.uuid4()),
            meeting_id,
            insight_type,
            content_json,
            source,
            model_used,
        )

    # ------------------------------------------------------------------ #
    # Speaker Storage
    # ------------------------------------------------------------------ #

    async def store_speaker(
        self,
        meeting_id: str,
        speaker_name: str,
        email: str | None = None,
        talk_time_seconds: float = 0.0,
        word_count: int = 0,
        sentiment_score: float | None = None,
        analytics: dict[str, Any] | None = None,
    ) -> None:
        """Upsert speaker metadata."""
        await self._ensure_pool()
        analytics_json = json.dumps(analytics) if analytics else None

        await self._pool.execute(
            """
            INSERT INTO meeting_speakers (id, meeting_id, speaker_name, email,
                                          talk_time_seconds, word_count,
                                          sentiment_score, analytics)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (meeting_id, speaker_name) DO UPDATE SET
                email = COALESCE(EXCLUDED.email, meeting_speakers.email),
                talk_time_seconds = EXCLUDED.talk_time_seconds,
                word_count = EXCLUDED.word_count,
                sentiment_score = EXCLUDED.sentiment_score,
                analytics = COALESCE(EXCLUDED.analytics, meeting_speakers.analytics)
            """,
            str(uuid.uuid4()),
            meeting_id,
            speaker_name,
            email,
            talk_time_seconds,
            word_count,
            sentiment_score,
            analytics_json,
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    async def search_meetings(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across chunks and sentences."""
        await self._ensure_pool()
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT m.id, m.title, m.created_at, m.status, m.source,
                   ts_rank(to_tsvector('english', mc.text), plainto_tsquery('english', $1)) as rank
            FROM meetings m
            JOIN meeting_chunks mc ON mc.meeting_id = m.id
            WHERE to_tsvector('english', mc.text) @@ plainto_tsquery('english', $1)
            UNION
            SELECT DISTINCT m.id, m.title, m.created_at, m.status, m.source,
                   ts_rank(to_tsvector('english', ms.text), plainto_tsquery('english', $1)) as rank
            FROM meetings m
            JOIN meeting_sentences ms ON ms.meeting_id = m.id
            WHERE to_tsvector('english', ms.text) @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2
            """,
            query,
            limit,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Insights Retrieval
    # ------------------------------------------------------------------ #

    async def get_meeting_insights(self, meeting_id: str) -> list[dict[str, Any]]:
        """Get all insights for a meeting."""
        await self._ensure_pool()
        rows = await self._pool.fetch(
            """
            SELECT * FROM meeting_data_insights
            WHERE meeting_id = $1::uuid
            ORDER BY created_at
            """,
            meeting_id,
        )
        return [dict(r) for r in rows]

    async def get_meeting_speakers(self, meeting_id: str) -> list[dict[str, Any]]:
        """Get all speakers for a meeting."""
        await self._ensure_pool()
        rows = await self._pool.fetch(
            """
            SELECT * FROM meeting_speakers
            WHERE meeting_id = $1::uuid
            ORDER BY talk_time_seconds DESC
            """,
            meeting_id,
        )
        return [dict(r) for r in rows]
