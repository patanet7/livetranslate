from __future__ import annotations

from dataclasses import dataclass, field

from translation.segment_record import SegmentPhase, SegmentRecord


@dataclass
class SegmentStore:
    """Per-session segment lifecycle tracker.

    Tracks each segment_id through draft/final phases.
    Replaces _stable_text_buffer and _last_translated_stable.

    Sentence accumulation: non-final finals (is_draft=False, is_final=False)
    append their text to _pending_sentence. When an is_final=True segment
    arrives, the accumulated sentence is returned for translation. This
    replaces the old _stable_text_buffer without any text-level dedup.
    """

    _records: dict[int, SegmentRecord] = field(default_factory=dict)
    _pending_sentence: str = ""
    _pending_segment_ids: list[int] = field(default_factory=list)

    def on_draft_received(
        self, segment_id: int, text: str, source_lang: str, target_lang: str,
    ) -> SegmentRecord:
        rec = SegmentRecord(
            segment_id=segment_id,
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            phase=SegmentPhase.DRAFT_RECEIVED,
        )
        self._records[segment_id] = rec
        return rec

    def on_draft_translated(
        self, segment_id: int, translation: str,
    ) -> SegmentRecord | None:
        rec = self._records.get(segment_id)
        if rec is None:
            return None
        rec.draft_translation = translation
        rec.phase = SegmentPhase.DRAFT_TRANSLATED
        return rec

    def on_final_received(
        self, segment_id: int, text: str, is_final: bool,
        source_lang: str, target_lang: str,
    ) -> tuple[SegmentRecord, str]:
        """Register a non-draft segment. Returns (record, translate_text).

        If is_final=False: accumulates text into _pending_sentence,
        returns empty translate_text (don't translate yet).

        If is_final=True: flushes _pending_sentence + this segment's text,
        returns the full accumulated sentence for translation.
        """
        rec = self._records.get(segment_id)
        if rec is None:
            rec = SegmentRecord(
                segment_id=segment_id,
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                phase=SegmentPhase.FINAL_RECEIVED,
            )
            self._records[segment_id] = rec
        else:
            rec.source_text = text
            rec.phase = SegmentPhase.FINAL_RECEIVED

        if not is_final:
            # Accumulate — sentence not complete yet
            self._pending_sentence += (" " if self._pending_sentence else "") + text.strip()
            self._pending_segment_ids.append(segment_id)
            return rec, ""

        # Sentence boundary: flush accumulated text + this segment
        accumulated = self._pending_sentence
        if accumulated:
            translate_text = (accumulated + " " + text.strip()).strip()
        else:
            translate_text = text.strip()

        self._pending_sentence = ""
        self._pending_segment_ids.clear()
        return rec, translate_text

    def on_final_translated(
        self, segment_id: int, translation: str,
    ) -> SegmentRecord | None:
        rec = self._records.get(segment_id)
        if rec is None:
            return None
        rec.final_translation = translation
        rec.phase = SegmentPhase.FINAL_TRANSLATED
        return rec

    def is_final_translated(self, segment_id: int) -> bool:
        rec = self._records.get(segment_id)
        return rec is not None and rec.phase == SegmentPhase.FINAL_TRANSLATED

    def get(self, segment_id: int) -> SegmentRecord | None:
        return self._records.get(segment_id)

    def flush_pending(self) -> str:
        """Force-flush any pending accumulated text (e.g., on session end)."""
        text = self._pending_sentence
        self._pending_sentence = ""
        self._pending_segment_ids.clear()
        return text

    def reset(self) -> None:
        self._records.clear()
        self._pending_sentence = ""
        self._pending_segment_ids.clear()

    def evict_old(self, keep_last: int = 50) -> None:
        if len(self._records) <= keep_last:
            return
        sorted_ids = sorted(self._records)
        for sid in sorted_ids[:-keep_last]:
            del self._records[sid]
