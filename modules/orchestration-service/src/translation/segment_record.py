from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentPhase(str, Enum):
    DRAFT_RECEIVED = "draft_received"
    DRAFT_TRANSLATED = "draft_translated"
    FINAL_RECEIVED = "final_received"
    FINAL_TRANSLATED = "final_translated"


@dataclass
class SegmentRecord:
    segment_id: int
    source_text: str
    source_lang: str
    target_lang: str
    phase: SegmentPhase = SegmentPhase.DRAFT_RECEIVED
    draft_translation: str | None = None
    final_translation: str | None = None
