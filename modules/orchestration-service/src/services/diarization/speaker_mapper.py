"""
Speaker Mapper

Maps anonymous VibeVoice-ASR speaker IDs to human-readable names using one
or more identification strategies (e.g. Fireflies transcript cross-reference,
voice-print matching).  Multiple strategy results can be merged with the
highest-confidence assignment winning per speaker.
"""

from __future__ import annotations

from collections import defaultdict

from livetranslate_common.logging import get_logger

from models.diarization import SpeakerMapEntry, TranscribeSegment

logger = get_logger()


class SpeakerMapper:
    """Maps VibeVoice-ASR anonymous speaker IDs to real names.

    Strategies are applied independently and their results merged via
    :meth:`merge_maps`.  Each strategy produces a
    ``dict[int, SpeakerMapEntry]`` keyed by zero-based speaker index.
    """

    # ------------------------------------------------------------------
    # Strategy 1: Fireflies cross-reference
    # ------------------------------------------------------------------

    def crossref_fireflies(
        self,
        segments: list[TranscribeSegment],
        fireflies_sentences: list[dict],
    ) -> dict[int, SpeakerMapEntry]:
        """Map VibeVoice speakers to names via Fireflies timestamp overlap.

        For every VibeVoice speaker, we accumulate the total seconds of
        overlap between that speaker's segments and each Fireflies sentence.
        The Fireflies ``speaker_name`` with the greatest accumulated overlap
        wins the identity assignment.

        Confidence is defined as ``best_overlap / total_overlap`` across all
        Fireflies speakers for that VibeVoice speaker.  Speakers with zero
        total overlap are excluded from the result.

        Args:
            segments: Ordered list of diarized VibeVoice-ASR segments.
            fireflies_sentences: Raw Fireflies sentence dicts, each expected
                to contain at minimum ``speaker_name``, ``start_time``, and
                ``end_time`` keys.

        Returns:
            Mapping of VibeVoice speaker index to :class:`SpeakerMapEntry`.
            Only speakers that overlap with at least one Fireflies sentence
            are included.
        """
        if not segments or not fireflies_sentences:
            return {}

        # overlap_accumulator[speaker_id][ff_speaker_name] = total overlap seconds
        overlap_accumulator: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for segment in segments:
            seg_start = segment.start
            seg_end = segment.end

            for ff_sentence in fireflies_sentences:
                ff_start: float = ff_sentence.get("start_time", 0.0)
                ff_end: float = ff_sentence.get("end_time", 0.0)
                ff_speaker: str = ff_sentence.get("speaker_name", "")

                overlap = _interval_overlap(seg_start, seg_end, ff_start, ff_end)
                if overlap > 0.0:
                    overlap_accumulator[segment.speaker][ff_speaker] += overlap

        result: dict[int, SpeakerMapEntry] = {}

        for speaker_id, ff_overlaps in overlap_accumulator.items():
            if not ff_overlaps:
                continue

            total_overlap = sum(ff_overlaps.values())
            if total_overlap <= 0.0:
                continue

            best_ff_speaker = max(ff_overlaps, key=lambda k: ff_overlaps[k])
            best_overlap = ff_overlaps[best_ff_speaker]
            confidence = best_overlap / total_overlap

            result[speaker_id] = SpeakerMapEntry(
                name=best_ff_speaker,
                confidence=confidence,
                method="fireflies_crossref",
            )

            logger.debug(
                "speaker_mapped_via_fireflies",
                speaker_id=speaker_id,
                name=best_ff_speaker,
                confidence=confidence,
                best_overlap_seconds=best_overlap,
                total_overlap_seconds=total_overlap,
            )

        return result

    # ------------------------------------------------------------------
    # Map merging
    # ------------------------------------------------------------------

    def merge_maps(
        self,
        maps: list[dict[int, SpeakerMapEntry]],
    ) -> dict[int, SpeakerMapEntry]:
        """Merge multiple speaker maps, keeping the highest-confidence entry.

        Each input map may have been produced by a different identification
        strategy.  For each speaker ID that appears in more than one map,
        the entry with the highest ``confidence`` value wins.  Ties are
        broken in favour of whichever entry appears first.

        Args:
            maps: List of speaker maps to merge.  Order affects tie-breaking
                only (earlier entries win ties).

        Returns:
            Single merged map with the best entry per speaker ID.
        """
        merged: dict[int, SpeakerMapEntry] = {}

        for speaker_map in maps:
            for speaker_id, entry in speaker_map.items():
                existing = merged.get(speaker_id)
                if existing is None or entry.confidence > existing.confidence:
                    merged[speaker_id] = entry

        return merged

    # ------------------------------------------------------------------
    # Unmapped speaker detection
    # ------------------------------------------------------------------

    def find_unmapped(
        self,
        segments: list[TranscribeSegment],
        speaker_map: dict[int, SpeakerMapEntry],
    ) -> list[int]:
        """Return a sorted list of speaker IDs that have no mapping entry.

        Scans all segment speaker IDs, then filters out any that are present
        as keys in ``speaker_map``.

        Args:
            segments: All diarized segments for the recording.
            speaker_map: Current best-known speaker assignments.

        Returns:
            Sorted list of speaker IDs (integers) with no mapping.
        """
        all_speaker_ids = {segment.speaker for segment in segments}
        unmapped = sorted(all_speaker_ids - speaker_map.keys())

        if unmapped:
            logger.info(
                "unmapped_speakers_detected",
                count=len(unmapped),
                speaker_ids=unmapped,
            )

        return unmapped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interval_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Return the duration of overlap between two time intervals.

    Args:
        a_start: Start of interval A in seconds.
        a_end: End of interval A in seconds.
        b_start: Start of interval B in seconds.
        b_end: End of interval B in seconds.

    Returns:
        Overlap duration in seconds; 0.0 if the intervals do not overlap.
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)
