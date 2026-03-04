"""
Speaker Merge

Detects and fixes over-segmentation in diarization output.

Over-segmentation occurs when the diarizer assigns a separate speaker ID to what
is actually the same person — typically because a short interjection ("Yeah",
"Hmm") receives its own cluster.  This module provides two pure functions:

- ``detect_merge_candidates`` — identify speakers whose word share is below a
  configurable ratio threshold.
- ``apply_merge`` — rewrite a segment list, replacing one speaker ID with
  another without mutating the original.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from livetranslate_common.logging import get_logger

from models.diarization import TranscribeSegment

logger = get_logger()


def _word_counts(segments: list[TranscribeSegment]) -> dict[int, int]:
    """Return the total word count for each speaker across all segments.

    Args:
        segments: Ordered list of diarized transcription segments.

    Returns:
        Mapping of speaker index to total word count.
    """
    counts: dict[int, int] = defaultdict(int)
    for seg in segments:
        counts[seg.speaker] += len(seg.text.split())
    return dict(counts)


def detect_merge_candidates(
    segments: list[TranscribeSegment],
    min_word_ratio: float = 0.05,
) -> list[dict]:
    """Detect speakers with very few words relative to the total transcript.

    A speaker whose word share falls below ``min_word_ratio`` of the total word
    count is considered a merge candidate — likely an artefact of diarizer
    over-segmentation rather than a genuine distinct participant.

    Args:
        segments: Ordered list of diarized transcription segments.
        min_word_ratio: Minimum fraction of total words a speaker must have to
            be considered non-trivial.  Speakers below this threshold are
            returned as candidates.  Defaults to ``0.05`` (5%).

    Returns:
        List of candidate dicts, each with the keys:

        - ``"source"`` (int): Speaker index to merge away.
        - ``"suggested_target"`` (int): Speaker index with the most words —
          the most likely merge destination.
        - ``"word_count"`` (int): Absolute word count for the source speaker.
        - ``"word_ratio"`` (float): Fractional word share for the source speaker.

        Returns an empty list when there are no segments, only one distinct
        speaker, or no speakers fall below the threshold.
    """
    if not segments:
        return []

    counts = _word_counts(segments)

    # Need at least two speakers for a merge to be meaningful
    if len(counts) < 2:
        return []

    total_words = sum(counts.values())
    if total_words == 0:
        return []

    # Speaker with the most words is the default merge target
    dominant_speaker = max(counts, key=lambda spk: counts[spk])

    candidates: list[dict] = []
    for speaker, word_count in counts.items():
        ratio = word_count / total_words
        if ratio < min_word_ratio:
            logger.info(
                "speaker_merge.candidate_detected",
                source_speaker=speaker,
                word_count=word_count,
                word_ratio=round(ratio, 4),
                suggested_target=dominant_speaker,
            )
            candidates.append(
                {
                    "source": speaker,
                    "suggested_target": dominant_speaker,
                    "word_count": word_count,
                    "word_ratio": ratio,
                }
            )

    return candidates


def apply_merge(
    segments: list[TranscribeSegment],
    source_speaker: int,
    target_speaker: int,
) -> list[TranscribeSegment]:
    """Replace all occurrences of *source_speaker* with *target_speaker*.

    The function is pure — it returns a new list of ``TranscribeSegment``
    objects and never mutates the originals.  Segments that do not involve
    ``source_speaker`` are included in the output unchanged.

    Args:
        segments: Ordered list of diarized transcription segments.
        source_speaker: Speaker index to merge away (will not appear in output).
        target_speaker: Speaker index to merge into (retained in output).

    Returns:
        New list of ``TranscribeSegment`` objects with every ``source_speaker``
        replaced by ``target_speaker``.  Ordering and all other fields
        (``start``, ``end``, ``text``) are preserved exactly.
    """
    result: list[TranscribeSegment] = []
    for seg in segments:
        if seg.speaker == source_speaker:
            result.append(
                TranscribeSegment(
                    speaker=target_speaker,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                )
            )
        else:
            result.append(seg)

    if any(seg.speaker == source_speaker for seg in segments):
        logger.info(
            "speaker_merge.merge_applied",
            source_speaker=source_speaker,
            target_speaker=target_speaker,
            segments_affected=sum(1 for seg in segments if seg.speaker == source_speaker),
        )

    return result
