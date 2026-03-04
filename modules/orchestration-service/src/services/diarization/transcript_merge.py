"""
Transcript merge: aligns Fireflies sentences with VibeVoice-ASR segments.

For each Fireflies sentence, the segment with the greatest temporal overlap
is found in the VibeVoice output.  The matched speaker ID is resolved through
the provided speaker map to produce a human-readable name, and three
diarization fields are added to a copy of the sentence dict.

If no VibeVoice segment overlaps with a given sentence the original
``speaker_name`` is preserved unchanged and no diarization fields are added.
"""

from __future__ import annotations

from livetranslate_common.logging import get_logger

from models.diarization import SpeakerMapEntry, TranscribeSegment

logger = get_logger()


def _overlap_seconds(
    ff_start: float,
    ff_end: float,
    vv_start: float,
    vv_end: float,
) -> float:
    """Return the number of seconds that two time intervals share.

    Args:
        ff_start: Fireflies sentence start in seconds.
        ff_end: Fireflies sentence end in seconds.
        vv_start: VibeVoice segment start in seconds.
        vv_end: VibeVoice segment end in seconds.

    Returns:
        Overlap duration in seconds (0.0 if the intervals do not overlap).
    """
    overlap_start = max(ff_start, vv_start)
    overlap_end = min(ff_end, vv_end)
    return max(0.0, overlap_end - overlap_start)


def merge_transcripts(
    fireflies_sentences: list[dict],
    vibevoice_segments: list[TranscribeSegment],
    speaker_map: dict[int, SpeakerMapEntry],
) -> list[dict]:
    """Align Fireflies sentences with VibeVoice-ASR segments and remap speakers.

    For each Fireflies sentence the VibeVoice segment with the greatest
    temporal overlap is selected.  The matched speaker ID is looked up in
    ``speaker_map``; if found the human-readable name replaces
    ``speaker_name``, otherwise ``SPEAKER_<id>`` is used.  Three extra fields
    are appended to signal the diarization source:

    - ``diarization_source`` — always ``"vibevoice"`` when a match is found
    - ``diarization_speaker_id`` — integer speaker index from VibeVoice
    - ``diarization_confidence`` — confidence from the speaker-map entry, or
      ``0.0`` when the speaker is not in the map

    When no VibeVoice segment overlaps with a sentence the original dict is
    returned unchanged (no diarization fields added).

    The function never mutates the input dicts; each returned dict is a
    shallow copy with the relevant fields overwritten or appended.

    Args:
        fireflies_sentences: Ordered list of sentence dicts from the Fireflies
            API.  Each dict must contain at minimum ``speaker_name``,
            ``start_time``, and ``end_time`` keys.
        vibevoice_segments: Ordered list of :class:`TranscribeSegment` objects
            produced by the VibeVoice-ASR / Whisper diarization pipeline.
        speaker_map: Mapping from zero-based integer speaker index to a
            :class:`SpeakerMapEntry` with ``name`` and ``confidence``.

    Returns:
        A new list with one dict per input sentence.  Original fields are
        preserved on every dict; diarization fields are added only where a
        VibeVoice match was found.
    """
    if not vibevoice_segments:
        logger.info(
            "transcript_merge.no_vibevoice_segments",
            fireflies_count=len(fireflies_sentences),
        )
        return [dict(sentence) for sentence in fireflies_sentences]

    result: list[dict] = []

    for sentence in fireflies_sentences:
        ff_start: float = sentence["start_time"]
        ff_end: float = sentence["end_time"]

        best_segment: TranscribeSegment | None = None
        best_overlap: float = 0.0

        for seg in vibevoice_segments:
            overlap = _overlap_seconds(ff_start, ff_end, seg.start, seg.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = seg

        if best_segment is None:
            # No temporal overlap — emit a copy with no diarization fields
            logger.debug(
                "transcript_merge.no_overlap",
                start=ff_start,
                end=ff_end,
                original_speaker=sentence.get("speaker_name"),
            )
            result.append(dict(sentence))
            continue

        speaker_id: int = best_segment.speaker
        entry: SpeakerMapEntry | None = speaker_map.get(speaker_id)

        merged = dict(sentence)

        if entry is not None:
            merged["speaker_name"] = entry.name
            merged["diarization_confidence"] = entry.confidence
        else:
            merged["speaker_name"] = f"SPEAKER_{speaker_id}"
            merged["diarization_confidence"] = 0.0

        merged["diarization_source"] = "vibevoice"
        merged["diarization_speaker_id"] = speaker_id

        logger.debug(
            "transcript_merge.mapped",
            start=ff_start,
            end=ff_end,
            speaker_id=speaker_id,
            speaker_name=merged["speaker_name"],
            overlap_seconds=best_overlap,
        )

        result.append(merged)

    logger.info(
        "transcript_merge.complete",
        sentence_count=len(fireflies_sentences),
        matched_count=sum(1 for d in result if "diarization_source" in d),
    )
    return result
