"""
Behavioral tests for transcript merge logic.

Aligns Fireflies sentences with VibeVoice-ASR segments and applies
speaker-map substitution.  No mocks — tests exercise real logic only.
"""

import pytest

from models.diarization import SpeakerMapEntry, TranscribeSegment
from services.diarization.transcript_merge import merge_transcripts


# ---------------------------------------------------------------------------
# Basic merge / speaker replacement
# ---------------------------------------------------------------------------


def test_basic_merge_replaces_speakers():
    ff = [
        {"speaker_name": "Speaker 1", "start_time": 0.0, "end_time": 5.0, "text": "Hello there", "index": 0},
        {"speaker_name": "Speaker 2", "start_time": 5.0, "end_time": 10.0, "text": "Hi back", "index": 1},
    ]
    vv = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello there"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="Hi back"),
    ]
    sm = {
        0: SpeakerMapEntry(name="Eric", confidence=0.9, method="fireflies_crossref"),
        1: SpeakerMapEntry(name="Thomas", confidence=0.85, method="voice_profile"),
    }
    merged = merge_transcripts(ff, vv, sm)
    assert merged[0]["speaker_name"] == "Eric"
    assert merged[1]["speaker_name"] == "Thomas"
    assert merged[0]["text"] == "Hello there"


# ---------------------------------------------------------------------------
# All original fields are preserved
# ---------------------------------------------------------------------------


def test_merge_preserves_all_fields():
    ff = [
        {
            "speaker_name": "S1",
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Hello",
            "index": 0,
            "raw_text": "Hello",
            "speaker_id": 1,
        }
    ]
    vv = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello")]
    sm = {0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual")}
    merged = merge_transcripts(ff, vv, sm)
    assert merged[0]["index"] == 0
    assert merged[0]["raw_text"] == "Hello"
    assert merged[0]["speaker_id"] == 1
    assert merged[0]["diarization_source"] == "vibevoice"
    assert merged[0]["diarization_speaker_id"] == 0
    assert merged[0]["diarization_confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Unmapped speaker falls back to SPEAKER_N label
# ---------------------------------------------------------------------------


def test_unmapped_speaker_gets_speaker_id():
    ff = [{"speaker_name": "S1", "start_time": 0.0, "end_time": 5.0, "text": "Hello", "index": 0}]
    vv = [TranscribeSegment(speaker=3, start=0.0, end=5.0, text="Hello")]
    merged = merge_transcripts(ff, vv, {})
    assert merged[0]["speaker_name"] == "SPEAKER_3"
    assert merged[0]["diarization_source"] == "vibevoice"
    assert merged[0]["diarization_speaker_id"] == 3


# ---------------------------------------------------------------------------
# No temporal overlap keeps original speaker unchanged
# ---------------------------------------------------------------------------


def test_no_overlap_keeps_original():
    ff = [{"speaker_name": "Eric", "start_time": 100.0, "end_time": 105.0, "text": "Late", "index": 0}]
    vv = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Early")]
    sm = {0: SpeakerMapEntry(name="Thomas", confidence=0.9, method="manual")}
    merged = merge_transcripts(ff, vv, sm)
    assert merged[0]["speaker_name"] == "Eric"
    assert merged[0].get("diarization_source") is None


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_fireflies_returns_empty():
    vv = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hi")]
    sm = {0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual")}
    merged = merge_transcripts([], vv, sm)
    assert merged == []


def test_empty_vibevoice_keeps_all_originals():
    ff = [
        {"speaker_name": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "Hi", "index": 0},
        {"speaker_name": "Bob", "start_time": 5.0, "end_time": 10.0, "text": "Hey", "index": 1},
    ]
    merged = merge_transcripts(ff, [], {})
    assert len(merged) == 2
    assert merged[0]["speaker_name"] == "Alice"
    assert merged[1]["speaker_name"] == "Bob"
    assert merged[0].get("diarization_source") is None
    assert merged[1].get("diarization_source") is None


# ---------------------------------------------------------------------------
# Best-overlap selection when multiple VibeVoice segments overlap
# ---------------------------------------------------------------------------


def test_best_overlap_segment_selected():
    """When multiple VV segments overlap, the one with the largest overlap wins."""
    ff = [{"speaker_name": "S1", "start_time": 2.0, "end_time": 8.0, "text": "Overlap test", "index": 0}]
    vv = [
        # overlaps 2s (2.0–4.0)
        TranscribeSegment(speaker=0, start=0.0, end=4.0, text="short"),
        # overlaps 4s (4.0–8.0) — should win
        TranscribeSegment(speaker=1, start=4.0, end=10.0, text="long"),
    ]
    sm = {
        0: SpeakerMapEntry(name="Alice", confidence=0.8, method="manual"),
        1: SpeakerMapEntry(name="Bob", confidence=0.9, method="manual"),
    }
    merged = merge_transcripts(ff, vv, sm)
    assert merged[0]["speaker_name"] == "Bob"
    assert merged[0]["diarization_speaker_id"] == 1


# ---------------------------------------------------------------------------
# Output list length matches input Fireflies sentences
# ---------------------------------------------------------------------------


def test_output_length_matches_fireflies():
    ff = [
        {"speaker_name": f"S{i}", "start_time": float(i * 5), "end_time": float(i * 5 + 5), "text": f"word {i}", "index": i}
        for i in range(5)
    ]
    vv = [TranscribeSegment(speaker=0, start=0.0, end=25.0, text="all")]
    sm = {0: SpeakerMapEntry(name="Eric", confidence=0.95, method="manual")}
    merged = merge_transcripts(ff, vv, sm)
    assert len(merged) == len(ff)


# ---------------------------------------------------------------------------
# Original dict is not mutated
# ---------------------------------------------------------------------------


def test_original_fireflies_dicts_not_mutated():
    ff = [{"speaker_name": "Original", "start_time": 0.0, "end_time": 5.0, "text": "Hello", "index": 0}]
    vv = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello")]
    sm = {0: SpeakerMapEntry(name="Replaced", confidence=0.9, method="manual")}
    merge_transcripts(ff, vv, sm)
    # Original dict must be untouched
    assert ff[0]["speaker_name"] == "Original"
    assert "diarization_source" not in ff[0]
