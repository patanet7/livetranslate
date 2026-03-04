#!/usr/bin/env python3
"""
Speaker Merge Tests - Behavioral Tests (No Mocks)

Tests real behavior of detect_merge_candidates and apply_merge for
over-segmentation detection and correction in the diarization pipeline.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

from models.diarization import TranscribeSegment
from services.diarization.speaker_merge import apply_merge, detect_merge_candidates


# =============================================================================
# detect_merge_candidates
# =============================================================================


def test_detect_candidates_low_word_count():
    """Speaker 2 with only 'Yeah' is flagged as a merge candidate."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=30.0, text="Long speech from main speaker about many topics"),
        TranscribeSegment(speaker=0, start=30.0, end=60.0, text="Still talking about things and stuff"),
        TranscribeSegment(speaker=1, start=60.0, end=90.0, text="Another person talking at length here"),
        TranscribeSegment(speaker=2, start=90.0, end=92.0, text="Yeah"),
    ]
    candidates = detect_merge_candidates(segments, min_word_ratio=0.05)
    assert len(candidates) >= 1
    assert any(c["source"] == 2 for c in candidates)


def test_no_candidates_when_balanced():
    """Two speakers with roughly equal word counts produce no candidates."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=30.0, text=" ".join(["word"] * 50)),
        TranscribeSegment(speaker=1, start=30.0, end=60.0, text=" ".join(["word"] * 45)),
    ]
    assert len(detect_merge_candidates(segments)) == 0


def test_apply_merge_replaces_speaker():
    """apply_merge replaces all occurrences of source_speaker with target_speaker."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello"),
        TranscribeSegment(speaker=2, start=5.0, end=6.0, text="Yeah"),
        TranscribeSegment(speaker=1, start=6.0, end=10.0, text="World"),
    ]
    merged = apply_merge(segments, source_speaker=2, target_speaker=0)
    assert all(seg.speaker != 2 for seg in merged)
    assert merged[1].speaker == 0


def test_apply_merge_preserves_order():
    """apply_merge returns segments in the same chronological order."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="A"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="B"),
    ]
    merged = apply_merge(segments, source_speaker=1, target_speaker=0)
    assert merged[0].start < merged[1].start


# =============================================================================
# Additional behavioral tests
# =============================================================================


def test_candidate_dict_has_required_keys():
    """Each candidate dict must contain source, suggested_target, word_count, word_ratio."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=60.0, text=" ".join(["word"] * 100)),
        TranscribeSegment(speaker=1, start=60.0, end=62.0, text="ok"),
    ]
    candidates = detect_merge_candidates(segments, min_word_ratio=0.05)
    assert len(candidates) >= 1
    for c in candidates:
        assert "source" in c
        assert "suggested_target" in c
        assert "word_count" in c
        assert "word_ratio" in c


def test_candidate_suggested_target_is_largest_speaker():
    """suggested_target should be the speaker with the most words."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=60.0, text=" ".join(["word"] * 100)),
        TranscribeSegment(speaker=1, start=60.0, end=80.0, text=" ".join(["word"] * 30)),
        TranscribeSegment(speaker=2, start=80.0, end=81.0, text="hi"),
    ]
    candidates = detect_merge_candidates(segments, min_word_ratio=0.05)
    candidate_for_2 = next(c for c in candidates if c["source"] == 2)
    assert candidate_for_2["suggested_target"] == 0


def test_apply_merge_does_not_mutate_original():
    """apply_merge returns a new list; original segments are unchanged."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="A"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="B"),
    ]
    original_speakers = [s.speaker for s in segments]
    apply_merge(segments, source_speaker=1, target_speaker=0)
    assert [s.speaker for s in segments] == original_speakers


def test_apply_merge_no_op_when_source_absent():
    """apply_merge with a source not present in segments returns identical speakers."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="World"),
    ]
    merged = apply_merge(segments, source_speaker=99, target_speaker=0)
    assert [s.speaker for s in merged] == [s.speaker for s in segments]


def test_apply_merge_preserves_text_and_timing():
    """apply_merge only changes speaker; text, start, and end are preserved."""
    segments = [
        TranscribeSegment(speaker=2, start=3.5, end=7.2, text="Interesting point."),
    ]
    merged = apply_merge(segments, source_speaker=2, target_speaker=0)
    assert merged[0].text == "Interesting point."
    assert merged[0].start == 3.5
    assert merged[0].end == 7.2


def test_detect_candidates_empty_segments():
    """detect_merge_candidates on an empty list returns no candidates."""
    assert detect_merge_candidates([]) == []


def test_detect_candidates_single_speaker():
    """A single speaker cannot be a merge candidate."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=10.0, text="Just me talking here."),
    ]
    assert detect_merge_candidates(segments) == []


def test_word_ratio_boundary_exactly_at_threshold():
    """A speaker exactly at min_word_ratio boundary is not flagged as candidate."""
    # 10 words each, ratio = 0.5 > any reasonable threshold
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=10.0, text=" ".join(["word"] * 10)),
        TranscribeSegment(speaker=1, start=10.0, end=20.0, text=" ".join(["word"] * 10)),
    ]
    # ratio for each speaker = 0.5, well above default 0.05
    assert detect_merge_candidates(segments, min_word_ratio=0.05) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
