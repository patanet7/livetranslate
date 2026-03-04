"""
Behavioral tests for SpeakerMapper.

Tests the Fireflies cross-reference strategy, map merging, and unmapped speaker
detection — all without mocks, exercising real logic.
"""

import pytest

from models.diarization import SpeakerMapEntry, TranscribeSegment
from services.diarization.speaker_mapper import SpeakerMapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segments() -> list[TranscribeSegment]:
    return [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello from speaker zero"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="Hi from speaker one"),
        TranscribeSegment(speaker=0, start=10.0, end=15.0, text="More from zero"),
    ]


# ---------------------------------------------------------------------------
# crossref_fireflies
# ---------------------------------------------------------------------------


def test_crossref_by_timestamp_overlap():
    mapper = SpeakerMapper()
    segments = _make_segments()
    ff = [
        {"speaker_name": "Eric Chen", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
        {"speaker_name": "Thomas", "start_time": 5.0, "end_time": 10.0, "text": "Hi"},
    ]
    result = mapper.crossref_fireflies(segments, ff)
    assert result[0].name == "Eric Chen"
    assert result[0].method == "fireflies_crossref"
    assert result[1].name == "Thomas"


def test_crossref_no_overlap():
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=100.0, end=105.0, text="Late")]
    ff = [{"speaker_name": "Eric", "start_time": 0.0, "end_time": 5.0, "text": "Early"}]
    assert len(mapper.crossref_fireflies(segments, ff)) == 0


def test_crossref_picks_best_overlap():
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=0.0, end=10.0, text="Long")]
    ff = [
        {"speaker_name": "Eric", "start_time": 0.0, "end_time": 3.0, "text": "Short"},
        {"speaker_name": "Thomas", "start_time": 2.0, "end_time": 10.0, "text": "Longer"},
    ]
    result = mapper.crossref_fireflies(segments, ff)
    assert result[0].name == "Thomas"


def test_crossref_confidence_between_zero_and_one():
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=0.0, end=10.0, text="Overlap test")]
    ff = [
        {"speaker_name": "Alice", "start_time": 0.0, "end_time": 6.0, "text": "A"},
        {"speaker_name": "Bob", "start_time": 4.0, "end_time": 10.0, "text": "B"},
    ]
    result = mapper.crossref_fireflies(segments, ff)
    entry = result[0]
    assert 0.0 <= entry.confidence <= 1.0


def test_crossref_empty_segments():
    mapper = SpeakerMapper()
    result = mapper.crossref_fireflies([], [{"speaker_name": "X", "start_time": 0.0, "end_time": 1.0, "text": "x"}])
    assert result == {}


def test_crossref_empty_fireflies():
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hi")]
    result = mapper.crossref_fireflies(segments, [])
    assert result == {}


def test_crossref_multiple_segments_same_speaker():
    """Speaker 0 appears in two non-contiguous segments; accumulated overlap wins."""
    mapper = SpeakerMapper()
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="First"),
        TranscribeSegment(speaker=0, start=10.0, end=15.0, text="Second"),
    ]
    ff = [
        {"speaker_name": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "A"},
        {"speaker_name": "Bob", "start_time": 10.0, "end_time": 15.0, "text": "B"},
    ]
    # Both have equal overlap (5s each); the winner is whichever accumulates more.
    # Here Alice and Bob are tied — just assert the result has an entry and it's valid.
    result = mapper.crossref_fireflies(segments, ff)
    assert 0 in result
    assert result[0].method == "fireflies_crossref"


# ---------------------------------------------------------------------------
# merge_maps
# ---------------------------------------------------------------------------


def test_merge_maps():
    mapper = SpeakerMapper()
    a = {0: SpeakerMapEntry(name="Eric", confidence=0.7, method="fireflies_crossref")}
    b = {1: SpeakerMapEntry(name="Thomas", confidence=0.9, method="voice_profile")}
    merged = mapper.merge_maps([a, b])
    assert merged[0].name == "Eric"
    assert merged[1].name == "Thomas"


def test_merge_higher_confidence_wins():
    mapper = SpeakerMapper()
    low = {0: SpeakerMapEntry(name="Unknown", confidence=0.3, method="fireflies_crossref")}
    high = {0: SpeakerMapEntry(name="Eric", confidence=0.95, method="voice_profile")}
    merged = mapper.merge_maps([low, high])
    assert merged[0].name == "Eric"


def test_merge_order_does_not_matter_for_confidence():
    mapper = SpeakerMapper()
    high = {0: SpeakerMapEntry(name="Eric", confidence=0.95, method="voice_profile")}
    low = {0: SpeakerMapEntry(name="Unknown", confidence=0.3, method="fireflies_crossref")}
    # High-confidence map comes first this time
    merged = mapper.merge_maps([high, low])
    assert merged[0].name == "Eric"


def test_merge_empty_list():
    mapper = SpeakerMapper()
    assert mapper.merge_maps([]) == {}


def test_merge_single_map():
    mapper = SpeakerMapper()
    m = {0: SpeakerMapEntry(name="Solo", confidence=0.5, method="manual")}
    merged = mapper.merge_maps([m])
    assert merged[0].name == "Solo"


def test_merge_three_maps_highest_wins():
    mapper = SpeakerMapper()
    maps = [
        {0: SpeakerMapEntry(name="Low", confidence=0.2, method="a")},
        {0: SpeakerMapEntry(name="High", confidence=0.9, method="b")},
        {0: SpeakerMapEntry(name="Mid", confidence=0.5, method="c")},
    ]
    merged = mapper.merge_maps(maps)
    assert merged[0].name == "High"


# ---------------------------------------------------------------------------
# find_unmapped
# ---------------------------------------------------------------------------


def test_find_unmapped():
    mapper = SpeakerMapper()
    segments = _make_segments()
    speaker_map = {0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual")}
    assert mapper.find_unmapped(segments, speaker_map) == [1]


def test_find_unmapped_all_mapped():
    mapper = SpeakerMapper()
    segments = _make_segments()
    speaker_map = {
        0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual"),
        1: SpeakerMapEntry(name="Thomas", confidence=0.8, method="manual"),
    }
    assert mapper.find_unmapped(segments, speaker_map) == []


def test_find_unmapped_none_mapped():
    mapper = SpeakerMapper()
    segments = _make_segments()
    assert mapper.find_unmapped(segments, {}) == [0, 1]


def test_find_unmapped_returns_sorted():
    mapper = SpeakerMapper()
    segments = [
        TranscribeSegment(speaker=3, start=0.0, end=1.0, text="Three"),
        TranscribeSegment(speaker=1, start=1.0, end=2.0, text="One"),
        TranscribeSegment(speaker=2, start=2.0, end=3.0, text="Two"),
    ]
    result = mapper.find_unmapped(segments, {})
    assert result == [1, 2, 3]


def test_find_unmapped_empty_segments():
    mapper = SpeakerMapper()
    assert mapper.find_unmapped([], {}) == []
