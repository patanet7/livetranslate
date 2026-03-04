"""
Tests for VibeVoice-ASR vLLM HTTP client.

Pure parsing tests — no network calls required.
"""

import json

import pytest

from clients.vibevoice_client import VibeVoiceClient, VibeVoiceError
from models.diarization import TranscribeResponse


def test_client_init():
    c = VibeVoiceClient(base_url="http://localhost:8000/v1")
    assert c.base_url == "http://localhost:8000/v1"


def test_client_init_strips_trailing_slash():
    c = VibeVoiceClient(base_url="http://localhost:8000/v1/")
    assert c.base_url == "http://localhost:8000/v1"


def test_parse_segments():
    raw = json.dumps([
        {"speaker": 0, "start": 0.5, "end": 3.2, "text": "Hello there"},
        {"speaker": 1, "start": 3.5, "end": 7.0, "text": "Hi, how are you"},
    ])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw, duration_seconds=60.0, processing_time=10.0)
    assert isinstance(result, TranscribeResponse)
    assert len(result.segments) == 2
    assert result.num_speakers == 2


def test_parse_deduplicates_speakers():
    raw = json.dumps([
        {"speaker": 0, "start": 0.0, "end": 1.0, "text": "A"},
        {"speaker": 1, "start": 1.0, "end": 2.0, "text": "B"},
        {"speaker": 0, "start": 2.0, "end": 3.0, "text": "C"},
    ])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw, duration_seconds=3.0, processing_time=1.0)
    assert result.num_speakers == 2


def test_parse_empty():
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output("[]", duration_seconds=0.0, processing_time=0.0)
    assert len(result.segments) == 0
    assert result.num_speakers == 0


def test_parse_invalid_json():
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output("not json", duration_seconds=0.0, processing_time=0.0)
    assert len(result.segments) == 0


def test_vibevoice_error():
    err = VibeVoiceError("test error", status_code=500)
    assert str(err) == "test error"
    assert err.status_code == 500


def test_parse_segment_fields():
    """Verify segment fields are correctly mapped from VibeVoice JSON."""
    raw = json.dumps([
        {"speaker": 2, "start": 1.1, "end": 4.4, "text": "Testing fields"},
    ])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw, duration_seconds=10.0, processing_time=2.0)
    seg = result.segments[0]
    assert seg.speaker == 2
    assert seg.start == pytest.approx(1.1)
    assert seg.end == pytest.approx(4.4)
    assert seg.text == "Testing fields"


def test_parse_metadata_propagated():
    """Verify duration_seconds and processing_time_seconds are set on the response."""
    raw = json.dumps([{"speaker": 0, "start": 0.0, "end": 1.0, "text": "Hi"}])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw, duration_seconds=120.5, processing_time=30.25)
    assert result.duration_seconds == pytest.approx(120.5)
    assert result.processing_time_seconds == pytest.approx(30.25)


def test_vibevoice_error_no_status_code():
    """VibeVoiceError can be created without a status_code."""
    err = VibeVoiceError("something went wrong")
    assert str(err) == "something went wrong"
    assert err.status_code is None
