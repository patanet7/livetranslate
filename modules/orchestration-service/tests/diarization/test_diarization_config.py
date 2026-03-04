"""
Tests for DiarizationSettings configuration class.

Behavioral tests - no mocks. Tests exercise real Pydantic/pydantic-settings
validation and env-var parsing.
"""

import pytest

from config import DiarizationSettings


def test_default_settings():
    """DiarizationSettings has correct defaults when no env vars are set."""
    settings = DiarizationSettings()
    assert settings.vibevoice_url == "http://localhost:8000/v1"
    assert settings.enabled is False
    assert settings.max_concurrent_jobs == 1
    assert settings.auto_apply_threshold == 0.85


def test_env_override(monkeypatch):
    """DIARIZATION_* env vars are picked up and coerced to the right types."""
    monkeypatch.setenv("DIARIZATION_VIBEVOICE_URL", "http://192.168.1.50:8000/v1")
    monkeypatch.setenv("DIARIZATION_ENABLED", "true")
    settings = DiarizationSettings()
    assert settings.vibevoice_url == "http://192.168.1.50:8000/v1"
    assert settings.enabled is True


def test_hotwords_from_comma_string(monkeypatch):
    """DIARIZATION_HOTWORDS comma-separated string is parsed into a list."""
    monkeypatch.setenv("DIARIZATION_HOTWORDS", "sprint,deploy,LiveTranslate")
    settings = DiarizationSettings()
    assert settings.hotwords == ["sprint", "deploy", "LiveTranslate"]


def test_has_vibevoice_url():
    """has_vibevoice_url() returns True when URL is set and False when blank."""
    settings = DiarizationSettings()
    assert settings.has_vibevoice_url() is True
    settings.vibevoice_url = ""
    assert settings.has_vibevoice_url() is False


def test_hotwords_default_empty():
    """hotwords defaults to an empty list."""
    settings = DiarizationSettings()
    assert settings.hotwords == []


def test_hotwords_strips_whitespace(monkeypatch):
    """Whitespace around comma-separated hotwords is stripped."""
    monkeypatch.setenv("DIARIZATION_HOTWORDS", " sprint , deploy , LiveTranslate ")
    settings = DiarizationSettings()
    assert settings.hotwords == ["sprint", "deploy", "LiveTranslate"]


def test_hotwords_ignores_empty_segments(monkeypatch):
    """Empty segments (trailing/leading commas) are ignored."""
    monkeypatch.setenv("DIARIZATION_HOTWORDS", ",sprint,,deploy,")
    settings = DiarizationSettings()
    assert settings.hotwords == ["sprint", "deploy"]


def test_env_prefix_is_diarization(monkeypatch):
    """Settings only respond to DIARIZATION_* prefix, not unprefixed vars."""
    monkeypatch.setenv("ENABLED", "true")
    settings = DiarizationSettings()
    # ENABLED (no prefix) must not affect this settings class
    assert settings.enabled is False


def test_all_default_fields():
    """All field defaults match the spec exactly."""
    settings = DiarizationSettings()
    assert settings.min_confidence_auto_assign == 0.80
    assert settings.auto_enroll_speakers is True
    assert settings.fireflies_crossref_enabled is True


def test_has_vibevoice_url_whitespace_only():
    """has_vibevoice_url() returns False for whitespace-only URL."""
    settings = DiarizationSettings()
    settings.vibevoice_url = "   "
    assert settings.has_vibevoice_url() is False


def test_max_concurrent_jobs_override(monkeypatch):
    """DIARIZATION_MAX_CONCURRENT_JOBS env var is parsed as int."""
    monkeypatch.setenv("DIARIZATION_MAX_CONCURRENT_JOBS", "4")
    settings = DiarizationSettings()
    assert settings.max_concurrent_jobs == 4


def test_auto_apply_threshold_override(monkeypatch):
    """DIARIZATION_AUTO_APPLY_THRESHOLD env var is parsed as float."""
    monkeypatch.setenv("DIARIZATION_AUTO_APPLY_THRESHOLD", "0.90")
    settings = DiarizationSettings()
    assert settings.auto_apply_threshold == pytest.approx(0.90)
