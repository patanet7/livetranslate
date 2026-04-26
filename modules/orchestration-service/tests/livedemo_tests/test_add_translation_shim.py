"""Tests for the add_translation() shim — every existing caller gets paired-block rendering for free.

`bot_integration.py` and friends already pass `original_text` in the dict (line
1257). Before this change, `add_translation` ignored that field and only
rendered the translation. After: it routes through `add_caption()` so both
fields land as a paired block (B3) without any caller-site changes.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig
from livetranslate_common.theme import DisplayMode


def _mgr(*, show_diarization_ids: bool = False) -> VirtualWebcamManager:
    cfg = WebcamConfig(
        width=1280, height=720, fps=30,
        display_mode=DisplayMode.SUBTITLE,
        show_diarization_ids=show_diarization_ids,
    )
    return VirtualWebcamManager(cfg)


def test_add_translation_with_original_text_renders_paired_block():
    """B3 — when caller dict has both original_text + translated_text, both must render."""
    mgr = _mgr()
    mgr.add_translation({
        "translation_id": "x1",
        "speaker_name": "Alice",
        "speaker_id": "SPEAKER_00",
        "original_text": "Hello world",
        "translated_text": "你好世界",
        "source_language": "en",
        "target_language": "zh",
        "translation_confidence": 0.92,
        "is_original_transcription": False,
    })
    # Caption stored. Inspect the TranslationDisplay's text + original_text fields.
    assert len(mgr.current_translations) == 1
    cap = mgr.current_translations[0]
    # Primary text = translation
    assert cap.text == "你好世界"
    # Secondary = original
    assert getattr(cap, "original_text", None) == "Hello world"
    # B4: speaker label drops diarization id by default
    assert cap.speaker_name == "Alice"


def test_add_translation_with_only_translated_text_back_compat():
    """Old call sites that only pass translated_text must still work."""
    mgr = _mgr()
    mgr.add_translation({
        "translation_id": "x2",
        "speaker_name": "Bob",
        "translated_text": "再见",
        "source_language": "en",
        "target_language": "zh",
    })
    assert len(mgr.current_translations) == 1
    cap = mgr.current_translations[0]
    assert cap.text == "再见"
    # No original known → secondary should be None
    assert getattr(cap, "original_text", None) is None
    assert cap.speaker_name == "Bob"


def test_add_translation_diarization_default_hidden():
    """B4 — speaker_id like SPEAKER_00 must not leak when show_diarization_ids=False."""
    mgr = _mgr(show_diarization_ids=False)
    mgr.add_translation({
        "translation_id": "x3",
        "speaker_name": "Charlie",
        "speaker_id": "SPEAKER_42",
        "translated_text": "hi",
        "source_language": "en",
        "target_language": "en",
    })
    cap = mgr.current_translations[0]
    assert "SPEAKER_" not in (cap.speaker_name or "")
    assert cap.speaker_name == "Charlie"


def test_add_translation_diarization_on_when_enabled():
    mgr = _mgr(show_diarization_ids=True)
    mgr.add_translation({
        "translation_id": "x4",
        "speaker_name": "Charlie",
        "speaker_id": "SPEAKER_42",
        "translated_text": "hi",
        "source_language": "en",
        "target_language": "en",
    })
    cap = mgr.current_translations[0]
    assert "SPEAKER_42" in (cap.speaker_name or "")
