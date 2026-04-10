"""
PIL Frame Rendering Validation — VirtualWebcamManager at 1280x720.

Tests that each canonical DisplayMode (SUBTITLE, SPLIT, INTERPRETER) and
CJK text render a valid numpy frame of the correct shape without error.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig, TranslationDisplay
from livetranslate_common.theme import DisplayMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(mode: DisplayMode) -> VirtualWebcamManager:
    config = WebcamConfig(width=1280, height=720, fps=30, display_mode=mode)
    return VirtualWebcamManager(config)


def _make_translation(text: str = "Hello world", speaker: str = "Alice") -> TranslationDisplay:
    return TranslationDisplay(
        translation_id="test-id-001",
        text=text,
        source_language="en",
        target_language="es",
        speaker_name=speaker,
        confidence=0.95,
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_renders_subtitle_mode():
    """SUBTITLE mode generates a 1280x720 RGB frame."""
    manager = _make_manager(DisplayMode.SUBTITLE)
    manager.current_translations.append(_make_translation())
    manager._generate_frame()

    assert manager.current_frame is not None
    assert manager.current_frame.shape == (720, 1280, 3)


@pytest.mark.integration
def test_renders_split_mode():
    """SPLIT mode generates a 1280x720 RGB frame."""
    manager = _make_manager(DisplayMode.SPLIT)
    manager.current_translations.append(_make_translation())
    manager._generate_frame()

    assert manager.current_frame is not None
    assert manager.current_frame.shape == (720, 1280, 3)


@pytest.mark.integration
def test_renders_interpreter_mode():
    """INTERPRETER mode generates a 1280x720 RGB frame."""
    manager = _make_manager(DisplayMode.INTERPRETER)
    manager.current_translations.append(_make_translation())
    manager._generate_frame()

    assert manager.current_frame is not None
    assert manager.current_frame.shape == (720, 1280, 3)


@pytest.mark.integration
def test_renders_cjk_text():
    """Chinese text and speaker name render without error at 1280x720."""
    manager = _make_manager(DisplayMode.SUBTITLE)
    manager.current_translations.append(
        _make_translation(text="你好世界", speaker="张三")
    )
    manager._generate_frame()

    assert manager.current_frame is not None
    assert manager.current_frame.shape == (720, 1280, 3)


@pytest.mark.integration
def test_renders_waiting_frame():
    """With no translations, the waiting-frame path still produces a valid frame."""
    manager = _make_manager(DisplayMode.SUBTITLE)
    # No translations added — current_translations is empty
    manager._generate_frame()

    assert manager.current_frame is not None
    assert manager.current_frame.shape == (720, 1280, 3)
