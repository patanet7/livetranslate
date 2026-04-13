"""Tests for ScreenCaptureAudioSource.

Covers:
- is_available() returns True when binary is on PATH
- is_available() returns False when binary is missing
- start() fails and calls on_error when binary is missing
- stop() is idempotent
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — make src importable from the orchestration-service root
# ---------------------------------------------------------------------------

_ORCH_ROOT = Path(__file__).parent.parent.parent
_SRC_PATH = _ORCH_ROOT / "src"
for _p in [str(_ORCH_ROOT), str(_SRC_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.audio.screencapture_source import (  # noqa: E402
    CAPTURE_BINARY,
    ScreenCaptureAudioSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_audio(_chunk):  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_binary_exists(self, tmp_path):
        """is_available() returns True when shutil.which finds the binary."""
        fake_binary = tmp_path / CAPTURE_BINARY
        fake_binary.write_text("#!/bin/sh\n")
        fake_binary.chmod(0o755)

        with patch("shutil.which", return_value=str(fake_binary)):
            assert ScreenCaptureAudioSource.is_available() is True

    def test_returns_false_when_binary_missing(self):
        """is_available() returns False when shutil.which returns None."""
        with patch("shutil.which", return_value=None):
            assert ScreenCaptureAudioSource.is_available() is False


class TestStart:
    @pytest.mark.asyncio
    async def test_raises_and_calls_on_error_when_binary_missing(self):
        """start() raises RuntimeError and invokes on_error when binary absent."""
        errors: list[Exception] = []

        source = ScreenCaptureAudioSource(
            on_audio=_noop_audio,
            on_error=lambda exc: errors.append(exc),
        )

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match=CAPTURE_BINARY):
                await source.start()

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert CAPTURE_BINARY in str(errors[0])

    @pytest.mark.asyncio
    async def test_raises_without_on_error_when_binary_missing(self):
        """start() raises RuntimeError even when no on_error callback is set."""
        source = ScreenCaptureAudioSource(on_audio=_noop_audio)

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                await source.start()


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_is_idempotent_before_start(self):
        """stop() must not raise when called before start()."""
        source = ScreenCaptureAudioSource(on_audio=_noop_audio)
        # Should complete without error.
        await source.stop()
        await source.stop()

    @pytest.mark.asyncio
    async def test_stop_twice_after_failed_start_is_safe(self):
        """stop() called twice after a failed start does not raise."""
        errors: list[Exception] = []
        source = ScreenCaptureAudioSource(
            on_audio=_noop_audio,
            on_error=lambda exc: errors.append(exc),
        )

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                await source.start()

        # _running is still False because start() failed before setting it.
        await source.stop()
        await source.stop()
