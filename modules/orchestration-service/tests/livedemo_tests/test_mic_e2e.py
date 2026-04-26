"""Real-network behavioral E2E for MicSource.

Connects to the *running* orchestration WS (default ws://localhost:3000/api/audio/stream)
and pumps a known fixture WAV through it. Asserts ≥1 CaptionEvent comes back with
non-empty translation. Skipped if orchestration isn't reachable.

Run conditions:
- Orchestration running on LIVEDEMO_ORCHESTRATION_WS_URL (default ws://localhost:3000/api/audio/stream)
- Transcription service reachable from orchestration
- LLM endpoint reachable
- LIVEDEMO_E2E_FIXTURE_WAV env points to a 16kHz mono int16 WAV ≥3s long with English speech
"""
from __future__ import annotations

import os
import socket
import wave
from pathlib import Path

import pytest

from livedemo.sources.mic import MicSource


def _orchestration_reachable() -> bool:
    s = socket.socket()
    try:
        s.settimeout(0.5)
        s.connect(("127.0.0.1", 3000))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _fixture_wav() -> Path | None:
    p = os.environ.get("LIVEDEMO_E2E_FIXTURE_WAV")
    return Path(p) if p else None


_skip_reason = (
    None
    if (_orchestration_reachable() and _fixture_wav() and _fixture_wav().exists())
    else "orchestration not running on :3000 or LIVEDEMO_E2E_FIXTURE_WAV missing"
)


async def _wav_audio_provider(path: Path):
    """Yield 20ms (320-sample) int16 chunks from the fixture wav."""
    import asyncio

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000, "fixture must be 16kHz"
        assert wf.getnchannels() == 1, "fixture must be mono"
        assert wf.getsampwidth() == 2, "fixture must be int16"
        chunk_size = 320 * 2  # 320 samples × 2 bytes
        while True:
            data = wf.readframes(320)
            if not data:
                return
            yield data
            await asyncio.sleep(0.02)  # real-time pacing


@pytest.mark.e2e
@pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")
@pytest.mark.asyncio
async def test_mic_e2e_real_orchestration_yields_translation():
    ws_url = os.environ.get("LIVEDEMO_ORCHESTRATION_WS_URL", "ws://localhost:3000/api/audio/stream")
    src = MicSource(
        ws_url=ws_url,
        target_language="zh",
        source_language="en",
        audio_provider=lambda: _wav_audio_provider(_fixture_wav()),
    )
    out = []
    import asyncio
    try:
        async def collect():
            async for evt in src.stream():
                out.append(evt)
                if len(out) >= 1:
                    return
        await asyncio.wait_for(collect(), timeout=30.0)
    except asyncio.TimeoutError:
        pytest.fail(f"No CaptionEvent received within 30s (orchestration at {ws_url})")
    assert len(out) >= 1
    assert out[0].translated_text
    assert out[0].target_lang == "zh"
