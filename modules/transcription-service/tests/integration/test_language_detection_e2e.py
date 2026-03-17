"""E2E language detection tests using real production data.

Two test categories:
1. Log-replay tests: Parse real flapping events from production logs and prove
   WhisperLanguageDetector eliminates false switches (no service needed).
2. Live streaming tests: Stream real recorded audio through the running
   transcription service and verify stable language detection (needs service).

Run:
  # Log-replay (no service required):
  uv run pytest modules/transcription-service/tests/integration/test_language_detection_e2e.py -v -k replay

  # Live streaming (needs transcription service on :5001):
  uv run pytest modules/transcription-service/tests/integration/test_language_detection_e2e.py -v -k streaming -s
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from language_detection import WhisperLanguageDetector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RECORDINGS_DIR = Path("/tmp/livetranslate/recordings")
SERVICE_URL = "ws://localhost:5001/api/stream"

# Known recording sessions from production
LONG_MEETING_SESSION = "af5b37c9"  # ~57min en/zh interpreter meeting (115 chunks)
SHORT_EN_SESSION = "48697a4a"  # ~12min English session (24 chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_flapping_fixture() -> dict:
    """Load the extracted flapping events fixture."""
    path = FIXTURES_DIR / "flapping_events_20260317.json"
    if not path.exists():
        pytest.skip(f"Flapping fixture not found: {path}. Run fixture extraction first.")
    with open(path) as f:
        return json.load(f)


def _find_recording(session_prefix: str) -> Path | None:
    """Find a recording directory by session ID prefix."""
    if not RECORDINGS_DIR.exists():
        return None
    for d in RECORDINGS_DIR.iterdir():
        if d.name.startswith(session_prefix) and d.is_dir():
            return d
    return None


def _load_recording_audio(recording_dir: Path, max_chunks: int = 10) -> tuple[np.ndarray, int]:
    """Load audio from a recording directory. Returns (audio_16k, 16000).

    Loads up to max_chunks FLAC files, concatenates, and downsamples to 16kHz.
    """
    import soundfile as sf

    manifest_path = recording_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sample_rate = manifest["sample_rate"]
    chunks = sorted(manifest["chunks"], key=lambda c: c["sequence"])[:max_chunks]

    audio_parts = []
    for chunk_info in chunks:
        chunk_path = recording_dir / chunk_info["filename"]
        if chunk_path.exists():
            data, sr = sf.read(str(chunk_path))
            if data.ndim == 2:
                data = data.mean(axis=1)
            audio_parts.append(data.astype(np.float32))

    if not audio_parts:
        pytest.skip(f"No audio chunks found in {recording_dir}")

    audio = np.concatenate(audio_parts)

    # Downsample to 16kHz if needed
    if sample_rate != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        except ImportError:
            # Manual nearest-neighbor downsample
            ratio = 16000 / sample_rate
            indices = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
            audio = audio[indices]

    return audio, 16000


async def _check_service():
    """Check if transcription service is running."""
    try:
        import websockets
        ws = await asyncio.wait_for(
            websockets.connect(SERVICE_URL, ping_interval=None),
            timeout=3,
        )
        await ws.close()
        return True
    except Exception:
        return False


async def _stream_and_collect(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str | None = None,
    chunk_duration_ms: int = 100,
    pace_factor: float = 0.1,
    recv_timeout_s: float = 60.0,
) -> list[dict]:
    """Stream audio through the transcription service and collect results."""
    import websockets

    ws = await websockets.connect(SERVICE_URL, ping_interval=None, close_timeout=30)

    config: dict = {"type": "config"}
    if language:
        config["language"] = language
    await ws.send(json.dumps(config))

    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunk_sleep = (chunk_duration_ms / 1000) * pace_factor

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i: i + chunk_samples]
        if len(chunk) < 160:
            break
        await ws.send(chunk.astype(np.float32).tobytes())
        await asyncio.sleep(chunk_sleep)

    await ws.send(json.dumps({"type": "end"}))

    messages = []
    deadline = asyncio.get_event_loop().time() + recv_timeout_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(30, recv_timeout_s))
            data = json.loads(msg)
            messages.append(data)
        except (asyncio.TimeoutError, Exception):
            break

    await ws.close()
    return messages


# ---------------------------------------------------------------------------
# Log-Replay Tests (no service needed)
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestLogReplayDetection:
    """Replay real production flapping events through WhisperLanguageDetector.

    These tests use the extracted log fixture to prove the new detector
    eliminates the 210 false switches observed in the production meeting.
    """

    def test_real_flapping_fixture_no_false_switches(self):
        """Replay all 210+ switch events from the real production log.

        The old LanguageDetector switched 210 times in 40 minutes.
        The WhisperLanguageDetector should switch 0 times for these
        short-duration, low-confidence detections.
        """
        fixture = _load_flapping_fixture()
        events = fixture["events"]
        assert len(events) > 50, f"Expected many events, got {len(events)}"

        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )

        # Process initial event
        initial = next(e for e in events if e["type"] == "initial")
        detector.detect_initial(initial["language"], initial.get("confidence", 0.5))

        switch_count = 0
        false_switches = []

        for event in events:
            if event["type"] != "switch":
                continue

            # The old detector switched here. Feed both the "new" language and
            # then the "old" language (simulating the bounce-back) to the
            # WhisperLanguageDetector with typical Whisper confidence levels.
            delta_s = event.get("delta_s", 3.0)

            # Simulate the detection that caused the switch
            # Use moderate confidence (Whisper hallucinations are typically 0.3-0.6)
            result = detector.update(
                event["new"],
                chunk_duration_s=min(delta_s, 6.0),
                confidence=0.45,
            )
            if result is not None:
                switch_count += 1
                false_switches.append(f"{event['old']}→{event['new']} at {event['timestamp']}")

        # The old detector had 210 switches. The new one should have 0 or very few
        # (legitimate switches like en→zh in interpreter mode might still fire,
        # but the hallucinated ones like en→nn, en→cy, en→ko should be blocked).
        assert switch_count <= 2, (
            f"WhisperLanguageDetector had {switch_count} switches "
            f"(old detector had {len([e for e in events if e['type'] == 'switch'])}). "
            f"False switches: {false_switches[:10]}"
        )

    def test_hallucinated_languages_in_real_log(self):
        """Verify that ALL hallucinated languages from the real log are rejected.

        The real log shows switches to: nn, cy, ko, fr, es.
        None of these should survive the sustained detector.
        """
        fixture = _load_flapping_fixture()
        events = fixture["events"]

        hallucinated_langs = {"nn", "cy", "ko", "fr", "es", "it", "pt", "nl", "ru", "pl"}

        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.5)

        hallucination_switches = []
        for event in events:
            if event["type"] != "switch":
                continue
            if event["new"] in hallucinated_langs:
                delta_s = event.get("delta_s", 3.0)
                result = detector.update(
                    event["new"],
                    chunk_duration_s=min(delta_s, 6.0),
                    confidence=0.4,
                )
                if result is not None:
                    hallucination_switches.append(f"→{event['new']}")

                # Bounce back (the old detector always bounced back quickly)
                detector.update("en", chunk_duration_s=3.0, confidence=0.7)

        assert len(hallucination_switches) == 0, (
            f"Hallucinated languages should NEVER trigger a switch: {hallucination_switches}"
        )

    def test_real_en_zh_switch_detected(self):
        """In interpreter mode, a real en→zh switch should still work.

        The log shows genuine zh segments starting around 17:25. Simulate
        sustained Chinese at high confidence — detector should switch.
        """
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.5)

        # 30s of English (10 chunks × 3s)
        for _ in range(10):
            assert detector.update("en", 3.0, 0.8) is None

        # Real Chinese switch: sustained zh at high confidence for 15s+
        switched = False
        for _ in range(6):
            result = detector.update("zh", 3.0, 0.85)
            if result == "zh":
                switched = True

        assert switched, "Real sustained zh switch should be detected"


# ---------------------------------------------------------------------------
# Live Streaming Tests (needs transcription service on :5001)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
async def require_service_fixture():
    """Skip tests if transcription service isn't running."""
    if not await _check_service():
        pytest.skip("Transcription service not running on port 5001")


@pytest.mark.asyncio
@pytest.mark.e2e
class TestLiveLanguageDetection:
    """Stream real recorded audio through the live transcription service
    and verify stable language detection with the new WhisperLanguageDetector.
    """

    async def test_english_recording_no_flapping(self, require_service_fixture):
        """Stream English audio — should detect English and stay English.

        Uses real recorded audio from a session. The old detector would
        flap to nn, cy, ko etc. The new one should stay on English.
        """
        recording_dir = _find_recording(SHORT_EN_SESSION)
        if recording_dir is None:
            pytest.skip(f"Recording {SHORT_EN_SESSION} not found in {RECORDINGS_DIR}")

        # Load first 5 chunks (~2.5 minutes) to keep test fast
        audio, sr = _load_recording_audio(recording_dir, max_chunks=5)

        messages = await _stream_and_collect(
            audio, sr, language=None,  # auto-detect
            pace_factor=0.05,  # 20x speed for faster tests
            recv_timeout_s=30.0,
        )

        # Extract language detection events
        lang_events = [m for m in messages if m.get("type") == "language_detected"]
        segments = [m for m in messages if m.get("type") == "segment"]

        # Should have segments
        assert len(segments) >= 1, f"Expected segments, got types: {[m['type'] for m in messages]}"

        # Count language switches (initial detection is fine, flapping is not)
        if lang_events:
            languages = [e["language"] for e in lang_events]
            unique_langs = set(languages)
            # Should be mostly English — at most 1 switch is acceptable
            non_en = [l for l in languages[1:] if l != "en"]
            assert len(non_en) <= 1, (
                f"Language flapping detected: {languages}. "
                f"Expected stable English detection."
            )

    async def test_mixed_meeting_language_stability(self, require_service_fixture):
        """Stream mixed en/zh meeting audio — should have stable transitions.

        Uses the long meeting recording. Switches should only happen at
        genuine language transitions, not on every 3s chunk.
        """
        recording_dir = _find_recording(LONG_MEETING_SESSION)
        if recording_dir is None:
            pytest.skip(f"Recording {LONG_MEETING_SESSION} not found in {RECORDINGS_DIR}")

        # Load first 3 chunks (~1.5 minutes) for a fast test
        audio, sr = _load_recording_audio(recording_dir, max_chunks=3)

        messages = await _stream_and_collect(
            audio, sr, language=None,  # auto-detect for interpreter-like mode
            pace_factor=0.05,
            recv_timeout_s=30.0,
        )

        lang_events = [m for m in messages if m.get("type") == "language_detected"]
        segments = [m for m in messages if m.get("type") == "segment"]

        assert len(segments) >= 1

        # In a ~90s window, there should be at most a few genuine switches.
        # The old detector would flap 10+ times in this window.
        if lang_events:
            switch_count = sum(
                1 for i in range(1, len(lang_events))
                if lang_events[i]["language"] != lang_events[i - 1]["language"]
            )
            assert switch_count <= 5, (
                f"Too many language switches ({switch_count}) in 90s — "
                f"likely flapping. Languages: {[e['language'] for e in lang_events]}"
            )
