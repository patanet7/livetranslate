import io
import os
import uuid

import httpx
import numpy as np
import pytest

VERIFY_LIVE = os.getenv("VERIFY_LIVE_SERVICES") == "1"

pytestmark = pytest.mark.skipif(
    not VERIFY_LIVE,
    reason="Set VERIFY_LIVE_SERVICES=1 to run live service contract checks",
)


def _make_wav_bytes(duration: float = 0.5, freq: float = 440.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.3 * np.sin(2 * np.pi * freq * t)
    samples = np.int16(waveform * 32767)
    buffer = io.BytesIO()
    import wave

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_whisper_live_endpoints():
    base_url = os.getenv("AUDIO_SERVICE_URL")
    assert base_url, "AUDIO_SERVICE_URL must be set"

    audio_bytes = _make_wav_bytes()
    session_id = f"contract-{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        health = await client.get(f"{base_url}/health")
        assert health.status_code == 200

        models = await client.get(f"{base_url}/api/models")
        assert models.status_code == 200

        device_info = await client.get(f"{base_url}/api/device-info")
        assert device_info.status_code == 200

        # Start realtime session
        start_resp = await client.post(
            f"{base_url}/api/realtime/start", json={"session_id": session_id}
        )
        assert start_resp.status_code == 200

        # Send audio chunk
        chunk_resp = await client.post(
            f"{base_url}/api/realtime/audio",
            data={"session_id": session_id},
            files={"audio": ("chunk.wav", audio_bytes, "audio/wav")},
        )
        assert chunk_resp.status_code == 200

        # Check status
        status_resp = await client.get(f"{base_url}/api/realtime/status/{session_id}")
        assert status_resp.status_code == 200

        # Stop session
        stop_resp = await client.post(
            f"{base_url}/api/realtime/stop", json={"session_id": session_id}
        )
        assert stop_resp.status_code == 200

        # Audio analysis
        analyze_resp = await client.post(
            f"{base_url}/api/analyze",
            files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
        )
        assert analyze_resp.status_code == 200

        # Pipeline processing
        pipeline_resp = await client.post(
            f"{base_url}/api/process-pipeline",
            data={"request_id": f"req-{session_id}"},
            files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
        )
        assert pipeline_resp.status_code == 200

        # Stream results (may be empty but should respond)
        results_resp = await client.get(
            f"{base_url}/api/stream-results/{session_id}", params={"limit": 5}
        )
        assert results_resp.status_code == 200

        stats_resp = await client.get(f"{base_url}/api/processing-stats")
        assert stats_resp.status_code == 200

        # Download (optional; accept 200 or 404)
        download_resp = await client.get(
            f"{base_url}/api/download/{session_id}", params={"format": "text"}
        )
        assert download_resp.status_code in {200, 404}


@pytest.mark.asyncio
async def test_translation_live_endpoints():
    base_url = os.getenv("TRANSLATION_SERVICE_URL")
    assert base_url, "TRANSLATION_SERVICE_URL must be set"

    async with httpx.AsyncClient(timeout=10.0) as client:
        health = await client.get(f"{base_url}/api/health")
        assert health.status_code == 200

        status = await client.get(f"{base_url}/api/status")
        assert status.status_code == 200

        device_info = await client.get(f"{base_url}/api/device-info")
        assert device_info.status_code == 200

        languages = await client.get(f"{base_url}/api/languages")
        assert languages.status_code == 200
        assert languages.json().get("languages"), "Expected language list"

        detect = await client.post(f"{base_url}/api/detect", json={"text": "hello"})
        assert detect.status_code == 200
        payload = detect.json()
        assert "language" in payload

        translate = await client.post(
            f"{base_url}/api/translate",
            json={"text": "hello", "target_language": "es"},
        )
        assert translate.status_code == 200
        assert "translated_text" in translate.json()

        perf = await client.get(f"{base_url}/api/performance")
        assert perf.status_code == 200
