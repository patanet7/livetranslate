"""Smoke tests for FakeWhisperServer.

Mirrors test_fakes_smoke.py for the new Whisper fake — verifies the multipart
upload path captures form fields + audio bytes, plus auth + failure knobs.
"""

from __future__ import annotations

import io
import wave

import httpx
import pytest


def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a tiny silent WAV file in-memory."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


@pytest.mark.integration
@pytest.mark.asyncio
class TestFakeWhisperServerSmoke:
    async def test_health_endpoint(self, fake_whisper_server) -> None:
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    async def test_models_endpoint(self, fake_whisper_server) -> None:
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert any(
                m["id"] == "mlx-community/whisper-large-v3-turbo" for m in data["data"]
            )

    async def test_transcribe_default_verbose_json(
        self, fake_whisper_server
    ) -> None:
        fake_whisper_server.set_response_text("hello world")
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            resp = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={
                    "model": "mlx-community/whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                    "language": "en",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"
        assert isinstance(body["segments"], list)
        assert body["segments"][0]["text"] == "hello world"

    async def test_form_fields_recorded(self, fake_whisper_server) -> None:
        """The fake captures form fields so tests can assert request shape."""
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={
                    "model": "mlx-community/whisper-medium",
                    "language": "zh",
                    "prompt": "上一段。",
                    "temperature": "0.0",
                    "response_format": "verbose_json",
                },
            )
        recorded = fake_whisper_server.recorded_requests[-1]
        assert recorded["form_fields"]["model"] == "mlx-community/whisper-medium"
        assert recorded["form_fields"]["language"] == "zh"
        assert recorded["form_fields"]["prompt"] == "上一段。"
        assert recorded["form_fields"]["response_format"] == "verbose_json"
        assert recorded["audio_size"] == len(wav)

    async def test_require_api_key_rejects_missing(
        self, fake_whisper_server
    ) -> None:
        fake_whisper_server.require_api_key("secret123")
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            # No Authorization header
            resp = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": "m"},
            )
        assert resp.status_code == 401

    async def test_require_api_key_accepts_correct(
        self, fake_whisper_server
    ) -> None:
        fake_whisper_server.require_api_key("secret123")
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(
            base_url=fake_whisper_server.base_url,
            headers={"Authorization": "Bearer secret123"},
        ) as client:
            resp = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": "m"},
            )
        assert resp.status_code == 200

    async def test_fail_n_times(self, fake_whisper_server) -> None:
        fake_whisper_server.fail_n_times(2, status=503)
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            r1 = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", wav, "audio/wav")},
                data={"model": "m"},
            )
            r2 = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", wav, "audio/wav")},
                data={"model": "m"},
            )
            r3 = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", wav, "audio/wav")},
                data={"model": "m"},
            )
        assert r1.status_code == 503
        assert r2.status_code == 503
        assert r3.status_code == 200

    async def test_custom_segments(self, fake_whisper_server) -> None:
        fake_whisper_server.set_segments(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "first ",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.02,
                    "compression_ratio": 1.2,
                },
                {
                    "id": 1,
                    "start": 2.0,
                    "end": 4.0,
                    "text": "second.",
                    "avg_logprob": -0.15,
                    "no_speech_prob": 0.03,
                    "compression_ratio": 1.3,
                },
            ]
        )
        wav = _make_wav_bytes()
        async with httpx.AsyncClient(base_url=fake_whisper_server.base_url) as client:
            resp = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", wav, "audio/wav")},
                data={"model": "m", "response_format": "verbose_json"},
            )
        body = resp.json()
        assert len(body["segments"]) == 2
        assert body["segments"][1]["text"] == "second."
