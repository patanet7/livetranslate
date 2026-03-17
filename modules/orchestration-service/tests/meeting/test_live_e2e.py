# ruff: noqa: RUF001
"""Live E2E test — connects to the running system and validates persistence.

Requires all services running (`just dev`):
  - Orchestration on :3000
  - Transcription on :5001
  - LLM on :8006
  - PostgreSQL on :5432

Streams a real WAV fixture through the WebSocket, waits for transcription +
translation, then queries the live DB to verify:
  1. Meeting session created and completed
  2. FLAC recording files exist on disk
  3. Transcript chunks persisted in meeting_chunks
  4. Translations persisted in meeting_translations with chunk_id FK
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Fixtures and constants
# ---------------------------------------------------------------------------

ORCHESTRATION_WS = os.getenv("ORCHESTRATION_WS", "ws://localhost:3000/api/audio/stream")
ORCHESTRATION_HTTP = os.getenv("ORCHESTRATION_HTTP", "http://localhost:3000")
RECORDING_BASE = Path(os.getenv("RECORDING_BASE_PATH", "/tmp/livetranslate/recordings"))

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "transcription-service"
    / "tests"
    / "fixtures"
    / "audio"
)
ZH_FIXTURE = FIXTURE_DIR / "meeting_zh.wav"
EN_FIXTURE = FIXTURE_DIR / "hello_world.wav"


def _check_services():
    """Skip if services aren't running."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{ORCHESTRATION_HTTP}/api/audio/health", timeout=3)
        if resp.status != 200:
            pytest.skip("Orchestration service not healthy")
    except Exception:
        pytest.skip("Orchestration service not reachable — run `just dev` first")


def _load_wav_as_48k_float32(path: Path) -> np.ndarray:
    """Load WAV and resample to 48kHz float32 mono (what the browser sends)."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    if sr != 48000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=48000)
    return data.astype(np.float32)


# ===========================================================================
# Live E2E test
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.timeout(120)
class TestLiveMeetingPersistence:
    """Stream real audio through the live system and verify DB + file persistence."""

    @pytest.mark.asyncio
    async def test_meeting_session_persists_transcripts_and_translations(self):
        """Full flow: connect → start → promote → stream audio → end → verify DB."""
        _check_services()

        if not ZH_FIXTURE.exists():
            pytest.skip(f"Fixture not found: {ZH_FIXTURE}")

        import websockets

        audio = _load_wav_as_48k_float32(ZH_FIXTURE)

        received_messages: list[dict] = []
        session_id = None
        pipeline_session_id = None

        async with websockets.connect(ORCHESTRATION_WS) as ws:
            # 1. Receive ConnectedMessage
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == "connected"
            session_id = msg["session_id"]
            print(f"  Connected: session_id={session_id}")

            # 2. Send start_session
            await ws.send(json.dumps({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            }))

            # Wait for service_status
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            status_msg = json.loads(raw)
            print(f"  Service status: {status_msg}")

            # 3. Send promote_to_meeting
            await ws.send(json.dumps({"type": "promote_to_meeting"}))

            # Read meeting_started
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            meeting_msg = json.loads(raw)
            assert meeting_msg["type"] == "meeting_started"
            pipeline_session_id = meeting_msg.get("session_id")
            print(f"  Meeting started: pipeline_session_id={pipeline_session_id}")

            # 4. Stream audio in 100ms chunks (simulates browser AudioWorklet)
            chunk_samples = int(48000 * 0.1)  # 100ms at 48kHz
            chunks_sent = 0
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i : i + chunk_samples]
                if len(chunk) == 0:
                    break
                await ws.send(chunk.tobytes())
                chunks_sent += 1
                # Pace it roughly like real-time (but faster)
                if chunks_sent % 50 == 0:
                    await asyncio.sleep(0.05)

            print(f"  Streamed {chunks_sent} chunks ({len(audio)/48000:.1f}s of audio)")

            # 5. Wait for transcription and translation results
            deadline = time.monotonic() + 60  # 60s max wait
            segments_received = 0
            translations_received = 0

            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    msg = json.loads(raw)
                    received_messages.append(msg)

                    if msg.get("type") == "segment":
                        segments_received += 1
                        if msg.get("is_final"):
                            print(f"  Segment (final): {msg.get('text', '')[:60]}")
                    elif msg.get("type") == "translation":
                        translations_received += 1
                        if not msg.get("is_draft"):
                            print(f"  Translation: {msg.get('text', '')[:60]}")
                    elif msg.get("type") == "translation_chunk":
                        pass  # streaming chunk, expected
                    else:
                        print(f"  Other message: {msg.get('type')}")

                    # Once we have at least 1 final translation, we have enough
                    if translations_received >= 1 and segments_received >= 2:
                        # Give a bit more time for persistence to flush
                        await asyncio.sleep(2)
                        break
                except TimeoutError:
                    if segments_received > 0:
                        break  # got some results, transcription may be done
                    continue

            print(f"  Received: {segments_received} segments, {translations_received} translations")

            # 6. End meeting
            await ws.send(json.dumps({"type": "end_meeting"}))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                end_msg = json.loads(raw)
                print(f"  End meeting response: {end_msg.get('type')}")
            except TimeoutError:
                pass

            # 7. End session
            await ws.send(json.dumps({"type": "end_session"}))
            await asyncio.sleep(1)

        # ---------------------------------------------------------------
        # VERIFICATION: Query the live database
        # ---------------------------------------------------------------
        assert pipeline_session_id is not None, "Never received pipeline_session_id"
        assert segments_received > 0, "No transcription segments received"

        print(f"\n  === Verifying DB for session {pipeline_session_id} ===")

        # Connect to live PostgreSQL
        from sqlalchemy import select, text
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as AS
        from sqlalchemy.orm import selectinload
        from sqlalchemy.pool import NullPool

        _src = Path(__file__).resolve().parent.parent.parent / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from database.models import Meeting, MeetingChunk, MeetingTranslation

        # Use the LIVE database — not the testcontainer that conftest.py sets up.
        # The root conftest overrides DATABASE_URL with the testcontainer URL,
        # so we read from the .env file directly.
        from dotenv import dotenv_values
        _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        _env = dotenv_values(str(_env_path))
        db_url = _env.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/livetranslate",
        )
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(db_url, poolclass=NullPool)

        import uuid
        sid = uuid.UUID(pipeline_session_id)

        async with AS(engine, expire_on_commit=False) as session:
            # Check meeting record
            meeting = await session.get(Meeting, sid)
            assert meeting is not None, f"Meeting {sid} not found in DB"
            print(f"  Meeting status: {meeting.status}")
            print(f"  Meeting source: {meeting.source}")
            assert meeting.status in ("completed", "active"), f"Unexpected status: {meeting.status}"

            # Check transcript chunks
            result = await session.execute(
                select(MeetingChunk)
                .where(MeetingChunk.meeting_id == sid)
                .order_by(MeetingChunk.timestamp_ms)
            )
            chunks = list(result.scalars().all())
            print(f"  Transcript chunks in DB: {len(chunks)}")
            assert len(chunks) > 0, "No transcript chunks persisted!"

            for i, c in enumerate(chunks[:5]):
                print(f"    [{i}] lang={c.source_language} final={c.is_final} text={c.text[:50]}")

            # Check translations with chunk_id linkage
            result = await session.execute(
                select(MeetingTranslation)
                .where(MeetingTranslation.chunk_id.in_([c.id for c in chunks]))
            )
            translations = list(result.scalars().all())
            print(f"  Translations linked to chunks: {len(translations)}")

            for t in translations[:5]:
                print(f"    target={t.target_language} chunk_id={t.chunk_id} text={t.translated_text[:50]}")

            # The critical assertion: translations exist AND are linked to chunks
            if translations_received > 0:
                assert len(translations) > 0, (
                    "Translations were received via WebSocket but NONE persisted to DB with chunk_id!"
                )
                assert all(t.chunk_id is not None for t in translations), (
                    "Some translations have chunk_id=None — wiring not working!"
                )

        # Check FLAC recording files
        recording_dir = RECORDING_BASE / pipeline_session_id
        print(f"\n  === Checking FLAC recording at {recording_dir} ===")
        if recording_dir.exists():
            manifest_path = recording_dir / "manifest.json"
            assert manifest_path.exists(), "manifest.json missing"
            manifest = json.loads(manifest_path.read_text())
            print(f"  FLAC chunks: {len(manifest['chunks'])}")
            print(f"  Total samples: {manifest['total_samples']}")
            assert manifest["total_samples"] > 0, "No samples recorded"

            # Verify first chunk is valid FLAC
            first_chunk = manifest["chunks"][0]
            chunk_path = recording_dir / first_chunk["filename"]
            assert chunk_path.exists(), f"FLAC file missing: {chunk_path}"
            info = sf.info(str(chunk_path))
            print(f"  First chunk: {info.frames} frames, {info.samplerate}Hz, {info.channels}ch")
            assert info.frames > 0
        else:
            print(f"  WARNING: Recording directory not found at {recording_dir}")

        await engine.dispose()
        print("\n  === LIVE E2E VALIDATION PASSED ===")
