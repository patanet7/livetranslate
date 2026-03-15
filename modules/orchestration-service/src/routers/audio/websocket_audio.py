"""WebSocket audio streaming endpoint.

Handles the full lifecycle:
  1. Frontend connects → ConnectedMessage
  2. start_session → create MeetingPipeline + connect to transcription service
  3. Binary audio → downsample via pipeline → forward to transcription service
  4. Transcription results → SegmentMessage → forward to frontend
  5. Final segments → translation via TranslationService → TranslationMessage
  6. promote_to_meeting / end_meeting → MeetingPipeline lifecycle
  7. end_session → cleanup

Dual-pipe compatible: the WebSocketTranscriptionClient used here can also be
instantiated by the Fireflies/Google Meet adapters for their audio flow.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from livetranslate_common.logging import get_logger
from livetranslate_common.models.ws_messages import (
    ConnectedMessage,
    EndMeetingMessage,
    EndSessionMessage,
    LanguageDetectedMessage,
    MeetingStartedMessage,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranslationMessage,
    parse_ws_message,
)

from clients.transcription_client import WebSocketTranscriptionClient
from meeting.downsampler import downsample_to_16k

logger = get_logger()

router = APIRouter()

# Module-level session tracking (for health check and observability)
_active_sessions: dict[str, dict] = {}

# Configurable via environment
TRANSCRIPTION_HOST = os.getenv("TRANSCRIPTION_HOST", "localhost")
TRANSCRIPTION_PORT = int(os.getenv("TRANSCRIPTION_PORT", "5001"))
RECORDING_BASE_PATH = Path(os.getenv("RECORDING_BASE_PATH", "/tmp/livetranslate/recordings"))
TARGET_LANGUAGE = os.getenv("DEFAULT_TARGET_LANGUAGE", "en")
DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "48000"))
DEFAULT_CHANNELS = int(os.getenv("DEFAULT_CHANNELS", "1"))


async def _try_create_pipeline(
    sample_rate: int,
    channels: int,
) -> "MeetingPipeline | None":
    """Create a MeetingPipeline if database is available, else return None.

    Uses the shared DatabaseManager singleton (connection-pooled) instead of
    creating a per-connection engine.  When the database isn't configured
    (e.g. unit tests, dev without postgres), audio still flows — just without
    session persistence or recording.
    """
    try:
        from dependencies import get_database_manager
        from meeting.pipeline import MeetingPipeline
        from meeting.session_manager import MeetingSessionManager

        db_manager = get_database_manager()
        session = await db_manager.get_session_direct()

        session_mgr = MeetingSessionManager(
            db=session,
            recording_base_path=RECORDING_BASE_PATH,
        )
        pipeline = MeetingPipeline(
            session_manager=session_mgr,
            recording_base_path=RECORDING_BASE_PATH,
            source_type="loopback",
            sample_rate=sample_rate,
            channels=channels,
        )
        # Attach session for cleanup
        pipeline._db_session = session
        return pipeline
    except Exception as exc:
        logger.debug("pipeline_creation_skipped", reason=str(exc))
        return None


@router.websocket("/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """WebSocket endpoint for frontend audio streaming.

    Endpoint: ws://orchestration:3000/api/audio/stream

    Binary frames:  raw float32 PCM audio (48 kHz from browser)
    Text frames:    JSON messages (start_session, end_session, promote_to_meeting, etc.)
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info("ws_audio_connected", session_id=session_id)

    # Track this session for health/observability
    _active_sessions[session_id] = {
        "connected_at": datetime.now(UTC).isoformat(),
        "source": "loopback",
        "is_meeting": False,
    }

    # Send ConnectedMessage immediately
    await websocket.send_text(
        ConnectedMessage(session_id=session_id).model_dump_json()
    )

    pipeline = None
    transcription_client = None
    translation_service = None
    session_tasks: set[asyncio.Task] = set()
    segment_counter = 0
    sample_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    source_language: str | None = None
    target_language: str | None = TARGET_LANGUAGE

    async def handle_transcription_segment(data: dict) -> None:
        """Transform transcription result → SegmentMessage → forward to frontend."""
        nonlocal segment_counter, source_language
        segment_counter += 1

        msg = SegmentMessage(
            segment_id=segment_counter,
            text=data.get("text", ""),
            language=data.get("language", ""),
            confidence=data.get("confidence", 0.0),
            stable_text=data.get("stable_text", ""),
            unstable_text=data.get("unstable_text", ""),
            is_final=data.get("is_final", False),
            speaker_id=data.get("speaker_id"),
            start_ms=data.get("start_ms"),
            end_ms=data.get("end_ms"),
        )

        # Track source language before send (don't lose it if send fails)
        if msg.language:
            source_language = msg.language

        try:
            await websocket.send_text(msg.model_dump_json())
        except Exception:
            return  # frontend disconnected

        # Trigger translation for final segments
        if msg.is_final and translation_service and target_language and source_language != target_language:
            task = asyncio.create_task(
                _translate_and_send(
                    websocket,
                    translation_service,
                    segment_id=msg.segment_id,
                    text=msg.text,
                    source_lang=msg.language,
                    target_lang=target_language,
                    speaker_name=msg.speaker_id,
                )
            )
            session_tasks.add(task)
            task.add_done_callback(session_tasks.discard)

    async def handle_language_detected(data: dict) -> None:
        """Forward language_detected to frontend."""
        nonlocal source_language
        source_language = data.get("language", source_language)
        try:
            await websocket.send_text(
                LanguageDetectedMessage(
                    language=data.get("language", ""),
                    confidence=data.get("confidence", 0.0),
                ).model_dump_json()
            )
        except Exception:
            pass

    async def handle_transcription_error(data: dict) -> None:
        """Forward errors from transcription service to frontend."""
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": data.get("message", "Transcription error"),
                "recoverable": data.get("recoverable", True),
            }))
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive()

            # Binary frame = audio data
            if "bytes" in data and data["bytes"]:
                if transcription_client and transcription_client.connected:
                    audio = np.frombuffer(data["bytes"], dtype=np.float32)

                    if pipeline:
                        downsampled = await pipeline.process_audio(audio)
                    else:
                        # Fallback: downsample directly without pipeline
                        downsampled = downsample_to_16k(
                            audio, source_rate=sample_rate, channels=channels
                        )

                    if len(downsampled) > 0:
                        try:
                            await transcription_client.send_audio(downsampled.tobytes())
                        except Exception as exc:
                            logger.warning("audio_forward_failed", error=str(exc))
                continue

            # Text frame = JSON control message
            if "text" not in data or not data["text"]:
                continue

            raw_text = data["text"]
            msg = parse_ws_message(raw_text)
            if msg is None:
                # Handle unregistered message types (ping, etc.)
                try:
                    raw = json.loads(raw_text)
                    msg_type = raw.get("type", "")
                    if msg_type == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }))
                    else:
                        logger.warning("unknown_ws_message", type=msg_type)
                except json.JSONDecodeError:
                    pass
                continue

            # --- start_session ---
            if isinstance(msg, StartSessionMessage):
                # Tear down previous session if start_session sent twice (reconnect)
                if transcription_client:
                    try:
                        await transcription_client.send_end()
                        await transcription_client.close()
                    except Exception:
                        pass
                    transcription_client = None
                if pipeline:
                    try:
                        await pipeline.end()
                    except Exception:
                        pass
                    pipeline = None

                sample_rate = msg.sample_rate
                channels = msg.channels

                # Create MeetingPipeline (optional — works without DB)
                pipeline = await _try_create_pipeline(sample_rate, channels)
                if pipeline:
                    try:
                        await pipeline.start()
                        logger.info(
                            "pipeline_started",
                            session_id=session_id,
                            pipeline_session=str(pipeline.session_id),
                        )
                    except Exception as exc:
                        logger.warning("pipeline_start_failed", error=str(exc)[:200])
                        pipeline = None  # fall back to pipeline-less mode

                # Connect to transcription service
                transcription_client = WebSocketTranscriptionClient(
                    host=TRANSCRIPTION_HOST,
                    port=TRANSCRIPTION_PORT,
                )
                transcription_client.on_segment(handle_transcription_segment)
                transcription_client.on_language_detected(handle_language_detected)
                transcription_client.on_error(handle_transcription_error)

                try:
                    await transcription_client.connect()

                    # Optionally init translation service + warm up LLM
                    translation_service = _try_create_translation_service()
                    if translation_service:
                        asyncio.create_task(_warm_up_llm(translation_service))

                    # Send service status
                    await websocket.send_text(
                        ServiceStatusMessage(
                            transcription="up",
                            translation="up" if translation_service else "down",
                        ).model_dump_json()
                    )
                except Exception as exc:
                    logger.error("transcription_connect_failed", error=str(exc))
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Cannot reach transcription service: {exc}",
                        "recoverable": False,
                    }))
                    await websocket.send_text(
                        ServiceStatusMessage(
                            transcription="down",
                            translation="down",
                        ).model_dump_json()
                    )

            # --- promote_to_meeting ---
            elif isinstance(msg, PromoteToMeetingMessage):
                if pipeline:
                    try:
                        await pipeline.promote_to_meeting()
                        _active_sessions.get(session_id, {})["is_meeting"] = True
                        await websocket.send_text(
                            MeetingStartedMessage(
                                session_id=str(pipeline.session_id),
                                started_at=datetime.now(UTC).isoformat(),
                            ).model_dump_json()
                        )
                        await websocket.send_text(
                            RecordingStatusMessage(
                                recording=True,
                                chunks_written=0,
                            ).model_dump_json()
                        )
                    except Exception as exc:
                        logger.error("promote_failed", error=str(exc))
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Failed to promote to meeting: {exc}",
                            "recoverable": True,
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Meeting pipeline not available (no database configured)",
                        "recoverable": False,
                    }))

            # --- end_meeting ---
            elif isinstance(msg, EndMeetingMessage):
                if pipeline and pipeline.is_meeting:
                    await pipeline.end()
                    await websocket.send_text(
                        RecordingStatusMessage(
                            recording=False,
                            chunks_written=0,
                        ).model_dump_json()
                    )

            # --- end_session ---
            elif isinstance(msg, EndSessionMessage):
                # Signal end to transcription service — it will flush remaining
                # audio and close the connection. Wait for the receive loop to
                # finish so the final segment callback fires before we exit.
                if transcription_client and transcription_client.connected:
                    await transcription_client.send_end()
                    if transcription_client._receive_task:
                        try:
                            await asyncio.wait_for(
                                transcription_client._receive_task, timeout=15
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                break

    except WebSocketDisconnect:
        logger.info("ws_audio_disconnected", session_id=session_id)

    except RuntimeError as exc:
        # Starlette raises RuntimeError when calling receive() after disconnect
        if "disconnect" in str(exc).lower():
            logger.info("ws_audio_disconnected", session_id=session_id)
        else:
            logger.exception("ws_audio_error", session_id=session_id, error=str(exc))

    except Exception as exc:
        logger.exception("ws_audio_error", session_id=session_id, error=str(exc))

    finally:
        # Wait for in-flight translations to complete (up to 30s) before cleanup.
        # end_session means "flush everything" — don't discard pending translations.
        if session_tasks:
            done, pending = await asyncio.wait(session_tasks, timeout=30.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # Cleanup: end pipeline and close transcription client
        if transcription_client:
            try:
                await transcription_client.send_end()
                await transcription_client.close()
            except Exception:
                pass

        if pipeline:
            try:
                await pipeline.end()
            except Exception:
                pass
            # Close the DB session obtained from the shared pool
            db_session = getattr(pipeline, "_db_session", None)
            if db_session:
                try:
                    await db_session.close()
                except Exception:
                    pass

        if translation_service:
            try:
                await translation_service.close()
            except Exception:
                pass

        _active_sessions.pop(session_id, None)
        logger.info("ws_audio_cleaned_up", session_id=session_id)


def _try_create_translation_service() -> "TranslationService | None":
    """Create a TranslationService if LLM config is available."""
    try:
        from translation.config import TranslationConfig
        from translation.service import TranslationService

        config = TranslationConfig.from_env()
        return TranslationService(config)
    except Exception as exc:
        logger.debug("translation_service_unavailable", reason=str(exc))
        return None


async def _translate_and_send(
    websocket: WebSocket,
    translation_service: "TranslationService",
    segment_id: int,
    text: str,
    source_lang: str,
    target_lang: str,
    speaker_name: str | None,
) -> None:
    """Translate a final segment and send the result to the frontend.

    Runs as a fire-and-forget task. Errors are silently logged — translation
    failure should never crash the main audio pipeline.
    """
    try:
        from livetranslate_common.models import TranslationRequest

        request = TranslationRequest(
            text=text,
            source_language=source_lang,
            target_language=target_lang,
            speaker_name=speaker_name,
        )
        response = await translation_service.translate(request)

        context_entries = translation_service.get_context(speaker_name)
        msg = TranslationMessage(
            text=response.translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            transcript_id=segment_id,
            context_used=len(context_entries),
        )
        await websocket.send_text(msg.model_dump_json())

    except Exception as exc:
        logger.warning(
            "translation_failed",
            segment_id=segment_id,
            error=str(exc),
        )


async def _warm_up_llm(translation_service: "TranslationService") -> None:
    """Fire-and-forget LLM warm-up — loads the model into Ollama's memory.

    Ollama cold-starts can take 10-15s for the first inference. By sending a
    trivial translation at session start, the model is loaded before the first
    real final segment arrives.
    """
    try:
        from livetranslate_common.models import TranslationRequest

        request = TranslationRequest(
            text="hello",
            source_language="en",
            target_language="es",
        )
        await translation_service.translate(request)
        # Clear the warm-up entry from the context window
        translation_service.clear_context()
        logger.info("llm_warm_up_complete")
    except Exception as exc:
        logger.debug("llm_warm_up_failed", error=str(exc))


# Health check endpoint
@router.get("/health")
async def websocket_audio_health():
    """Health check for WebSocket audio streaming."""
    return {
        "status": "healthy",
        "active_sessions": len(_active_sessions),
        "sessions": list(_active_sessions.keys()),
        "transcription_service": f"{TRANSCRIPTION_HOST}:{TRANSCRIPTION_PORT}",
    }
