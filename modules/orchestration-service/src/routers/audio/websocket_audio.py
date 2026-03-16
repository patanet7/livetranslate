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
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from livetranslate_common.logging import get_logger
from livetranslate_common.models.ws_messages import (
    ConfigMessage,
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


async def _strip_think_and_stream(raw_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Filter out <think>...</think> blocks from streaming LLM output.

    Buffers the first ~30 characters to detect and strip think blocks.
    After the buffer phase (or if no think block is found), switches to
    passthrough mode yielding tokens immediately.
    """
    buffer = ""
    in_think = False
    BUFFER_LIMIT = 30

    async for delta in raw_stream:
        if in_think:
            # Inside a think block — accumulate until we find </think>
            buffer += delta
            end_idx = buffer.find("</think>")
            if end_idx != -1:
                # Think block closed — yield anything after </think>
                remainder = buffer[end_idx + 8:]
                buffer = ""
                in_think = False
                if remainder:
                    yield remainder
            continue

        if len(buffer) < BUFFER_LIMIT:
            # Still buffering — accumulate
            buffer += delta

            # Check if buffer starts with <think>
            if buffer.startswith("<think>"):
                in_think = True
                # Check if </think> already arrived in the same buffer
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    remainder = buffer[end_idx + 8:]
                    buffer = ""
                    in_think = False
                    if remainder:
                        yield remainder
                continue

            # If we've buffered enough and no <think> prefix, flush
            if len(buffer) >= BUFFER_LIMIT:
                yield buffer
                buffer = ""
        else:
            # Past buffer phase — passthrough
            yield delta

    # Flush any remaining buffer (handles: no think block, unclosed think block)
    if buffer and not in_think:
        yield buffer


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
    _ws_send_lock = asyncio.Lock()
    _disconnected = False
    segment_counter = 0  # fallback counter for backends that don't send segment_id
    sample_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    source_language: str | None = None
    target_language: str | None = TARGET_LANGUAGE

    # Accumulator for stable text across segments (for sentence-level translation)
    _stable_text_buffer: str = ""
    _last_translated_stable: str = ""

    # Lock for draft translations — non-blocking, drop when busy.
    # Lock (not Semaphore) because Lock has .locked() for non-blocking check.
    _draft_lock = asyncio.Lock()

    async def safe_send(msg: str) -> bool:
        """Send a text frame with disconnect protection.

        Returns True if sent, False if the WebSocket is dead.
        Serializes sends via lock to prevent interleaved frames
        from concurrent translation tasks.
        """
        nonlocal _disconnected
        if _disconnected:
            return False
        try:
            async with _ws_send_lock:
                await websocket.send_text(msg)
            return True
        except Exception:
            _disconnected = True
            return False

    async def handle_transcription_segment(data: dict) -> None:
        """Transform transcription result → SegmentMessage → forward to frontend."""
        nonlocal segment_counter, source_language, _stable_text_buffer, _last_translated_stable

        # Preserve transcription service's segment_id for draft→final matching.
        # The transcription service sends draft (seg_id=N) then final (seg_id=N)
        # with matching IDs. Only fall back to our counter for backends that
        # don't include segment_id.
        seg_id = data.get("segment_id")
        if seg_id is None:
            segment_counter += 1
            seg_id = segment_counter

        msg = SegmentMessage(
            segment_id=seg_id,
            text=data.get("text", ""),
            language=data.get("language", ""),
            confidence=data.get("confidence", 0.0),
            stable_text=data.get("stable_text", ""),
            unstable_text=data.get("unstable_text", ""),
            is_final=data.get("is_final", False),
            is_draft=data.get("is_draft", False),
            speaker_id=data.get("speaker_id"),
            start_ms=data.get("start_ms"),
            end_ms=data.get("end_ms"),
        )

        logger.debug(
            "segment_forward",
            session_id=session_id,
            seg_id=msg.segment_id,
            is_draft=msg.is_draft,
            is_final=msg.is_final,
            text_len=len(msg.text),
            stable_len=len(msg.stable_text),
            unstable_len=len(msg.unstable_text),
        )

        # Track source language before send (don't lose it if send fails)
        if msg.language:
            source_language = msg.language

        if not await safe_send(msg.model_dump_json()):
            return  # frontend disconnected

        # Translation trigger: need translation_service + language pair
        if not (translation_service and target_language and source_language != target_language):
            return

        # --- Draft path: bypass stable_text, translate msg.text directly ---
        if msg.is_draft:
            # Non-blocking check: drop if another draft is in-flight.
            # Safe because no await between this check and create_task below
            # (single-threaded asyncio — no concurrent coroutine can acquire between).
            if _draft_lock.locked():
                logger.debug(
                    "draft_translation_dropped",
                    session_id=session_id,
                    seg_id=msg.segment_id,
                    is_draft=True,
                )
                return

            logger.debug(
                "translation_trigger",
                session_id=session_id,
                seg_id=msg.segment_id,
                translate_text=msg.text[:80],
                is_draft=True,
            )

            async def _draft_translate(lock, svc, seg_id, text, lang, tgt, spk, cfg):
                """Draft translation with lock + timeout."""
                await lock.acquire()
                try:
                    await asyncio.wait_for(
                        _translate_and_send(
                            safe_send,
                            svc,
                            segment_id=seg_id,
                            text=text,
                            source_lang=lang,
                            target_lang=tgt,
                            speaker_name=spk,
                            pipeline=pipeline,
                            is_draft=True,
                        ),
                        timeout=cfg.draft_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        "draft_translation_timeout",
                        session_id=session_id,
                        seg_id=seg_id,
                        is_draft=True,
                        timeout_s=cfg.draft_timeout_s,
                    )
                finally:
                    lock.release()

            task = asyncio.create_task(
                _draft_translate(
                    _draft_lock,
                    translation_service,
                    msg.segment_id,
                    msg.text,
                    msg.language,
                    target_language,
                    msg.speaker_id,
                    translation_service.config,
                )
            )
            session_tasks.add(task)
            task.add_done_callback(session_tasks.discard)
            return

        # --- Final path: stable_text accumulation + sentence boundary detection ---
        translate_text = ""
        if msg.stable_text:
            import re
            incoming = msg.stable_text
            # Guard: strip overlap with already-translated text to prevent
            # sending duplicate context to the LLM
            if _last_translated_stable:
                # Fast path: if incoming is entirely contained in what we
                # already translated, skip it (handles duplicate segment_ids)
                incoming_norm = re.sub(r"[^\w\s]", "", incoming).lower()
                last_norm = re.sub(r"[^\w\s]", "", _last_translated_stable).lower()
                if incoming_norm and incoming_norm in last_norm:
                    incoming = ""
                else:
                    # Word-level prefix dedup for partial overlaps
                    prev_words = last_norm.split()
                    new_words_raw = incoming.split()
                    new_words = [re.sub(r"[^\w\s]", "", w).lower() for w in new_words_raw]
                    max_check = min(len(prev_words), len(new_words), 12)
                    for overlap_len in range(max_check, 0, -1):
                        if prev_words[-overlap_len:] == new_words[:overlap_len]:
                            incoming = " ".join(new_words_raw[overlap_len:])
                            break
            if incoming.strip():
                _stable_text_buffer += (" " if _stable_text_buffer else "") + incoming.strip()
            # Check for sentence boundaries (period, question mark, exclamation)
            if re.search(r"[.!?。！？]$", _stable_text_buffer.strip()):  # noqa: RUF001
                translate_text = _stable_text_buffer.strip()
                # Accumulate translated text (keep last ~500 chars for substring dedup)
                _last_translated_stable = (_last_translated_stable + " " + translate_text)[-500:]
                _stable_text_buffer = ""
        elif msg.is_final:
            # Fallback: use is_final when stable_text is not populated
            translate_text = msg.text

        if translate_text:
            logger.debug(
                "translation_trigger",
                session_id=session_id,
                seg_id=msg.segment_id,
                translate_text=translate_text[:80],
                stable_text=msg.stable_text[:40] if msg.stable_text else "",
                last_translated=_last_translated_stable[:40] if _last_translated_stable else "",
                buffer_len=len(_stable_text_buffer),
                is_draft=False,
            )
            task = asyncio.create_task(
                _translate_and_send(
                    safe_send,
                    translation_service,
                    segment_id=msg.segment_id,
                    text=translate_text,
                    source_lang=msg.language,
                    target_lang=target_language,
                    speaker_name=msg.speaker_id,
                    pipeline=pipeline,
                    is_draft=False,
                )
            )
            session_tasks.add(task)
            task.add_done_callback(session_tasks.discard)

    async def handle_language_detected(data: dict) -> None:
        """Forward language_detected to frontend."""
        nonlocal source_language
        source_language = data.get("language", source_language)
        await safe_send(
            LanguageDetectedMessage(
                language=data.get("language", ""),
                confidence=data.get("confidence", 0.0),
            ).model_dump_json()
        )

    async def handle_transcription_error(data: dict) -> None:
        """Forward errors from transcription service to frontend."""
        await safe_send(json.dumps({
            "type": "error",
            "message": data.get("message", "Transcription error"),
            "recoverable": data.get("recoverable", True),
        }))

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
                        await safe_send(json.dumps({
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
                        # Skip warm-up if already warmed at startup (lifespan)
                        already_warm = getattr(websocket.app.state, "translation_service_warmed", False)
                        if not already_warm:
                            asyncio.create_task(_warm_up_llm(translation_service))

                    # Send service status
                    await safe_send(
                        ServiceStatusMessage(
                            transcription="up",
                            translation="up" if translation_service else "down",
                        ).model_dump_json()
                    )
                except Exception as exc:
                    logger.error("transcription_connect_failed", error=str(exc))
                    await safe_send(json.dumps({
                        "type": "error",
                        "message": f"Cannot reach transcription service: {exc}",
                        "recoverable": False,
                    }))
                    await safe_send(
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
                        await safe_send(
                            MeetingStartedMessage(
                                session_id=str(pipeline.session_id),
                                started_at=datetime.now(UTC).isoformat(),
                            ).model_dump_json()
                        )
                        await safe_send(
                            RecordingStatusMessage(
                                recording=True,
                                chunks_written=0,
                            ).model_dump_json()
                        )
                    except Exception as exc:
                        logger.error("promote_failed", error=str(exc))
                        await safe_send(json.dumps({
                            "type": "error",
                            "message": f"Failed to promote to meeting: {exc}",
                            "recoverable": True,
                        }))
                else:
                    await safe_send(json.dumps({
                        "type": "error",
                        "message": "Meeting pipeline not available (no database configured)",
                        "recoverable": False,
                    }))

            # --- end_meeting ---
            elif isinstance(msg, EndMeetingMessage):
                if pipeline and pipeline.is_meeting:
                    await pipeline.end()
                    await safe_send(
                        RecordingStatusMessage(
                            recording=False,
                            chunks_written=0,
                        ).model_dump_json()
                    )

            # --- config (language, model changes from toolbar) ---
            elif isinstance(msg, ConfigMessage):
                if msg.language:
                    source_language = msg.language
                    if transcription_client and transcription_client.connected:
                        await transcription_client.send_config(language=msg.language)

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
    safe_send: "Callable[[str], Awaitable[bool]]",
    translation_service: "TranslationService",
    segment_id: int,
    text: str,
    source_lang: str,
    target_lang: str,
    speaker_name: str | None,
    pipeline: "MeetingPipeline | None" = None,
    is_draft: bool = False,
) -> None:
    """Translate a segment and forward to the frontend.

    Draft path (is_draft=True):
      - Non-streaming LLMClient.translate() with draft_max_tokens, max_retries=0
      - No context window write, no DB persistence
      - Sends TranslationMessage(is_draft=True)

    Final path (is_draft=False):
      - Streaming translate_stream() with max_tokens from config
      - Writes to context window, persists to DB in meeting mode
      - Sends translation_chunk messages + TranslationMessage(is_draft=False)
    """
    import time as _time

    try:
        start = _time.monotonic()

        if is_draft:
            # --- Draft: non-streaming, fail-fast, no context ---
            translation = await translation_service._client.translate(
                text=text,
                source_language=source_lang,
                target_language=target_lang,
                max_tokens=translation_service.config.draft_max_tokens,
                max_retries=0,
            )
            latency_ms = (_time.monotonic() - start) * 1000

            logger.info(
                "translation_complete",
                segment_id=segment_id,
                is_draft=True,
                latency_ms=round(latency_ms, 1),
                input_len=len(text),
                output_len=len(translation),
            )

            await safe_send(
                TranslationMessage(
                    text=translation,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    transcript_id=segment_id,
                    context_used=0,
                    is_draft=True,
                ).model_dump_json()
            )
            return

        # --- Final: streaming with context ---
        context = translation_service.get_context(speaker_name)
        full_text = ""
        translation = ""
        try:
            async for delta in _strip_think_and_stream(
                translation_service._client.translate_stream(
                    text, source_lang, target_lang, context,
                    max_tokens=translation_service.config.max_tokens,
                )
            ):
                full_text += delta
                sent = await safe_send(json.dumps({
                    "type": "translation_chunk",
                    "transcript_id": segment_id,
                    "delta": delta,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }))
                if not sent:
                    return  # WebSocket dead, stop wasting LLM tokens

            logger.debug(
                "stream_complete",
                segment_id=segment_id,
                is_draft=False,
                full_text_len=len(full_text),
                full_text_preview=full_text[:100],
            )
            translation = translation_service._client._extract_translation(full_text)
        except Exception as stream_exc:
            logger.warning(
                "translation_stream_fallback",
                segment_id=segment_id,
                is_draft=False,
                error=str(stream_exc),
            )
            from livetranslate_common.models import TranslationRequest

            request = TranslationRequest(
                text=text,
                source_language=source_lang,
                target_language=target_lang,
                speaker_name=speaker_name,
            )
            response = await translation_service.translate(request)
            translation = response.translated_text

        latency_ms = (_time.monotonic() - start) * 1000

        # Add to context window (finals only — never drafts)
        translation_service._get_context_window(speaker_name).add(text, translation)

        logger.info(
            "translation_complete",
            segment_id=segment_id,
            is_draft=False,
            latency_ms=round(latency_ms, 1),
            input_len=len(text),
            output_len=len(translation),
        )

        await safe_send(
            TranslationMessage(
                text=translation,
                source_lang=source_lang,
                target_lang=target_lang,
                transcript_id=segment_id,
                context_used=len(context),
                is_draft=False,
            ).model_dump_json()
        )

        # Persist to DB when in meeting mode (finals only)
        if pipeline and pipeline.is_meeting and pipeline.session_manager:
            try:
                await pipeline.session_manager.save_translation(
                    chunk_id=None,
                    translated_text=translation,
                    source_language=source_lang,
                    target_language=target_lang,
                    model_used=translation_service.config.model,
                    translation_time_ms=round(latency_ms, 1),
                )
            except Exception as persist_exc:
                logger.warning(
                    "translation_persist_failed",
                    segment_id=segment_id,
                    is_draft=False,
                    error=str(persist_exc),
                )

    except Exception as exc:
        logger.warning(
            "translation_failed",
            segment_id=segment_id,
            is_draft=is_draft,
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
