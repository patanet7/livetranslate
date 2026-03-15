"""Transcription service FastAPI application.

Endpoints:
  GET  /health              → service health + loaded backends
  GET  /api/models          → list of available models
  GET  /api/registry        → current registry config
  POST /api/registry/reload → hot-reload registry from disk
  WS   /api/stream          → binary audio in, text results out
  POST /api/transcribe      → batch transcription (file upload)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from livetranslate_common.logging import get_logger

from backends.manager import BackendManager
from language_detection import LanguageDetector
from registry import ModelRegistry
from vac_online_processor import VACOnlineProcessor

logger = get_logger()


@dataclass
class SessionState:
    session_id: str
    language: str | None = None
    initial_prompt: str | None = None
    glossary_terms: list[str] | None = None
    current_backend_key: str | None = None
    lang_detector: LanguageDetector = field(default_factory=LanguageDetector)
    vac_processor: VACOnlineProcessor | None = None


def create_app(registry_path: Path | None = None) -> FastAPI:
    if registry_path and registry_path.exists():
        registry = ModelRegistry(registry_path)
        manager = BackendManager(max_vram_mb=registry.vram_budget_mb)
    else:
        registry = None
        manager = BackendManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Shutdown: unload all backends
        for key in list(manager.loaded_backends.keys()):
            backend = manager.loaded_backends[key]
            await backend.unload_model()
        logger.info("all_backends_unloaded")

    app = FastAPI(title="Transcription Service", lifespan=lifespan)

    @app.get("/health")
    async def health():
        status = "ok" if registry is not None else "degraded"
        return {
            "status": status,
            "loaded_backends": list(manager.loaded_backends.keys()),
            "vram_usage_mb": manager.current_vram_mb,
        }

    @app.get("/api/models")
    async def list_models():
        return [
            b.get_model_info().model_dump()
            for b in manager.loaded_backends.values()
        ]

    @app.get("/api/registry")
    async def get_registry():
        if registry is None:
            return {"error": "No registry loaded"}
        return registry._data

    @app.post("/api/registry/reload")
    async def reload_registry():
        if registry is None:
            return JSONResponse({"error": "No registry loaded"}, status_code=400)
        success = registry.reload()
        if not success:
            return JSONResponse({"error": "Reload failed — check logs"}, status_code=500)
        return {"status": "reloaded", "version": registry.version}

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        session_id = str(uuid.uuid4())[:8]
        logger.info("ws_connected", session_id=session_id)

        if registry is None:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "No registry loaded — service cannot transcribe",
                "recoverable": False,
            }))
            await ws.close()
            return

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=16)
        state = SessionState(session_id=session_id)

        async def producer():
            """Read frames from WebSocket and route to queue or handle control messages."""
            try:
                while True:
                    data = await ws.receive()

                    if "bytes" in data and data["bytes"]:
                        try:
                            audio_queue.put_nowait(data["bytes"])
                        except asyncio.QueueFull:
                            logger.warning("audio_frame_dropped", session_id=session_id, reason="backpressure")

                    elif "text" in data and data["text"]:
                        try:
                            msg = json.loads(data["text"])
                        except json.JSONDecodeError:
                            logger.warning("invalid_json_control_frame", session_id=session_id)
                            continue

                        msg_type = msg.get("type")
                        if msg_type == "config":
                            state.language = msg.get("language")
                            state.initial_prompt = msg.get("initial_prompt")
                            state.glossary_terms = msg.get("glossary_terms")
                        elif msg_type == "end":
                            await audio_queue.put(None)  # sentinel
                            return
            except WebSocketDisconnect:
                await audio_queue.put(None)  # sentinel on disconnect

        async def consumer():
            """Process audio frames from queue and send results."""
            while True:
                raw_audio = await audio_queue.get()
                if raw_audio is None:
                    break  # sentinel received

                audio = np.frombuffer(raw_audio, dtype=np.float32)

                if len(audio) < 1600:  # < 100ms at 16kHz
                    continue

                try:
                    lang = state.language or state.lang_detector.current_language or "en"
                    config = registry.get_config(lang)
                    transcription_backend = await manager.get_backend(config)

                    new_backend_key = f"{config.backend}:{config.model}"
                    if state.current_backend_key is not None and new_backend_key != state.current_backend_key:
                        # Release the old backend before switching
                        manager.release_backend(state.current_backend_key)
                        await ws.send_text(json.dumps({
                            "type": "backend_switched",
                            "from": state.current_backend_key,
                            "to": new_backend_key,
                            "reason": f"language changed to {lang}",
                        }))
                    state.current_backend_key = new_backend_key

                    # Lazily initialize VACOnlineProcessor on first frame
                    if state.vac_processor is None:
                        state.vac_processor = VACOnlineProcessor(
                            prebuffer_s=config.prebuffer_s,
                            overlap_s=config.overlap_s,
                            stride_s=config.stride_s,
                        )

                    await state.vac_processor.feed_audio(audio)
                    if not state.vac_processor.ready_for_inference():
                        continue  # not enough audio yet

                    inference_audio = state.vac_processor.get_inference_audio()

                    effective_prompt = state.initial_prompt
                    if state.glossary_terms:
                        glossary_str = ", ".join(state.glossary_terms)
                        if effective_prompt:
                            effective_prompt = f"{glossary_str}. {effective_prompt}"
                        else:
                            effective_prompt = glossary_str

                    result = await asyncio.wait_for(
                        transcription_backend.transcribe(
                            inference_audio,
                            language=state.language or state.lang_detector.current_language,
                            beam_size=config.beam_size,
                            initial_prompt=effective_prompt,
                        ),
                        timeout=30.0,
                    )

                    # language_detected fires BEFORE segment
                    if state.lang_detector.current_language is None:
                        detected = state.lang_detector.detect_initial(result.language, result.confidence)
                        await ws.send_text(json.dumps({
                            "type": "language_detected",
                            "language": detected,
                            "confidence": result.confidence,
                        }))
                    else:
                        chunk_duration_s = len(inference_audio) / 16000.0
                        switched = state.lang_detector.update(result.language, chunk_duration_s)
                        if switched:
                            await ws.send_text(json.dumps({
                                "type": "language_detected",
                                "language": switched,
                                "confidence": result.confidence,
                            }))

                    await ws.send_text(json.dumps({
                        "type": "segment",
                        **result.model_dump(include={"text", "language", "confidence", "is_final", "segments"}),
                    }))

                except asyncio.TimeoutError:
                    logger.error("transcribe_timeout", session_id=session_id)
                    try:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": "Transcription timed out",
                            "recoverable": True,
                        }))
                    except Exception:
                        break
                except Exception as exc:
                    logger.exception("transcribe_error", session_id=session_id, error=str(exc))
                    try:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": str(exc),
                            "recoverable": True,
                        }))
                    except Exception:
                        break  # WS already closed

        try:
            await asyncio.gather(producer(), consumer())
        except WebSocketDisconnect:
            pass
        finally:
            # Release backend reference on session end
            if state.current_backend_key is not None:
                manager.release_backend(state.current_backend_key)

        logger.info("ws_session_ended", session_id=session_id)

    return app
