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

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from livetranslate_common.logging import get_logger

from backends.manager import BackendManager
from language_detection import LanguageDetector
from registry import ModelRegistry

logger = get_logger()


def create_app(registry_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Transcription Service")

    if registry_path and registry_path.exists():
        registry = ModelRegistry(registry_path)
        manager = BackendManager(max_vram_mb=registry.vram_budget_mb)
    else:
        registry = None
        manager = BackendManager()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
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
            return {"error": "No registry loaded"}
        registry.reload()
        return {"status": "reloaded", "version": registry.version}

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        logger.info("ws_connected")
        session_language: str | None = None
        session_initial_prompt: str | None = None
        session_glossary_terms: list[str] | None = None
        lang_detector = LanguageDetector()
        current_backend_key: str | None = None

        try:
            while True:
                data = await ws.receive()

                if "bytes" in data and data["bytes"]:
                    audio = np.frombuffer(data["bytes"], dtype=np.float32)

                    # Fix #8: minimum frame size validation (< 100ms at 16kHz)
                    if len(audio) < 1600:
                        continue

                    if registry is None:
                        continue

                    # Fix #1: error boundary around all per-frame inference logic
                    try:
                        lang = session_language or lang_detector.current_language or "en"
                        config = registry.get_config(lang)
                        transcription_backend = await manager.get_backend(config)

                        new_backend_key = f"{config.backend}:{config.model}"
                        if current_backend_key is not None and new_backend_key != current_backend_key:
                            await ws.send_text(json.dumps({
                                "type": "backend_switched",
                                "from": current_backend_key,
                                "to": new_backend_key,
                                "reason": f"language changed to {lang}",
                            }))
                        current_backend_key = new_backend_key

                        effective_prompt = session_initial_prompt
                        if session_glossary_terms:
                            glossary_str = ", ".join(session_glossary_terms)
                            if effective_prompt:
                                effective_prompt = f"{glossary_str}. {effective_prompt}"
                            else:
                                effective_prompt = glossary_str

                        result = await transcription_backend.transcribe(
                            audio,
                            language=session_language or lang_detector.current_language,
                            beam_size=config.beam_size,
                            batch_profile=config.batch_profile,
                            initial_prompt=effective_prompt,
                        )

                        # Fix #3: language_detected fires BEFORE segment
                        if lang_detector.current_language is None:
                            detected = lang_detector.detect_initial(
                                result.language, result.confidence
                            )
                            await ws.send_text(json.dumps({
                                "type": "language_detected",
                                "language": detected,
                                "confidence": result.confidence,
                            }))
                        else:
                            chunk_duration_s = len(audio) / 16000.0
                            switched = lang_detector.update(result.language, chunk_duration_s)
                            if switched:
                                await ws.send_text(json.dumps({
                                    "type": "language_detected",
                                    "language": switched,
                                    "confidence": result.confidence,
                                }))

                        # Fix #4: filter model_dump() to public fields only
                        await ws.send_text(json.dumps({
                            "type": "segment",
                            **result.model_dump(include={"text", "language", "confidence", "is_final", "segments"}),
                        }))

                    except Exception as exc:
                        logger.exception("transcribe_error", error=str(exc))
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": str(exc),
                            "recoverable": True,
                        }))
                        continue

                elif "text" in data and data["text"]:
                    # Fix #5: guard against malformed JSON control frames
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        logger.warning("invalid_json_control_frame")
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "config":
                        session_language = msg.get("language")
                        session_initial_prompt = msg.get("initial_prompt")
                        session_glossary_terms = msg.get("glossary_terms")

                    elif msg_type == "end":
                        break

        except WebSocketDisconnect:
            logger.info("ws_disconnected")
            return

        logger.info("ws_session_ended")

    return app
