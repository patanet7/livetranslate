import os
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile


def _generate_session_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}-{suffix}"


def _base_metadata() -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": _generate_session_id("req"),
    }


def build_whisper_app() -> FastAPI:
    app = FastAPI(title="Mock Whisper Service", version="0.1.0")

    models = [
        {"name": "whisper-tiny", "size": "39MB"},
        {"name": "whisper-base", "size": "74MB"},
        {"name": "whisper-small", "size": "244MB"},
    ]

    @app.get("/health")
    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "service": "whisper-mock", **_base_metadata()}

    @app.get("/api/models")
    async def list_models() -> Dict[str, List[Dict[str, str]]]:
        return {"available_models": models}

    @app.post("/transcribe/{model_name}")
    @app.post("/api/transcribe")
    async def transcribe(
        model_name: str = "whisper-base",  # pragma: allowlist secret
        audio: UploadFile = File(...),
    ) -> Dict[str, Any]:
        _ = await audio.read()  # discard content; mock result
        return {
            "text": "This is a mock transcription.",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.5,
                    "text": "This is a mock transcription.",
                    "confidence": 0.72,
                }
            ],
            "speakers": [{"speaker": "S1", "text": "This is a mock transcription."}],
            "processing_time": 0.42,
            "confidence": 0.72,
            "model": model_name,
            "metadata": _base_metadata(),
        }

    return app


def build_translation_app() -> FastAPI:
    app = FastAPI(title="Mock Translation Service", version="0.1.0")

    languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
    ]

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "service": "translation-mock", **_base_metadata()}

    @app.get("/api/device-info")
    async def device_info() -> Dict[str, Any]:
        return {
            "device": "cpu",
            "status": "ready",
            "backend": "mock",
            "metadata": _base_metadata(),
        }

    @app.get("/api/languages")
    async def supported_languages() -> Dict[str, List[Dict[str, str]]]:
        return {"languages": languages}

    @app.post("/api/translate")
    async def translate(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        target = payload.get("target_language", "en")
        session_id = payload.get("session_id") or _generate_session_id("session")
        return {
            "translated_text": f"{text} (translated to {target})",
            "source_language": payload.get("source_language", "auto"),
            "target_language": target,
            "confidence": 0.9,
            "processing_time": 0.25,
            "model_used": payload.get("model", "mock-translator"),
            "backend_used": "mock-backend",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": _base_metadata(),
        }

    return app


def create_app() -> FastAPI:
    service = os.getenv("MOCK_SERVICE", "whisper").lower()
    if service == "translation":
        return build_translation_app()
    return build_whisper_app()
