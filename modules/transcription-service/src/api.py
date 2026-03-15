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
import os
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


class SimpleStabilityTracker:
    """Track which words are stable (seen in 2+ consecutive decodes).

    Words behind (chunk_start + overlap_s) in the previous decode are considered
    stable — they've been seen by two overlapping inference windows and are
    unlikely to change. Words in the last overlap window are unstable.
    """

    def __init__(self, overlap_s: float = 1.0) -> None:
        self._overlap_s = overlap_s

    def split(self, text: str, segments: list) -> tuple[str, str]:
        """Split text into (stable, unstable) based on segment timing.

        If there are timed segments, words whose end_ms falls before the overlap
        boundary are stable. Otherwise, fall back to keeping the last ~20% as
        unstable.
        """
        if not text:
            return "", ""

        overlap_ms = int(self._overlap_s * 1000)

        # Use segment timing if available
        if segments:
            last_end_ms = max(s.end_ms for s in segments)
            cutoff_ms = last_end_ms - overlap_ms
            stable_parts = []
            unstable_parts = []
            for seg in segments:
                if seg.end_ms <= cutoff_ms:
                    stable_parts.append(seg.text)
                else:
                    unstable_parts.append(seg.text)
            return " ".join(stable_parts), " ".join(unstable_parts)

        # Fallback: last 20% of words are unstable
        words = text.split()
        if len(words) <= 2:
            return "", text
        cut = max(1, len(words) - max(1, len(words) // 5))
        return " ".join(words[:cut]), " ".join(words[cut:])


@dataclass
class SessionState:
    session_id: str
    language: str | None = None
    initial_prompt: str | None = None
    glossary_terms: list[str] | None = None
    current_backend_key: str | None = None
    lang_detector: LanguageDetector = field(default_factory=LanguageDetector)
    vac_processor: VACOnlineProcessor | None = None


def _dedup_overlap(prev_text: str, new_text: str) -> str:
    """Remove overlapping text between consecutive segments.

    The VAC processor retains ``overlap_s`` seconds of audio for context,
    which causes the start of the new segment to repeat the end of the
    previous one.

    For CJK text (no spaces): character-level suffix/prefix matching.
    For Latin text: word-level suffix/prefix matching.
    """
    if not prev_text or not new_text:
        return new_text

    import re
    _strip_punct = re.compile(r"[^\w\s]")

    # Detect CJK: if >30% of chars are CJK, use character-level dedup
    cjk_count = sum(1 for c in new_text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    is_cjk = cjk_count > len(new_text) * 0.3

    if is_cjk:
        # Character-level dedup for CJK (no spaces between words)
        # Normalize: strip punctuation for matching
        prev_chars = re.sub(r'[^\w]', '', prev_text)
        new_chars = re.sub(r'[^\w]', '', new_text)
        # Cap at 20 chars (~1.5s of CJK at ~3 chars/s per speaker)
        max_overlap = min(len(prev_chars), len(new_chars), 20)
        for overlap_len in range(max_overlap, 2, -1):  # min 3 chars to avoid false matches
            suffix = prev_chars[-overlap_len:]
            if new_chars.startswith(suffix):
                # Found character overlap — find where it ends in the original new_text
                # Strip from new_text by finding the suffix position
                stripped = new_text
                for i in range(len(new_text)):
                    candidate = re.sub(r'[^\w]', '', new_text[:i + 1])
                    if len(candidate) >= overlap_len:
                        stripped = new_text[i + 1:].lstrip()
                        break
                return stripped if stripped else new_text
        return new_text

    # Word-level dedup for Latin scripts
    def _normalize(words: list[str]) -> list[str]:
        return [_strip_punct.sub("", w).lower() for w in words]

    prev_words = prev_text.split()
    new_words = new_text.split()
    prev_norm = _normalize(prev_words)
    new_norm = _normalize(new_words)

    max_overlap = min(len(prev_norm), len(new_norm), 12)
    for overlap_len in range(max_overlap, 0, -1):
        suffix = prev_norm[-overlap_len:]
        prefix = new_norm[:overlap_len]
        if suffix == prefix:
            return " ".join(new_words[overlap_len:])

    return new_text


def _register_backend_factories(manager: BackendManager, registry: ModelRegistry) -> None:
    """Dynamically import and register backend factories from the registry YAML.

    The registry's ``backends`` section maps backend names to module/class:
      whisper:
        module: backends.whisper
        class: WhisperBackend
    """
    import importlib

    for backend_name in registry._data.get("backends", {}):
        info = registry.get_backend_module(backend_name)
        module_path = info["module"]
        class_name = info["class"]

        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            def make_factory(klass):
                def factory(config: "BackendConfig"):
                    return klass(
                        model_name=config.model,
                        compute_type=config.compute_type,
                        beam_size=config.beam_size,
                    )
                return factory

            manager.register_factory(backend_name, make_factory(cls))
            logger.info("backend_factory_registered", backend=backend_name, cls=class_name)
        except Exception:
            logger.exception("backend_factory_registration_failed", backend=backend_name)


def create_app(registry_path: Path | None = None) -> FastAPI:
    if registry_path and registry_path.exists():
        registry = ModelRegistry(registry_path)
        manager = BackendManager(max_vram_mb=registry.vram_budget_mb)
        _register_backend_factories(manager, registry)
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

        # Queue size must absorb frames during inference. At 100ms/frame and
        # ~4s inference on CPU, 40 frames arrive per cycle. 128 gives ~3 inference
        # windows of headroom. On GPU where inference < 0.5s, 16 would suffice.
        queue_size = int(os.getenv("AUDIO_QUEUE_SIZE", "128"))
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_size)
        state = SessionState(session_id=session_id)
        # Track previous segment text for overlap dedup + context passing
        _prev_segment_text: str = ""
        _stability_tracker: SimpleStabilityTracker | None = None
        # Track session-level sample count for absolute timestamps
        _session_samples_consumed: int = 0

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
            except RuntimeError as exc:
                # Starlette raises RuntimeError when calling receive() after disconnect
                if "disconnect" in str(exc).lower():
                    await audio_queue.put(None)
                else:
                    raise

        async def _run_inference(inference_audio: np.ndarray) -> None:
            """Run transcription inference and send results. Runs as background task
            so the consumer can keep feeding VAC during inference."""
            nonlocal _prev_segment_text, _session_samples_consumed

            try:
                lang = state.language or state.lang_detector.current_language or "en"
                config = registry.get_config(lang)
                transcription_backend = await manager.get_backend(config)

                # Build prompt: glossary + user prompt + previous transcription context
                # CJK languages: limit prompt to 80 chars (longer prompts cause decoder copying)
                is_cjk = lang in ("zh", "ja", "ko")
                max_prompt_chars = 80 if is_cjk else 200
                effective_prompt = state.initial_prompt
                if state.glossary_terms:
                    glossary_str = ", ".join(state.glossary_terms)
                    effective_prompt = f"{glossary_str}. {effective_prompt}" if effective_prompt else glossary_str
                if _prev_segment_text:
                    context_tail = _prev_segment_text[-max_prompt_chars:]
                    # Trim to last complete sentence boundary for CJK
                    # Keep full tail if trimming would produce empty context
                    if is_cjk:
                        for sep in ("。", "！", "？", "，"):  # noqa: RUF001 — CJK punctuation
                            idx = context_tail.rfind(sep)
                            if idx >= 0 and idx < len(context_tail) - 1:
                                context_tail = context_tail[idx + 1:]
                                break
                    effective_prompt = f"{effective_prompt} {context_tail}" if effective_prompt else context_tail

                try:
                    result = await asyncio.wait_for(
                        transcription_backend.transcribe(
                            inference_audio,
                            language=state.language or state.lang_detector.current_language,
                            beam_size=config.beam_size,
                            initial_prompt=effective_prompt,
                        ),
                        timeout=30.0,
                    )
                    manager.record_success(config)
                except (TimeoutError, RuntimeError):
                    manager.record_failure(config)
                    raise

                # Gate: suppress if no_speech_prob indicates silence
                # Must fire BEFORE language detector update to avoid polluting LID state
                if result.no_speech_prob is not None and result.no_speech_prob > 0.6:
                    logger.debug("no_speech_filtered", no_speech_prob=result.no_speech_prob)
                    return

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

                # Language consistency filter: discard wrong-script hallucinations
                if (
                    state.language
                    and result.language != state.language
                    and result.confidence < 0.7
                ):
                    logger.debug(
                        "language_mismatch_filtered",
                        expected=state.language,
                        detected=result.language,
                        confidence=result.confidence,
                    )
                    return

                # Filter hallucinated low-confidence short segments
                if result.confidence < 0.4 and len(result.text.split()) <= 3:
                    logger.debug("low_confidence_filtered", text=result.text, confidence=result.confidence)
                    return

                result_data = result.model_dump(include={
                    "text", "language", "confidence", "is_final", "segments",
                    "stable_text", "unstable_text", "speaker_id",
                })
                # Convert chunk-relative timing to session-absolute
                chunk_offset_ms = int(_session_samples_consumed / 16)
                if result.segments:
                    result_data["start_ms"] = result.segments[0].start_ms + chunk_offset_ms
                    result_data["end_ms"] = result.segments[-1].end_ms + chunk_offset_ms
                else:
                    result_data["start_ms"] = chunk_offset_ms
                    result_data["end_ms"] = chunk_offset_ms
                overlap_samples = state.vac_processor._overlap_samples if state.vac_processor else 0
                _session_samples_consumed += max(0, len(inference_audio) - overlap_samples)

                # Dedup overlap
                deduped_text = _dedup_overlap(_prev_segment_text, result_data["text"])
                result_data["text"] = deduped_text
                _prev_segment_text = result.text

                # Stability tracking: split into stable/unstable text
                nonlocal _stability_tracker
                if _stability_tracker is None:
                    overlap = state.vac_processor.overlap_s if state.vac_processor else 1.0
                    _stability_tracker = SimpleStabilityTracker(overlap_s=overlap)
                stable, unstable = _stability_tracker.split(deduped_text, result.segments)
                result_data["stable_text"] = stable
                result_data["unstable_text"] = unstable

                if deduped_text.strip():
                    await ws.send_text(json.dumps({
                        "type": "segment",
                        **result_data,
                    }))

            except asyncio.TimeoutError:
                logger.error("transcribe_timeout", session_id=session_id)
                try:
                    await ws.send_text(json.dumps({
                        "type": "error", "message": "Transcription timed out", "recoverable": True,
                    }))
                except Exception:
                    pass
            except Exception as exc:
                logger.exception("transcribe_error", session_id=session_id, error=str(exc))
                try:
                    await ws.send_text(json.dumps({
                        "type": "error", "message": str(exc), "recoverable": True,
                    }))
                except Exception:
                    pass

        async def consumer():
            """Feed audio to VAC. When ready, spawn inference as background task
            so audio keeps flowing during model execution — real-time streaming."""
            nonlocal _prev_segment_text, _session_samples_consumed
            inference_task: asyncio.Task | None = None

            while True:
                raw_audio = await audio_queue.get()
                if raw_audio is None:
                    # Wait for any running inference to finish
                    if inference_task and not inference_task.done():
                        await inference_task
                    # End of stream — flush remaining VAC buffer
                    if state.vac_processor is not None and state.current_backend_key is not None:
                        remaining = state.vac_processor.get_inference_audio()
                        if len(remaining) >= 4800:
                            await _run_inference(remaining)
                    break

                audio = np.frombuffer(raw_audio, dtype=np.float32)
                if len(audio) < 1600:
                    continue

                try:
                    lang = state.language or state.lang_detector.current_language or "en"
                    config = registry.get_config(lang)
                    transcription_backend = await manager.get_backend(config)

                    new_backend_key = f"{config.backend}:{config.model}"
                    if state.current_backend_key is not None and new_backend_key != state.current_backend_key:
                        manager.release_backend(state.current_backend_key)
                        await ws.send_text(json.dumps({
                            "type": "backend_switched",
                            "backend": config.backend,
                            "model": config.model,
                            "language": lang,
                        }))
                    state.current_backend_key = new_backend_key

                    if state.vac_processor is None:
                        state.vac_processor = VACOnlineProcessor(
                            prebuffer_s=config.prebuffer_s,
                            overlap_s=config.overlap_s,
                            stride_s=config.stride_s,
                        )

                    await state.vac_processor.feed_audio(audio)
                    if not state.vac_processor.ready_for_inference():
                        continue

                    # Don't start new inference if previous is still running
                    if inference_task and not inference_task.done():
                        continue

                    inference_audio = state.vac_processor.get_inference_audio()
                    inference_task = asyncio.create_task(_run_inference(inference_audio))
                    # Yield to event loop so inference task can start immediately
                    await asyncio.sleep(0)

                except Exception as exc:
                    logger.exception("consumer_error", session_id=session_id, error=str(exc))
                    continue  # keep processing audio

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
