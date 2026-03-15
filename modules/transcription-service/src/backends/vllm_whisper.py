"""VLLMWhisperBackend — vllm-mlx OpenAI-compatible transcription API.

Uses vllm-mlx's /v1/audio/transcriptions endpoint (Whisper via mlx-audio).
Single server handles both transcription AND translation on Apple Silicon.

Start server:
  uv run vllm-mlx serve mlx-community/Qwen3.5-4B-4bit --port 8000

The Whisper model is loaded automatically by mlx-audio when the first
transcription request arrives.
"""
from __future__ import annotations

import asyncio
import math
import tempfile
import time
from typing import AsyncIterator

import httpx
import numpy as np
import soundfile as sf

from livetranslate_common.logging import get_logger
from livetranslate_common.models import ModelInfo, Segment, TranscriptionResult

logger = get_logger()

_WHISPER_LANGUAGES: frozenset[str] = frozenset(
    [
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
        "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
        "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
        "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
        "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
        "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
        "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
        "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
        "yi", "yo", "zh",
    ]
)

# Default vllm-mlx Whisper model
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


class VLLMWhisperBackend:
    """Transcription backend that calls vllm-mlx's OpenAI-compatible transcription API.

    This delegates inference to a running vllm-mlx server, which handles
    model loading, batching, and Metal GPU scheduling internally.

    Args:
        model_name: Whisper model identifier (passed to vllm-mlx).
        compute_type: Ignored (vllm-mlx handles quantization).
        base_url: vllm-mlx server URL (default http://localhost:8000).
        timeout_s: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        compute_type: str = "float16",
        base_url: str | None = None,
        timeout_s: int = 30,
        **kwargs,
    ) -> None:
        import os
        self._model_name = model_name
        self._compute_type = compute_type
        self._base_url = base_url or os.getenv("VLLM_MLX_URL", "http://localhost:8000")
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._loaded = False
        # Map short names to HF repos for the API
        self._whisper_model = self._resolve_model(model_name)

    @staticmethod
    def _resolve_model(name: str) -> str:
        """Resolve short model name to full HF repo path."""
        mapping = {
            "tiny": "mlx-community/whisper-tiny",
            "base": "mlx-community/whisper-base",
            "small": "mlx-community/whisper-small",
            "medium": "mlx-community/whisper-medium",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        return mapping.get(name, name)

    async def load_model(self, model_name: str, device: str = "mps") -> None:
        """Initialize the HTTP client and verify the server is reachable."""
        self._model_name = model_name
        self._whisper_model = self._resolve_model(model_name)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_s),
        )

        # Verify server is reachable
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            self._loaded = True
            logger.info(
                "vllm_whisper_backend.connected",
                base_url=self._base_url,
                model=self._whisper_model,
            )
        except Exception as exc:
            logger.warning(
                "vllm_whisper_backend.server_not_reachable",
                base_url=self._base_url,
                error=str(exc),
            )
            # Still mark as loaded — will fail on first transcribe
            self._loaded = True

    async def unload_model(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._loaded = False

    async def warmup(self) -> None:
        """Send a short silence to warm up the server-side model."""
        if not self._loaded or not self._client:
            return

        logger.info("vllm_whisper_backend.warmup.start")
        t0 = time.perf_counter()
        silence = np.zeros(16000, dtype=np.float32)
        try:
            await self.transcribe(silence, language="en")
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("vllm_whisper_backend.warmup.done", elapsed_ms=round(elapsed_ms, 1))

    async def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio via vllm-mlx's /v1/audio/transcriptions endpoint."""
        if not self._client:
            raise RuntimeError("VLLMWhisperBackend: not connected — call load_model() first")

        # Write audio to temp WAV (the API expects file upload)
        mono = audio.astype(np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, mono, 16000)
            f.seek(0)

            data = {"model": self._whisper_model}
            if language:
                data["language"] = language

            response = await self._client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", open(f.name, "rb"), "audio/wav")},
                data=data,
            )
            response.raise_for_status()

        result = response.json()

        # Parse OpenAI Whisper API response format
        text = result.get("text", "").strip()
        detected_language = result.get("language", language or "en")

        # Build segments from response (if available)
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                Segment(
                    text=seg.get("text", "").strip(),
                    start_ms=int(seg.get("start", 0) * 1000),
                    end_ms=int(seg.get("end", 0) * 1000),
                    confidence=max(0.0, min(1.0, math.exp(seg.get("avg_logprob", -1.0)))),
                )
            )

        overall_confidence = (
            float(np.mean([s.confidence for s in segments])) if segments else 0.5
        )

        return TranscriptionResult(
            text=text,
            language=detected_language,
            confidence=overall_confidence,
            segments=segments,
            is_final=True,
            is_draft=False,
        )

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None = None,
        **kwargs,
    ) -> AsyncIterator[TranscriptionResult]:
        """Yield incremental results (batch transcribe, yield per segment)."""
        result = await self.transcribe(audio, language=language, **kwargs)
        accumulated_text = ""
        accumulated_segments: list[Segment] = []
        confidence_sum = 0.0

        for seg in result.segments:
            accumulated_text = (accumulated_text + " " + seg.text).strip()
            accumulated_segments.append(seg)
            confidence_sum += seg.confidence

            yield TranscriptionResult(
                text=accumulated_text,
                language=result.language,
                confidence=confidence_sum / len(accumulated_segments),
                segments=list(accumulated_segments),
                is_final=False,
                is_draft=True,
            )

    def supports_language(self, lang: str) -> bool:
        return lang.lower() in _WHISPER_LANGUAGES

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            backend="vllm",
            languages=sorted(_WHISPER_LANGUAGES),
            vram_mb=0,  # managed by vllm-mlx server
            compute_type=self._compute_type,
        )

    def vram_usage_mb(self) -> int:
        return 0  # managed by vllm-mlx server
