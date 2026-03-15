"""MLXWhisperBackend — Apple Silicon MLX integration.

Implements the TranscriptionBackend protocol using mlx-whisper for
Metal-accelerated inference on Apple Silicon (M1/M2/M3/M4).

Useful for local development and testing without CUDA GPU.
Slower than faster-whisper on NVIDIA but runs natively on Mac.

Model names map to HuggingFace MLX community repos:
  "tiny"   → "mlx-community/whisper-tiny"
  "base"   → "mlx-community/whisper-base"
  "small"  → "mlx-community/whisper-small"
  "medium" → "mlx-community/whisper-medium"
  "large"  → "mlx-community/whisper-large-v3-turbo"
"""
from __future__ import annotations

import asyncio
import math
import tempfile
import time
from typing import AsyncIterator

import numpy as np
import soundfile as sf

from livetranslate_common.logging import get_logger
from livetranslate_common.models import ModelInfo, Segment, TranscriptionResult

logger = get_logger()

# Same language set as Whisper (mlx-whisper supports all OpenAI Whisper languages)
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

# Map short model names to HuggingFace MLX community repos
_MLX_MODEL_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# Approximate memory usage (unified memory on Apple Silicon)
_MODEL_MEM_MB: dict[str, int] = {
    "tiny": 75,
    "tiny.en": 75,
    "base": 145,
    "base.en": 145,
    "small": 480,
    "small.en": 480,
    "medium": 1500,
    "medium.en": 1500,
    "large-v3-turbo": 1600,
    "turbo": 1600,
    "large-v3": 2900,
}


def _log_prob_to_confidence(avg_log_prob: float) -> float:
    """Convert avg_log_prob to a [0, 1] confidence score."""
    return max(0.0, min(1.0, math.exp(avg_log_prob)))


class MLXWhisperBackend:
    """Transcription backend powered by mlx-whisper on Apple Silicon.

    Args:
        model_name: Short model name (e.g. ``"base"``, ``"medium"``) or
            full HuggingFace repo path.
        beam_size: Beam width for decoding (not directly supported by
            mlx-whisper's simple API, but passed as ``beam_size`` kwarg).
    """

    def __init__(
        self,
        model_name: str = "base",
        compute_type: str = "float16",
        beam_size: int = 5,
        **kwargs,
    ) -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._repo: str | None = None
        self._loaded = False

    async def load_model(self, model_name: str, device: str = "mps") -> None:
        """Load the MLX Whisper model.

        mlx-whisper downloads and caches models on first use, so load_model
        triggers a warmup call to pull the weights.
        """
        self._model_name = model_name
        self._repo = _MLX_MODEL_REPOS.get(model_name, model_name)

        logger.info(
            "mlx_whisper_backend.load_model.start",
            model=model_name,
            repo=self._repo,
            device="apple_silicon",
        )

        # mlx-whisper downloads on first transcribe call.
        # We trigger a short warmup to pull the weights now.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._preload)

        self._loaded = True
        logger.info(
            "mlx_whisper_backend.load_model.done",
            model=model_name,
            mem_mb=self.vram_usage_mb(),
        )

    def _preload(self) -> None:
        """Preload model weights by running a tiny transcription."""
        import mlx_whisper

        silence = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=self._repo,
            language="en",
        )

    async def unload_model(self) -> None:
        """Mark model as unloaded. MLX manages memory automatically."""
        self._loaded = False
        self._repo = None
        logger.info("mlx_whisper_backend.unload_model", model=self._model_name)

    async def warmup(self) -> None:
        """Run a short silent inference to warm up MLX kernels."""
        if not self._loaded:
            logger.warning("mlx_whisper_backend.warmup.skipped", reason="model_not_loaded")
            return

        logger.info("mlx_whisper_backend.warmup.start", model=self._model_name)
        t0 = time.perf_counter()
        silence = np.zeros(16000, dtype=np.float32)
        await self.transcribe(silence, language="en")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "mlx_whisper_backend.warmup.done",
            model=self._model_name,
            elapsed_ms=round(elapsed_ms, 1),
        )

    async def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe an audio array using mlx-whisper.

        Args:
            audio: Float32 PCM audio at 16 kHz, shape ``(samples,)``.
            language: BCP-47 language hint or ``None`` for auto-detect.

        Returns:
            TranscriptionResult with segments.
        """
        if not self._loaded:
            raise RuntimeError("MLXWhisperBackend: model not loaded — call load_model() first")

        repo = self._repo
        beam = self._beam_size

        def _run():
            import mlx_whisper

            # Ensure mono 1-D float32 (stereo arrays cause OOM in MLX)
            mono = audio.astype(np.float32)
            if mono.ndim == 2:
                mono = mono.mean(axis=1)

            transcribe_kwargs = {
                "path_or_hf_repo": repo,
                "verbose": False,
            }
            if language:
                transcribe_kwargs["language"] = language

            result = mlx_whisper.transcribe(mono, **transcribe_kwargs)
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run)

        full_text = result.get("text", "").strip()
        detected_language = result.get("language", language or "en")

        seg_models = []
        for seg in result.get("segments", []):
            avg_logprob = seg.get("avg_logprob", -1.0)
            seg_models.append(
                Segment(
                    text=seg.get("text", "").strip(),
                    start_ms=int(seg.get("start", 0) * 1000),
                    end_ms=int(seg.get("end", 0) * 1000),
                    confidence=_log_prob_to_confidence(avg_logprob),
                )
            )

        overall_confidence = (
            float(np.mean([s.confidence for s in seg_models])) if seg_models else 0.0
        )

        return TranscriptionResult(
            text=full_text,
            language=detected_language,
            confidence=overall_confidence,
            segments=seg_models,
            is_final=True,
            is_draft=False,
        )

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None = None,
        **kwargs,
    ) -> AsyncIterator[TranscriptionResult]:
        """Yield incremental results (one per segment).

        mlx-whisper doesn't support true streaming, so we transcribe
        the full chunk and yield segments incrementally.
        """
        result = await self.transcribe(audio, language=language, **kwargs)

        accumulated_text = ""
        accumulated_segments: list[Segment] = []
        confidence_sum = 0.0

        for seg in result.segments:
            accumulated_text = (accumulated_text + " " + seg.text).strip()
            accumulated_segments.append(seg)
            confidence_sum += seg.confidence
            avg_confidence = confidence_sum / len(accumulated_segments)

            yield TranscriptionResult(
                text=accumulated_text,
                language=result.language,
                confidence=avg_confidence,
                segments=list(accumulated_segments),
                is_final=False,
                is_draft=True,
            )

    def supports_language(self, lang: str) -> bool:
        return lang.lower() in _WHISPER_LANGUAGES

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            backend="mlx",
            languages=sorted(_WHISPER_LANGUAGES),
            vram_mb=self._estimate_mem(),
            compute_type=self._compute_type,
        )

    def vram_usage_mb(self) -> int:
        if not self._loaded:
            return 0
        return self._estimate_mem()

    def _estimate_mem(self) -> int:
        key = self._model_name.split("/")[-1]
        return _MODEL_MEM_MB.get(key, 500)
