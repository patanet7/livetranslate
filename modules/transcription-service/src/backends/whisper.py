"""WhisperBackend — faster-whisper CTranslate2 integration.

Implements the TranscriptionBackend protocol using faster-whisper for
CTranslate2-optimised inference on GPU/CPU.
"""
from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from livetranslate_common.logging import get_logger
from livetranslate_common.models import ModelInfo, Segment, TranscriptionResult

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = get_logger()

# All 99 languages faster-whisper/Whisper supports (BCP-47 two-letter codes)
_WHISPER_LANGUAGES: frozenset[str] = frozenset(
    [
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "zh",
    ]
)

# Approximate VRAM usage per model size (MB) at float16
_MODEL_VRAM_MB: dict[str, int] = {
    "tiny": 150,
    "tiny.en": 150,
    "base": 290,
    "base.en": 290,
    "small": 480,
    "small.en": 480,
    "medium": 1500,
    "medium.en": 1500,
    "large": 2900,
    "large-v1": 2900,
    "large-v2": 2900,
    "large-v3": 2900,
    "large-v3-turbo": 1600,
    "turbo": 1600,
    "distil-large-v2": 1500,
    "distil-large-v3": 1500,
    "distil-medium.en": 750,
    "distil-small.en": 350,
}


def _log_prob_to_confidence(avg_log_prob: float) -> float:
    """Convert faster-whisper avg_log_prob to a [0, 1] confidence score.

    avg_log_prob is the mean log-probability over the segment tokens.
    Applying exp() converts it to a linear probability; clamping guards
    against numerical outliers (log-prob > 0 or extreme negatives).
    """
    return max(0.0, min(1.0, math.exp(avg_log_prob)))


class WhisperBackend:
    """Transcription backend powered by faster-whisper (CTranslate2).

    Args:
        model_name: Whisper model identifier (e.g. ``"tiny"``, ``"base"``,
            ``"large-v3"``).
        compute_type: CTranslate2 quantisation type — ``"float16"``,
            ``"int8_float16"``, ``"int8"``, ``"float32"``, or ``"auto"``.
        device: Compute device — ``"cuda"``, ``"cpu"``, or ``"auto"``.
        device_index: CUDA device index (ignored for CPU).
        cpu_threads: Number of CPU threads (0 = auto).
        num_workers: Number of parallel data-loading workers.
        beam_size: Beam width for decoding.
        vad_filter: Enable Silero VAD pre-filtering in faster-whisper.
        vad_parameters: Extra kwargs forwarded to the Silero VAD call.
    """

    def __init__(
        self,
        model_name: str = "base",
        compute_type: str = "float16",
        device: str = "cuda",
        device_index: int = 0,
        cpu_threads: int = 0,
        num_workers: int = 1,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
    ) -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._device = device
        self._device_index = device_index
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._vad_parameters: dict = vad_parameters or {}
        self._model: WhisperModel | None = None

    # ------------------------------------------------------------------
    # TranscriptionBackend protocol
    # ------------------------------------------------------------------

    async def load_model(self, model_name: str, device: str = "cuda") -> None:
        """Load the faster-whisper model into memory.

        Args:
            model_name: Whisper model identifier to load (overrides the
                instance-level ``model_name``).
            device: Compute device (overrides the instance-level device).
        """
        from faster_whisper import WhisperModel  # noqa: PLC0415 (deferred import)

        self._model_name = model_name
        self._device = device

        logger.info(
            "whisper_backend.load_model.start",
            model=model_name,
            device=device,
            compute_type=self._compute_type,
        )

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                model_name,
                device=device,
                device_index=self._device_index,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
            ),
        )

        logger.info(
            "whisper_backend.load_model.done",
            model=model_name,
            device=device,
            vram_mb=self.vram_usage_mb(),
        )

    async def unload_model(self) -> None:
        """Release model memory."""
        if self._model is not None:
            logger.info("whisper_backend.unload_model", model=self._model_name)
            self._model = None

    async def warmup(self) -> None:
        """Run a short silent inference to warm up the model / CUDA kernels."""
        if self._model is None:
            logger.warning("whisper_backend.warmup.skipped", reason="model_not_loaded")
            return

        logger.info("whisper_backend.warmup.start", model=self._model_name)
        silence = np.zeros(16000, dtype=np.float32)
        await self.transcribe(silence, language="en")
        logger.info("whisper_backend.warmup.done", model=self._model_name)

    async def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe an audio array.

        Args:
            audio: Float32 PCM audio at 16 kHz, shape ``(samples,)``.
            language: BCP-47 language hint or ``None`` for auto-detect.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`faster_whisper.WhisperModel.transcribe`.

        Returns:
            :class:`~livetranslate_common.models.TranscriptionResult`

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError("WhisperBackend: model is not loaded — call load_model() first")

        loop = asyncio.get_event_loop()
        segments_iter, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                vad_parameters=self._vad_parameters or None,
                **kwargs,
            ),
        )

        segments_list = list(segments_iter)
        full_text = "".join(seg.text for seg in segments_list).strip()

        seg_models = [
            Segment(
                text=seg.text.strip(),
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                confidence=_log_prob_to_confidence(seg.avg_logprob),
            )
            for seg in segments_list
        ]

        overall_confidence = (
            float(np.mean([s.confidence for s in seg_models])) if seg_models else 0.0
        )

        detected_language = info.language if info.language else (language or "en")

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
        """Yield incremental transcription results for a single audio chunk.

        For faster-whisper this is implemented by yielding one
        :class:`TranscriptionResult` per decoded segment so that callers
        receive partial output as quickly as possible.

        Args:
            audio: Float32 PCM audio at 16 kHz.
            language: BCP-47 language hint or ``None`` for auto-detect.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`faster_whisper.WhisperModel.transcribe`.

        Yields:
            :class:`~livetranslate_common.models.TranscriptionResult` per
            segment decoded.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError("WhisperBackend: model is not loaded — call load_model() first")

        loop = asyncio.get_event_loop()
        segments_iter, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                vad_parameters=self._vad_parameters or None,
                **kwargs,
            ),
        )

        detected_language = info.language if info.language else (language or "en")
        accumulated_text = ""

        for seg in segments_iter:
            seg_text = seg.text.strip()
            accumulated_text = (accumulated_text + " " + seg_text).strip()
            confidence = _log_prob_to_confidence(seg.avg_logprob)
            segment_model = Segment(
                text=seg_text,
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                confidence=confidence,
            )
            yield TranscriptionResult(
                text=accumulated_text,
                language=detected_language,
                confidence=confidence,
                segments=[segment_model],
                is_final=False,
                is_draft=True,
            )

    def supports_language(self, lang: str) -> bool:
        """Return ``True`` if Whisper supports the given BCP-47 language code.

        Args:
            lang: BCP-47 language code (e.g. ``"en"``, ``"zh"``).

        Returns:
            Whether the language is in Whisper's supported set.
        """
        return lang.lower() in _WHISPER_LANGUAGES

    def get_model_info(self) -> ModelInfo:
        """Return metadata about this backend/model configuration.

        Returns:
            :class:`~livetranslate_common.models.ModelInfo`
        """
        return ModelInfo(
            name=self._model_name,
            backend="whisper",
            languages=sorted(_WHISPER_LANGUAGES),
            vram_mb=self._estimate_vram(),
            compute_type=self._compute_type,
        )

    def vram_usage_mb(self) -> int:
        """Return estimated VRAM usage in megabytes.

        Returns ``0`` if the model is not loaded, otherwise returns the
        lookup-table estimate for the current model / compute-type
        combination.

        Returns:
            VRAM usage in MB.
        """
        if self._model is None:
            return 0
        return self._estimate_vram()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_vram(self) -> int:
        """Estimate VRAM requirement from the model name and compute type.

        The base estimate comes from :data:`_MODEL_VRAM_MB`. For ``int8``
        quantisation the estimate is halved; for ``float32`` it is doubled
        relative to the float16 baseline.

        Returns:
            Estimated VRAM in MB.
        """
        # Normalise to the base model size token (strip suffixes like
        # "-ct2", "-openai-ct2" etc. that sometimes appear in paths)
        key = self._model_name.split("/")[-1]
        base_mb = _MODEL_VRAM_MB.get(key, 500)

        if "int8" in self._compute_type:
            return base_mb // 2
        if "float32" in self._compute_type:
            return base_mb * 2
        # float16, auto, bfloat16, etc.
        return base_mb
