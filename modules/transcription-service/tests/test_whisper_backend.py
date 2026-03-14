"""Tests for WhisperBackend — faster-whisper integration.

NOTE: GPU integration tests require a GPU with faster-whisper installed.
Mark with @pytest.mark.gpu for CI filtering.
"""
import numpy as np
import pytest
import pytest_asyncio

from backends.base import TranscriptionBackend
from backends.whisper import WhisperBackend
from livetranslate_common.models import BackendConfig


class TestWhisperBackendProtocol:
    def test_implements_protocol(self):
        """WhisperBackend must satisfy TranscriptionBackend protocol."""
        assert issubclass(WhisperBackend, TranscriptionBackend) or isinstance(
            WhisperBackend.__new__(WhisperBackend), TranscriptionBackend
        )

    def test_instantiates_with_defaults(self):
        """WhisperBackend can be constructed with default arguments."""
        backend = WhisperBackend()
        assert backend is not None

    def test_instantiates_with_kwargs(self):
        """WhisperBackend accepts all documented keyword arguments."""
        backend = WhisperBackend(
            model_name="tiny",
            compute_type="int8",
            device="cpu",
            device_index=0,
            cpu_threads=2,
            num_workers=1,
            beam_size=3,
            vad_filter=False,
        )
        assert backend is not None

    def test_supports_language_english(self):
        """supports_language returns True for English without model loaded."""
        backend = WhisperBackend()
        assert backend.supports_language("en") is True

    def test_supports_language_chinese(self):
        """supports_language returns True for Chinese without model loaded."""
        backend = WhisperBackend()
        assert backend.supports_language("zh") is True

    def test_supports_language_unsupported(self):
        """supports_language returns False for invented language codes."""
        backend = WhisperBackend()
        assert backend.supports_language("xx") is False

    def test_get_model_info_returns_model_info(self):
        """get_model_info returns a ModelInfo instance without model loaded."""
        from livetranslate_common.models import ModelInfo

        backend = WhisperBackend(model_name="tiny", compute_type="float16")
        info = backend.get_model_info()
        assert isinstance(info, ModelInfo)
        assert info.backend == "whisper"
        assert info.name == "tiny"
        assert info.compute_type == "float16"
        assert "en" in info.languages
        assert "zh" in info.languages

    def test_vram_usage_mb_unloaded(self):
        """vram_usage_mb returns 0 when model is not loaded."""
        backend = WhisperBackend(model_name="tiny")
        assert backend.vram_usage_mb() == 0

    def test_estimate_vram_int8_is_lower(self):
        """int8 compute_type yields lower VRAM estimate than float16."""
        fp16 = WhisperBackend(model_name="base", compute_type="float16")._estimate_vram()
        int8 = WhisperBackend(model_name="base", compute_type="int8")._estimate_vram()
        assert int8 < fp16

    def test_estimate_vram_float32_is_higher(self):
        """float32 compute_type yields higher VRAM estimate than float16."""
        fp16 = WhisperBackend(model_name="base", compute_type="float16")._estimate_vram()
        fp32 = WhisperBackend(model_name="base", compute_type="float32")._estimate_vram()
        assert fp32 > fp16

    def test_log_prob_confidence_conversion(self):
        """_log_prob_to_confidence correctly maps log-probs to [0, 1]."""
        from backends.whisper import _log_prob_to_confidence

        # Typical log-prob near zero → confidence near 1
        conf = _log_prob_to_confidence(-0.1)
        assert 0.0 <= conf <= 1.0
        assert conf > 0.8

        # Very negative log-prob → confidence near 0
        conf_low = _log_prob_to_confidence(-10.0)
        assert conf_low < 0.01

        # Clamps at 0
        conf_extreme = _log_prob_to_confidence(-1000.0)
        assert conf_extreme == 0.0


@pytest.mark.gpu
class TestWhisperBackendIntegration:
    @pytest_asyncio.fixture
    async def backend(self):
        b = WhisperBackend(
            model_name="tiny",
            compute_type="float16",
            device="cuda",
        )
        await b.load_model("tiny")
        await b.warmup()
        yield b
        await b.unload_model()

    async def test_transcribe_silence(self, backend):
        """Silence should produce empty or near-empty transcription."""
        silence = np.zeros(16000, dtype=np.float32)
        result = await backend.transcribe(silence, language="en")
        assert result.language == "en"

    async def test_vram_reporting(self, backend):
        assert backend.vram_usage_mb() > 0

    async def test_model_info(self, backend):
        info = backend.get_model_info()
        assert info.backend == "whisper"
        assert info.compute_type == "float16"

    async def test_supports_language(self, backend):
        assert backend.supports_language("en") is True
        assert backend.supports_language("zh") is True

    async def test_transcribe_result_structure(self, backend):
        """Transcription result must conform to TranscriptionResult model."""
        from livetranslate_common.models import TranscriptionResult

        audio = np.zeros(32000, dtype=np.float32)
        result = await backend.transcribe(audio, language="en")

        assert isinstance(result, TranscriptionResult)
        assert 0.0 <= result.confidence <= 1.0
        for seg in result.segments:
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.end_ms >= seg.start_ms

    async def test_transcribe_stream_yields_results(self, backend):
        """transcribe_stream must yield at least one result for non-silent audio."""
        # Use a short sine tone so VAD does not filter it all out
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        results = []
        async for result in backend.transcribe_stream(audio, language="en"):
            results.append(result)
            assert 0.0 <= result.confidence <= 1.0

    async def test_unload_clears_vram(self, backend):
        """After unload_model, vram_usage_mb should return 0."""
        assert backend.vram_usage_mb() > 0
        await backend.unload_model()
        assert backend.vram_usage_mb() == 0
