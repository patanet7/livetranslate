"""Tests for BackendManager VRAM budget enforcement and LRU eviction."""
import asyncio

import numpy as np
import pytest

from backends.base import TranscriptionBackend
from backends.manager import BackendManager
from livetranslate_common.models import BackendConfig, ModelInfo, TranscriptionResult


class FakeBackend:
    """Test double implementing TranscriptionBackend protocol."""

    def __init__(self, name: str, vram_mb: int, languages: list[str]):
        self._name = name
        self._vram_mb = vram_mb
        self._languages = languages
        self._loaded = False

    async def transcribe(self, audio: np.ndarray, language: str | None = None, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="fake", language=language or "en", confidence=0.9,
            is_final=True, is_draft=False, speaker_id=None,
        )

    async def transcribe_stream(self, audio, language=None, **kwargs):
        yield await self.transcribe(audio, language, **kwargs)

    def supports_language(self, lang: str) -> bool:
        return lang in self._languages

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name, backend="fake",
            languages=self._languages, vram_mb=self._vram_mb,
            compute_type="float16",
        )

    async def load_model(self, model_name: str, device: str = "cuda") -> None:
        self._loaded = True

    async def unload_model(self) -> None:
        self._loaded = False

    async def warmup(self) -> None:
        pass

    def vram_usage_mb(self) -> int:
        return self._vram_mb if self._loaded else 0


class TestBackendManager:
    @pytest.fixture
    def manager(self):
        return BackendManager(max_vram_mb=10000)

    def test_initial_state(self, manager):
        assert manager.current_vram_mb == 0
        assert len(manager.loaded_backends) == 0

    @pytest.mark.asyncio
    async def test_load_backend(self, manager):
        backend = FakeBackend("test-model", vram_mb=3000, languages=["en"])
        config = BackendConfig(
            backend="fake", model="test-model", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        manager.register_factory("fake", lambda cfg: backend)

        result = await manager.get_backend(config)
        assert result is backend
        assert manager.current_vram_mb == 3000

    @pytest.mark.asyncio
    async def test_lru_eviction(self, manager):
        """When loading a new backend exceeds budget, LRU backend is evicted."""
        b1 = FakeBackend("model-a", vram_mb=6000, languages=["en"])
        b2 = FakeBackend("model-b", vram_mb=6000, languages=["zh"])

        manager.register_factory("fake_a", lambda cfg: b1)
        manager.register_factory("fake_b", lambda cfg: b2)

        cfg_a = BackendConfig(
            backend="fake_a", model="model-a", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        cfg_b = BackendConfig(
            backend="fake_b", model="model-b", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.0, overlap_s=1.0,
            vad_threshold=0.45, beam_size=5, prebuffer_s=0.5,
            batch_profile="realtime",
        )

        await manager.get_backend(cfg_a)
        assert manager.current_vram_mb == 6000

        # Loading b (6000MB) would exceed 10000MB budget, so a must be evicted
        await manager.get_backend(cfg_b)
        assert b1._loaded is False  # evicted
        assert b2._loaded is True
        assert manager.current_vram_mb == 6000
