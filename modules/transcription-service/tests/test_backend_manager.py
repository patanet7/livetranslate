"""Tests for BackendManager VRAM budget enforcement, LRU eviction, and CircuitBreaker."""
import asyncio
import time

import numpy as np
import pytest

from backends.base import TranscriptionBackend
from backends.manager import BackendManager, BackendUnavailableError, CircuitBreaker
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

    @pytest.mark.asyncio
    async def test_ref_counting_increments_on_get(self, manager):
        """get_backend should increment ref count for the backend key."""
        backend = FakeBackend("test-model", vram_mb=3000, languages=["en"])
        config = BackendConfig(
            backend="fake", model="test-model", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        manager.register_factory("fake", lambda cfg: backend)

        await manager.get_backend(config)
        key = "fake:test-model"
        assert manager._ref_counts.get(key, 0) == 1

        # Second get increments to 2
        await manager.get_backend(config)
        assert manager._ref_counts.get(key, 0) == 2

    @pytest.mark.asyncio
    async def test_release_decrements_ref_count(self, manager):
        """release_backend should decrement the ref count."""
        backend = FakeBackend("test-model", vram_mb=3000, languages=["en"])
        config = BackendConfig(
            backend="fake", model="test-model", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        manager.register_factory("fake", lambda cfg: backend)

        await manager.get_backend(config)
        key = "fake:test-model"
        assert manager._ref_counts.get(key, 0) == 1

        manager.release_backend(key)
        assert manager._ref_counts.get(key, 0) == 0

    @pytest.mark.asyncio
    async def test_in_use_backend_skipped_during_eviction(self, manager):
        """LRU eviction should skip backends with ref_count > 0."""
        b1 = FakeBackend("model-a", vram_mb=5000, languages=["en"])
        b2 = FakeBackend("model-b", vram_mb=5000, languages=["zh"])
        b3 = FakeBackend("model-c", vram_mb=5000, languages=["fr"])

        manager.register_factory("fake_a", lambda cfg: b1)
        manager.register_factory("fake_b", lambda cfg: b2)
        manager.register_factory("fake_c", lambda cfg: b3)

        cfg_a = BackendConfig(
            backend="fake_a", model="model-a", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        cfg_b = BackendConfig(
            backend="fake_b", model="model-b", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        cfg_c = BackendConfig(
            backend="fake_c", model="model-c", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )

        # Budget is 10000MB. Load a (5000) then b (5000) = 10000MB used.
        await manager.get_backend(cfg_a)  # ref_count[a] = 1
        await manager.get_backend(cfg_b)  # ref_count[b] = 1

        # a is LRU but still in use (ref_count=1), so loading c should evict b instead
        # First release b so it can be evicted, keep a in use
        manager.release_backend("fake_b:model-b")  # ref_count[b] = 0

        await manager.get_backend(cfg_c)  # should evict b (oldest unreferenced)
        assert b2._loaded is False  # b evicted
        assert b1._loaded is True   # a still loaded (was in use)


class TestCircuitBreaker:
    """Behavioral tests for the CircuitBreaker class."""

    def test_initially_closed(self):
        """A fresh circuit breaker must be closed."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
        assert cb.is_open is False

    def test_stays_closed_below_threshold(self):
        """Two consecutive failures with threshold=3 must leave the breaker closed."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

    def test_opens_after_exactly_threshold_failures(self):
        """The breaker must open on the third (threshold) consecutive failure."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True

    def test_remains_open_during_cooldown(self):
        """The breaker must stay open until cooldown elapses."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        # Not enough time has passed — still open
        assert cb.is_open is True

    def test_is_open_returns_false_after_cooldown_elapses(self):
        """is_open must return False once cooldown_s has elapsed (probe window)."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=0.05)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.1)
        assert cb.is_open is False

    def test_record_success_resets_failure_count(self):
        """record_success must reset consecutive failure count to zero."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Two more failures should not open the breaker (count was reset)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

    def test_record_success_clears_opened_at(self):
        """record_success after cooldown must clear _opened_at so the breaker closes."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=0.05)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        # Probe window: is_open returns False
        assert cb.is_open is False
        cb.record_success()
        # After success _opened_at must be cleared
        assert cb._opened_at is None
        assert cb._consecutive_failures == 0

    def test_backend_manager_auto_creates_circuit_breaker_for_new_key(self):
        """get_circuit_breaker must create a fresh breaker for an unseen key."""
        manager = BackendManager(max_vram_mb=10000)
        key = "fake:model-x"
        assert key not in manager._circuit_breakers
        cb = manager.get_circuit_breaker(key)
        assert key in manager._circuit_breakers
        assert isinstance(cb, CircuitBreaker)
        assert cb.is_open is False

    def test_backend_manager_returns_same_breaker_for_same_key(self):
        """get_circuit_breaker must return the identical object on repeated calls."""
        manager = BackendManager(max_vram_mb=10000)
        key = "fake:model-y"
        cb1 = manager.get_circuit_breaker(key)
        cb2 = manager.get_circuit_breaker(key)
        assert cb1 is cb2

    @pytest.mark.asyncio
    async def test_get_backend_raises_when_circuit_open(self):
        """get_backend must raise BackendUnavailableError when the circuit is open."""
        manager = BackendManager(max_vram_mb=10000)
        config = BackendConfig(
            backend="fake", model="test-model", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        # Open the breaker manually via record_failure three times
        manager.record_failure(config)
        manager.record_failure(config)
        manager.record_failure(config)
        with pytest.raises(BackendUnavailableError):
            await manager.get_backend(config)

    def test_recovery_after_cooldown_record_success_closes_circuit(self):
        """After cooldown, record_success must fully close the circuit breaker."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_s=0.05)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.1)
        # Probe window — allowed through (is_open is False)
        assert cb.is_open is False
        cb.record_success()
        # Breaker fully reset — three new failures required to reopen
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False
