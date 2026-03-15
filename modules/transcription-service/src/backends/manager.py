"""BackendManager — VRAM-aware backend lifecycle with LRU eviction + circuit breaker."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable

from livetranslate_common.logging import get_logger
from livetranslate_common.models import BackendConfig

from backends.base import TranscriptionBackend

logger = get_logger()


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's circuit breaker is open (too many consecutive failures)."""


class CircuitBreaker:
    """Simple circuit breaker: opens after N consecutive failures, probes after cooldown."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._consecutive_failures: int = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return False  # allow probe
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            logger.warning(
                "circuit_breaker_opened",
                failures=self._consecutive_failures,
                cooldown_s=self.cooldown_s,
            )


class BackendManager:
    def __init__(self, max_vram_mb: int = 10000):
        self.max_vram_mb = max_vram_mb
        self.loaded_backends: OrderedDict[str, TranscriptionBackend] = OrderedDict()
        self._factories: dict[str, Callable[[BackendConfig], TranscriptionBackend]] = {}
        self._load_lock = asyncio.Lock()
        self._ref_counts: dict[str, int] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def register_factory(
        self, backend_name: str, factory: Callable[[BackendConfig], TranscriptionBackend]
    ) -> None:
        self._factories[backend_name] = factory

    @property
    def current_vram_mb(self) -> int:
        return sum(b.vram_usage_mb() for b in self.loaded_backends.values())

    def _backend_key(self, config: BackendConfig) -> str:
        return f"{config.backend}:{config.model}"

    def get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a backend key."""
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker()
        return self._circuit_breakers[key]

    def record_success(self, config: BackendConfig) -> None:
        """Record a successful inference for circuit breaker tracking."""
        key = self._backend_key(config)
        self.get_circuit_breaker(key).record_success()

    def record_failure(self, config: BackendConfig) -> None:
        """Record a failed inference for circuit breaker tracking."""
        key = self._backend_key(config)
        self.get_circuit_breaker(key).record_failure()

    async def get_backend(self, config: BackendConfig) -> TranscriptionBackend:
        """Return a loaded backend for the given config, evicting LRU if needed.

        Raises BackendUnavailableError if the backend's circuit breaker is open.
        """
        key = self._backend_key(config)
        cb = self.get_circuit_breaker(key)
        if cb.is_open:
            raise BackendUnavailableError(
                f"Backend {key} circuit breaker is open — "
                f"will probe again in {cb.cooldown_s}s"
            )

        async with self._load_lock:
            if key in self.loaded_backends:
                self.loaded_backends.move_to_end(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                return self.loaded_backends[key]

            factory = self._factories.get(config.backend)
            if factory is None:
                raise ValueError(f"No factory registered for backend '{config.backend}'")

            backend = factory(config)
            needed = backend.get_model_info().vram_mb
            while self.current_vram_mb + needed > self.max_vram_mb and self.loaded_backends:
                await self._evict_lru()

            device = getattr(config, "device", "auto") or "auto"
            await backend.load_model(config.model, device=device)
            await backend.warmup()
            self.loaded_backends[key] = backend
            self._ref_counts[key] = 1

            logger.info(
                "backend_loaded",
                backend=config.backend,
                model=config.model,
                vram_mb=backend.vram_usage_mb(),
                total_vram_mb=self.current_vram_mb,
            )
            return backend

    def release_backend(self, key: str) -> None:
        """Decrement the reference count for a backend key.

        Call this when a session finishes using a backend so it becomes
        eligible for LRU eviction again.
        """
        if key in self._ref_counts and self._ref_counts[key] > 0:
            self._ref_counts[key] -= 1
            logger.info("backend_released", key=key, ref_count=self._ref_counts[key])

    async def _evict_lru(self) -> None:
        if not self.loaded_backends:
            return
        # Skip in-use backends (ref_count > 0), evict the oldest unused one
        for key in list(self.loaded_backends.keys()):
            if self._ref_counts.get(key, 0) > 0:
                continue
            backend = self.loaded_backends.pop(key)
            self._ref_counts.pop(key, None)
            logger.info("backend_evicted", key=key, freed_mb=backend.vram_usage_mb())
            await backend.unload_model()
            return
        # All backends are in use — evict the absolute LRU as last resort
        key, backend = self.loaded_backends.popitem(last=False)
        self._ref_counts.pop(key, None)
        logger.warning("backend_evicted_in_use", key=key, freed_mb=backend.vram_usage_mb())
        await backend.unload_model()
