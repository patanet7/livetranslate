"""BackendManager — VRAM-aware backend lifecycle with LRU eviction."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Callable

from livetranslate_common.logging import get_logger
from livetranslate_common.models import BackendConfig

from backends.base import TranscriptionBackend

logger = get_logger()


class BackendManager:
    def __init__(self, max_vram_mb: int = 10000):
        self.max_vram_mb = max_vram_mb
        self.loaded_backends: OrderedDict[str, TranscriptionBackend] = OrderedDict()
        self._factories: dict[str, Callable[[BackendConfig], TranscriptionBackend]] = {}
        self._load_lock = asyncio.Lock()

    def register_factory(
        self, backend_name: str, factory: Callable[[BackendConfig], TranscriptionBackend]
    ) -> None:
        self._factories[backend_name] = factory

    @property
    def current_vram_mb(self) -> int:
        return sum(b.vram_usage_mb() for b in self.loaded_backends.values())

    def _backend_key(self, config: BackendConfig) -> str:
        return f"{config.backend}:{config.model}"

    async def get_backend(self, config: BackendConfig) -> TranscriptionBackend:
        """Return a loaded backend for the given config, evicting LRU if needed."""
        key = self._backend_key(config)

        if key in self.loaded_backends:
            self.loaded_backends.move_to_end(key)
            return self.loaded_backends[key]

        async with self._load_lock:
            if key in self.loaded_backends:
                self.loaded_backends.move_to_end(key)
                return self.loaded_backends[key]

            factory = self._factories.get(config.backend)
            if factory is None:
                raise ValueError(f"No factory registered for backend '{config.backend}'")

            backend = factory(config)
            needed = backend.get_model_info().vram_mb
            while self.current_vram_mb + needed > self.max_vram_mb and self.loaded_backends:
                await self._evict_lru()

            await backend.load_model(config.model)
            await backend.warmup()
            self.loaded_backends[key] = backend

            logger.info(
                "backend_loaded",
                backend=config.backend,
                model=config.model,
                vram_mb=backend.vram_usage_mb(),
                total_vram_mb=self.current_vram_mb,
            )
            return backend

    async def _evict_lru(self) -> None:
        if not self.loaded_backends:
            return
        key, backend = self.loaded_backends.popitem(last=False)
        logger.info("backend_evicted", key=key, freed_mb=backend.vram_usage_mb())
        await backend.unload_model()
