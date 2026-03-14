"""ModelRegistry — YAML-based language→backend+model routing.

The registry is the single source of truth for "what model handles what
language with what parameters." Changeable without code changes.
Supports hot-reload via reload() or POST /api/registry/reload.
"""
from __future__ import annotations

import signal
from pathlib import Path

import yaml
from livetranslate_common.logging import get_logger
from livetranslate_common.models import BackendConfig

logger = get_logger()


class ModelRegistry:
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._data: dict = {}
        self._routing: dict[str, BackendConfig] = {}
        self.version: int = 0
        self.vram_budget_mb: int = 10000
        self._load()

    def _load(self) -> None:
        raw = yaml.safe_load(self._config_path.read_text())
        version = raw.get("version", 1)
        vram_budget_mb = raw.get("vram_budget_mb", 10000)

        routing: dict[str, BackendConfig] = {}
        for lang, entry in raw.get("language_routing", {}).items():
            routing[lang] = BackendConfig.model_validate(entry)

        # Only commit if all validation succeeded
        self._data = raw
        self.version = version
        self.vram_budget_mb = vram_budget_mb
        self._routing = routing

        logger.info(
            "registry_loaded",
            version=self.version,
            languages=list(self._routing.keys()),
            path=str(self._config_path),
        )

    def reload(self) -> None:
        try:
            self._load()
        except Exception:
            logger.exception("registry_reload_failed", path=str(self._config_path))

    def get_config(self, language: str) -> BackendConfig:
        """Get the BackendConfig for a language, falling back to '*'."""
        if language in self._routing:
            return self._routing[language]
        if "*" in self._routing:
            return self._routing["*"]
        raise KeyError(f"No registry entry for language '{language}' and no wildcard '*' fallback")

    def get_backend_module(self, backend_name: str) -> dict:
        """Return the module/class info for a backend name."""
        backends = self._data.get("backends", {})
        if backend_name not in backends:
            raise KeyError(f"Unknown backend: {backend_name}")
        return backends[backend_name]

    @property
    def all_languages(self) -> list[str]:
        return [k for k in self._routing if k != "*"]


def register_sighup_handler(registry: ModelRegistry) -> None:
    """Register SIGHUP handler for hot-reloading the registry from disk.

    Usage: send `kill -HUP <pid>` to reload without restarting the service.
    Also available via POST /api/registry/reload.
    """
    def _handler(signum, frame):
        logger.info("sighup_received", action="reloading_registry")
        registry.reload()

    signal.signal(signal.SIGHUP, _handler)
    logger.info("sighup_handler_registered")
