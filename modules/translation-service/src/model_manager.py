#!/usr/bin/env python3
"""
RuntimeModelManager - Dynamic Model Loading/Switching Without Service Restart

This module provides a manager for dynamically loading, switching, and managing
translation models at runtime. Key features:

- Switch between Ollama models without restart
- Model caching for fast switching between previously loaded models
- Preloading models in the background for instant switching
- Thread-safe operations with async locking
- Graceful fallback to previously working models on failure

Usage:
    manager = RuntimeModelManager()
    await manager.initialize_default()  # Load default model from env

    # Switch model at runtime
    success = await manager.switch_model("llama2:7b", "ollama")

    # Preload for faster switching
    await manager.preload_model("mistral:latest", "ollama")

    # Get current translator for translation
    translator = await manager.get_current_translator()
    result = await translator.translate(...)
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from livetranslate_common.logging import get_logger
from openai_compatible_translator import OpenAICompatibleConfig, OpenAICompatibleTranslator

logger = get_logger()


@dataclass
class ModelInfo:
    """Information about a loaded model"""

    model_name: str
    backend: str
    loaded_at: datetime
    last_used: datetime
    request_count: int = 0
    error_count: int = 0


class RuntimeModelManager:
    """
    Manages dynamic model loading/unloading without service restart.

    This manager maintains a cache of initialized translators and allows
    switching between them at runtime. It supports:

    - Ollama (local)
    - Groq (cloud)
    - vLLM (local/remote server)
    - OpenAI (cloud)
    - Any OpenAI-compatible endpoint
    """

    def __init__(self):
        self.current_model: str | None = None
        self.current_backend: str | None = None
        self.current_translator: OpenAICompatibleTranslator | None = None

        # Cache of initialized translators by "{backend}:{model}" key
        self.model_cache: dict[str, OpenAICompatibleTranslator] = {}
        self.model_info: dict[str, ModelInfo] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Configuration for backends
        self._backend_configs: dict[str, dict[str, Any]] = {}

        # Load backend configs from environment
        self._load_backend_configs()

        logger.info("RuntimeModelManager initialized")

    def _load_backend_configs(self):
        """Load backend configurations from environment variables"""
        # Ollama configuration
        self._backend_configs["ollama"] = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": "",  # Ollama doesn't require API key
            "timeout": float(os.getenv("OLLAMA_TIMEOUT", "60.0")),
        }

        # Groq configuration
        self._backend_configs["groq"] = {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "timeout": 30.0,
        }

        # vLLM configuration
        self._backend_configs["vllm"] = {
            "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": "",
            "timeout": 60.0,
        }

        # OpenAI configuration
        self._backend_configs["openai"] = {
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": 30.0,
        }

    def _get_cache_key(self, model_name: str, backend: str) -> str:
        """Generate cache key for model+backend combination"""
        return f"{backend}:{model_name}"

    def _create_config(self, model_name: str, backend: str) -> OpenAICompatibleConfig:
        """Create OpenAI-compatible config for a model and backend"""
        backend_config = self._backend_configs.get(backend, {})

        if not backend_config:
            raise ValueError(f"Unknown backend: {backend}")

        return OpenAICompatibleConfig(
            name=f"{backend}-{model_name}",
            base_url=backend_config["base_url"],
            model=model_name,
            api_key=backend_config.get("api_key", ""),
            timeout=backend_config.get("timeout", 60.0),
        )

    async def initialize_default(self) -> bool:
        """
        Initialize with default model from environment variables.

        Reads OLLAMA_MODEL and OLLAMA_BASE_URL from environment.
        """
        default_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
        default_backend = "ollama"

        logger.info(f"Initializing default model: {default_model} on {default_backend}")

        return await self.switch_model(default_model, default_backend)

    async def switch_model(self, model_name: str, backend: str = "ollama") -> bool:
        """
        Switch to a different model at runtime.

        Args:
            model_name: Name of the model (e.g., "llama2:7b", "mistral:latest")
            backend: Backend to use ("ollama", "groq", "vllm", "openai")

        Returns:
            True if switch successful, False otherwise
        """
        async with self._lock:
            cache_key = self._get_cache_key(model_name, backend)

            # Check if already using this model
            if self.current_model == model_name and self.current_backend == backend:
                logger.info(f"Already using model {model_name} on {backend}")
                return True

            # Check cache first (instant switch)
            if cache_key in self.model_cache:
                translator = self.model_cache[cache_key]

                # Verify translator is still ready
                if translator.is_ready:
                    self.current_translator = translator
                    self.current_model = model_name
                    self.current_backend = backend
                    self.model_info[cache_key].last_used = datetime.now(UTC)

                    logger.info(f"Switched to cached model: {model_name} on {backend}")
                    return True
                else:
                    # Remove stale entry
                    del self.model_cache[cache_key]
                    del self.model_info[cache_key]

            # Initialize new translator
            try:
                config = self._create_config(model_name, backend)
                new_translator = OpenAICompatibleTranslator(config)

                logger.info(f"Initializing new model: {model_name} on {backend}...")

                if await new_translator.initialize():
                    # Cache the translator
                    self.model_cache[cache_key] = new_translator
                    self.model_info[cache_key] = ModelInfo(
                        model_name=model_name,
                        backend=backend,
                        loaded_at=datetime.now(UTC),
                        last_used=datetime.now(UTC),
                    )

                    # Set as current
                    self.current_translator = new_translator
                    self.current_model = model_name
                    self.current_backend = backend

                    logger.info(f"Successfully switched to model: {model_name} on {backend}")
                    return True
                else:
                    logger.error(f"Failed to initialize model: {model_name} on {backend}")
                    return False

            except Exception as e:
                logger.error(f"Error switching to model {model_name} on {backend}: {e}")
                return False

    async def preload_model(self, model_name: str, backend: str = "ollama") -> bool:
        """
        Pre-load a model in background for faster switching later.

        This loads and caches the model but doesn't switch to it.

        Args:
            model_name: Name of the model to preload
            backend: Backend to use

        Returns:
            True if preload successful
        """
        cache_key = self._get_cache_key(model_name, backend)

        # Check if already cached
        if cache_key in self.model_cache and self.model_cache[cache_key].is_ready:
            logger.info(f"Model already cached: {model_name} on {backend}")
            return True

        try:
            config = self._create_config(model_name, backend)
            translator = OpenAICompatibleTranslator(config)

            logger.info(f"Preloading model: {model_name} on {backend}...")

            if await translator.initialize():
                async with self._lock:
                    self.model_cache[cache_key] = translator
                    self.model_info[cache_key] = ModelInfo(
                        model_name=model_name,
                        backend=backend,
                        loaded_at=datetime.now(UTC),
                        last_used=datetime.now(UTC),
                    )

                logger.info(f"Successfully preloaded model: {model_name} on {backend}")
                return True
            else:
                logger.error(f"Failed to preload model: {model_name} on {backend}")
                return False

        except Exception as e:
            logger.error(f"Error preloading model {model_name} on {backend}: {e}")
            return False

    async def get_current_translator(self) -> OpenAICompatibleTranslator | None:
        """Get currently active translator"""
        async with self._lock:
            if self.current_translator and self.current_translator.is_ready:
                # Update usage stats
                cache_key = self._get_cache_key(self.current_model, self.current_backend)
                if cache_key in self.model_info:
                    self.model_info[cache_key].request_count += 1
                    self.model_info[cache_key].last_used = datetime.now(UTC)
                return self.current_translator
            return None

    async def get_translator(
        self, model_name: str, backend: str = "ollama"
    ) -> OpenAICompatibleTranslator | None:
        """
        Get a specific translator, loading if necessary.

        This is useful when you want to use a specific model for a request
        without switching the default.
        """
        cache_key = self._get_cache_key(model_name, backend)

        # Check cache
        if cache_key in self.model_cache and self.model_cache[cache_key].is_ready:
            return self.model_cache[cache_key]

        # Load the model
        if await self.preload_model(model_name, backend):
            return self.model_cache.get(cache_key)

        return None

    async def unload_model(self, model_name: str, backend: str) -> bool:
        """
        Unload a model from cache to free resources.

        Cannot unload the currently active model.
        """
        cache_key = self._get_cache_key(model_name, backend)

        async with self._lock:
            if model_name == self.current_model and backend == self.current_backend:
                logger.warning("Cannot unload currently active model")
                return False

            if cache_key in self.model_cache:
                translator = self.model_cache[cache_key]
                await translator.close()
                del self.model_cache[cache_key]
                del self.model_info[cache_key]
                logger.info(f"Unloaded model: {model_name} on {backend}")
                return True

            return False

    async def get_available_models(self, backend: str = "ollama") -> list[str]:
        """
        Get list of available models from a backend.

        For Ollama, this queries the local Ollama instance.
        """
        # Try to use existing translator for this backend
        for cache_key, translator in self.model_cache.items():
            if cache_key.startswith(f"{backend}:") and translator.is_ready:
                return await translator.get_available_models()

        # Create temporary translator to query models
        try:
            config = self._create_config("temp", backend)
            temp_translator = OpenAICompatibleTranslator(config)
            if await temp_translator.initialize():
                models = await temp_translator.get_available_models()
                await temp_translator.close()
                return models
        except Exception as e:
            logger.error(f"Failed to get available models from {backend}: {e}")

        return []

    def get_status(self) -> dict[str, Any]:
        """Get current model manager status"""
        cached_models = []
        for _cache_key, info in self.model_info.items():
            cached_models.append(
                {
                    "model": info.model_name,
                    "backend": info.backend,
                    "loaded_at": info.loaded_at.isoformat(),
                    "last_used": info.last_used.isoformat(),
                    "request_count": info.request_count,
                    "error_count": info.error_count,
                    "is_current": (
                        info.model_name == self.current_model
                        and info.backend == self.current_backend
                    ),
                }
            )

        return {
            "current_model": self.current_model,
            "current_backend": self.current_backend,
            "is_ready": self.current_translator is not None and self.current_translator.is_ready,
            "cached_models": cached_models,
            "cache_size": len(self.model_cache),
            "supported_backends": list(self._backend_configs.keys()),
        }

    async def close(self):
        """Close all translators and cleanup"""
        async with self._lock:
            for cache_key, translator in self.model_cache.items():
                try:
                    await translator.close()
                except Exception as e:
                    logger.error(f"Error closing translator {cache_key}: {e}")

            self.model_cache.clear()
            self.model_info.clear()
            self.current_translator = None
            self.current_model = None
            self.current_backend = None

            logger.info("RuntimeModelManager closed")


# Global instance for use across the service
_global_model_manager: RuntimeModelManager | None = None


def get_model_manager() -> RuntimeModelManager:
    """Get or create the global RuntimeModelManager instance"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = RuntimeModelManager()
    return _global_model_manager


async def initialize_model_manager() -> RuntimeModelManager:
    """Initialize the global model manager with default settings"""
    manager = get_model_manager()
    await manager.initialize_default()
    return manager
