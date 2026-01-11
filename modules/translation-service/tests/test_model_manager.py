#!/usr/bin/env python3
"""
Unit tests for RuntimeModelManager - Dynamic Model Switching

Tests the ability to switch translation models at runtime without service restart.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_manager import (
    RuntimeModelManager,
    ModelInfo,
    get_model_manager,
    initialize_model_manager,
)


@pytest.fixture
def model_manager():
    """Create a fresh RuntimeModelManager for each test"""
    return RuntimeModelManager()


@pytest.fixture
def mock_translator():
    """Create a mock translator"""
    translator = AsyncMock()
    translator.is_ready = True
    translator.initialize = AsyncMock(return_value=True)
    translator.close = AsyncMock()
    translator.get_available_models = AsyncMock(return_value=["model1", "model2"])
    return translator


class TestRuntimeModelManager:
    """Test RuntimeModelManager core functionality"""

    def test_init(self, model_manager):
        """Test RuntimeModelManager initialization"""
        assert model_manager.current_model is None
        assert model_manager.current_backend is None
        assert model_manager.current_translator is None
        assert len(model_manager.model_cache) == 0
        assert len(model_manager.model_info) == 0

    def test_backend_configs_loaded(self, model_manager):
        """Test that backend configs are loaded from environment"""
        assert "ollama" in model_manager._backend_configs
        assert "groq" in model_manager._backend_configs
        assert "vllm" in model_manager._backend_configs
        assert "openai" in model_manager._backend_configs

        # Check ollama config structure
        ollama_config = model_manager._backend_configs["ollama"]
        assert "base_url" in ollama_config
        assert "api_key" in ollama_config
        assert "timeout" in ollama_config

    def test_get_cache_key(self, model_manager):
        """Test cache key generation"""
        key = model_manager._get_cache_key("llama2:7b", "ollama")
        assert key == "ollama:llama2:7b"

        key = model_manager._get_cache_key("gpt-4", "openai")
        assert key == "openai:gpt-4"

    def test_create_config(self, model_manager):
        """Test OpenAI-compatible config creation"""
        config = model_manager._create_config("mistral:latest", "ollama")
        assert config.model == "mistral:latest"
        assert "ollama" in config.name
        assert config.base_url == "http://localhost:11434/v1"

    def test_create_config_unknown_backend(self, model_manager):
        """Test that unknown backend raises ValueError"""
        with pytest.raises(ValueError, match="Unknown backend"):
            model_manager._create_config("model", "unknown_backend")

    @pytest.mark.asyncio
    async def test_switch_model_success(self, model_manager, mock_translator):
        """Test successful model switch"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            success = await model_manager.switch_model("llama2:7b", "ollama")

        assert success is True
        assert model_manager.current_model == "llama2:7b"
        assert model_manager.current_backend == "ollama"
        assert model_manager.current_translator == mock_translator
        assert len(model_manager.model_cache) == 1
        assert "ollama:llama2:7b" in model_manager.model_cache

    @pytest.mark.asyncio
    async def test_switch_model_same_model(self, model_manager, mock_translator):
        """Test switching to the same model returns True immediately"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            # First switch
            await model_manager.switch_model("llama2:7b", "ollama")

            # Second switch to same model
            success = await model_manager.switch_model("llama2:7b", "ollama")

        assert success is True
        # Should only have one model in cache
        assert len(model_manager.model_cache) == 1

    @pytest.mark.asyncio
    async def test_switch_model_cached(self, model_manager, mock_translator):
        """Test switching to a cached model is instant"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            # Load first model
            await model_manager.switch_model("llama2:7b", "ollama")

            # Create second mock translator
            mock_translator2 = AsyncMock()
            mock_translator2.is_ready = True
            mock_translator2.initialize = AsyncMock(return_value=True)

        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator2):
            # Load second model
            await model_manager.switch_model("mistral:latest", "ollama")

        # Now both models are cached
        assert len(model_manager.model_cache) == 2

        # Switch back to first model (should use cache)
        success = await model_manager.switch_model("llama2:7b", "ollama")
        assert success is True
        assert model_manager.current_model == "llama2:7b"

    @pytest.mark.asyncio
    async def test_switch_model_failure(self, model_manager):
        """Test model switch failure handling"""
        mock_translator = AsyncMock()
        mock_translator.initialize = AsyncMock(return_value=False)

        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            success = await model_manager.switch_model("nonexistent", "ollama")

        assert success is False
        assert model_manager.current_model is None
        assert len(model_manager.model_cache) == 0

    @pytest.mark.asyncio
    async def test_preload_model(self, model_manager, mock_translator):
        """Test preloading a model without switching to it"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            success = await model_manager.preload_model("llama2:7b", "ollama")

        assert success is True
        # Model should be cached but not current
        assert "ollama:llama2:7b" in model_manager.model_cache
        assert model_manager.current_model is None

    @pytest.mark.asyncio
    async def test_preload_model_already_cached(self, model_manager, mock_translator):
        """Test preloading an already cached model"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.preload_model("llama2:7b", "ollama")
            success = await model_manager.preload_model("llama2:7b", "ollama")

        assert success is True
        # Should still only have one entry
        assert len(model_manager.model_cache) == 1

    @pytest.mark.asyncio
    async def test_get_current_translator(self, model_manager, mock_translator):
        """Test getting current translator"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.switch_model("llama2:7b", "ollama")
            translator = await model_manager.get_current_translator()

        assert translator == mock_translator

    @pytest.mark.asyncio
    async def test_get_current_translator_not_ready(self, model_manager):
        """Test getting translator when none is ready"""
        translator = await model_manager.get_current_translator()
        assert translator is None

    @pytest.mark.asyncio
    async def test_unload_model(self, model_manager, mock_translator):
        """Test unloading a cached model"""
        mock_translator2 = AsyncMock()
        mock_translator2.is_ready = True
        mock_translator2.initialize = AsyncMock(return_value=True)
        mock_translator2.close = AsyncMock()

        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.switch_model("llama2:7b", "ollama")

        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator2):
            await model_manager.preload_model("mistral:latest", "ollama")

        # Unload non-current model
        success = await model_manager.unload_model("mistral:latest", "ollama")
        assert success is True
        assert len(model_manager.model_cache) == 1

    @pytest.mark.asyncio
    async def test_unload_current_model_fails(self, model_manager, mock_translator):
        """Test that unloading current model fails"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.switch_model("llama2:7b", "ollama")

        success = await model_manager.unload_model("llama2:7b", "ollama")
        assert success is False
        # Model should still be cached
        assert len(model_manager.model_cache) == 1

    @pytest.mark.asyncio
    async def test_get_status(self, model_manager, mock_translator):
        """Test getting model manager status"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.switch_model("llama2:7b", "ollama")

        status = model_manager.get_status()

        assert status["current_model"] == "llama2:7b"
        assert status["current_backend"] == "ollama"
        assert status["is_ready"] is True
        assert status["cache_size"] == 1
        assert "ollama" in status["supported_backends"]
        assert len(status["cached_models"]) == 1

        cached_model = status["cached_models"][0]
        assert cached_model["model"] == "llama2:7b"
        assert cached_model["backend"] == "ollama"
        assert cached_model["is_current"] is True

    @pytest.mark.asyncio
    async def test_close(self, model_manager, mock_translator):
        """Test closing model manager"""
        with patch('model_manager.OpenAICompatibleTranslator', return_value=mock_translator):
            await model_manager.switch_model("llama2:7b", "ollama")

        await model_manager.close()

        assert len(model_manager.model_cache) == 0
        assert len(model_manager.model_info) == 0
        assert model_manager.current_model is None
        assert model_manager.current_translator is None
        mock_translator.close.assert_called_once()


class TestModelInfo:
    """Test ModelInfo dataclass"""

    def test_model_info_creation(self):
        """Test creating ModelInfo"""
        now = datetime.utcnow()
        info = ModelInfo(
            model_name="llama2:7b",
            backend="ollama",
            loaded_at=now,
            last_used=now,
        )

        assert info.model_name == "llama2:7b"
        assert info.backend == "ollama"
        assert info.request_count == 0
        assert info.error_count == 0


class TestGlobalFunctions:
    """Test global helper functions"""

    def test_get_model_manager_singleton(self):
        """Test that get_model_manager returns singleton"""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
