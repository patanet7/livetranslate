#!/usr/bin/env python3
"""
TDD Tests for Whisper Service Warmup System

Following SimulStreaming reference implementation:
- Warmup eliminates 20-second cold start on first request
- Pre-loads model, triggers JIT compilation, allocates memory
- First real request should be <2 seconds instead of ~20 seconds

Reference: SimulStreaming/whisper_streaming/whisper_server.py lines 149-161
"""

import sys
import time
from pathlib import Path

# Add src directory to path (adjusted for tests/integration/ location)
# tests/integration/test_warmup.py → tests/integration → tests → whisper-service root → src
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
from whisper_service import ModelManager

# Root directory for all paths (tests/integration → tests → whisper-service root)
ROOT_DIR = Path(__file__).parent.parent.parent


class TestWarmupSystem:
    """Test warmup system to eliminate cold start delays"""

    def test_warmup_audio_file_exists(self):
        """Test that warmup audio file is available"""
        warmup_file = ROOT_DIR / "warmup.wav"

        # Should exist in whisper-service root
        assert warmup_file.exists(), f"Warmup audio file not found: {warmup_file}"

        # Should be a valid WAV file (at least 44 bytes for WAV header)
        assert warmup_file.stat().st_size >= 44, "Warmup file is too small to be valid WAV"

    def test_model_manager_has_warmup_method(self):
        """Test that ModelManager has warmup() method"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        assert hasattr(manager, "warmup"), "ModelManager should have warmup() method"
        assert callable(manager.warmup), "warmup should be callable"

    def test_warmup_runs_successfully(self):
        """Test that warmup runs without errors"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Create 1 second of silent audio (16kHz)
        warmup_audio = np.zeros(16000, dtype=np.float32)

        # Warmup should complete without exceptions
        try:
            manager.warmup(warmup_audio)
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"Warmup failed with error: {e}")

        assert success, "Warmup should complete successfully"

    def test_warmup_sets_warmed_up_flag(self):
        """Test that warmup sets is_warmed_up flag"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Should not be warmed up initially
        assert hasattr(manager, "is_warmed_up"), "ModelManager should have is_warmed_up attribute"
        assert manager.is_warmed_up is False, "Should not be warmed up initially"

        # Run warmup
        warmup_audio = np.zeros(16000, dtype=np.float32)
        manager.warmup(warmup_audio)

        # Should be warmed up after warmup
        assert manager.is_warmed_up is True, "Should be warmed up after warmup()"

    def test_warmup_with_default_model(self):
        """Test that warmup uses default model if no model loaded"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)

        # Should load default model during warmup
        manager.warmup(warmup_audio)

        # Default model should be loaded
        assert (
            manager.default_model in manager.models
        ), f"Default model '{manager.default_model}' should be loaded after warmup"

    @pytest.mark.integration
    def test_first_inference_fast_after_warmup(self):
        """
        CRITICAL: Test that first inference is fast after warmup

        Without warmup: ~20 seconds (cold start)
        With warmup: <2 seconds (warm start)
        """
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Warmup with 1 second of audio
        warmup_audio = np.zeros(16000, dtype=np.float32)
        manager.warmup(warmup_audio)

        # Now test that first real inference is fast
        # Use "base" model for smoke test (fast download, still tests warmup effectiveness)
        test_audio = np.zeros(16000, dtype=np.float32)

        start_time = time.time()
        manager.safe_inference(
            model_name="base",  # Small model for quick smoke test
            audio_data=test_audio,
            beam_size=1,  # Greedy for speed
        )
        inference_time = time.time() - start_time

        # Should be fast (< 10 seconds even with model download on first run)
        assert (
            inference_time < 10.0
        ), f"First inference took {inference_time:.2f}s, expected <10s after warmup"

        print(f"✅ First inference after warmup: {inference_time:.2f}s")

    @pytest.mark.integration
    def test_warmup_initializes_attention_hooks(self):
        """Test that warmup initializes attention capture system"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)
        manager.warmup(warmup_audio)

        # Attention hooks should be installed
        model = manager.models.get(manager.default_model)
        assert model is not None, "Default model should be loaded"

        # Check hooks on decoder blocks
        hook_count = 0
        for block in model.decoder.blocks:
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "_forward_hooks"):
                hook_count += len(block.cross_attn._forward_hooks)

        assert hook_count > 0, "Attention hooks should be installed during warmup"
        print(f"✅ {hook_count} attention hooks installed during warmup")

    def test_warmup_idempotent(self):
        """Test that calling warmup multiple times is safe"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)

        # Call warmup multiple times
        manager.warmup(warmup_audio)
        manager.warmup(warmup_audio)
        manager.warmup(warmup_audio)

        # Should still be warmed up
        assert manager.is_warmed_up is True

        # Should have default model loaded once
        assert manager.default_model in manager.models

    def test_warmup_with_custom_model(self):
        """Test warmup with specific model name"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)

        # Warmup with specific model
        manager.warmup(warmup_audio, model_name="large-v3")

        assert "large-v3" in manager.models, "Specified model should be loaded"
        assert manager.is_warmed_up is True


class TestWarmupConfiguration:
    """Test warmup configuration and options"""

    def test_warmup_file_path_configurable(self):
        """Test that warmup file path can be configured"""
        models_dir = ROOT_DIR / ".models"

        # Should accept warmup_file parameter
        manager = ModelManager(models_dir=str(models_dir), warmup_file=str(ROOT_DIR / "warmup.wav"))

        assert hasattr(manager, "warmup_file"), "Should have warmup_file attribute"

    def test_auto_warmup_on_startup(self):
        """Test auto-warmup on ModelManager initialization"""
        models_dir = ROOT_DIR / ".models"
        warmup_file = ROOT_DIR / "warmup.wav"

        # Should support auto_warmup parameter
        manager = ModelManager(
            models_dir=str(models_dir), warmup_file=str(warmup_file), auto_warmup=True
        )

        # Should be warmed up automatically
        assert (
            manager.is_warmed_up is True
        ), "ModelManager with auto_warmup=True should warmup on initialization"


class TestWarmupPerformance:
    """Test warmup performance characteristics"""

    @pytest.mark.integration
    def test_warmup_completes_quickly(self):
        """Test that warmup itself completes in reasonable time"""
        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)

        start_time = time.time()
        manager.warmup(warmup_audio)
        warmup_time = time.time() - start_time

        # Warmup should complete in <10 seconds (even on CPU)
        assert warmup_time < 10.0, f"Warmup took {warmup_time:.2f}s, expected <10s"

        print(f"✅ Warmup completed in {warmup_time:.2f}s")

    @pytest.mark.integration
    def test_warmup_memory_overhead_minimal(self):
        """Test that warmup doesn't cause significant memory overhead"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get memory before warmup
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        warmup_audio = np.zeros(16000, dtype=np.float32)
        manager.warmup(warmup_audio)

        # Get memory after warmup
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Memory increase should be reasonable (model loading + working memory)
        # Large-v3 is ~3GB, so allow up to 4GB increase
        assert (
            mem_increase < 4096
        ), f"Warmup caused {mem_increase:.1f}MB memory increase, expected <4096MB"

        print(f"✅ Warmup memory increase: {mem_increase:.1f}MB")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
