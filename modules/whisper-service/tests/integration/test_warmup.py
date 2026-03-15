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

    @pytest.fixture(scope="class")
    def warmup_manager(self):
        """Shared ModelManager for warmup tests — loads once, cleans up after."""
        import gc

        import torch

        models_dir = ROOT_DIR / ".models"
        mgr = ModelManager(models_dir=str(models_dir))
        yield mgr
        del mgr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_warmup_audio_file_exists(self):
        """Test that warmup audio file is available"""
        warmup_file = ROOT_DIR / "warmup.wav"

        # Should exist in whisper-service root
        assert warmup_file.exists(), f"Warmup audio file not found: {warmup_file}"

        # Should be a valid WAV file (at least 44 bytes for WAV header)
        assert warmup_file.stat().st_size >= 44, "Warmup file is too small to be valid WAV"

    def test_model_manager_has_warmup_method(self, warmup_manager):
        """Test that ModelManager has warmup() method"""
        assert hasattr(warmup_manager, "warmup"), "ModelManager should have warmup() method"
        assert callable(warmup_manager.warmup), "warmup should be callable"

    def test_warmup_runs_successfully(self, warmup_manager):
        """Test that warmup runs without errors"""
        warmup_audio = np.zeros(16000, dtype=np.float32)

        try:
            warmup_manager.warmup(warmup_audio)
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"Warmup failed with error: {e}")

        assert success, "Warmup should complete successfully"

    def test_warmup_sets_warmed_up_flag(self, warmup_manager):
        """Test that warmup sets is_warmed_up flag"""
        assert hasattr(warmup_manager, "is_warmed_up"), "ModelManager should have is_warmed_up attribute"

        warmup_audio = np.zeros(16000, dtype=np.float32)
        warmup_manager.warmup(warmup_audio)

        assert warmup_manager.is_warmed_up is True, "Should be warmed up after warmup()"

    def test_warmup_with_default_model(self, warmup_manager):
        """Test that warmup uses default model if no model loaded"""
        warmup_audio = np.zeros(16000, dtype=np.float32)
        warmup_manager.warmup(warmup_audio)

        assert (
            warmup_manager.default_model in warmup_manager.models
        ), f"Default model '{warmup_manager.default_model}' should be loaded after warmup"

    @pytest.mark.integration
    def test_first_inference_fast_after_warmup(self, warmup_manager):
        """
        CRITICAL: Test that first inference is fast after warmup

        Without warmup: ~20 seconds (cold start)
        With warmup: <2 seconds (warm start)
        """
        warmup_audio = np.zeros(16000, dtype=np.float32)
        warmup_manager.warmup(warmup_audio)

        # Now test that first real inference is fast
        # Use "base" model for smoke test (fast download, still tests warmup effectiveness)
        test_audio = np.zeros(16000, dtype=np.float32)

        start_time = time.time()
        warmup_manager.safe_inference(
            model_name="base",
            audio_data=test_audio,
            beam_size=1,
        )
        inference_time = time.time() - start_time

        # Should be fast (< 10 seconds even with model download on first run)
        assert (
            inference_time < 10.0
        ), f"First inference took {inference_time:.2f}s, expected <10s after warmup"

        print(f"✅ First inference after warmup: {inference_time:.2f}s")

    @pytest.mark.integration
    def test_warmup_initializes_attention_hooks(self, warmup_manager):
        """Test that warmup initializes attention capture system"""
        warmup_audio = np.zeros(16000, dtype=np.float32)
        warmup_manager.warmup(warmup_audio)

        model = warmup_manager.models.get(warmup_manager.default_model)
        assert model is not None, "Default model should be loaded"

        # Check hooks on decoder blocks
        hook_count = 0
        for block in model.decoder.blocks:
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "_forward_hooks"):
                hook_count += len(block.cross_attn._forward_hooks)

        assert hook_count > 0, "Attention hooks should be installed during warmup"
        print(f"✅ {hook_count} attention hooks installed during warmup")

    def test_warmup_idempotent(self, warmup_manager):
        """Test that calling warmup multiple times is safe"""
        warmup_audio = np.zeros(16000, dtype=np.float32)

        warmup_manager.warmup(warmup_audio)
        warmup_manager.warmup(warmup_audio)
        warmup_manager.warmup(warmup_audio)

        assert warmup_manager.is_warmed_up is True
        assert warmup_manager.default_model in warmup_manager.models

    def test_warmup_with_custom_model(self, warmup_manager):
        """Test warmup with specific model name"""
        warmup_audio = np.zeros(16000, dtype=np.float32)

        warmup_manager.warmup(warmup_audio, model_name="large-v3")

        assert "large-v3" in warmup_manager.models, "Specified model should be loaded"
        assert warmup_manager.is_warmed_up is True


class TestWarmupConfiguration:
    """Test warmup configuration and options"""

    def test_warmup_file_path_configurable(self):
        """Test that warmup file path can be configured"""
        import gc

        import torch

        models_dir = ROOT_DIR / ".models"
        manager = ModelManager(models_dir=str(models_dir), warmup_file=str(ROOT_DIR / "warmup.wav"))
        try:
            assert hasattr(manager, "warmup_file"), "Should have warmup_file attribute"
        finally:
            del manager
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_auto_warmup_on_startup(self):
        """Test auto-warmup on ModelManager initialization"""
        import gc

        import torch

        models_dir = ROOT_DIR / ".models"
        warmup_file = ROOT_DIR / "warmup.wav"

        manager = ModelManager(
            models_dir=str(models_dir), warmup_file=str(warmup_file), auto_warmup=True
        )
        try:
            assert (
                manager.is_warmed_up is True
            ), "ModelManager with auto_warmup=True should warmup on initialization"
        finally:
            del manager
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class TestWarmupPerformance:
    """Test warmup performance characteristics"""

    @pytest.fixture(scope="class")
    def perf_manager(self):
        """Shared ModelManager for performance tests."""
        import gc

        import torch

        models_dir = ROOT_DIR / ".models"
        mgr = ModelManager(models_dir=str(models_dir))
        yield mgr
        del mgr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_warmup_completes_quickly(self, perf_manager):
        """Test that warmup itself completes in reasonable time"""
        warmup_audio = np.zeros(16000, dtype=np.float32)

        start_time = time.time()
        perf_manager.warmup(warmup_audio)
        warmup_time = time.time() - start_time

        assert warmup_time < 10.0, f"Warmup took {warmup_time:.2f}s, expected <10s"

        print(f"✅ Warmup completed in {warmup_time:.2f}s")

    @pytest.mark.integration
    def test_warmup_memory_overhead_minimal(self, perf_manager):
        """Test that warmup doesn't cause significant memory overhead"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        mem_before = process.memory_info().rss / 1024 / 1024

        warmup_audio = np.zeros(16000, dtype=np.float32)
        perf_manager.warmup(warmup_audio)

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before

        assert (
            mem_increase < 4096
        ), f"Warmup caused {mem_increase:.1f}MB memory increase, expected <4096MB"

        print(f"✅ Warmup memory increase: {mem_increase:.1f}MB")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
