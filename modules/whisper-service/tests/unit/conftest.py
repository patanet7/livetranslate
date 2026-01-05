#!/usr/bin/env python3
"""
Global test configuration and fixtures for whisper-service unit tests.

Provides:
- Session-level cleanup for ML models
- Memory management fixtures
- Shared model fixtures (module-scoped)
"""

import pytest
import torch
import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    """
    Auto-cleanup after ALL tests complete.

    Ensures ML models and GPU memory are properly released.
    This runs automatically for every test session.
    """
    yield

    # Final cleanup after all tests
    logger.info("[TEST CLEANUP] Cleaning up ML models and GPU memory...")

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[TEST CLEANUP] ✓ Cleared CUDA cache")

    # Force garbage collection
    gc.collect()
    logger.info("[TEST CLEANUP] ✓ Garbage collection complete")
    logger.info("[TEST CLEANUP] Session cleanup finished")


@pytest.fixture(autouse=True)
def cleanup_after_each_test():
    """
    Cleanup after EACH individual test.

    Provides lightweight cleanup between tests to prevent memory accumulation.
    This runs automatically for every test function.
    """
    yield

    # Minor cleanup between tests
    gc.collect()


# Shared Model Fixtures (Module-scoped)
# These load models ONCE per module and share across tests

@pytest.fixture(scope="module")
def shared_vad_model():
    """
    Load Silero VAD model ONCE per test module.

    This prevents redundant model loading across test classes.
    All tests in test_vad.py share this single model instance.

    Returns:
        tuple: (model, utils) from torch.hub.load
    """
    logger.info("[FIXTURE] Loading shared Silero VAD model...")

    # Set custom cache directory to .models/silero-vad
    cache_dir = Path(__file__).parent.parent.parent / '.models' / 'silero-vad'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set torch hub directory to our custom cache
    torch.hub.set_dir(str(cache_dir.parent))

    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    model.eval()

    logger.info("[FIXTURE] ✓ Shared Silero VAD model loaded")

    yield model

    # Cleanup after module
    logger.info("[FIXTURE] Cleaning up shared Silero VAD model...")
    del model
    gc.collect()
    logger.info("[FIXTURE] ✓ Silero VAD model cleanup complete")


@pytest.fixture(scope="module")
def shared_whisper_manager():
    """
    Create shared ModelManager with large-v3 loaded ONCE per test module.

    This prevents loading the ~3GB model multiple times.
    Used for integration tests that require real inference.

    Each test can reset context and use different prompts via:
    - manager.rolling_context.text = ""
    - manager.static_prompt = "new prompt"
    - manager.init_context()

    Returns:
        ModelManager instance with large-v3 model loaded
    """
    from pathlib import Path
    import sys

    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from whisper_service import ModelManager

    logger.info("[FIXTURE] Loading shared Whisper ModelManager with large-v3...")

    models_dir = Path(__file__).parent.parent / ".models"
    manager = ModelManager(models_dir=str(models_dir))
    manager.init_context()
    model = manager.load_model("large-v3")

    logger.info("[FIXTURE] ✓ Shared ModelManager with large-v3 loaded")

    yield manager

    # Cleanup after module
    logger.info("[FIXTURE] Cleaning up shared ModelManager...")
    del manager

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    logger.info("[FIXTURE] ✓ ModelManager cleanup complete")


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests requiring real models"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress tests (60+ minutes)"
    )
