"""
Shared fixtures for integration tests.

Provides session-scoped ML model fixtures to prevent OOM on Apple Silicon.
Models are loaded ONCE per test session and shared across all integration tests.

Rules:
- All fixtures use yield + teardown with gc.collect() (project hard rule)
- Models are explicitly deleted and caches emptied in teardown
- torch.mps.empty_cache() called on Apple Silicon to release MPS allocator pool
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
# Add _legacy so whisper_service imports resolve
_LEGACY_DIR = SRC_DIR / "_legacy"
if _LEGACY_DIR.exists():
    sys.path.insert(1, str(_LEGACY_DIR))

MODELS_DIR = Path(__file__).parent.parent / ".models"


def _empty_device_cache():
    """Flush both CUDA and MPS caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Session-scoped model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def shared_vad_model():
    """Load Silero VAD model ONCE for the entire integration test session.

    ~30MB on Apple Silicon.  Shared across all tests that need VAD.
    """
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    model.eval()

    yield model

    del model
    _empty_device_cache()
    gc.collect()


@pytest.fixture(scope="session")
def shared_model_manager():
    """Create a single ModelManager with Whisper large-v3 loaded.

    ~3GB on Apple Silicon.  Every integration test that needs Whisper
    inference should request this fixture instead of creating its own
    ModelManager.

    The manager caches the model in ``self.models["large-v3"]``.
    Tests can call ``manager.load_model("large-v3")`` safely -- it returns
    the cached instance without re-loading.
    """
    from whisper_service import ModelManager

    manager = ModelManager(models_dir=str(MODELS_DIR))
    manager.load_model("large-v3")

    yield manager

    # Teardown: drop the manager and its cached model(s)
    del manager
    _empty_device_cache()
    gc.collect()


@pytest.fixture(scope="session")
def shared_whisper_model(shared_model_manager):
    """Convenience: return the loaded Whisper model object directly.

    Equivalent to ``shared_model_manager.models["large-v3"]``.
    """
    return shared_model_manager.models["large-v3"]


# ---------------------------------------------------------------------------
# Per-test cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _gc_after_each_integration_test():
    """Lightweight GC + MPS cache flush after every integration test.

    This ensures any transient tensors created during a test body are
    released before the next test starts, preventing accumulation.
    """
    yield
    gc.collect()
    _empty_device_cache()
