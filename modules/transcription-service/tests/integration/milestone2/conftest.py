"""
Fixtures for milestone2 integration tests.

Provides per-test GC cleanup to prevent OOM from inline SessionRestartTranscriber
loads that each bring in ~1.5GB of Whisper model weights.
"""

import gc

import pytest
import torch


def _empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


@pytest.fixture(autouse=True)
def _gc_after_milestone2_test():
    """Force aggressive cleanup after each milestone2 test.

    These tests create SessionRestartTranscriber inline (no shared fixture)
    because each test needs specific config. The GC + cache flush ensures
    the ~1.5GB model is released before the next test loads its own.
    """
    yield
    gc.collect()
    _empty_device_cache()
    gc.collect()  # Second pass to catch weak-ref cycles
