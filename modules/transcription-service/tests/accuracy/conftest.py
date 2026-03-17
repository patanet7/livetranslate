"""
Fixtures for accuracy tests.

Accuracy tests load SessionRestartTranscriber per test class via yield fixtures.
This conftest adds a safety-net GC pass after each test to release transient
tensors and prevent accumulation across classes.
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
def _gc_after_accuracy_test():
    """GC + MPS cache flush after every accuracy test."""
    yield
    gc.collect()
    _empty_device_cache()
