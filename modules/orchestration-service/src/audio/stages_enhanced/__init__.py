#!/usr/bin/env python3
"""
Enhanced Audio Processing Stages - Phase 1 Implementation

This package contains enhanced audio processing stages that use industry-standard
libraries to replace custom DSP implementations. These stages are drop-in
replacements for the original stages with improved quality and reliability.

Phase 1 (Foundation Libraries):
- LUFSNormalizationStage - Using pyloudnorm (ITU-R BS.1770-4 compliant)
- CompressionStage - Using pedalboard (Spotify's audio library)
- LimiterStage - Using pedalboard
- EqualizerStage - Using pedalboard (planned)
- VADStage - Using webrtcvad (planned)

All enhanced stages maintain the same interface as the original stages,
enabling seamless A/B testing and gradual migration.
"""

from typing import TYPE_CHECKING
import importlib.util

# Version info
__version__ = "1.0.0"

# Check library availability WITHOUT importing the heavy classes
# This uses importlib.util.find_spec which checks if a module exists
# without actually loading it into memory


def _check_library_available(lib_name):
    """Check if a library is available without importing it."""
    try:
        spec = importlib.util.find_spec(lib_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# Check library availability (lightweight check - doesn't load libraries)
HAS_PYLOUDNORM = _check_library_available("pyloudnorm")
HAS_PEDALBOARD = _check_library_available("pedalboard")
HAS_WEBRTCVAD = _check_library_available("webrtcvad")

# Feature availability flags (based on library availability)
AVAILABLE_FEATURES = {
    "lufs_normalization": HAS_PYLOUDNORM,
    "compression": HAS_PEDALBOARD,
    "limiter": HAS_PEDALBOARD,
    "equalizer": False,  # TODO: Implement with pedalboard
    "vad": HAS_WEBRTCVAD,
}

PHASE_1_COMPLETE = HAS_PYLOUDNORM and HAS_PEDALBOARD

# Lazy imports - only import classes when actually used
# This prevents loading heavy libraries just to check availability
__all__ = []


def __getattr__(name):
    """Lazy import of enhanced stage classes."""
    if name == "LUFSNormalizationStageEnhanced":
        if not HAS_PYLOUDNORM:
            raise ImportError("pyloudnorm is required. Install with: poetry install")
        from .lufs_normalization_enhanced import LUFSNormalizationStageEnhanced

        return LUFSNormalizationStageEnhanced

    elif name == "CompressionStageEnhanced":
        if not HAS_PEDALBOARD:
            raise ImportError("pedalboard is required. Install with: poetry install")
        from .compression_enhanced import CompressionStageEnhanced

        return CompressionStageEnhanced

    elif name == "LimiterStageEnhanced":
        if not HAS_PEDALBOARD:
            raise ImportError("pedalboard is required. Install with: poetry install")
        from .limiter_enhanced import LimiterStageEnhanced

        return LimiterStageEnhanced

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
