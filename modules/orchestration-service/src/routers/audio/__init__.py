"""
Audio Router Package

Modular audio processing router split from the original monolithic audio.py file.

Modules:
- _shared: Common imports, utilities, and shared components
- audio_core: Core processing endpoints (process, upload, health, models)
- audio_analysis: Analysis endpoints (FFT, LUFS, spectrum, quality)
- audio_stages: Stage processing endpoints (individual stages, pipeline)
- audio_presets: Preset management endpoints (CRUD operations, comparison)
"""

# Export individual modules for importing
from . import audio_core, audio_analysis, audio_stages, audio_presets

__all__ = ["audio_core", "audio_analysis", "audio_stages", "audio_presets"]