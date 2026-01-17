"""
Generic Transcription Pipeline Module

Source-agnostic pipeline coordinator with adapters for different transcript sources.
All business logic (aggregation, context windows, glossary, captions) is DRY and shared.

Usage:
    from services.pipeline import (
        TranscriptionPipelineCoordinator,
        PipelineConfig,
        FirefliesChunkAdapter,
        GoogleMeetChunkAdapter,
    )

    # Create coordinator with appropriate adapter
    coordinator = TranscriptionPipelineCoordinator(
        config=PipelineConfig(...),
        adapter=FirefliesChunkAdapter(),
        ...
    )
"""

from .config import PipelineConfig, PipelineStats
from .coordinator import TranscriptionPipelineCoordinator
from .adapters import (
    ChunkAdapter,
    TranscriptChunk,
    FirefliesChunkAdapter,
    GoogleMeetChunkAdapter,
    ImportChunkAdapter,
)

__all__ = [
    # Config
    "PipelineConfig",
    "PipelineStats",
    # Coordinator
    "TranscriptionPipelineCoordinator",
    # Adapters
    "ChunkAdapter",
    "TranscriptChunk",
    "FirefliesChunkAdapter",
    "GoogleMeetChunkAdapter",
    "ImportChunkAdapter",
]
