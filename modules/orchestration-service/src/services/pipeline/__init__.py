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
        AudioUploadChunkAdapter,
    )

    # Create coordinator with appropriate adapter
    coordinator = TranscriptionPipelineCoordinator(
        config=PipelineConfig(...),
        adapter=FirefliesChunkAdapter(),
        ...
    )
"""

from .adapters import (
    AudioUploadChunkAdapter,
    ChunkAdapter,
    FirefliesChunkAdapter,
    GoogleMeetChunkAdapter,
    ImportChunkAdapter,
    TranscriptChunk,
)
from .config import PipelineConfig, PipelineStats
from .coordinator import TranscriptionPipelineCoordinator

__all__ = [
    "AudioUploadChunkAdapter",
    # Adapters
    "ChunkAdapter",
    "FirefliesChunkAdapter",
    "GoogleMeetChunkAdapter",
    "ImportChunkAdapter",
    # Config
    "PipelineConfig",
    "PipelineStats",
    "TranscriptChunk",
    # Coordinator
    "TranscriptionPipelineCoordinator",
]
