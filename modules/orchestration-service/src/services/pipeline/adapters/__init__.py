"""
Pipeline Adapters Module

Source-specific adapters that convert raw chunks to unified TranscriptChunk format.
Each adapter handles the nuances of its source API while presenting a consistent
interface to the TranscriptionPipelineCoordinator.
"""

from .base import ChunkAdapter, TranscriptChunk
from .fireflies_adapter import FirefliesChunkAdapter
from .google_meet_adapter import GoogleMeetChunkAdapter
from .import_adapter import ImportChunkAdapter
from .audio_adapter import AudioUploadChunkAdapter

__all__ = [
    "ChunkAdapter",
    "TranscriptChunk",
    "FirefliesChunkAdapter",
    "GoogleMeetChunkAdapter",
    "ImportChunkAdapter",
    "AudioUploadChunkAdapter",
]
