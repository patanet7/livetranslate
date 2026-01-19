#!/usr/bin/env python3
"""
Orchestration Response Formatter

Formats API responses for orchestration service integration.
Extracted from whisper_service.py for better modularity and testability.
"""

from typing import Any

from transcription import TranscriptionResult


def format_success_response(
    chunk_id: str,
    session_id: str,
    result: TranscriptionResult,
    processing_time: float,
    chunk_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Format a successful orchestration chunk processing response.

    Args:
        chunk_id: Unique identifier for the audio chunk
        session_id: Session identifier
        result: Transcription result from Whisper
        processing_time: Processing time in seconds
        chunk_metadata: Original chunk metadata

    Returns:
        Formatted response dict for orchestration API
    """
    return {
        "chunk_id": chunk_id,
        "session_id": session_id,
        "status": "success",
        "transcription": {
            "text": result.text,
            "language": result.language,
            "confidence_score": result.confidence_score,
            "segments": result.segments,
            "timestamp": result.timestamp,
        },
        "processing_info": {
            "model_used": result.model_used,
            "device_used": result.device_used,
            "processing_time": processing_time,
            "chunk_metadata": chunk_metadata,
            "service_mode": "orchestration",
        },
        "chunk_sequence": chunk_metadata.get("sequence_number", 0),
        "chunk_timing": {
            "start_time": chunk_metadata.get("start_time", 0.0),
            "end_time": chunk_metadata.get("end_time", 0.0),
            "duration": chunk_metadata.get("duration", 0.0),
            "overlap_start": chunk_metadata.get("overlap_start", 0.0),
            "overlap_end": chunk_metadata.get("overlap_end", 0.0),
        },
    }


def format_error_response(
    chunk_id: str,
    session_id: str,
    error: Exception,
    processing_time: float,
    chunk_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Format an error response for failed chunk processing.

    Args:
        chunk_id: Unique identifier for the audio chunk
        session_id: Session identifier
        error: Exception that occurred
        processing_time: Processing time in seconds
        chunk_metadata: Original chunk metadata

    Returns:
        Formatted error response dict for orchestration API
    """
    return {
        "chunk_id": chunk_id,
        "session_id": session_id,
        "status": "error",
        "error": str(error),
        "error_type": "orchestration_processing_error",
        "processing_time": processing_time,
        "chunk_metadata": chunk_metadata,
    }
