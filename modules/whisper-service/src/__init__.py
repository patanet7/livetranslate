"""
Whisper Service Module

This module provides NPU-optimized speech-to-text transcription with real-time streaming 
capabilities. Extracted from the existing whisper-npu-server with enhanced modular architecture.

Key Components:
- WhisperService: Main service class with NPU/GPU/CPU acceleration
- ModelManager: Model loading and device management with fallback support
- AudioBufferManager: Rolling buffers and Voice Activity Detection
- SessionManager: Session persistence and transcription history
- API Server: Flask-based REST API with WebSocket streaming

Features:
- NPU/GPU/CPU acceleration with automatic fallback
- Real-time streaming with rolling buffers
- Voice Activity Detection (VAD)
- Session management and persistence  
- Model management with memory optimization
- Threading safety for concurrent requests

Usage:
    from whisper_service import create_whisper_service, TranscriptionRequest
    
    # Create service
    service = await create_whisper_service()
    
    # Create request
    request = TranscriptionRequest(
        audio_data=audio_array,
        model_name="whisper-base"
    )
    
    # Transcribe
    result = await service.transcribe(request)
    print(result.text)
"""

from .whisper_service import (
    WhisperService,
    TranscriptionRequest,
    TranscriptionResult,
    ModelManager,
    AudioBufferManager,
    SessionManager,
    create_whisper_service
)

from .api_server import create_app

__version__ = "1.0.0"
__author__ = "LiveTranslate Team"

# Public API
__all__ = [
    "WhisperService",
    "TranscriptionRequest", 
    "TranscriptionResult",
    "ModelManager",
    "AudioBufferManager", 
    "SessionManager",
    "create_whisper_service",
    "create_app"
]

# Factory function for easy module usage
async def create_service(config=None):
    """
    Convenience factory function to create a whisper service
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized WhisperService instance
    """
    return await create_whisper_service(config)

# Module-level configuration
DEFAULT_CONFIG = {
    "models_dir": "~/.whisper/models",
    "default_model": "whisper-base",
    "sample_rate": 16000,
    "buffer_duration": 6.0,
    "inference_interval": 3.0,
    "enable_vad": True,
    "min_inference_interval": 0.2,
    "max_concurrent_requests": 10
} 