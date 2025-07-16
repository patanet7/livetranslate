"""
NPU-Optimized Whisper Service

Intel NPU-accelerated speech-to-text transcription service extracted and specialized
from the original whisper-service with proven NPU functionality.

Features:
- Intel NPU acceleration with OpenVINO
- Automatic hardware fallback (NPU → GPU → CPU)
- Power management and thermal optimization
- Real-time streaming transcription
- Production-tested NPU model management
"""

__version__ = "1.0.0-npu"
__author__ = "LiveTranslate Team"

# Import key components for easy access
try:
    from .core.npu_model_manager import ModelManager
    from .core.npu_engine import WhisperService, create_whisper_service
    from .utils.device_detection import DeviceDetector, detect_hardware
    from .utils.power_manager import PowerManager, PowerProfile
    
    __all__ = [
        "ModelManager",
        "WhisperService", 
        "create_whisper_service",
        "DeviceDetector",
        "detect_hardware",
        "PowerManager",
        "PowerProfile"
    ]
    
except ImportError as e:
    # Graceful handling if dependencies aren't available
    import logging
    logging.getLogger(__name__).warning(f"Some NPU components not available: {e}")
    __all__ = []