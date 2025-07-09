#!/usr/bin/env python3
"""
Fixed Whisper Service with proper model path handling and fallbacks
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import time
import numpy as np
import librosa

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """Fixed model manager with proper path handling"""
    
    def __init__(self, models_dir: Optional[str] = None):
        # Fix model path resolution
        if models_dir:
            self.models_dir = models_dir
        elif os.path.exists("/app/models"):  # Docker container path
            self.models_dir = "/app/models"
        else:
            self.models_dir = os.path.expanduser("~/.whisper/models")
        
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._detect_device()
        self.pipelines = {}
        
        logger.info(f"ModelManager initialized - Models dir: {self.models_dir}, Device: {self.device}")
        
        # Check for available models
        self._check_available_models()
    
    def _detect_device(self) -> str:
        """Detect available device with proper fallback"""
        try:
            # Check environment override
            if env_device := os.getenv("OPENVINO_DEVICE"):
                logger.info(f"Using device from environment: {env_device}")
                return env_device
            
            # Try to import OpenVINO and detect devices
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                logger.info(f"Available devices: {devices}")
                
                # Prefer NPU > GPU > CPU
                if "NPU" in devices:
                    return "NPU"
                elif "GPU" in devices:
                    return "GPU"
                else:
                    return "CPU"
            except ImportError:
                logger.warning("OpenVINO not available, falling back to CPU simulation")
                return "CPU_SIMULATION"
                
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            return "CPU_SIMULATION"
    
    def _check_available_models(self):
        """Check what models are actually available"""
        available = self.list_models()
        if available:
            logger.info(f"Found {len(available)} models: {available}")
        else:
            logger.warning(f"No models found in {self.models_dir}")
            logger.info("To download models, run: docker-compose --profile download-models up model-downloader")
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            if not self.models_dir.exists():
                return []
            
            models = []
            for item in self.models_dir.iterdir():
                if item.is_dir() and any(item.glob("*.bin")):  # Has model files
                    models.append(item.name)
            
            return sorted(models)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def load_model(self, model_name: str):
        """Load model with proper error handling"""
        if model_name in self.pipelines:
            return self.pipelines[model_name]
        
        model_path = self.models_dir / model_name
        
        # Check if model exists
        if not model_path.exists():
            available = self.list_models()
            if available:
                raise FileNotFoundError(f"Model '{model_name}' not found. Available: {available}")
            else:
                raise FileNotFoundError(f"No models found. Please download models first.")
        
        logger.info(f"Loading model: {model_name} from {model_path}")
        
        try:
            if self.device == "CPU_SIMULATION":
                # Create mock pipeline for development/testing
                pipeline = MockWhisperPipeline(model_name)
                logger.info(f"✓ Created mock pipeline for {model_name}")
            else:
                # Try real OpenVINO pipeline
                import openvino_genai
                pipeline = openvino_genai.WhisperPipeline(str(model_path), device=self.device)
                logger.info(f"✓ Loaded real model {model_name} on {self.device}")
            
            self.pipelines[model_name] = pipeline
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
            # Fallback to simulation
            if self.device != "CPU_SIMULATION":
                logger.info("Falling back to simulation mode...")
                pipeline = MockWhisperPipeline(model_name)
                self.pipelines[model_name] = pipeline
                return pipeline
            else:
                raise


class MockWhisperPipeline:
    """Mock pipeline for development and testing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, audio_data: np.ndarray):
        """Generate mock transcription"""
        # Simulate processing time based on audio length
        duration = len(audio_data) / 16000  # Assume 16kHz
        time.sleep(min(duration * 0.1, 2.0))  # Max 2 seconds
        
        # Create mock result
        class MockResult:
            def __init__(self):
                self.text = f"Mock transcription from {self.model_name}: [Audio duration: {duration:.1f}s]"
                self.language = "en"
                self.segments = [
                    {"start": 0.0, "end": duration, "text": self.text}
                ]
        
        return MockResult()


# Update the main WhisperService class to use fixed model manager
class WhisperService:
    """Fixed WhisperService with better error handling"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        
        # Use fixed model manager
        self.model_manager = ModelManager(self.config.get("models_dir"))
        
        # Initialize other components...
        logger.info("WhisperService initialized with fixes")
    
    def _load_config(self) -> Dict:
        """Load configuration with better defaults"""
        return {
            "models_dir": os.getenv("WHISPER_MODELS_DIR", "/app/models"),
            "default_model": os.getenv("WHISPER_DEFAULT_MODEL", "whisper-base"),
            "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
            "device": os.getenv("OPENVINO_DEVICE"),
        }
    
    def get_service_status(self) -> Dict:
        """Get service status with model information"""
        return {
            "device": self.model_manager.device,
            "models_dir": str(self.model_manager.models_dir),
            "available_models": self.model_manager.list_models(),
            "loaded_models": list(self.model_manager.pipelines.keys()),
            "status": "ready" if self.model_manager.list_models() else "no_models"
        }


# Factory function
async def create_whisper_service(config: Optional[Dict] = None) -> WhisperService:
    """Create fixed whisper service"""
    return WhisperService(config)