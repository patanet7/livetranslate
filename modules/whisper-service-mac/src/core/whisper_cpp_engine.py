#!/usr/bin/env python3
"""
whisper.cpp Engine for macOS

Native whisper.cpp integration with Apple Silicon optimizations including
Metal GPU acceleration, Core ML + ANE support, and word-level timestamps.
"""

import os
import sys
import subprocess
import logging
import threading
import time
import json
import tempfile
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class WhisperResult:
    """Result from whisper.cpp inference"""
    text: str
    language: str
    segments: List[Dict[str, Any]]
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    processing_time: float = 0.0
    model_name: str = ""
    confidence: float = 0.0


class WhisperCppError(Exception):
    """whisper.cpp specific errors"""
    pass


class WhisperCppEngine:
    """
    Native whisper.cpp engine with Apple Silicon optimizations
    
    Provides high-performance speech-to-text using whisper.cpp with:
    - Metal GPU acceleration on Apple Silicon
    - Core ML + Apple Neural Engine support
    - Word-level timestamps
    - GGML model support with quantization
    """
    
    def __init__(self, models_dir: str = "../models", whisper_cpp_path: str = "whisper_cpp"):
        """Initialize whisper.cpp engine"""
        self.models_dir = Path(models_dir)
        self.whisper_cpp_path = Path(whisper_cpp_path)
        self.ggml_dir = self.models_dir / "ggml"
        self.coreml_cache_dir = self.models_dir / "cache" / "coreml"
        
        # Engine state
        self.current_model = None
        self.available_models = []
        self.capabilities = self._detect_capabilities()
        
        # Thread safety
        self.inference_lock = threading.Lock()
        
        # Performance tracking
        self.inference_times = []
        self.model_load_times = {}
        
        logger.info(f"Initialized whisper.cpp engine with capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect whisper.cpp build capabilities"""
        capabilities = {
            "metal": False,
            "coreml": False,
            "accelerate": False,
            "neon": False,
            "avx": False,
            "word_timestamps": True,  # Always available in whisper.cpp
            "quantization": True
        }
        
        # Check if whisper-cli binary exists
        whisper_cli = self.whisper_cpp_path / "build" / "bin" / "whisper-cli"
        if not whisper_cli.exists():
            logger.warning(f"whisper-cli not found at {whisper_cli}")
            return capabilities
        
        try:
            # Run whisper-cli with a test to see system info
            result = subprocess.run(
                [str(whisper_cli), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse capabilities from help output or try a quick system info run
                try:
                    # Try to get system info by running with no args (should show system info)
                    test_result = subprocess.run(
                        [str(whisper_cli)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    output = test_result.stderr + test_result.stdout
                    
                    # Parse system info output
                    if "METAL = 1" in output:
                        capabilities["metal"] = True
                    if "COREML = 1" in output:
                        capabilities["coreml"] = True
                    if "NEON = 1" in output:
                        capabilities["neon"] = True
                    if "AVX" in output and "AVX = 1" in output:
                        capabilities["avx"] = True
                    if "BLAS = 1" in output or "Accelerate" in output:
                        capabilities["accelerate"] = True
                        
                except Exception as e:
                    logger.debug(f"Could not detect detailed capabilities: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not detect whisper.cpp capabilities: {e}")
        
        # Detect Apple Silicon for defaults
        if os.uname().machine == "arm64":
            capabilities["neon"] = True
            capabilities["accelerate"] = True
            # Assume Metal/Core ML are available on Apple Silicon builds
            if capabilities.get("metal") is None:
                capabilities["metal"] = True
        
        return capabilities
    
    def initialize(self) -> bool:
        """Initialize the engine and scan for models"""
        try:
            # Ensure directories exist
            self.ggml_dir.mkdir(parents=True, exist_ok=True)
            self.coreml_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Scan for available models
            self._scan_models()
            
            logger.info(f"whisper.cpp engine initialized with {len(self.available_models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize whisper.cpp engine: {e}")
            return False
    
    def _scan_models(self):
        """Scan for available GGML models"""
        self.available_models = []
        
        if not self.ggml_dir.exists():
            logger.warning(f"GGML models directory not found: {self.ggml_dir}")
            return
        
        # Look for GGML model files
        for model_file in self.ggml_dir.glob("*.bin"):
            model_name = model_file.stem
            
            # Extract base model name (remove ggml- prefix and quantization suffix)
            base_name = model_name
            if base_name.startswith("ggml-"):
                base_name = base_name[5:]  # Remove "ggml-" prefix
            
            # Check for quantization suffix (e.g., -q5_0)
            quantization = None
            for q_type in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
                if base_name.endswith(f"-{q_type}"):
                    quantization = q_type
                    base_name = base_name[:-len(f"-{q_type}")]
                    break
            
            model_info = {
                "name": base_name,
                "file_path": str(model_file),
                "file_name": model_file.name,
                "quantization": quantization,
                "size": model_file.stat().st_size,
                "coreml_available": self._check_coreml_model(base_name)
            }
            
            self.available_models.append(model_info)
            logger.debug(f"Found model: {model_info}")
        
        self.available_models.sort(key=lambda x: x["name"])
        logger.info(f"Found {len(self.available_models)} GGML models")
    
    def _check_coreml_model(self, model_name: str) -> bool:
        """Check if Core ML model exists for this model"""
        coreml_path = self.coreml_cache_dir / f"ggml-{model_name}-encoder.mlmodelc"
        return coreml_path.exists()
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return self.available_models.copy()
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model (actually just validates it exists)"""
        try:
            # Find the model
            model_info = None
            for model in self.available_models:
                if model["name"] == model_name or model["file_name"] == model_name:
                    model_info = model
                    break
            
            if not model_info:
                raise WhisperCppError(f"Model {model_name} not found")
            
            # Verify the file exists
            model_path = Path(model_info["file_path"])
            if not model_path.exists():
                raise WhisperCppError(f"Model file not found: {model_path}")
            
            self.current_model = model_info
            logger.info(f"Model {model_name} ready for inference")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    @contextmanager
    def _temp_audio_file(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Create temporary WAV file for whisper.cpp"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Write audio data as 16-bit WAV (whisper.cpp requirement)
            sf.write(temp_path, audio_data, sample_rate, subtype='PCM_16')
            yield temp_path
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def transcribe(self, 
                  audio_data: Union[np.ndarray, str], 
                  model_name: Optional[str] = None,
                  language: Optional[str] = None,
                  task: str = "transcribe",
                  word_timestamps: bool = False,
                  **kwargs) -> WhisperResult:
        """
        Transcribe audio using whisper.cpp
        
        Args:
            audio_data: Audio data as numpy array or file path
            model_name: Model to use (if None, uses current_model)
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            word_timestamps: Enable word-level timestamps
            **kwargs: Additional whisper.cpp parameters
        """
        start_time = time.time()
        
        with self.inference_lock:
            try:
                # Select model
                if model_name:
                    model_info = None
                    for model in self.available_models:
                        if model["name"] == model_name:
                            model_info = model
                            break
                    if not model_info:
                        raise WhisperCppError(f"Model {model_name} not found")
                else:
                    model_info = self.current_model
                    if not model_info:
                        raise WhisperCppError("No model loaded")
                
                model_path = model_info["file_path"]
                
                # Prepare audio input and call whisper.cpp
                if isinstance(audio_data, str):
                    # File path provided
                    audio_file = audio_data
                    temp_file = None
                    
                    # Call whisper.cpp
                    result = self._run_whisper_cpp(
                        model_path=model_path,
                        audio_file=audio_file,
                        language=language,
                        task=task,
                        word_timestamps=word_timestamps,
                        **kwargs
                    )
                else:
                    # Audio data provided - create temp file
                    sample_rate = kwargs.get("sample_rate", 16000)
                    with self._temp_audio_file(audio_data, sample_rate) as temp_path:
                        audio_file = temp_path
                        temp_file = temp_path
                        
                        # Call whisper.cpp
                        result = self._run_whisper_cpp(
                            model_path=model_path,
                            audio_file=audio_file,
                            language=language,
                            task=task,
                            word_timestamps=word_timestamps,
                            **kwargs
                        )
                
                # Add metadata
                result.processing_time = time.time() - start_time
                result.model_name = model_info["name"]
                
                # Track performance
                self.inference_times.append(result.processing_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
                
                return result
                
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                raise WhisperCppError(f"Transcription failed: {e}")
    
    def _run_whisper_cpp(self, 
                        model_path: str,
                        audio_file: str,
                        language: Optional[str] = None,
                        task: str = "transcribe",
                        word_timestamps: bool = False,
                        **kwargs) -> WhisperResult:
        """Run whisper.cpp binary with specified parameters"""
        
        whisper_cli = self.whisper_cpp_path / "build" / "bin" / "whisper-cli"
        
        if not whisper_cli.exists():
            raise WhisperCppError(f"whisper-cli not found: {whisper_cli}")
        
        # Build command arguments
        cmd = [
            str(whisper_cli),
            "-m", model_path,
            "-f", audio_file,
            "--output-json"  # Request JSON output
        ]
        
        # Add language if specified
        if language:
            cmd.extend(["-l", language])
        
        # Add task
        if task == "translate":
            cmd.append("--translate")
        
        # Add word timestamps
        if word_timestamps:
            cmd.extend(["-ml", "1"])  # Max length 1 for word-level
        
        # Add other parameters
        threads = kwargs.get("threads", 4)
        cmd.extend(["-t", str(threads)])
        
        if kwargs.get("verbose", False):
            cmd.append("--verbose")
        
        # Add any custom whisper.cpp arguments
        custom_args = kwargs.get("whisper_args", [])
        if custom_args:
            cmd.extend(custom_args)
        
        logger.debug(f"Running whisper.cpp: {' '.join(cmd)}")
        
        try:
            # Run whisper.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 300)  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise WhisperCppError(f"whisper.cpp failed: {error_msg}")
            
            # Parse output
            return self._parse_whisper_output(result.stdout, result.stderr)
            
        except subprocess.TimeoutExpired:
            raise WhisperCppError("whisper.cpp timed out")
        except Exception as e:
            raise WhisperCppError(f"Failed to run whisper.cpp: {e}")
    
    def _parse_whisper_output(self, stdout: str, stderr: str) -> WhisperResult:
        """Parse whisper.cpp output into WhisperResult"""
        
        # Try to parse JSON output first
        try:
            # Look for JSON in stdout
            lines = stdout.strip().split('\n')
            json_output = None
            
            for line in lines:
                if line.strip().startswith('{'):
                    try:
                        json_output = json.loads(line.strip())
                        break
                    except json.JSONDecodeError:
                        continue
            
            if json_output:
                return self._parse_json_output(json_output)
                
        except Exception as e:
            logger.debug(f"Could not parse JSON output: {e}")
        
        # Fallback to text parsing
        return self._parse_text_output(stdout, stderr)
    
    def _parse_json_output(self, json_data: Dict[str, Any]) -> WhisperResult:
        """Parse JSON output from whisper.cpp"""
        
        # Extract text
        text = ""
        segments = []
        word_timestamps = []
        
        if "transcription" in json_data:
            for segment in json_data["transcription"]:
                text += segment.get("text", "")
                
                seg_info = {
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", ""),
                    "confidence": segment.get("confidence", 0.0)
                }
                segments.append(seg_info)
                
                # Extract word-level timestamps if available
                if "words" in segment:
                    for word in segment["words"]:
                        word_info = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "confidence": word.get("confidence", 0.0)
                        }
                        word_timestamps.append(word_info)
        
        # Extract language
        language = json_data.get("language", "en")
        
        return WhisperResult(
            text=text.strip(),
            language=language,
            segments=segments,
            word_timestamps=word_timestamps if word_timestamps else None
        )
    
    def _parse_text_output(self, stdout: str, stderr: str) -> WhisperResult:
        """Parse text output from whisper.cpp (fallback)"""
        
        # Look for transcription in stdout
        lines = stdout.strip().split('\n')
        
        text_lines = []
        segments = []
        language = "en"
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and system info
            if not line or line.startswith('whisper_') or 'system_info:' in line:
                continue
            
            # Look for timestamp format: [00:00:00.000 --> 00:00:00.000]  text
            if line.startswith('[') and ']' in line:
                try:
                    # Extract timestamp and text
                    timestamp_part, text_part = line.split(']', 1)
                    timestamp_part = timestamp_part[1:]  # Remove [
                    text_part = text_part.strip()
                    
                    if '-->' in timestamp_part:
                        start_str, end_str = timestamp_part.split('-->')
                        start_time = self._parse_timestamp(start_str.strip())
                        end_time = self._parse_timestamp(end_str.strip())
                        
                        if text_part:  # Only add non-empty text
                            text_lines.append(text_part)
                            segments.append({
                                "start": start_time,
                                "end": end_time,
                                "text": text_part,
                                "confidence": 0.0  # Not available in text output
                            })
                            
                except Exception as e:
                    logger.debug(f"Could not parse line: {line} - {e}")
                    continue
            
            # Look for language detection
            if 'detected language:' in line.lower():
                try:
                    language = line.split(':')[1].strip()
                except:
                    pass
        
        text = " ".join(text_lines)
        
        return WhisperResult(
            text=text,
            language=language,
            segments=segments,
            word_timestamps=None
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to seconds"""
        try:
            # Format: HH:MM:SS.mmm
            parts = timestamp_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
            "min_inference_time": min(self.inference_times),
            "max_inference_time": max(self.inference_times),
            "total_inferences": len(self.inference_times),
            "capabilities": self.capabilities,
            "current_model": self.current_model["name"] if self.current_model else None
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device and capability information"""
        import platform
        
        return {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "capabilities": self.capabilities,
            "whisper_cpp_path": str(self.whisper_cpp_path),
            "models_count": len(self.available_models),
            "current_model": self.current_model["name"] if self.current_model else None
        }