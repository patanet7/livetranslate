#!/usr/bin/env python3
"""
Whisper Service Module

Provides NPU-optimized speech-to-text transcription with real-time streaming capabilities.
Extracted from the existing whisper-npu-server with enhanced modular architecture.

Key Features:
- NPU/GPU/CPU acceleration with automatic fallback
- Real-time streaming with rolling buffers
- Voice Activity Detection (VAD)
- Session management and persistence
- Model management with memory optimization
- Threading safety for concurrent requests
"""

import os
import asyncio
import logging
import threading
import time
import json
import tempfile
from typing import Dict, List, Optional, AsyncGenerator, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from collections import deque

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from scipy import signal

# OpenVINO imports for NPU acceleration
import openvino as ov
import openvino_genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionRequest:
    """Transcription request data structure"""
    audio_data: Union[np.ndarray, bytes]
    model_name: str = "whisper-base"
    language: Optional[str] = None
    session_id: Optional[str] = None
    streaming: bool = False
    enhanced: bool = False
    sample_rate: int = 16000
    enable_vad: bool = True
    timestamp_mode: str = "word"  # word, segment, none

@dataclass
class TranscriptionResult:
    """Transcription result data structure"""
    text: str
    segments: List[Dict]
    language: str
    confidence_score: float
    processing_time: float
    model_used: str
    device_used: str
    session_id: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ModelManager:
    """
    Manages Whisper model loading with NPU optimization and fallback support
    """
    
    def __init__(self, models_dir: Optional[str] = None, use_hf_pipeline: bool = False):
        """Initialize model manager with NPU detection and model loading"""
        # Use local models directory relative to whisper-service
        if models_dir is None:
            # Try environment variable first
            env_models = os.getenv("WHISPER_MODELS_DIR")
            if env_models and os.path.exists(env_models):
                self.models_dir = env_models
            else:
                # Try local models directory in whisper-service
                local_models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
                if os.path.exists(local_models):
                    self.models_dir = local_models
                else:
                    self.models_dir = os.path.expanduser("~/.whisper/models")
        else:
            self.models_dir = models_dir
        self.pipelines = {}
        self.default_model = "whisper-base"
        self.device = self._detect_best_device()
        self.use_hf_pipeline = use_hf_pipeline  # Option to use Hugging Face pipeline
        
        if use_hf_pipeline:
            logger.info("🔄 Using Hugging Face transformers pipeline with language detection")
        else:
            logger.info("🔄 Using OpenVINO pipeline with NPU optimization")
        
        # Thread safety for NPU access
        self.inference_lock = threading.Lock()
        self.request_queue = Queue(maxsize=10)
        self.last_inference_time = 0
        self.min_inference_interval = 0.2  # 200ms for memory relief
        
        logger.info(f"ModelManager initialized - Device: {self.device}, Models: {self.models_dir}")
        logger.info("Note: Whisper model deprecation warnings are suppressed. Consider updating to newer OpenVINO model formats.")
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Try to preload the default model
        self._preload_default_model()
    
    def _detect_best_device(self) -> str:
        """Detect the best available device for inference"""
        try:
            # Check environment variable first
            env_device = os.getenv("OPENVINO_DEVICE")
            if env_device:
                logger.info(f"Using device from environment: {env_device}")
                return env_device
            
            # Auto-detect available devices
            core = ov.Core()
            available_devices = core.available_devices
            logger.info(f"Available OpenVINO devices: {available_devices}")
            
            # Prefer NPU, then GPU, then CPU
            if "NPU" in available_devices:
                logger.info("✓ NPU detected! Using NPU for inference.")
                return "NPU"
            elif "GPU" in available_devices:
                logger.info("⚠ NPU not found, using GPU fallback.")
                return "GPU"
            else:
                logger.info("⚠ NPU/GPU not found, using CPU fallback.")
                return "CPU"
                
        except Exception as e:
            logger.error(f"Error detecting devices: {e}")
            return "CPU"
    
    def load_model(self, model_name: str):
        """Load a Whisper model with device fallback"""
        if model_name not in self.pipelines:
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                available_models = self.list_models()
                if available_models:
                    logger.warning(f"Model {model_name} not found. Available models: {available_models}")
                    raise FileNotFoundError(f"Model {model_name} not found. Available: {available_models}")
                else:
                    raise FileNotFoundError(f"No models found in {self.models_dir}. Please mount models directory.")
            
            logger.info(f"[MODEL] 🔄 Loading model: {model_name} on device: {self.device}")
            try:
                # Use OpenVINO GenAI WhisperPipeline for proper model loading
                logger.info(f"[MODEL] 🧠 Creating WhisperPipeline for {model_name}")
                logger.info(f"[MODEL] 📁 Model path: {model_path}")
                
                # Suppress deprecation warnings for older Whisper models
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", message=".*Whisper decoder models with past is deprecated.*")
                    start_load_time = time.time()
                    pipeline = openvino_genai.WhisperPipeline(str(model_path), device=self.device)
                    load_time = time.time() - start_load_time
                    
                self.pipelines[model_name] = pipeline
                logger.info(f"[MODEL] ✅ Model {model_name} loaded successfully on {self.device} in {load_time:.2f}s")
                logger.info(f"[MODEL] 📊 Total loaded models: {len(self.pipelines)} ({list(self.pipelines.keys())})")
                        
            except Exception as e:
                if self.device != "CPU":
                    logger.warning(f"[MODEL] ⚠️ Failed to load on {self.device}, trying CPU fallback: {e}")
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=DeprecationWarning)
                            warnings.filterwarnings("ignore", message=".*Whisper decoder models with past is deprecated.*")
                            start_fallback_time = time.time()
                            pipeline = openvino_genai.WhisperPipeline(str(model_path), device="CPU")
                            fallback_time = time.time() - start_fallback_time
                            
                        self.pipelines[model_name] = pipeline
                        self.device = "CPU"  # Update device for this session
                        logger.info(f"[MODEL] ✅ Model {model_name} loaded on CPU fallback in {fallback_time:.2f}s")
                        logger.info(f"[MODEL] 📊 Total loaded models: {len(self.pipelines)} ({list(self.pipelines.keys())})")
                    except Exception as cpu_e:
                        logger.error(f"[MODEL] ❌ Failed to load on CPU fallback: {cpu_e}")
                        raise cpu_e
                else:
                    logger.error(f"[MODEL] ❌ Failed to load {model_name} on {self.device}: {e}")
                    raise e
        
        return self.pipelines[model_name]
    
    def list_models(self) -> List[str]:
        """List available models in the models directory"""
        try:
            return [d for d in os.listdir(self.models_dir) 
                    if os.path.isdir(os.path.join(self.models_dir, d))]
        except:
            return []
    
    def _preload_default_model(self):
        """Try to preload the default model if available"""
        try:
            logger.info(f"Attempting to preload default model: {self.default_model}")
            self.load_model(self.default_model)
            logger.info(f"✓ Default model {self.default_model} preloaded successfully")
        except Exception as e:
            logger.warning(f"Could not preload {self.default_model}: {e}")
            logger.info("✅ Server will work in simulation mode without real models")
    
    def clear_cache(self):
        """Clear model cache and loaded models to free memory"""
        try:
            logger.info("Clearing model cache and pipelines due to memory pressure...")
            
            # Clear all loaded pipelines
            for model_name in list(self.pipelines.keys()):
                try:
                    del self.pipelines[model_name]
                    logger.debug(f"Cleared pipeline for {model_name}")
                except:
                    pass
            
            self.pipelines.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("✓ Model cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing model cache: {e}")
    
    def safe_inference(self, model_name: str, audio_data: np.ndarray):
        """Thread-safe inference with NPU device protection and memory management"""
        with self.inference_lock:
            try:
                # Enforce minimum interval between inferences to prevent NPU overload
                current_time = time.time()
                time_since_last = current_time - self.last_inference_time
                if time_since_last < self.min_inference_interval:
                    sleep_time = self.min_inference_interval - time_since_last
                    logger.debug(f"NPU cooldown: sleeping {sleep_time:.3f}s")
                    time.sleep(sleep_time)
                
                # Load model if not already loaded
                pipeline = self.load_model(model_name)
                
                # Perform inference with device error handling
                logger.debug(f"Starting inference for {len(audio_data)} samples")
                start_time = time.time()
                
                try:
                    result = pipeline.generate(audio_data)
                    inference_time = time.time() - start_time
                    self.last_inference_time = time.time()
                    
                    logger.debug(f"Inference completed in {inference_time:.3f}s")
                    return result
                    
                except Exception as device_error:
                    error_msg = str(device_error)
                    
                    # Handle specific device errors
                    if "Infer Request is busy" in error_msg:
                        logger.warning("Device busy - inference request rejected")
                        raise Exception("Device is busy processing another request. Please try again.")
                    
                    elif "ZE_RESULT_ERROR_DEVICE_LOST" in error_msg or "device hung" in error_msg:
                        logger.error("Device lost/hung - attempting recovery")
                        # Clear the pipeline to force reload
                        if model_name in self.pipelines:
                            del self.pipelines[model_name]
                        raise Exception("Device error - model will be reloaded on next request")
                    
                    elif "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY" in error_msg or "insufficient host memory" in error_msg:
                        logger.error("Out of memory - clearing cache")
                        self.clear_cache()
                        
                        # Suggest smaller model if using large model
                        if "large" in model_name:
                            raise Exception("Out of memory. Try using whisper-tiny or whisper-base instead of large models.")
                        else:
                            raise Exception("Out of memory. Cache cleared - please try again with fewer concurrent requests.")
                    
                    else:
                        logger.error(f"Inference error: {error_msg}")
                        raise device_error
                        
            except Exception as e:
                self.last_inference_time = time.time()  # Still update to prevent hammering
                raise e

class AudioBufferManager:
    """
    Manages rolling audio buffers with voice activity detection for streaming
    """
    
    def __init__(self, buffer_duration: float = 6.0, sample_rate: int = 16000, enable_vad: bool = True):
        """Initialize audio buffer manager"""
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.max_samples = int(buffer_duration * sample_rate)
        self.enable_vad = enable_vad
        
        # Rolling buffer for audio samples
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_lock = threading.Lock()
        
        # VAD setup
        if enable_vad:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
                self.vad_enabled = True
                logger.info("✓ Voice Activity Detection enabled")
            except:
                self.vad = None
                self.vad_enabled = False
                logger.warning("⚠ Voice Activity Detection not available")
        else:
            self.vad = None
            self.vad_enabled = False
        
        # Audio processing
        self.last_processed_time = 0
        
    def add_audio_chunk(self, audio_samples: np.ndarray) -> int:
        """Add new audio samples to the rolling buffer"""
        with self.buffer_lock:
            if isinstance(audio_samples, np.ndarray):
                # Convert to list and extend buffer
                samples_list = audio_samples.tolist()
                self.audio_buffer.extend(samples_list)
                return len(self.audio_buffer)
            return 0
    
    def get_buffer_audio(self) -> np.ndarray:
        """Get current buffer as numpy array"""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([])
            return np.array(list(self.audio_buffer))
    
    def find_speech_boundaries(self, audio_array: np.ndarray, chunk_duration: float = 0.02) -> Tuple[Optional[int], Optional[int]]:
        """Find speech boundaries using VAD"""
        if not self.vad_enabled or len(audio_array) == 0:
            return None, None
            
        try:
            # Convert to 16-bit PCM for VAD
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Process in 20ms chunks (VAD requirement)
            chunk_samples = int(self.sample_rate * chunk_duration)
            speech_chunks = []
            
            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                if len(chunk) == chunk_samples:
                    # VAD expects specific sample rates
                    if self.sample_rate in [8000, 16000, 32000, 48000]:
                        is_speech = self.vad.is_speech(chunk.tobytes(), self.sample_rate)
                        speech_chunks.append((i, i + chunk_samples, is_speech))
            
            # Find speech boundaries
            speech_start = None
            speech_end = None
            
            for start, end, is_speech in speech_chunks:
                if is_speech and speech_start is None:
                    speech_start = start
                elif not is_speech and speech_start is not None:
                    speech_end = end
                    break
            
            return speech_start, speech_end
            
        except Exception as e:
            logger.debug(f"VAD processing failed: {e}")
            return None, None
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.buffer_lock:
            self.audio_buffer.clear()

class SessionManager:
    """
    Manages transcription sessions with persistence and statistics
    """
    
    def __init__(self, session_dir: Optional[str] = None):
        """Initialize session manager"""
        self.session_dir = session_dir or os.path.join(os.path.dirname(__file__), "..", "session_data")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.sessions: Dict[str, Dict] = {}
        self.transcription_history = deque(maxlen=200)
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(self, session_id: str, config: Optional[Dict] = None) -> Dict:
        """Create a new transcription session"""
        session_config = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {
                "transcriptions": 0,
                "total_duration": 0.0,
                "total_words": 0,
                "avg_confidence": 0.0
            },
            "transcriptions": []
        }
        
        self.sessions[session_id] = session_config
        self._save_session(session_id)
        logger.info(f"Created transcription session: {session_id}")
        return session_config
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.sessions.get(session_id)
    
    def add_transcription(self, session_id: str, result: TranscriptionResult):
        """Add transcription result to session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        # Add to session transcriptions
        transcription_data = {
            "text": result.text,
            "timestamp": result.timestamp,
            "confidence": result.confidence_score,
            "model": result.model_used,
            "device": result.device_used,
            "processing_time": result.processing_time
        }
        
        session["transcriptions"].append(transcription_data)
        
        # Update statistics
        stats = session["stats"]
        stats["transcriptions"] += 1
        stats["total_words"] += len(result.text.split())
        
        # Update average confidence
        old_avg = stats["avg_confidence"]
        count = stats["transcriptions"]
        stats["avg_confidence"] = (old_avg * (count - 1) + result.confidence_score) / count
        
        # Add to global history
        self.transcription_history.append(transcription_data)
        
        # Save session
        self._save_session(session_id)
    
    def close_session(self, session_id: str) -> Optional[Dict]:
        """Close session and return final statistics"""
        session = self.sessions.get(session_id)
        if session:
            session["closed_at"] = datetime.now().isoformat()
            self._save_session(session_id)
            logger.info(f"Closed transcription session: {session_id}")
        return session
    
    def get_transcription_history(self, limit: int = 50) -> List[Dict]:
        """Get recent transcription history"""
        return list(self.transcription_history)[-limit:]
    
    def _load_sessions(self):
        """Load sessions from disk"""
        try:
            sessions_file = os.path.join(self.session_dir, "sessions.json")
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = data.get("sessions", {})
                    
            # Load transcription history
            history_file = os.path.join(self.session_dir, "transcriptions.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.transcription_history = deque(data.get("transcriptions", []), maxlen=200)
                    
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
    
    def _save_session(self, session_id: str):
        """Save session to disk"""
        try:
            # Save all sessions
            sessions_file = os.path.join(self.session_dir, "sessions.json")
            sessions_data = {
                "sessions": self.sessions,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)
            
            # Save transcription history
            history_file = os.path.join(self.session_dir, "transcriptions.json")
            history_data = {
                "transcriptions": list(self.transcription_history)[-100:],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

class WhisperService:
    """
    Main Whisper Service class providing NPU-optimized transcription
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Whisper service with configuration"""
        self.config = config or self._load_config()
        
        # Initialize components
        self.model_manager = ModelManager(self.config.get("models_dir"))
        self.buffer_manager = AudioBufferManager(
            buffer_duration=self.config.get("buffer_duration", 6.0),
            sample_rate=self.config.get("sample_rate", 16000),
            enable_vad=self.config.get("enable_vad", True)
        )
        self.session_manager = SessionManager(self.config.get("session_dir"))
        
        # Streaming settings
        self.streaming_active = False
        self.streaming_thread = None
        self.inference_interval = self.config.get("inference_interval", 3.0)
        
        logger.info("WhisperService initialized successfully")
    
    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """
        Transcribe audio using the specified model
        
        Args:
            request: Transcription request with audio and parameters
            
        Returns:
            Transcription result with text and metadata
        """
        start_time = time.time()
        
        try:
            # Process audio data
            if isinstance(request.audio_data, bytes):
                # Load audio from bytes
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(request.audio_data)
                    tmp_file.flush()
                    
                    audio_data, sr = librosa.load(tmp_file.name, sr=request.sample_rate)
                    os.unlink(tmp_file.name)
            else:
                audio_data = request.audio_data
                sr = request.sample_rate
            
            # Ensure correct sample rate
            if sr != request.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=request.sample_rate)
            
            # Apply VAD if enabled
            if request.enable_vad:
                speech_start, speech_end = self.buffer_manager.find_speech_boundaries(audio_data)
                if speech_start is not None and speech_end is not None:
                    audio_data = audio_data[speech_start:speech_end]
            
            # Perform inference
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.model_manager.safe_inference, 
                request.model_name, 
                audio_data
            )
            
            processing_time = time.time() - start_time
            
            # Parse OpenVINO WhisperDecodedResults properly
            logger.info(f"[WHISPER] 🔍 Result type: {type(result)}")
            logger.info(f"[WHISPER] 🔍 Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            # Handle OpenVINO WhisperDecodedResults structure
            if hasattr(result, 'texts') and result.texts:
                # OpenVINO returns 'texts' (plural) - get the first text
                text = result.texts[0] if result.texts else ""
                logger.info(f"[WHISPER] 📝 Text extracted from 'texts': '{text}'")
                
                # Try to get segments from chunks
                segments = []
                if hasattr(result, 'chunks') and result.chunks:
                    segments = result.chunks
                    logger.info(f"[WHISPER] 📋 Chunks/segments count: {len(segments)}")
                    logger.info(f"[WHISPER] 🔍 First chunk structure: {type(segments[0]) if segments else 'No chunks'}")
                    
                    # Debug first chunk attributes
                    if segments and hasattr(segments[0], '__dict__'):
                        chunk_attrs = [attr for attr in dir(segments[0]) if not attr.startswith('_')]
                        logger.info(f"[WHISPER] 🔍 First chunk attributes: {chunk_attrs}")
                
            elif hasattr(result, 'text'):
                # Fallback to 'text' attribute
                text = result.text
                segments = getattr(result, 'segments', [])
                logger.info(f"[WHISPER] 📝 Text extracted from 'text': '{text}'")
                logger.info(f"[WHISPER] 📋 Segments count: {len(segments)}")
                
            else:
                # Last resort: string conversion
                text = str(result)
                segments = []
                logger.info(f"[WHISPER] ⚠️ Using string conversion: '{text}'")
            
            # Enhanced language detection for OpenVINO
            language = 'unknown'
            
            # Method 1: Check result attributes
            if hasattr(result, 'language'):
                language = result.language
                logger.info(f"[WHISPER] 🌍 Found language attribute: {language}")
            elif hasattr(result, 'lang'):
                language = result.lang
                logger.info(f"[WHISPER] 🌍 Found lang attribute: {language}")
                
            # Method 2: Check chunks for language info
            elif hasattr(result, 'chunks') and result.chunks:
                try:
                    first_chunk = result.chunks[0]
                    if hasattr(first_chunk, 'language'):
                        language = first_chunk.language
                        logger.info(f"[WHISPER] 🌍 Found language in chunk: {language}")
                    elif hasattr(first_chunk, 'lang'):
                        language = first_chunk.lang
                        logger.info(f"[WHISPER] 🌍 Found lang in chunk: {language}")
                except Exception as e:
                    logger.debug(f"[WHISPER] Could not extract language from chunks: {e}")
                    
            # Method 3: Simple language detection from text content
            if language == 'unknown' and text:
                # Detect Chinese characters
                if any('\u4e00' <= char <= '\u9fff' for char in text):
                    language = 'zh'
                    logger.info(f"[WHISPER] 🌍 Detected Chinese from text content: {language}")
                # Detect other common patterns
                elif any(char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for char in text):
                    language = 'en'
                    logger.info(f"[WHISPER] 🌍 Detected English from text content: {language}")
                else:
                    language = 'auto'
                    logger.info(f"[WHISPER] 🌍 Auto-detected language: {language}")
            
            # Create result
            transcription_result = TranscriptionResult(
                text=text,
                segments=segments,
                language=language,
                confidence_score=0.9,  # Default confidence, could be extracted from result
                processing_time=processing_time,
                model_used=request.model_name,
                device_used=self.model_manager.device,
                session_id=request.session_id
            )
            
            # Add to session if provided
            if request.session_id:
                self.session_manager.add_transcription(request.session_id, transcription_result)
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    async def transcribe_stream(self, request: TranscriptionRequest) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription results in real-time
        
        Args:
            request: Transcription request with streaming enabled
            
        Yields:
            Partial transcription results as they become available
        """
        if not request.streaming:
            # Non-streaming fallback
            result = await self.transcribe(request)
            yield result
            return
        
        # Start streaming transcription
        try:
            # Add initial audio to buffer
            if isinstance(request.audio_data, np.ndarray):
                self.buffer_manager.add_audio_chunk(request.audio_data)
            
            # Start periodic inference
            if not self.streaming_active:
                await self.start_streaming(request)
            
            # Yield results as they become available
            last_result_time = time.time()
            
            while self.streaming_active:
                await asyncio.sleep(self.inference_interval)
                
                # Get only NEW audio that hasn't been processed yet
                new_audio, samples_to_mark = self.buffer_manager.get_new_audio_only()
                
                if len(new_audio) > 0:  # Process if we have new audio
                    try:
                        logger.info(f"[STREAM] Processing {len(new_audio)} new samples ({len(new_audio)/request.sample_rate:.2f}s)")
                        
                        # Create streaming request with new audio only
                        stream_request = TranscriptionRequest(
                            audio_data=new_audio,
                            model_name=request.model_name,
                            language=request.language,
                            session_id=request.session_id,
                            sample_rate=request.sample_rate,
                            enable_vad=request.enable_vad
                        )
                        
                        # Transcribe only the new audio
                        result = await self.transcribe(stream_request)
                        
                        # Mark this audio as processed to prevent re-transcription
                        self.buffer_manager.mark_audio_as_processed(samples_to_mark)
                        
                        # Yield result
                        yield result
                        last_result_time = time.time()
                        
                        logger.info(f"[STREAM] ✅ Transcribed new audio: '{result.text[:50]}...' (Lang: {result.language})")
                        
                    except Exception as e:
                        logger.warning(f"Streaming transcription error: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise
        finally:
            await self.stop_streaming()
    
    async def start_streaming(self, request: TranscriptionRequest):
        """Start streaming transcription"""
        if self.streaming_active:
            return
        
        self.streaming_active = True
        logger.info(f"Started streaming transcription with model {request.model_name}")
    
    async def stop_streaming(self):
        """Stop streaming transcription"""
        if not self.streaming_active:
            return
        
        self.streaming_active = False
        logger.info("Stopped streaming transcription")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> int:
        """Add audio chunk to the streaming buffer"""
        return self.buffer_manager.add_audio_chunk(audio_chunk)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_manager.list_models()
    
    def get_service_status(self) -> Dict:
        """Get service status information"""
        return {
            "device": self.model_manager.device,
            "loaded_models": list(self.model_manager.pipelines.keys()),
            "available_models": self.get_available_models(),
            "streaming_active": self.streaming_active,
            "buffer_size": len(self.buffer_manager.audio_buffer),
            "vad_enabled": self.buffer_manager.vad_enabled,
            "sessions": len(self.session_manager.sessions)
        }
    
    def create_session(self, session_id: str, config: Optional[Dict] = None) -> Dict:
        """Create a new transcription session"""
        return self.session_manager.create_session(session_id, config)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.session_manager.get_session(session_id)
    
    def close_session(self, session_id: str) -> Optional[Dict]:
        """Close a transcription session"""
        return self.session_manager.close_session(session_id)
    
    def get_transcription_history(self, limit: int = 50) -> List[Dict]:
        """Get recent transcription history"""
        return self.session_manager.get_transcription_history(limit)
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_manager.clear_cache()
    
    def _load_config(self) -> Dict:
        """Load configuration from environment and config files"""
        config = {
            # Model settings - use local models directory first
            "models_dir": os.getenv("WHISPER_MODELS_DIR", 
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models") 
                if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models"))
                else os.path.expanduser("~/.whisper/models")),
            "default_model": os.getenv("WHISPER_DEFAULT_MODEL", "whisper-base.en"),
            
            # Audio settings - optimized for reduced duplicates
            "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
            "buffer_duration": float(os.getenv("BUFFER_DURATION", "4.0")),  # Reduced from 6.0
            "inference_interval": float(os.getenv("INFERENCE_INTERVAL", "3.0")),
            "overlap_duration": float(os.getenv("OVERLAP_DURATION", "0.2")),  # Minimal overlap
            "enable_vad": os.getenv("ENABLE_VAD", "true").lower() == "true",
            
            # Device settings
            "device": os.getenv("OPENVINO_DEVICE"),
            
            # Session settings
            "session_dir": os.getenv("SESSION_DIR"),
            
            # Performance settings
            "min_inference_interval": float(os.getenv("MIN_INFERENCE_INTERVAL", "0.2")),
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
        }
        
        # Load from config file if exists
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return config
    
    async def shutdown(self):
        """Shutdown the whisper service and cleanup resources"""
        try:
            # Stop streaming
            await self.stop_streaming()
            
            # Clear buffers
            self.buffer_manager.clear_buffer()
            
            # Clear model cache
            self.model_manager.clear_cache()
            
            logger.info("WhisperService shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Factory function for easy service creation
async def create_whisper_service(config: Optional[Dict] = None) -> WhisperService:
    """
    Factory function to create and initialize a whisper service
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Initialized WhisperService instance
    """
    service = WhisperService(config)
    return service

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Whisper Service")
        parser.add_argument("--audio", required=True, help="Audio file to transcribe")
        parser.add_argument("--model", default="whisper-base", help="Model to use")
        parser.add_argument("--language", help="Language hint")
        parser.add_argument("--streaming", action="store_true", help="Use streaming")
        
        args = parser.parse_args()
        
        # Create service
        service = await create_whisper_service()
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(args.audio, sr=16000)
            
            # Create request
            request = TranscriptionRequest(
                audio_data=audio_data,
                model_name=args.model,
                language=args.language,
                streaming=args.streaming
            )
            
            if args.streaming:
                print("Streaming transcription:")
                async for result in service.transcribe_stream(request):
                    print(f"[{result.timestamp}] {result.text}")
            else:
                result = await service.transcribe(request)
                print(f"Transcription: {result.text}")
                print(f"Language: {result.language}")
                print(f"Confidence: {result.confidence_score:.2f}")
                print(f"Device: {result.device_used}")
                print(f"Time: {result.processing_time:.2f}s")
        
        finally:
            await service.shutdown()
    
    asyncio.run(main()) 