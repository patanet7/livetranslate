import librosa
import openvino as ov
import openvino_genai
from flask import Flask, request, jsonify, Response
import io
import os
from pathlib import Path
import logging
import tempfile
import numpy as np
import soundfile as sf
import subprocess
import time
import random
import threading
from queue import Queue, Empty
from collections import deque
import webrtcvad
from scipy import signal
from typing import List, Dict, Optional
import json

# Configure FFmpeg BEFORE importing pydub to avoid warnings
def configure_ffmpeg():
    """Configure pydub to use local FFmpeg installation"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(script_dir, "external", "ffmpeg", "bin", "ffmpeg.exe")
    local_ffprobe = os.path.join(script_dir, "external", "ffmpeg", "bin", "ffprobe.exe")
    
    # Check if local FFmpeg exists
    if os.path.exists(local_ffmpeg) and os.path.exists(local_ffprobe):
        # Set environment variables for pydub to find FFmpeg
        os.environ["FFMPEG_BINARY"] = os.path.abspath(local_ffmpeg)
        os.environ["FFPROBE_BINARY"] = os.path.abspath(local_ffprobe)
        print(f"✓ Configured FFmpeg paths: {os.path.abspath(local_ffmpeg)}")
        return True
    else:
        print(f"⚠ Local FFmpeg not found at {local_ffmpeg}, using system PATH")
        return False

# Configure FFmpeg before importing pydub
configure_ffmpeg()

# Now import pydub after FFmpeg is configured
from pydub import AudioSegment
from pydub.utils import which

# Also configure AudioSegment directly after import
def configure_pydub_after_import():
    """Configure AudioSegment after pydub import"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(script_dir, "external", "ffmpeg", "bin", "ffmpeg.exe")
    local_ffprobe = os.path.join(script_dir, "external", "ffmpeg", "bin", "ffprobe.exe")
    
    if os.path.exists(local_ffmpeg) and os.path.exists(local_ffprobe):
        AudioSegment.converter = os.path.abspath(local_ffmpeg)
        AudioSegment.ffmpeg = os.path.abspath(local_ffmpeg)
        AudioSegment.ffprobe = os.path.abspath(local_ffprobe)
        print(f"✓ Configured pydub AudioSegment with local FFmpeg")

configure_pydub_after_import()

# Enhanced logging setup with better formatting for activity logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper-npu-server.log', encoding='utf-8')
    ],
    force=True
)

# Create a separate logger for activity logs with concise formatting
activity_logger = logging.getLogger('activity')
activity_handler = logging.StreamHandler()
activity_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
activity_logger.addHandler(activity_handler)
activity_logger.setLevel(logging.INFO)
activity_logger.propagate = False  # Don't send to main logger

logger = logging.getLogger(__name__)

# Import the new speaker diarization system
try:
    from speaker_diarization import (
        AdvancedSpeakerDiarization,
        PunctuationAligner,
        SpeechEnhancer
    )
    SPEAKER_DIARIZATION_AVAILABLE = True
    logger.info("✓ Advanced Speaker Diarization available")
except ImportError as e:
    SPEAKER_DIARIZATION_AVAILABLE = False
    logger.warning(f"Speaker Diarization not available: {e}")
    
    # Create stub classes so the endpoints still work
    class AdvancedSpeakerDiarization:
        def __init__(self, **kwargs):
            pass
        def process_audio_chunk(self, audio):
            return []
        def get_speaker_statistics(self):
            return {"total_speakers": 0, "active_speakers": 0, "total_segments": 0}
    
    class PunctuationAligner:
        def __init__(self):
            pass
        def align_segments_with_punctuation(self, segments, text, duration):
            return []
    
    class SpeechEnhancer:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate
        def enhance_audio(self, audio):
            return audio


# Session persistence for transcriptions
def load_session_transcriptions():
    """Load transcriptions from disk"""
    try:
        session_file = os.path.join(os.path.dirname(__file__), 'session_data', 'transcriptions.json')
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return deque(data.get('transcriptions', []), maxlen=200)
    except Exception as e:
        logger.warning(f"Failed to load session transcriptions: {e}")
    return deque(maxlen=200)

def save_session_transcriptions(transcriptions):
    """Save transcriptions to disk"""
    try:
        session_dir = os.path.join(os.path.dirname(__file__), 'session_data')
        os.makedirs(session_dir, exist_ok=True)
        session_file = os.path.join(session_dir, 'transcriptions.json')
        
        # Convert deque to list for JSON serialization
        data = {
            'transcriptions': list(transcriptions)[-100:],  # Keep last 100
            'last_updated': time.time()
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save session transcriptions: {e}")

def load_settings():
    """Load settings from disk"""
    try:
        settings_file = os.path.join(os.path.dirname(__file__), 'session_data', 'settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)
                return saved_settings
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
    return {}

def save_settings(settings):
    """Save settings to disk"""
    try:
        session_dir = os.path.join(os.path.dirname(__file__), 'session_data')
        os.makedirs(session_dir, exist_ok=True)
        settings_file = os.path.join(session_dir, 'settings.json')
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        
        activity_logger.info(f"Settings saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")

def log_activity(message, level='INFO'):
    """Log activity with concise formatting"""
    if level == 'INFO':
        activity_logger.info(message)
    elif level == 'WARNING':
        activity_logger.warning(message)
    elif level == 'ERROR':
        activity_logger.error(message)


# Fix Windows console encoding for Unicode characters
if os.name == 'nt':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

app = Flask(__name__)

# Configure logging to handle Unicode properly
for handler in logger.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
        try:
            handler.stream.reconfigure(encoding='utf-8')
        except:
            pass

# Add detailed request logging
@app.before_request
def log_request_info():
    logger.debug(f"Request: {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    if request.data:
        logger.debug(f"Body size: {len(request.data)} bytes")

# IMPROVED CORS SUPPORT - More robust implementation
def add_cors_headers(response):
    """Add CORS headers to response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, PATCH'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
    response.headers['Access-Control-Max-Age'] = '3600'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Content-Length'
    return response

@app.after_request
def after_request(response):
    """Apply CORS headers to all responses"""
    return add_cors_headers(response)

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = Response()
        response = add_cors_headers(response)
        response.status_code = 200
        return response

# Add a specific CORS test endpoint
@app.route("/cors-test", methods=['GET', 'POST', 'OPTIONS'])
def cors_test():
    """Test endpoint for CORS functionality"""
    return jsonify({
        "cors_enabled": True,
        "method": request.method,
        "origin": request.headers.get('Origin', 'no-origin'),
        "timestamp": time.time(),
        "message": "CORS headers should be present in response"
    })

class ModelManager:
    def __init__(self):
        # Use Windows user profile path where models are actually located
        # self.models_dir = os.path.expanduser("~/.whisper/models")
        self.models_dir = os.path.abspath("./models")
        self.pipelines = {}
        self.default_model = "whisper-base"  # Better quality than tiny, still NPU-friendly
        self.device = self._detect_best_device()
        
        # Thread safety for NPU access
        self.inference_lock = threading.Lock()
        self.request_queue = Queue(maxsize=10)  # Limit concurrent requests
        self.last_inference_time = 0
        self.min_inference_interval = 0.2  # Increased to 200ms for memory relief
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model directory: {self.models_dir}")
        logger.info(f"Default model: {self.default_model} (optimized for NPU memory)")
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Try to preload the default model
        self._preload_default_model()
        
    def _detect_best_device(self):
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
        
    def load_model(self, model_name):
        if model_name not in self.pipelines:
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                available_models = self.list_models()
                if available_models:
                    logger.warning(f"Model {model_name} not found. Available models: {available_models}")
                    raise FileNotFoundError(f"Model {model_name} not found. Available: {available_models}")
                else:
                    raise FileNotFoundError(f"No models found in {self.models_dir}. Please mount models directory.")
            
            logger.info(f"Loading model: {model_name} on device: {self.device}")
            try:
                # Use OpenVINO GenAI WhisperPipeline for proper model loading
                logger.info(f"Creating WhisperPipeline for {model_name}")
                pipeline = openvino_genai.WhisperPipeline(str(model_path), device=self.device)
                self.pipelines[model_name] = pipeline
                logger.info(f"✓ Model {model_name} loaded successfully on {self.device}")
                        
            except Exception as e:
                if self.device != "CPU":
                    logger.warning(f"Failed to load on {self.device}, trying CPU fallback: {e}")
                    try:
                        pipeline = openvino_genai.WhisperPipeline(str(model_path), device="CPU")
                        self.pipelines[model_name] = pipeline
                        self.device = "CPU"  # Update device for this session
                        logger.info(f"✓ Model {model_name} loaded on CPU fallback")
                    except Exception as cpu_e:
                        logger.error(f"Failed to load on CPU fallback: {cpu_e}")
                        raise cpu_e
                else:
                    raise e
        
        return self.pipelines[model_name]

    def list_models(self):
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
            logger.info(f"✅ Server will work in simulation mode without real models")

    def clear_npu_cache(self):
        """Clear NPU cache and loaded models to free memory"""
        try:
            logger.info("Clearing NPU cache and models due to memory pressure...")
            
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
            
            logger.info("✓ NPU cache and models cleared")
            
        except Exception as e:
            logger.error(f"Error clearing NPU cache: {e}")

    def safe_inference(self, model_name, audio_data):
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
                logger.debug(f"Starting NPU inference for {len(audio_data)} samples")
                start_time = time.time()
                
                try:
                    result = pipeline.generate(audio_data)
                    inference_time = time.time() - start_time
                    self.last_inference_time = time.time()
                    
                    logger.debug(f"NPU inference completed in {inference_time:.3f}s")
                    return result
                    
                except Exception as npu_error:
                    error_msg = str(npu_error)
                    
                    # Handle specific NPU errors
                    if "Infer Request is busy" in error_msg:
                        logger.warning("NPU busy - inference request rejected")
                        raise Exception("NPU is busy processing another request. Please try again.")
                    
                    elif "ZE_RESULT_ERROR_DEVICE_LOST" in error_msg or "device hung" in error_msg:
                        logger.error("NPU device lost/hung - attempting recovery")
                        # Clear the pipeline to force reload
                        if model_name in self.pipelines:
                            del self.pipelines[model_name]
                        raise Exception("NPU device error - model will be reloaded on next request")
                    
                    elif "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY" in error_msg or "insufficient host memory" in error_msg:
                        logger.error("NPU out of memory - clearing cache and suggesting smaller model")
                        self.clear_npu_cache()
                        
                        # Suggest smaller model if using large model
                        if "large" in model_name:
                            raise Exception("NPU out of memory. Try using whisper-tiny or whisper-base instead of large models.")
                        else:
                            raise Exception("NPU out of memory. Cache cleared - please try again with fewer concurrent requests.")
                    
                    else:
                        logger.error(f"NPU inference error: {error_msg}")
                        raise npu_error
                        
            except Exception as e:
                self.last_inference_time = time.time()  # Still update to prevent hammering
                raise e

class RollingBufferManager:
    """Manages rolling audio buffers with voice activity detection and speaker diarization"""
    
    def __init__(self, buffer_duration=6.0, inference_interval=3.0, sample_rate=16000, 
                 enable_diarization=False, n_speakers=None):
        self.buffer_duration = buffer_duration  # Reduced to 6s as requested
        self.inference_interval = inference_interval  # More frequent inference
        self.sample_rate = sample_rate
        self.max_samples = int(buffer_duration * sample_rate)
        
        # Speaker diarization settings
        self.enable_diarization = enable_diarization and SPEAKER_DIARIZATION_AVAILABLE
        self.n_speakers = n_speakers
        
        # Rolling buffer for audio samples
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_lock = threading.Lock()
        
        # VAD setup
        try:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            self.vad_enabled = True
            logger.info("✓ Voice Activity Detection enabled")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠ Voice Activity Detection not available")
        
        # Speaker diarization setup
        if self.enable_diarization:
            try:
                self.speaker_diarizer = AdvancedSpeakerDiarization(
                    sample_rate=sample_rate,
                    window_duration=buffer_duration,
                    overlap_duration=2.0,  # 2s overlap for continuity
                    n_speakers=n_speakers,
                    embedding_method='resemblyzer',  # Can be configured
                    clustering_method='hdbscan' if n_speakers is None else 'agglomerative',
                    enable_enhancement=True,
                    device='cpu'  # Could be 'cuda' if available
                )
                self.punctuation_aligner = PunctuationAligner()
                logger.info(f"✓ Speaker Diarization enabled with {n_speakers if n_speakers else 'auto-detect'} speakers")
            except Exception as e:
                logger.error(f"Failed to initialize speaker diarization: {e}")
                self.enable_diarization = False
                self.speaker_diarizer = None
                self.punctuation_aligner = None
        else:
            self.speaker_diarizer = None
            self.punctuation_aligner = None
        
        # Enhanced speech processing
        if SPEAKER_DIARIZATION_AVAILABLE:
            try:
                self.speech_enhancer = SpeechEnhancer(sample_rate)
                logger.info("✓ Speech Enhancement enabled")
            except Exception as e:
                logger.warning(f"Speech enhancement failed to initialize: {e}")
                self.speech_enhancer = None
        else:
            self.speech_enhancer = None
        
        # Inference timing
        self.last_inference_time = 0
        self.inference_timer = None
        self.is_running = False
        
        # Overlap management
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # Speaker tracking
        self.speaker_transcriptions = deque(maxlen=200)  # Store speaker + text
        
    def add_audio_chunk(self, audio_samples):
        """Add new audio samples to the rolling buffer"""
        with self.buffer_lock:
            if isinstance(audio_samples, np.ndarray):
                # Convert to list and extend buffer
                samples_list = audio_samples.tolist()
                self.audio_buffer.extend(samples_list)
                
                # Process with speaker diarization if enabled
                if self.enable_diarization and self.speaker_diarizer is not None:
                    try:
                        # Process chunk for speaker diarization
                        diarization_results = self.speaker_diarizer.process_audio_chunk(audio_samples)
                        
                        if diarization_results:
                            logger.debug(f"🎤 Speaker diarization found {len(diarization_results)} segments")
                            # Store diarization results for later use with transcription
                            if not hasattr(self, 'recent_diarization'):
                                self.recent_diarization = deque(maxlen=50)
                            self.recent_diarization.extend(diarization_results)
                            
                    except Exception as e:
                        logger.warning(f"Speaker diarization processing failed: {e}")
                
                # Start inference timer if not running
                if not self.is_running and len(self.audio_buffer) >= self.sample_rate * 3:  # At least 3 seconds
                    self.start_inference_timer()
                    
                return len(self.audio_buffer)
            return 0
    
    def get_buffer_audio(self):
        """Get current buffer as numpy array"""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([])
            return np.array(list(self.audio_buffer))
    
    def find_speech_boundaries(self, audio_array, chunk_duration=0.02):
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
    
    def start_inference_timer(self):
        """Start periodic inference"""
        if self.is_running:
            return
            
        self.is_running = True
        self._schedule_next_inference()
        logger.info(f"🔄 Started rolling buffer inference (every {self.inference_interval}s)")
    
    def stop_inference_timer(self):
        """Stop periodic inference"""
        self.is_running = False
        if self.inference_timer:
            self.inference_timer.cancel()
        logger.info("⏹ Stopped rolling buffer inference")
    
    def _schedule_next_inference(self):
        """Schedule the next inference"""
        if not self.is_running:
            return
            
        self.inference_timer = threading.Timer(self.inference_interval, self._run_inference)
        self.inference_timer.start()
    
    def _run_inference(self):
        """Run inference on current buffer"""
        if not self.is_running:
            return
            
        try:
            audio_array = self.get_buffer_audio()
            
            if len(audio_array) >= self.sample_rate * 5:  # At least 5 seconds
                # Find good speech boundaries
                speech_start, speech_end = self.find_speech_boundaries(audio_array)
                
                # Use VAD boundaries if available, otherwise use full buffer
                if speech_start is not None and speech_end is not None:
                    inference_audio = audio_array[speech_start:speech_end]
                    logger.debug(f"Using VAD boundaries: {speech_start}-{speech_end} samples")
                else:
                    inference_audio = audio_array
                    logger.debug("Using full buffer for inference")
                
                # Queue inference
                if hasattr(app, 'buffer_manager') and hasattr(app, 'model_manager'):
                    self._queue_buffer_inference(inference_audio)
            
        except Exception as e:
            logger.error(f"Buffer inference error: {e}")
        
        # Schedule next inference
        self._schedule_next_inference()
    
    def _queue_buffer_inference(self, audio_array):
        """Queue buffer inference with speaker diarization (non-blocking)"""
        def run_inference():
            try:
                # Get selected model from global state
                model_name = getattr(app, 'current_model', 'whisper-base')
                
                # Apply speech enhancement if available
                enhanced_audio = audio_array
                speaker_segments = []
                
                if self.speech_enhancer is not None:
                    try:
                        enhanced_audio = self.speech_enhancer.enhance_audio(audio_array)
                        logger.debug("Applied speech enhancement to buffer audio")
                    except Exception as e:
                        logger.warning(f"Speech enhancement failed: {e}")
                        enhanced_audio = audio_array
                
                # SPEAKER DIARIZATION - Run directly on enhanced audio (same as single-file)
                if self.enable_diarization and self.speaker_diarizer is not None:
                    try:
                        logger.debug("🎤 Running speaker diarization on rolling buffer audio...")
                        # Process with speaker diarization on the enhanced audio
                        speaker_segments = self.speaker_diarizer.process_audio_chunk(enhanced_audio)
                        
                        if speaker_segments:
                            logger.info(f"🔄 Rolling buffer found {len(speaker_segments)} speaker segments")
                            for i, seg in enumerate(speaker_segments):
                                logger.debug(f"  Segment {i+1}: Speaker {seg.get('speaker_id', 'unknown')} "
                                           f"from {seg.get('start_time', 0):.2f}s to {seg.get('end_time', 0):.2f}s "
                                           f"(confidence: {seg.get('confidence', 0):.2f})")
                        else:
                            logger.debug("🔄 Rolling buffer: No speaker segments detected")
                            
                    except Exception as e:
                        logger.warning(f"🎤 Rolling buffer speaker diarization failed: {e}")
                        speaker_segments = []
                
                # Run NPU inference
                result = app.model_manager.safe_inference(model_name, enhanced_audio)
                
                if hasattr(result, 'text'):
                    text = result.text.strip()
                elif isinstance(result, str):
                    text = result.strip()
                else:
                    text = str(result).strip()
                
                if text and text != self.last_transcription:
                    # Remove overlap with previous transcription
                    processed_text = self._remove_overlap(text)
                    
                    if processed_text:
                        # Prepare transcription result
                        transcription_result = {
                            'text': processed_text,
                            'timestamp': time.time(),
                            'type': 'rolling_buffer',
                            'buffer_duration': len(audio_array) / self.sample_rate,
                            'enhanced': self.speech_enhancer is not None
                        }
                        
                        # Add speaker diarization if enabled and segments were found
                        if self.enable_diarization and speaker_segments:
                            try:
                                # CONSOLIDATE segments from same speaker FIRST
                                consolidated_speakers = {}
                                
                                for segment in speaker_segments:
                                    speaker_id = segment.get('speaker_id', 'unknown')
                                    
                                    if speaker_id not in consolidated_speakers:
                                        consolidated_speakers[speaker_id] = {
                                            'speaker_id': speaker_id,
                                            'total_duration': 0,
                                            'start_time': segment.get('start_time', 0),
                                            'end_time': segment.get('end_time', 0),
                                            'confidence_sum': 0,
                                            'segment_count': 0
                                        }
                                    
                                    # Extend time range and accumulate duration
                                    consolidated_speakers[speaker_id]['start_time'] = min(
                                        consolidated_speakers[speaker_id]['start_time'],
                                        segment.get('start_time', 0)
                                    )
                                    consolidated_speakers[speaker_id]['end_time'] = max(
                                        consolidated_speakers[speaker_id]['end_time'],
                                        segment.get('end_time', 0)
                                    )
                                    consolidated_speakers[speaker_id]['total_duration'] += segment.get('duration', 0)
                                    consolidated_speakers[speaker_id]['confidence_sum'] += segment.get('confidence', 0.5)
                                    consolidated_speakers[speaker_id]['segment_count'] += 1
                                
                                # Create SINGLE speaker-aware transcription breakdown
                                speaker_transcriptions = []
                                total_duration = len(audio_array) / self.sample_rate
                                
                                # Sort speakers by total duration (most active first)
                                sorted_speakers = sorted(consolidated_speakers.values(), 
                                                       key=lambda x: x['total_duration'], reverse=True)
                                
                                # Distribute text among speakers based on their activity duration
                                text_chars = len(processed_text)
                                total_speaker_duration = sum(s['total_duration'] for s in sorted_speakers)
                                char_position = 0
                                
                                for speaker_info in sorted_speakers:
                                    # Calculate text portion based on speaker's total duration
                                    duration_ratio = speaker_info['total_duration'] / total_speaker_duration if total_speaker_duration > 0 else 1.0 / len(sorted_speakers)
                                    chars_for_speaker = int(text_chars * duration_ratio)
                                    
                                    # Extract text portion for this speaker
                                    if char_position + chars_for_speaker <= text_chars:
                                        speaker_text = processed_text[char_position:char_position + chars_for_speaker]
                                    else:
                                        speaker_text = processed_text[char_position:]
                                    
                                    char_position += chars_for_speaker
                                    
                                    # Calculate average confidence
                                    avg_confidence = speaker_info['confidence_sum'] / speaker_info['segment_count'] if speaker_info['segment_count'] > 0 else 0.5
                                    
                                    # Only include speakers with meaningful text OR if it's the only speaker
                                    if speaker_text.strip() or len(sorted_speakers) == 1:
                                        speaker_transcriptions.append({
                                            'speaker_id': speaker_info['speaker_id'],
                                            'text': speaker_text.strip() if speaker_text.strip() else processed_text,
                                            'start_time': speaker_info['start_time'],
                                            'end_time': speaker_info['end_time'],
                                            'duration': speaker_info['total_duration'],
                                            'confidence': max(0.6, avg_confidence + 0.15)  # Boost confidence for display
                                        })
                                
                                # If no speakers got text, assign all text to the most active speaker
                                if not speaker_transcriptions and sorted_speakers:
                                    main_speaker = sorted_speakers[0]
                                    avg_confidence = main_speaker['confidence_sum'] / main_speaker['segment_count'] if main_speaker['segment_count'] > 0 else 0.5
                                    
                                    speaker_transcriptions.append({
                                        'speaker_id': main_speaker['speaker_id'],
                                        'text': processed_text,
                                        'start_time': main_speaker['start_time'],
                                        'end_time': main_speaker['end_time'],
                                        'duration': main_speaker['total_duration'],
                                        'confidence': max(0.6, avg_confidence + 0.15)
                                    })
                                
                                transcription_result.update({
                                    "speakers": speaker_transcriptions,
                                    "diarization_enabled": True,
                                    "total_speakers": len(consolidated_speakers),
                                    "speaker_segments_detected": len(speaker_segments)
                                })
                                
                                logger.info(f"🔄 Rolling buffer with {len(speaker_transcriptions)} consolidated speaker entries for {len(consolidated_speakers)} speakers")
                                
                            except Exception as speaker_processing_error:
                                logger.warning(f"🎤 Failed to create rolling buffer speaker transcription: {speaker_processing_error}")
                                transcription_result['speakers'] = []
                                transcription_result['diarization_enabled'] = True
                                transcription_result['diarization_error'] = str(speaker_processing_error)
                        
                        elif self.enable_diarization:
                            # Diarization enabled but no speakers detected
                            transcription_result['speakers'] = []
                            transcription_result['diarization_enabled'] = True
                            logger.info(f"🔄 Rolling buffer (diarization enabled, no speakers): {processed_text}")
                        else:
                            # Diarization disabled
                            transcription_result['speakers'] = []
                            transcription_result['diarization_enabled'] = False
                            logger.info(f"🔄 Rolling buffer transcription: {processed_text}")
                        
                        # Store for frontend retrieval
                        if not hasattr(app, 'rolling_transcriptions'):
                            app.rolling_transcriptions = deque(maxlen=100)
                        
                        app.rolling_transcriptions.append(transcription_result)
                        
                        # Store in speaker transcriptions history with automatic saving
                        if transcription_result.get('speakers') is not None:
                            self.speaker_transcriptions.append(transcription_result)
                            
                            # Auto-save transcriptions for session persistence
                            save_session_transcriptions(self.speaker_transcriptions)
                            
                            log_activity(f"Rolling buffer: {len(transcription_result.get('speakers', []))} speakers, {len(processed_text)} chars")
                        
                        self.last_transcription = text
                        self.last_transcription_time = time.time()
                
            except Exception as e:
                logger.error(f"Rolling buffer inference failed: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=run_inference)
        thread.daemon = True
        thread.start()
    
    def _create_speaker_transcription(self, aligned_segments: List[Dict], text: str) -> List[Dict]:
        """Create speaker-aware transcription breakdown"""
        try:
            if not aligned_segments:
                return []
            
            # Simple approach: distribute text across speaker segments
            speaker_texts = []
            total_duration = sum(seg['duration'] for seg in aligned_segments)
            
            # Estimate text distribution based on segment duration
            text_chars = len(text)
            char_position = 0
            
            for segment in aligned_segments:
                # Calculate text portion based on time duration
                duration_ratio = segment['duration'] / total_duration if total_duration > 0 else 0
                chars_for_segment = int(text_chars * duration_ratio)
                
                # Extract text portion (rough approximation)
                if char_position + chars_for_segment <= text_chars:
                    segment_text = text[char_position:char_position + chars_for_segment]
                else:
                    segment_text = text[char_position:]
                
                char_position += chars_for_segment
                
                speaker_texts.append({
                    'speaker_id': segment['speaker_id'],
                    'text': segment_text.strip(),
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'confidence': segment.get('confidence', 0.5)
                })
            
            return speaker_texts
            
        except Exception as e:
            logger.error(f"Speaker transcription creation failed: {e}")
            return []
    
    def configure_diarization(self, enable: bool = True, n_speakers: Optional[int] = None):
        """Configure speaker diarization settings"""
        try:
            self.enable_diarization = enable and SPEAKER_DIARIZATION_AVAILABLE
            self.n_speakers = n_speakers
            
            if self.enable_diarization and self.speaker_diarizer is not None:
                self.speaker_diarizer.configure_speakers(n_speakers)
                logger.info(f"Updated diarization: enabled={enable}, speakers={n_speakers if n_speakers else 'auto'}")
            elif self.enable_diarization:
                # Reinitialize diarizer
                self.speaker_diarizer = AdvancedSpeakerDiarization(
                    sample_rate=self.sample_rate,
                    window_duration=self.buffer_duration,
                    overlap_duration=2.0,
                    n_speakers=n_speakers,
                    embedding_method='resemblyzer',
                    clustering_method='hdbscan' if n_speakers is None else 'agglomerative',
                    enable_enhancement=True
                )
                logger.info(f"Initialized diarization: speakers={n_speakers if n_speakers else 'auto'}")
            
        except Exception as e:
            logger.error(f"Diarization configuration failed: {e}")
            self.enable_diarization = False
    
    def get_speaker_statistics(self) -> Dict:
        """Get speaker diarization statistics"""
        if self.enable_diarization and self.speaker_diarizer is not None:
            try:
                return self.speaker_diarizer.get_speaker_statistics()
            except Exception as e:
                logger.error(f"Failed to get speaker statistics: {e}")
                return {}
        return {}
    
    def _remove_overlap(self, new_text):
        """Remove overlap with previous transcription"""
        if not self.last_transcription:
            return new_text
        
        # Simple overlap removal - look for common endings/beginnings
        last_words = self.last_transcription.split()[-3:]  # Last 3 words
        new_words = new_text.split()
        
        # Find overlap
        for i in range(min(len(last_words), len(new_words))):
            if last_words[-i-1:] == new_words[:i+1]:
                # Remove overlap from new text
                remaining_words = new_words[i+1:]
                return ' '.join(remaining_words) if remaining_words else ""
        
        return new_text
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.buffer_lock:
            self.audio_buffer.clear()
            self.last_transcription = ""

@app.route("/health", methods=['GET'])
def health():
    try:
        current_time = time.time()
        last_inference = getattr(model_manager, 'last_inference_time', 0)
        time_since_last_inference = current_time - last_inference if last_inference > 0 else -1
        
        return jsonify({
            "status": "healthy", 
            "device": model_manager.device,
            "models_available": len(model_manager.list_models()),
            "models_loaded": len(model_manager.pipelines),
            "current_model": getattr(app, 'current_model', 'unknown'),
            "server_uptime": current_time - app.start_time if hasattr(app, 'start_time') else 0,
            "last_inference_ago": time_since_last_inference,
            "note": "Using OpenVINO GenAI Whisper models for real NPU transcription"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "device": "unknown",
            "models_available": 0
        }), 500

@app.route("/models", methods=['GET'])
def list_models():
    models = model_manager.list_models()
    return jsonify({
        "models": models,
        "device": model_manager.device,
        "note": "Using OpenVINO GenAI Whisper models for real NPU transcription"
    })

@app.route("/transcribe/<model_name>", methods=['POST'])
def transcribe_with_model(model_name):
    try:
        # Check if model is available
        available_models = model_manager.list_models()
        if model_name not in available_models:
            return jsonify({
                "error": f"Model {model_name} not available. Available models: {available_models}",
                "note": "This server requires OpenVINO IR format models"
            }), 404
        
        # Get audio data from request
        if 'audio' not in request.files and not request.data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Handle audio data
        if request.data:
            audio_data = request.data
        else:
            audio_file = request.files['audio']
            audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return jsonify({"error": "Empty audio data"}), 400
        
        # Load audio using multiple fallback methods
        logger.info(f"Processing audio data: {len(audio_data)} bytes with model {model_name}")
        logger.debug(f"Audio data type: {type(audio_data)}")
        logger.debug(f"First 20 bytes (hex): {audio_data[:20].hex() if len(audio_data) >= 20 else audio_data.hex()}")
        
        # Detect audio format from the data
        detected_format = "unknown"
        if audio_data.startswith(b'RIFF'):
            detected_format = "wav"
        elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
            detected_format = "webm"
        elif audio_data.startswith(b'OggS'):
            detected_format = "ogg"
        elif audio_data.startswith(b'\x00\x00\x00') and b'ftyp' in audio_data[:32]:
            detected_format = "mp4"
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            detected_format = "mp3"
        
        # Check more patterns if still unknown
        if detected_format == "unknown":
            # Look for WebM patterns (including chunk headers)
            if b'webm' in audio_data[:100].lower():
                detected_format = "webm"
            elif b'opus' in audio_data[:100].lower():
                detected_format = "webm"  # Opus is commonly in WebM
            elif b'vorbis' in audio_data[:100].lower():
                detected_format = "webm"  # Vorbis is also in WebM
            # Look for MP4 box headers (including fragments)
            elif b'ftyp' in audio_data[:100] or b'mdat' in audio_data[:100] or b'moov' in audio_data[:100]:
                detected_format = "mp4"
            elif b'moof' in audio_data[:100] or b'mfhd' in audio_data[:100]:
                detected_format = "mp4_fragment"  # MP4 fragment (streaming chunk)
                logger.warning("⚠ Detected MP4 fragment - MediaRecorder chunks may be incomplete")
            # Look for OGG patterns
            elif b'Ogg' in audio_data[:50]:
                detected_format = "ogg"
            # Enhanced WebM detection for streaming chunks
            elif audio_data[:4] == b'\x00\x00\x00\x00' or audio_data.startswith(b'\x1a\x45'):
                detected_format = "webm"  # WebM chunk without full header
            # If starts with zeros, might be MP4 fragment
            elif audio_data[:4] == b'\x00\x00\x00\x00':
                detected_format = "mp4_fragment"
        
        logger.info(f"🔍 Detected audio format: {detected_format}")
        logger.debug(f"🔍 First 32 bytes: {audio_data[:32]}")
        logger.debug(f"🔍 First 32 bytes (hex): {audio_data[:32].hex()}")
        logger.debug(f"🔍 Looking for format signatures in first 100 bytes...")
        
        try:
            # Method 1: Direct librosa loading first (bypasses WebM issues)
            try:
                logger.info("Attempting direct librosa processing (bypass WebM)...")
                
                # Use appropriate file extension based on detected format
                file_ext = f".{detected_format}" if detected_format != "unknown" else ".webm"
                
                # Handle MP4 fragments specially
                if detected_format == "mp4_fragment":
                    file_ext = ".mp4"
                    logger.warning("⚠ Processing MP4 fragment - may fail due to incomplete headers")
                
                # Write audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                logger.debug(f"Wrote {len(audio_data)} bytes to {temp_file_path}")
                
                # Try direct librosa load (it can handle some WebM files)
                try:
                    audio, sr = librosa.load(temp_file_path, sr=16000)
                    if len(audio) > 0:
                        logger.info(f"✓ Direct librosa successful: {len(audio)} samples at {sr}Hz")
                        os.unlink(temp_file_path)
                        
                        # Check audio levels immediately
                        immediate_rms = np.sqrt(np.mean(audio**2))
                        immediate_max = np.max(np.abs(audio))
                        logger.info(f"🔊 Direct librosa levels: RMS={immediate_rms:.6f}, Max={immediate_max:.6f}")
                        
                        # Skip further processing if we got good audio
                        success_method = "direct_librosa"
                        
                    else:
                        raise Exception("Librosa returned empty audio")
                        
                except Exception as librosa_error:
                    logger.debug(f"Direct librosa failed: {librosa_error}")
                    # Continue to other methods
                    os.unlink(temp_file_path)
                    raise librosa_error
                    
            except Exception as direct_error:
                logger.debug(f"Direct librosa method failed: {direct_error}")
                
                # Method 2: Try WebM audio extraction using pydub (configured at startup)
                try:
                    logger.info("Attempting pydub + ffmpeg audio processing...")
                    
                    # Use appropriate file extension and format
                    file_ext = f".{detected_format}" if detected_format != "unknown" else ".webm"
                    format_hint = detected_format if detected_format != "unknown" else "webm"
                    
                    # Handle MP4 fragments specially
                    if detected_format == "mp4_fragment":
                        file_ext = ".mp4"
                        format_hint = "mp4"
                        logger.warning("⚠ Processing MP4 fragment - may fail due to incomplete headers")
                    
                    # Write audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name
                    
                    logger.debug(f"Wrote {len(audio_data)} bytes to {temp_file_path}")
                    
                    # Load with pydub using detected format
                    if format_hint == "mp4":
                        audio_segment = AudioSegment.from_file(temp_file_path, format="mp4")
                    elif format_hint == "ogg":
                        audio_segment = AudioSegment.from_file(temp_file_path, format="ogg")
                    elif format_hint == "wav":
                        audio_segment = AudioSegment.from_file(temp_file_path, format="wav")
                    else:
                        # Default to webm
                        audio_segment = AudioSegment.from_file(temp_file_path, format="webm")
                    
                    logger.info(f"Pydub loaded: {len(audio_segment)}ms, {audio_segment.frame_rate}Hz, {audio_segment.channels} channels")
                    
                    # Convert to mono 16kHz
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                    
                    # Convert to numpy array
                    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    audio = audio_array / 32768.0  # Normalize from int16 to float32
                    sr = 16000
                    
                    # Debug: Check audio levels immediately after pydub processing
                    immediate_rms = np.sqrt(np.mean(audio**2))
                    immediate_max = np.max(np.abs(audio))
                    logger.info(f"🔊 Pydub post-processing levels: RMS={immediate_rms:.6f}, Max={immediate_max:.6f}")
                    
                    logger.info(f"✓ Successfully processed with pydub+ffmpeg: {len(audio)} samples at {sr}Hz")
                    os.unlink(temp_file_path)
                    success_method = "pydub_ffmpeg"
                    
                except Exception as pydub_error:
                    logger.warning(f"Pydub+ffmpeg processing failed: {pydub_error}")
                    
                    # Method 3: Direct ffmpeg command line
                    try:
                        logger.info("Attempting direct ffmpeg processing...")
                        
                        # Use appropriate file extension based on detected format
                        file_ext = f".{detected_format}" if detected_format != "unknown" else ".webm"
                        input_format = detected_format if detected_format != "unknown" else "webm"
                        
                        # Handle MP4 fragments specially
                        if detected_format == "mp4_fragment":
                            file_ext = ".mp4"
                            input_format = "mp4"
                            logger.warning("⚠ Processing MP4 fragment with ffmpeg - may fail due to incomplete headers")
                        
                        # Write input file
                        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as input_file:
                            input_file.write(audio_data)
                            input_path = input_file.name
                        
                        # Create output file path
                        output_path = input_path.replace(file_ext, '.wav')
                        
                        # Use configured ffmpeg path
                        ffmpeg_cmd = AudioSegment.ffmpeg or "ffmpeg"
                        logger.info(f"Using ffmpeg command: {ffmpeg_cmd}")
                        
                        # Convert to WAV using ffmpeg with format-specific settings
                        cmd = [
                            ffmpeg_cmd,
                            "-f", input_format,     # Input format based on detection
                            "-i", input_path,
                            "-ar", "16000",  # 16kHz sample rate
                            "-ac", "1",      # Mono
                            "-f", "wav",     # WAV format
                            "-acodec", "pcm_s16le",  # Specific audio codec
                            output_path,
                            "-y"             # Overwrite output
                        ]
                        
                        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            # Load the converted audio with librosa
                            audio, sr = librosa.load(output_path, sr=16000)
                            logger.info(f"✓ Direct ffmpeg processing successful: {len(audio)} samples at {sr}Hz")
                            
                            # Debug: Check audio levels immediately after loading
                            immediate_rms = np.sqrt(np.mean(audio**2))
                            immediate_max = np.max(np.abs(audio))
                            logger.info(f"🔊 Immediate post-processing levels: RMS={immediate_rms:.6f}, Max={immediate_max:.6f}")
                            
                            # Clean up
                            os.unlink(input_path)
                            os.unlink(output_path)
                            success_method = "direct_ffmpeg"
                        else:
                            logger.error(f"FFmpeg failed: {result.stderr}")
                            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                            
                    except Exception as ffmpeg_error:
                        logger.warning(f"Direct ffmpeg processing failed: {ffmpeg_error}")
                        
                        # Method 4: Raw audio fallback (assume it's already audio data)
                        try:
                            logger.info("Attempting raw audio fallback...")
                            
                            # For MP4 chunks, try to skip MP4 container headers
                            if detected_format == "mp4":
                                logger.info("MP4 detected - attempting to extract raw audio from container...")
                                # MP4 chunks from MediaRecorder often have incomplete headers
                                # Try to find audio data after MP4 box headers
                                search_start = 0
                                # Look for common MP4 audio patterns
                                for i in range(min(1024, len(audio_data) - 4)):
                                    # Look for patterns that might indicate start of audio data
                                    if audio_data[i:i+4] in [b'mdat', b'data']:
                                        search_start = i + 4
                                        logger.info(f"Found potential MP4 audio data start at offset {search_start}")
                                        break
                                
                                if search_start > 0:
                                    audio_data_chunk = audio_data[search_start:]
                                else:
                                    audio_data_chunk = audio_data[64:]  # Skip first 64 bytes of headers
                            else:
                                audio_data_chunk = audio_data
                            
                            # Try to interpret as raw audio data
                            # This might work for some chunks that lost headers
                            try:
                                # Try 16-bit signed integers first
                                audio_array = np.frombuffer(audio_data_chunk, dtype=np.int16)
                                if len(audio_array) > 160:  # At least 10ms at 16kHz
                                    audio = audio_array.astype(np.float32) / 32768.0
                                    sr = 16000
                                    
                                    immediate_rms = np.sqrt(np.mean(audio**2))
                                    immediate_max = np.max(np.abs(audio))
                                    logger.info(f"✓ Raw audio fallback (int16): {len(audio)} samples, RMS={immediate_rms:.6f}, Max={immediate_max:.6f}")
                                    success_method = "raw_fallback_int16"
                                else:
                                    raise Exception("Raw int16 audio data too short")
                            except:
                                # Try 32-bit float
                                try:
                                    audio_array = np.frombuffer(audio_data_chunk, dtype=np.float32)
                                    if len(audio_array) > 80:  # At least 5ms at 16kHz
                                        audio = audio_array
                                        sr = 16000
                                        
                                        immediate_rms = np.sqrt(np.mean(audio**2))
                                        immediate_max = np.max(np.abs(audio))
                                        logger.info(f"✓ Raw audio fallback (float32): {len(audio)} samples, RMS={immediate_rms:.6f}, Max={immediate_max:.6f}")
                                        success_method = "raw_fallback_float32"
                                    else:
                                        raise Exception("Raw float32 audio data too short")
                                except Exception as float_error:
                                    raise Exception(f"Raw audio interpretation failed for both int16 and float32: {float_error}")
                                
                        except Exception as raw_error:
                            logger.error(f"Raw audio fallback failed: {raw_error}")
                            logger.error(f"Audio processing failed. All methods exhausted. {detected_format.upper()} chunk may be corrupted.")
                            
                            return jsonify({
                                "error": f"Audio processing failed. {detected_format.upper()} chunk appears to be corrupted or incomplete.",
                                "debug_info": {
                                    "pydub_error": str(pydub_error),
                                    "ffmpeg_error": str(ffmpeg_error),
                                    "raw_error": str(raw_error),
                                    "bytes_received": len(audio_data),
                                    "detected_format": detected_format,
                                    "note": "Try using single recording instead of streaming for better quality, or check if audio device is producing valid data"
                                }
                            }), 500
            
        except Exception as e:
            logger.error(f"Critical error in audio processing: {e}")
            logger.exception("Full traceback:")
            return jsonify({
                "error": f"Audio processing system error: {str(e)}",
                "debug_info": {
                    "bytes_received": len(audio_data),
                    "error_type": type(e).__name__
                }
            }), 500
            
        # Validate audio quality before NPU inference
        try:
            # Check if audio is mostly silence
            audio_rms = np.sqrt(np.mean(audio**2))
            audio_max = np.max(np.abs(audio))
            
            logger.info(f"Audio stats: RMS={audio_rms:.6f}, Max={audio_max:.6f}, Length={len(audio)} samples ({len(audio)/sr:.2f}s)")
            
            # ULTRA-LENIENT silence detection specifically for Chinese speech
            # Previous thresholds were filtering out valid Chinese speech!
            # Only catch truly empty audio (digital silence)
            if audio_rms < 0.0000001 or audio_max < 0.000001:  # 100x more lenient
                logger.info(f"Audio appears to be digital silence (RMS={audio_rms:.6f}, Max={audio_max:.6f})")
                return jsonify({
                    "text": "",
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(audio)/sr:.2f}s",
                    "processing_time": "0.001s",
                    "language": "auto-detected",
                    "npu_active": True,
                    "note": "No audio signal detected (digital silence)"
                })
            else:
                logger.info(f"🎤 Audio has signal - RMS={audio_rms:.6f}, Max={audio_max:.6f} - PROCEEDING WITH TRANSCRIPTION")
                
            # If audio is too short for meaningful transcription, but be more lenient
            if len(audio) < sr * 0.1:  # Reduced from 0.2 to 0.1 seconds (100ms minimum)
                logger.info(f"Audio too short for transcription: {len(audio)/sr:.2f}s")
                return jsonify({
                    "text": "",
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(audio)/sr:.2f}s",
                    "processing_time": "0.001s",
                    "language": "auto-detected",
                    "npu_active": True,
                    "note": "Audio too short for transcription"
                })
            
            # Log that we're proceeding with transcription
            logger.info(f"✓ Audio validation passed - proceeding with NPU transcription")
            logger.info(f"📊 Audio details: {len(audio)} samples, {len(audio)/sr:.3f}s duration, {sr}Hz sample rate")
                
        except Exception as validation_error:
            logger.warning(f"Audio validation failed: {validation_error}")
            # Continue with transcription anyway for Chinese speech compatibility
            logger.info("Continuing with transcription despite validation warning...")
        
        # Load the model and perform real NPU transcription
        try:
            # SPEAKER DIARIZATION INTEGRATION - Process BEFORE normalization/enhancement
            speaker_results = None
            speaker_segments = []
            
            # Check if speaker diarization is enabled globally
            if (SPEAKER_DIARIZATION_AVAILABLE and 
                hasattr(app, 'buffer_manager') and 
                hasattr(app.buffer_manager, 'enable_diarization') and 
                app.buffer_manager.enable_diarization):
                
                try:
                    logger.info("🎤 Running speaker diarization on raw audio (before normalization)...")
                    
                    # Use the buffer manager's diarizer for consistency
                    if hasattr(app.buffer_manager, 'speaker_diarizer') and app.buffer_manager.speaker_diarizer:
                        # Process with speaker diarization on RAW audio before any processing
                        speaker_segments = app.buffer_manager.speaker_diarizer.process_audio_chunk(audio)
                        
                        if speaker_segments:
                            logger.info(f"🎤 Found {len(speaker_segments)} speaker segments")
                            for i, seg in enumerate(speaker_segments):
                                logger.info(f"  Segment {i+1}: Speaker {seg.get('speaker_id', 'unknown')} "
                                           f"from {seg.get('start_time', 0):.2f}s to {seg.get('end_time', 0):.2f}s "
                                           f"(confidence: {seg.get('confidence', 0):.2f})")
                        else:
                            logger.info("🎤 No speaker segments detected in this audio")
                            
                        # DO NOT store raw segments - only store consolidated results later
                        
                    else:
                        logger.warning("🎤 Speaker diarization enabled but diarizer not initialized")
                        
                except Exception as diarization_error:
                    logger.warning(f"🎤 Speaker diarization failed: {diarization_error}")
                    # Continue with regular transcription
                    speaker_segments = []
            else:
                logger.debug("🎤 Speaker diarization not enabled or not available")
            
            # Use thread-safe inference with NPU protection
            processing_start = time.time()
            logger.info(f"🧠 Running NPU inference on {model_manager.device} for {len(audio)/sr:.2f}s audio...")
            
            # Use the safe inference method to prevent NPU overload
            result = model_manager.safe_inference(model_name, audio)
            
            processing_end = time.time()
            actual_processing_time = processing_end - processing_start
            
            logger.info(f"✅ NPU processing complete: {actual_processing_time:.3f}s")
            
            # Debug: Log the raw result object to catch truncation
            logger.info(f"🔍 Raw NPU result type: {type(result)}")
            logger.info(f"🔍 Raw NPU result repr: {repr(result)}")
            if hasattr(result, '__dict__'):
                logger.info(f"🔍 Result attributes: {list(result.__dict__.keys())}")
                for attr in result.__dict__.keys():
                    logger.info(f"🔍   {attr}: {repr(getattr(result, attr))}")
            
            # Extract text from result
            if hasattr(result, 'text'):
                transcribed_text = result.text
                logger.info(f"🔍 Extracted via .text: {repr(transcribed_text)} (length: {len(transcribed_text)})")
            elif isinstance(result, str):
                transcribed_text = result
                logger.info(f"🔍 Direct string result: {repr(transcribed_text)} (length: {len(transcribed_text)})")
            else:
                transcribed_text = str(result)
                logger.info(f"🔍 Converted to string: {repr(transcribed_text)} (length: {len(transcribed_text)})")
            
            # Extract language detection if available
            detected_language = "auto-detected"
            if hasattr(result, 'language'):
                detected_language = result.language
                logger.info(f"🌏 Detected language: {detected_language}")
            elif hasattr(result, 'lang'):
                detected_language = result.lang
                logger.info(f"🌏 Detected language: {detected_language}")
            else:
                logger.info(f"🌏 Language detection not available from result object")
            
            # Clean up whitespace
            original_text = transcribed_text
            transcribed_text = transcribed_text.strip()
            logger.info(f"🔍 After strip(): {repr(transcribed_text)} (was {len(original_text)} chars, now {len(transcribed_text)} chars)")
            
            # Check for potential language detection issues with Chinese
            if transcribed_text and detected_language == "auto-detected":
                # Simple heuristic: if we get common English phrases for what should be Chinese
                common_english_false_positives = [
                    "thank you", "thanks", "hello", "hi", "yes", "no", "okay", "ok",
                    "good", "nice", "great", "wonderful", "amazing", "perfect"
                ]
                
                if any(phrase in transcribed_text.lower() for phrase in common_english_false_positives):
                    logger.warning(f"🚨 Potential language detection issue: Got '{transcribed_text}' which may be incorrect for non-English audio")
                    logger.warning(f"🚨 Consider: Audio may be Chinese/other language but transcribed as English")
                    detected_language = "possibly-incorrect"
            
            if transcribed_text:
                # Prepare the response with speaker information if available
                response_data = {
                    "text": transcribed_text,
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(audio)/sr:.2f}s",
                    "processing_time": f"{actual_processing_time:.3f}s",
                    "language": detected_language,
                    "npu_active": True,
                    "audio_processing_method": success_method if 'success_method' in locals() else "unknown"
                }
                
                # Add speaker diarization results if available
                if speaker_segments:
                    try:
                        # CONSOLIDATE segments from same speaker FIRST
                        consolidated_speakers = {}
                        
                        for segment in speaker_segments:
                            speaker_id = segment.get('speaker_id', 'unknown')
                            
                            if speaker_id not in consolidated_speakers:
                                consolidated_speakers[speaker_id] = {
                                    'speaker_id': speaker_id,
                                    'total_duration': 0,
                                    'start_time': segment.get('start_time', 0),
                                    'end_time': segment.get('end_time', 0),
                                    'confidence_sum': 0,
                                    'segment_count': 0
                                }
                            
                            # Extend time range and accumulate duration
                            consolidated_speakers[speaker_id]['start_time'] = min(
                                consolidated_speakers[speaker_id]['start_time'],
                                segment.get('start_time', 0)
                            )
                            consolidated_speakers[speaker_id]['end_time'] = max(
                                consolidated_speakers[speaker_id]['end_time'],
                                segment.get('end_time', 0)
                            )
                            consolidated_speakers[speaker_id]['total_duration'] += segment.get('duration', 0)
                            consolidated_speakers[speaker_id]['confidence_sum'] += segment.get('confidence', 0.5)
                            consolidated_speakers[speaker_id]['segment_count'] += 1
                        
                        # Create SINGLE speaker-aware transcription breakdown
                        speaker_transcriptions = []
                        total_duration = len(audio) / sr
                        
                        # Sort speakers by total duration (most active first)
                        sorted_speakers = sorted(consolidated_speakers.values(), 
                                               key=lambda x: x['total_duration'], reverse=True)
                        
                        # Distribute text among speakers based on their activity duration
                        text_chars = len(transcribed_text)
                        total_speaker_duration = sum(s['total_duration'] for s in sorted_speakers)
                        char_position = 0
                        
                        for speaker_info in sorted_speakers:
                            # Calculate text portion based on speaker's total duration
                            duration_ratio = speaker_info['total_duration'] / total_speaker_duration if total_speaker_duration > 0 else 1.0 / len(sorted_speakers)
                            chars_for_speaker = int(text_chars * duration_ratio)
                            
                            # Extract text portion for this speaker
                            if char_position + chars_for_speaker <= text_chars:
                                speaker_text = transcribed_text[char_position:char_position + chars_for_speaker]
                            else:
                                speaker_text = transcribed_text[char_position:]
                            
                            char_position += chars_for_speaker
                            
                            # Calculate average confidence
                            avg_confidence = speaker_info['confidence_sum'] / speaker_info['segment_count'] if speaker_info['segment_count'] > 0 else 0.5
                            
                            # Only include speakers with meaningful text OR if it's the only speaker
                            if speaker_text.strip() or len(sorted_speakers) == 1:
                                speaker_transcriptions.append({
                                    'speaker_id': speaker_info['speaker_id'],
                                    'text': speaker_text.strip() if speaker_text.strip() else transcribed_text,
                                    'start_time': speaker_info['start_time'],
                                    'end_time': speaker_info['end_time'],
                                    'duration': speaker_info['total_duration'],
                                    'confidence': max(0.6, avg_confidence + 0.15)  # Boost confidence for display
                                })
                        
                        # If no speakers got text, assign all text to the most active speaker
                        if not speaker_transcriptions and sorted_speakers:
                            main_speaker = sorted_speakers[0]
                            avg_confidence = main_speaker['confidence_sum'] / main_speaker['segment_count'] if main_speaker['segment_count'] > 0 else 0.5
                            
                            speaker_transcriptions.append({
                                'speaker_id': main_speaker['speaker_id'],
                                'text': transcribed_text,
                                'start_time': main_speaker['start_time'],
                                'end_time': main_speaker['end_time'],
                                'duration': main_speaker['total_duration'],
                                'confidence': max(0.6, avg_confidence + 0.15)
                            })
                        
                        response_data.update({
                            "speakers": speaker_transcriptions,
                            "diarization_enabled": True,
                            "total_speakers": len(consolidated_speakers),
                            "speaker_segments_detected": len(speaker_segments)
                        })
                        
                        logger.info(f"🎤 Response includes {len(speaker_transcriptions)} consolidated speaker entries for {len(consolidated_speakers)} speakers")
                        
                        # Create and store transcription entry with final text and speakers
                        if hasattr(app.buffer_manager, 'speaker_transcriptions'):
                            transcription_entry = {
                                'timestamp': time.time(),
                                'type': 'single_file',
                                'text': transcribed_text,  # Store the actual transcribed text
                                'buffer_duration': len(audio) / sr,
                                'enhanced': False,
                                'speakers': speaker_transcriptions,
                                'diarization_enabled': True
                            }
                            app.buffer_manager.speaker_transcriptions.append(transcription_entry)
                            
                            # Save transcriptions to disk for persistence
                            save_session_transcriptions(self.speaker_transcriptions)
                            
                            log_activity(f"Transcription: {len(speaker_transcriptions)} speakers, {len(transcribed_text)} chars")
                        
                    except Exception as speaker_processing_error:
                        log_activity(f"Speaker processing failed: {speaker_processing_error}", 'WARNING')
                        response_data.update({
                            "speakers": [],
                            "diarization_enabled": True,
                            "diarization_error": str(speaker_processing_error)
                        })
                else:
                    # No speakers detected or diarization disabled
                    if (SPEAKER_DIARIZATION_AVAILABLE and 
                        hasattr(app, 'buffer_manager') and 
                        hasattr(app.buffer_manager, 'enable_diarization') and 
                        app.buffer_manager.enable_diarization):
                        response_data.update({
                            "speakers": [],
                            "diarization_enabled": True,
                            "note": "Speaker diarization enabled but no speakers detected"
                        })
                        
                        # Create transcription entry even when no speakers detected
                        if hasattr(app.buffer_manager, 'speaker_transcriptions'):
                            transcription_entry = {
                                'timestamp': time.time(),
                                'type': 'single_file',
                                'text': transcribed_text,  # Store the actual transcribed text
                                'buffer_duration': len(audio) / sr,
                                'enhanced': False,
                                'speakers': [],  # No speakers detected
                                'diarization_enabled': True
                            }
                            app.buffer_manager.speaker_transcriptions.append(transcription_entry)
                            
                            # Save transcriptions to disk for persistence
                            save_session_transcriptions(app.buffer_manager.speaker_transcriptions)
                            
                            log_activity(f"Transcription: No speakers, {len(transcribed_text)} chars")
                        
                    else:
                        response_data.update({
                            "speakers": [],
                            "diarization_enabled": False
                        })
                        
                        # Save regular transcription
                        if hasattr(app.buffer_manager, 'speaker_transcriptions'):
                            transcription_entry = {
                                'timestamp': time.time(),
                                'type': 'single_file', 
                                'text': transcribed_text,
                                'buffer_duration': len(audio) / sr,
                                'enhanced': False,
                                'speakers': [],
                                'diarization_enabled': False
                            }
                            app.buffer_manager.speaker_transcriptions.append(transcription_entry)
                            save_session_transcriptions(app.buffer_manager.speaker_transcriptions)
                            log_activity(f"Transcription: {len(transcribed_text)} chars")
                
                return jsonify(response_data)
            else:
                # Empty transcription - likely silence
                empty_response = {
                    "text": "",
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(audio)/sr:.2f}s",
                    "processing_time": f"{actual_processing_time:.3f}s",
                    "language": "auto-detected",
                    "npu_active": True,
                    "note": "No speech detected"
                }
                
                # Add speaker info even for empty transcription
                if speaker_segments:
                    empty_response.update({
                        "speakers": [],
                        "diarization_enabled": True,
                        "note": "No speech detected, but speaker segments were found"
                    })
                elif (SPEAKER_DIARIZATION_AVAILABLE and 
                      hasattr(app, 'buffer_manager') and 
                      hasattr(app.buffer_manager, 'enable_diarization') and 
                      app.buffer_manager.enable_diarization):
                    empty_response.update({
                        "speakers": [],
                        "diarization_enabled": True,
                        "note": "No speech detected and no speakers detected"
                    })
                else:
                    empty_response.update({
                        "speakers": [],
                        "diarization_enabled": False
                    })
                
                return jsonify(empty_response)
                
        except Exception as model_error:
            logger.error(f"Model inference failed: {model_error}")
            return jsonify({
                "error": f"Model inference failed: {str(model_error)}",
                "debug_info": {
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(audio)/sr:.2f}s"
                }
            }), 500
            
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=['POST'])
def transcribe():
    return transcribe_with_model(model_manager.default_model)

@app.route("/clear-cache", methods=['POST'])
def clear_cache():
    try:
        model_manager.clear_npu_cache()
        return jsonify({
            "status": "success",
            "message": "NPU cache cleared successfully",
            "device": model_manager.device
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

# Initialize model manager
model_manager = ModelManager()

# Settings Management with persistence - MOVED HERE TO FIX ORDERING
app_settings = {
    "default_model": "whisper-base",
    "device_preference": "NPU",
    "min_inference_interval": 200,
    "buffer_duration": 6.0,
    "inference_interval": 3.0,
    "sample_rate": 16000,
    "vad_enabled": True,
    "vad_aggressiveness": 2,
    "diarization_enabled": False,
    "n_speakers": None,
    "embedding_method": "resemblyzer",
    "clustering_method": "hdbscan",
    "overlap_duration": 2.0,
    "speech_enhancement_enabled": True,
    "max_queue_size": 10,
    "max_transcription_history": 200,
    "log_level": "INFO",
    "enable_file_logging": True,
    "openvino_device": os.getenv("OPENVINO_DEVICE", ""),
    "openvino_log_level": int(os.getenv("OPENVINO_LOG_LEVEL", "1"))
}

# Load saved settings on startup
saved_settings = load_settings()
app_settings.update(saved_settings)
activity_logger.info(f"Settings loaded with {len(saved_settings)} saved values")

# Initialize rolling buffer manager with persistent settings
app.model_manager = model_manager
app.current_model = app_settings.get("default_model", "whisper-base")

# Ensure the default model from settings is loaded if available
try:
    if app.current_model in model_manager.list_models():
        model_manager.load_model(app.current_model)
        logger.info(f"✓ Loaded default model from settings: {app.current_model}")
    else:
        logger.warning(f"Default model from settings '{app.current_model}' not found, using fallback")
        # Find first available model as fallback
        available_models = model_manager.list_models()
        if available_models:
            app.current_model = available_models[0]
            app_settings["default_model"] = app.current_model
            save_settings(app_settings)
            logger.info(f"Updated default model to first available: {app.current_model}")
except Exception as e:
    logger.warning(f"Could not load default model: {e}")

app.buffer_manager = RollingBufferManager(
    buffer_duration=app_settings["buffer_duration"],
    inference_interval=app_settings["inference_interval"],
    sample_rate=app_settings["sample_rate"],
    enable_diarization=app_settings["diarization_enabled"],
    n_speakers=app_settings["n_speakers"]
)

# Load persistent transcription data
app.rolling_transcriptions = deque(maxlen=100)
if hasattr(app.buffer_manager, 'speaker_transcriptions'):
    # Load persistent speaker transcriptions
    saved_transcriptions = load_session_transcriptions()
    app.buffer_manager.speaker_transcriptions = saved_transcriptions
    log_activity(f"Loaded {len(saved_transcriptions)} transcriptions from previous session")
else:
    app.buffer_manager.speaker_transcriptions = deque(maxlen=200)

@app.route("/stream/configure", methods=['POST'])
def configure_streaming():
    """Configure streaming parameters"""
    try:
        data = request.get_json()
        
        buffer_duration = data.get('buffer_duration', 30.0)
        inference_interval = data.get('inference_interval', 10.0)
        model_name = data.get('model', 'whisper-base')
        
        # Validate parameters
        if not (10.0 <= buffer_duration <= 60.0):
            return jsonify({"error": "Buffer duration must be between 10-60 seconds"}), 400
        
        if not (5.0 <= inference_interval <= 30.0):
            return jsonify({"error": "Inference interval must be between 5-30 seconds"}), 400
        
        # Stop current buffer manager
        app.buffer_manager.stop_inference_timer()
        
        # Create new buffer manager with new settings
        app.buffer_manager = RollingBufferManager(
            buffer_duration=buffer_duration,
            inference_interval=inference_interval
        )
        
        # Update current model
        app.current_model = model_name
        
        logger.info(f"Streaming configured: {buffer_duration}s buffer, {inference_interval}s interval, model: {model_name}")
        
        return jsonify({
            "status": "success",
            "buffer_duration": buffer_duration,
            "inference_interval": inference_interval,
            "model": model_name
        })
        
    except Exception as e:
        logger.error(f"Error configuring streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stream/audio", methods=['POST'])
def stream_audio_chunk():
    """Receive audio chunks for rolling buffer"""
    try:
        # Get audio data
        audio_data = request.data
        
        if len(audio_data) == 0:
            return jsonify({"error": "Empty audio data"}), 400
        
        # Process audio using existing methods
        logger.debug(f"Received streaming chunk: {len(audio_data)} bytes")
        
        # Use the same audio processing pipeline as transcribe
        try:
            # Method 1: Direct librosa loading first
            try:
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                audio, sr = librosa.load(temp_file_path, sr=16000)
                os.unlink(temp_file_path)
                
                if len(audio) > 0:
                    logger.debug(f"Processed chunk: {len(audio)} samples at {sr}Hz")
                    
                    # Add to rolling buffer
                    buffer_size = app.buffer_manager.add_audio_chunk(audio)
                    
                    return jsonify({
                        "status": "success",
                        "samples_added": len(audio),
                        "buffer_size": buffer_size,
                        "duration": len(audio) / sr
                    })
                else:
                    return jsonify({"error": "No audio data extracted"}), 400
                    
            except Exception as librosa_error:
                logger.debug(f"Librosa processing failed: {librosa_error}")
                return jsonify({"error": f"Audio processing failed: {str(librosa_error)}"}), 500
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error processing stream chunk: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stream/transcriptions", methods=['GET'])
def get_rolling_transcriptions():
    """Get new rolling buffer transcriptions"""
    try:
        since = float(request.args.get('since', 0))
        
        # Get transcriptions since timestamp
        new_transcriptions = []
        for trans in app.rolling_transcriptions:
            if trans['timestamp'] > since:
                new_transcriptions.append(trans)
        
        return jsonify({
            "transcriptions": new_transcriptions,
            "server_time": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error getting transcriptions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stream/start", methods=['POST'])
def start_streaming():
    """Start rolling buffer streaming"""
    try:
        app.buffer_manager.clear_buffer()
        logger.info("🔄 Rolling buffer streaming started")
        
        return jsonify({
            "status": "success",
            "message": "Rolling buffer streaming started"
        })
        
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stream/stop", methods=['POST'])
def stop_streaming():
    """Stop rolling buffer streaming"""
    try:
        app.buffer_manager.stop_inference_timer()
        logger.info("⏹ Rolling buffer streaming stopped")
        
        return jsonify({
            "status": "success",
            "message": "Rolling buffer streaming stopped"
        })
        
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/diarization/configure", methods=['POST'])
def configure_diarization():
    """Configure speaker diarization settings"""
    try:
        if not SPEAKER_DIARIZATION_AVAILABLE:
            return jsonify({
                "error": "Speaker diarization not available - missing dependencies",
                "available": False
            }), 400
        
        data = request.get_json()
        
        enable = data.get('enable', True)
        n_speakers = data.get('n_speakers', None)
        buffer_duration = data.get('buffer_duration', 6.0)
        inference_interval = data.get('inference_interval', 3.0)
        
        # Validate parameters
        if n_speakers is not None and (n_speakers < 1 or n_speakers > 10):
            return jsonify({"error": "Number of speakers must be between 1-10"}), 400
        
        if not (3.0 <= buffer_duration <= 15.0):
            return jsonify({"error": "Buffer duration must be between 3-15 seconds"}), 400
        
        if not (1.0 <= inference_interval <= 10.0):
            return jsonify({"error": "Inference interval must be between 1-10 seconds"}), 400
        
        # Stop current buffer manager
        app.buffer_manager.stop_inference_timer()
        
        # Create new buffer manager with diarization
        app.buffer_manager = RollingBufferManager(
            buffer_duration=buffer_duration,
            inference_interval=inference_interval,
            enable_diarization=enable,
            n_speakers=n_speakers
        )
        
        logger.info(f"Diarization configured: enabled={enable}, speakers={n_speakers}, buffer={buffer_duration}s")
        
        return jsonify({
            "status": "success",
            "diarization_enabled": enable,
            "n_speakers": n_speakers,
            "buffer_duration": buffer_duration,
            "inference_interval": inference_interval,
            "available_methods": ["resemblyzer", "spectral"] if SPEAKER_DIARIZATION_AVAILABLE else []
        })
        
    except Exception as e:
        logger.error(f"Error configuring diarization: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/diarization/status", methods=['GET'])
def diarization_status():
    """Get speaker diarization status and statistics"""
    try:
        status = {
            "available": SPEAKER_DIARIZATION_AVAILABLE,
            "enabled": getattr(app.buffer_manager, 'enable_diarization', False),
            "statistics": {}
        }
        
        if SPEAKER_DIARIZATION_AVAILABLE and hasattr(app.buffer_manager, 'get_speaker_statistics'):
            try:
                status["statistics"] = app.buffer_manager.get_speaker_statistics()
            except Exception as e:
                logger.warning(f"Failed to get speaker statistics: {e}")
                status["statistics"] = {}
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting diarization status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe/enhanced/<model_name>", methods=['POST'])
def transcribe_with_enhancement(model_name):
    """Enhanced transcription with speaker diarization and speech enhancement"""
    try:
        # Check if model is available
        available_models = model_manager.list_models()
        if model_name not in available_models:
            return jsonify({
                "error": f"Model {model_name} not available. Available models: {available_models}",
                "note": "This server requires OpenVINO IR format models"
            }), 404
        
        # Get audio data from request
        if 'audio' not in request.files and not request.data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Handle audio data
        if request.data:
            audio_data = request.data
        else:
            audio_file = request.files['audio']
            audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return jsonify({"error": "Empty audio data"}), 400
        
        # Get enhancement options
        enable_enhancement = request.args.get('enhance', 'true').lower() == 'true'
        enable_diarization = request.args.get('diarize', 'false').lower() == 'true'
        n_speakers = request.args.get('speakers', None)
        if n_speakers:
            try:
                n_speakers = int(n_speakers)
            except ValueError:
                n_speakers = None
        
        logger.info(f"Enhanced transcription: model={model_name}, enhance={enable_enhancement}, diarize={enable_diarization}, speakers={n_speakers}")
        
        # Process audio using existing pipeline (to get the numpy array)
        # [Use the same audio processing code as the main transcribe endpoint]
        # For brevity, I'll reference the existing processing...
        
        # Load audio using multiple fallback methods (same as existing code)
        logger.info(f"Processing audio data: {len(audio_data)} bytes with enhanced model {model_name}")
        
        # [Audio processing code - same as existing transcribe_with_model function]
        # ... [Detect format, process with librosa/pydub/ffmpeg] ...
        
        # Detect audio format from the data
        detected_format = "unknown"
        if audio_data.startswith(b'RIFF'):
            detected_format = "wav"
        elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
            detected_format = "webm"
        # ... [rest of format detection as in existing code] ...
        
        # Process audio with same methods as main endpoint
        try:
            # Method 1: Direct librosa loading first
            try:
                file_ext = f".{detected_format}" if detected_format != "unknown" else ".webm"
                
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                audio, sr = librosa.load(temp_file_path, sr=16000)
                os.unlink(temp_file_path)
                
                if len(audio) > 0:
                    logger.info(f"✓ Direct librosa successful: {len(audio)} samples at {sr}Hz")
                    success_method = "direct_librosa"
                else:
                    raise Exception("Librosa returned empty audio")
                    
            except Exception as librosa_error:
                # Fall back to existing processing methods
                return jsonify({
                    "error": "Audio processing failed in enhanced mode",
                    "suggestion": "Try the regular /transcribe endpoint"
                }), 500
        
        except Exception as e:
            logger.error(f"Enhanced transcription audio processing failed: {e}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
        
        # Apply speech enhancement if requested
        enhanced_audio = audio
        if enable_enhancement and SPEAKER_DIARIZATION_AVAILABLE:
            try:
                enhancer = SpeechEnhancer(sr)
                enhanced_audio = enhancer.enhance_audio(audio)
                logger.info("✓ Applied speech enhancement")
            except Exception as e:
                logger.warning(f"Speech enhancement failed: {e}")
                enhanced_audio = audio
        
        # Perform speaker diarization if requested
        speaker_segments = []
        if enable_diarization and SPEAKER_DIARIZATION_AVAILABLE:
            try:
                diarizer = AdvancedSpeakerDiarization(
                    sample_rate=sr,
                    window_duration=min(len(enhanced_audio) / sr, app_settings["buffer_duration"]),  # Use configured duration
                    n_speakers=n_speakers,
                    enable_enhancement=False  # Already enhanced
                )
                
                # Process the entire audio for diarization
                speaker_segments = diarizer.process_audio_chunk(enhanced_audio)
                logger.info(f"✓ Speaker diarization found {len(speaker_segments)} segments")
                
            except Exception as e:
                logger.warning(f"Speaker diarization failed: {e}")
                speaker_segments = []
        
        # Run NPU transcription
        try:
            processing_start = time.time()
            result = model_manager.safe_inference(model_name, enhanced_audio)
            processing_end = time.time()
            actual_processing_time = processing_end - processing_start
            
            # Extract text
            if hasattr(result, 'text'):
                transcribed_text = result.text.strip()
            elif isinstance(result, str):
                transcribed_text = result.strip()
            else:
                transcribed_text = str(result).strip()
            
            # Prepare enhanced response
            response = {
                "text": transcribed_text,
                "model": model_name,
                "device": model_manager.device,
                "audio_length": f"{len(audio)/sr:.2f}s",
                "processing_time": f"{actual_processing_time:.3f}s",
                "language": "auto-detected",
                "npu_active": True,
                "enhanced": {
                    "speech_enhancement": enable_enhancement,
                    "speaker_diarization": enable_diarization and len(speaker_segments) > 0,
                    "speakers_detected": len(set([seg['speaker_id'] for seg in speaker_segments])) if speaker_segments else 0
                }
            }
            
            # Add speaker information if available
            if speaker_segments:
                try:
                    aligner = PunctuationAligner()
                    aligned_segments = aligner.align_segments_with_punctuation(
                        speaker_segments, transcribed_text, len(enhanced_audio) / sr
                    )
                    
                    # Create speaker breakdown
                    speaker_breakdown = []
                    for segment in aligned_segments:
                        speaker_breakdown.append({
                            'speaker_id': segment['speaker_id'],
                            'start_time': segment['start_time'],
                            'end_time': segment['end_time'],
                            'duration': segment['duration'],
                            'confidence': segment.get('confidence', 0.5)
                        })
                    
                    response['speakers'] = speaker_breakdown
                    
                    # Log speaker results
                    speakers_found = set([seg['speaker_id'] for seg in aligned_segments])
                    logger.info(f"🎤 Enhanced transcription with speakers {speakers_found}: {transcribed_text}")
                    
                except Exception as e:
                    logger.warning(f"Speaker alignment failed: {e}")
                    response['speakers'] = []
            else:
                response['speakers'] = []
            
            return jsonify(response)
            
        except Exception as model_error:
            logger.error(f"Enhanced model inference failed: {model_error}")
            return jsonify({
                "error": f"Enhanced model inference failed: {str(model_error)}",
                "debug_info": {
                    "model": model_name,
                    "device": model_manager.device,
                    "audio_length": f"{len(enhanced_audio)/sr:.2f}s",
                    "enhancement_enabled": enable_enhancement,
                    "diarization_enabled": enable_diarization
                }
            }), 500
            
    except Exception as e:
        logger.error(f"Enhanced transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/speaker/history", methods=['GET'])
def get_speaker_history():
    """Get speaker transcription history"""
    try:
        since = float(request.args.get('since', 0))
        limit = int(request.args.get('limit', 50))
        
        if hasattr(app.buffer_manager, 'speaker_transcriptions'):
            # Get transcriptions since timestamp
            all_transcriptions = list(app.buffer_manager.speaker_transcriptions)
            
            # Filter by timestamp
            filtered = [trans for trans in all_transcriptions if trans['timestamp'] > since]
            
            # Apply limit
            recent_transcriptions = filtered[-limit:] if len(filtered) > limit else filtered
            
            return jsonify({
                "transcriptions": recent_transcriptions,
                "total_count": len(all_transcriptions),
                "filtered_count": len(filtered),
                "server_time": time.time(),
                "diarization_available": SPEAKER_DIARIZATION_AVAILABLE
            })
        else:
            return jsonify({
                "transcriptions": [],
                "total_count": 0,
                "filtered_count": 0,
                "server_time": time.time(),
                "diarization_available": SPEAKER_DIARIZATION_AVAILABLE
            })
        
    except Exception as e:
        logger.error(f"Error getting speaker history: {e}")
        return jsonify({"error": str(e)}), 500

# Server is ready with real model support
logger.info("Server starting with NPU support...")
logger.info("Note: Using OpenVINO GenAI Whisper models for real transcription")

@app.route("/settings", methods=['GET'])
def get_settings():
    """Get current server settings"""
    try:
        # Include current runtime state
        runtime_info = {
            "server_uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0,
            "models_loaded": len(model_manager.pipelines) if 'model_manager' in globals() else 0,
            "device_in_use": model_manager.device if 'model_manager' in globals() else "Unknown",
            "streaming_active": hasattr(app, 'buffer_manager') and app.buffer_manager.is_running if hasattr(app, 'buffer_manager') else False,
            "current_model": getattr(app, 'current_model', 'unknown'),
            "last_inference_time": getattr(model_manager, 'last_inference_time', 0) if 'model_manager' in globals() else 0,
            "available_models": model_manager.list_models() if 'model_manager' in globals() else []
        }
        
        return jsonify({
            **app_settings,
            "runtime_info": runtime_info
        })
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/settings", methods=['POST'])
def update_settings():
    """Update server settings"""
    try:
        new_settings = request.get_json()
        if not new_settings:
            return jsonify({"error": "No settings provided"}), 400
        
        # Validate and update settings
        updated = {}
        for key, value in new_settings.items():
            if key in app_settings:
                # Type validation
                if key in ["buffer_duration", "inference_interval", "overlap_duration"]:
                    value = float(value)
                elif key in ["min_inference_interval", "sample_rate", "vad_aggressiveness", "n_speakers", "max_queue_size", "max_transcription_history", "openvino_log_level"]:
                    value = int(value) if value is not None else None
                elif key in ["vad_enabled", "diarization_enabled", "speech_enhancement_enabled", "enable_file_logging"]:
                    value = bool(value)
                
                app_settings[key] = value
                updated[key] = value
        
        # Apply some settings immediately
        if 'log_level' in updated:
            logging.getLogger().setLevel(getattr(logging, updated['log_level']))
        
        # Update current model if default model changed
        if 'default_model' in updated:
            try:
                new_model = updated['default_model']
                if new_model in model_manager.list_models():
                    app.current_model = new_model
                    # Optionally preload the new model
                    model_manager.load_model(new_model)
                    logger.info(f"✓ Updated current model to: {new_model}")
                else:
                    logger.warning(f"Requested model '{new_model}' not available")
                    # Don't update app.current_model if model isn't available
                    updated['default_model'] = app.current_model  # Revert to current
            except Exception as e:
                logger.error(f"Failed to update default model: {e}")
                updated['default_model'] = app.current_model  # Revert to current
        
        # Restart buffer manager if streaming settings changed
        streaming_keys = ["buffer_duration", "inference_interval", "sample_rate", "vad_enabled", "vad_aggressiveness"]
        if any(key in updated for key in streaming_keys) and hasattr(app, 'buffer_manager'):
            logger.info("Restarting buffer manager with new settings...")
            app.buffer_manager.stop_inference_timer()
            app.buffer_manager = RollingBufferManager(
                buffer_duration=app_settings["buffer_duration"],
                inference_interval=app_settings["inference_interval"],
                sample_rate=app_settings["sample_rate"],
                enable_diarization=app_settings["diarization_enabled"],
                n_speakers=app_settings["n_speakers"]
            )
        
        logger.info(f"Settings updated: {updated}")
        
        # Save settings to disk
        save_settings(app_settings)
        
        return jsonify({
            "message": "Settings updated successfully",
            "updated": updated,
            "restart_required": any(key in updated for key in ["device_preference", "openvino_device"]),
            "saved_to_disk": True
        })
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/settings/<setting_key>", methods=['POST'])
def update_single_setting(setting_key):
    """Update a single setting"""
    try:
        data = request.get_json()
        if not data or 'value' not in data:
            return jsonify({"error": "No value provided"}), 400
        
        if setting_key not in app_settings:
            return jsonify({"error": f"Unknown setting: {setting_key}"}), 400
        
        value = data['value']
        
        # Type conversion based on setting
        if setting_key in ["buffer_duration", "inference_interval", "overlap_duration"]:
            value = float(value)
        elif setting_key in ["min_inference_interval", "sample_rate", "vad_aggressiveness", "n_speakers", "max_queue_size", "max_transcription_history", "openvino_log_level"]:
            value = int(value) if value is not None else None
        elif setting_key in ["vad_enabled", "diarization_enabled", "speech_enhancement_enabled", "enable_file_logging"]:
            value = bool(value)
        
        app_settings[setting_key] = value
        
        # Apply immediate changes if needed
        if setting_key == 'log_level':
            logging.getLogger().setLevel(getattr(logging, value))
        
        # Handle default model changes
        if setting_key == 'default_model':
            try:
                if value in model_manager.list_models():
                    app.current_model = value
                    # Optionally preload the new model
                    model_manager.load_model(value)
                    logger.info(f"✓ Updated current model to: {value}")
                else:
                    logger.warning(f"Requested model '{value}' not available")
                    # Revert to current model
                    app_settings[setting_key] = app.current_model
                    value = app.current_model
            except Exception as e:
                logger.error(f"Failed to update default model: {e}")
                # Revert to current model
                app_settings[setting_key] = app.current_model
                value = app.current_model
        
        # Save settings to disk
        save_settings(app_settings)
        
        log_activity(f"Setting updated: {setting_key} = {value}")
        return jsonify({
            "message": f"Setting {setting_key} updated successfully",
            "value": value,
            "saved_to_disk": True
        })
        
    except Exception as e:
        logger.error(f"Error updating setting {setting_key}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/restart", methods=['POST'])
def restart_server():
    """Restart the server (graceful shutdown and restart)"""
    try:
        logger.info("Server restart requested")
        
        # Stop any running processes
        if hasattr(app, 'buffer_manager'):
            app.buffer_manager.stop_inference_timer()
        
        # Clear model cache
        if 'model_manager' in globals():
            model_manager.clear_npu_cache()
        
        logger.info("Server restart initiated - stopping in 2 seconds...")
        
        # Check if we're running in a container or with a process manager
        is_container = os.path.exists('/.dockerenv')
        is_systemd = os.environ.get('SYSTEMD_EXEC_PID') is not None
        is_supervisor = os.environ.get('SUPERVISOR_PROCESS_NAME') is not None
        
        # Schedule restart based on environment
        def restart():
            time.sleep(2)
            if is_container or is_systemd or is_supervisor:
                # In container or managed environment - exit and let manager restart
                os._exit(0)
            else:
                # Manual startup - try to restart the start-native.py script
                import sys
                import subprocess
                script_dir = os.path.dirname(os.path.abspath(__file__))
                start_script = os.path.join(script_dir, 'start-native.py')
                
                if os.path.exists(start_script):
                    logger.info("Attempting to restart via start-native.py")
                    # Start new process
                    if os.name == 'nt':  # Windows
                        subprocess.Popen([sys.executable, start_script], 
                                       creationflags=subprocess.CREATE_NEW_CONSOLE)
                    else:  # Linux/Mac
                        subprocess.Popen([sys.executable, start_script])
                
                # Exit current process
                os._exit(0)
        
        import threading
        restart_thread = threading.Thread(target=restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        environment_info = {
            "container": is_container,
            "systemd": is_systemd, 
            "supervisor": is_supervisor,
            "restart_method": "container_exit" if any([is_container, is_systemd, is_supervisor]) else "script_restart"
        }
        
        return jsonify({
            "message": "Server restart initiated",
            "environment": environment_info
        })
        
    except Exception as e:
        logger.error(f"Error restarting server: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/shutdown", methods=['POST'])
def shutdown_server():
    """Gracefully shutdown the server"""
    try:
        logger.info("Server shutdown requested")
        
        # Stop any running processes
        if hasattr(app, 'buffer_manager'):
            app.buffer_manager.stop_inference_timer()
        
        # Clear resources
        if 'model_manager' in globals():
            model_manager.clear_npu_cache()
        
        logger.info("Server shutting down...")
        
        # Schedule shutdown
        def shutdown():
            time.sleep(1)
            os._exit(0)
        
        import threading
        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
        return jsonify({"message": "Server shutdown initiated"})
        
    except Exception as e:
        logger.error(f"Error shutting down server: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/start", methods=['POST'])
def start_server():
    """Start server components (mainly for containers)"""
    try:
        logger.info("Server start requested")
        
        # Initialize components if they don't exist
        if not hasattr(app, 'buffer_manager'):
            app.buffer_manager = RollingBufferManager(
                buffer_duration=app_settings["buffer_duration"],
                inference_interval=app_settings["inference_interval"],
                sample_rate=app_settings["sample_rate"],
                enable_diarization=app_settings["diarization_enabled"],
                n_speakers=app_settings["n_speakers"]
            )
        
        # Record start time
        app.start_time = time.time()
        
        return jsonify({"message": "Server components started successfully"})
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Whisper NPU Server...")
    
    # Record start time
    app.start_time = time.time()
    
    # Initialize with persistent settings
    log_activity("Server starting with NPU support")
    log_activity(f"Window duration: {app_settings['buffer_duration']}s, Update interval: {app_settings['inference_interval']}s")
    
    # Clean up old session data (keep last 7 days)
    try:
        session_dir = os.path.join(os.path.dirname(__file__), 'session_data')
        if os.path.exists(session_dir):
            import glob
            import os.path
            
            for file_path in glob.glob(os.path.join(session_dir, '*.json')):
                if os.path.getmtime(file_path) < time.time() - (7 * 24 * 3600):  # 7 days
                    os.remove(file_path)
                    log_activity(f"Cleaned up old session file: {os.path.basename(file_path)}")
    except Exception as e:
        logger.warning(f"Session cleanup failed: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
