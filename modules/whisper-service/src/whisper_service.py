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
from typing import Dict, List, Optional, AsyncGenerator, Union, Tuple, Any
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from collections import deque

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from scipy import signal

# PyTorch and Whisper imports
import torch
import whisper
from whisper.decoding import DecodingOptions, DecodingResult
import torch.nn.functional as F

# Phase 2: SimulStreaming components
from beam_decoder import BeamSearchDecoder, BeamSearchConfig
from alignatt_decoder import AlignAttDecoder, AlignAttConfig, AlignAttState
from domain_prompt_manager import DomainPromptManager, create_domain_prompt
from vad_detector import SileroVAD, get_vad
from stability_tracker import StabilityTracker, StabilityConfig, TokenState

# Phase 1 Refactoring: Import PyTorch ModelManager from models package
from models.pytorch_manager import PyTorchModelManager

# Phase 2 Day 7-10: Import extracted components
from transcription import (
    TranscriptionRequest,
    TranscriptionResult,
    SimpleAudioBufferManager,
    detect_hallucination,
    find_stable_word_prefix,
    calculate_text_stability_score
)
from session import SessionManager
from config import load_whisper_config
from audio import load_audio_from_bytes, ensure_sample_rate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Disable SDPA to get attention weights for AlignAtt streaming
# PyTorch's scaled_dot_product_attention doesn't return attention weights (qk=None)
# We need the attention weights for AlignAtt frame-level decisions
try:
    whisper.model.MultiHeadAttention.use_sdpa = False
    logger.info("[STREAMING] ✓ Disabled SDPA to enable attention weight capture for AlignAtt")
except Exception as e:
    logger.warning(f"[STREAMING] Could not disable SDPA: {e}")

# Phase 2 Day 7: Dataclasses extracted to transcription/request_models.py
# TranscriptionRequest and TranscriptionResult are now imported above

# Phase 1 Refactoring: Use PyTorchModelManager from models package
# The ModelManager class has been extracted to models/pytorch_manager.py
# This alias maintains backwards compatibility
ModelManager = PyTorchModelManager

# Phase 2 Day 7: SimpleAudioBufferManager extracted to transcription/buffer_manager.py
# Phase 2 Day 7: SessionManager extracted to session/session_manager.py
# Both classes are now imported above
class WhisperService:
    """
    Main Whisper Service class providing NPU-optimized transcription
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Whisper service with configuration"""
        self.config = config or load_whisper_config()
        
        # Check if running in orchestration mode (disable internal chunking)
        self.orchestration_mode = self.config.get("orchestration_mode", False)
        
        # Initialize components
        self.model_manager = ModelManager(self.config.get("models_dir"))

        # Initialize per-session audio buffers (SimulStreaming pattern: one buffer per stream)
        # Each session gets its own buffer to avoid cross-contamination
        self.session_audio_buffers = {}  # session_id -> List[torch.Tensor]
        self.session_buffers_lock = threading.Lock()

        # Per-session VAD state tracking (VACOnlineProcessor pattern)
        # Tracks which sessions are currently in speech vs silence
        self.session_vad_states = {}  # session_id -> 'voice' | 'nonvoice' | None

        # Initialize Silero VAD for speech detection (prevents hallucinations on silence)
        # Following SimulStreaming: VAD is used as PRE-FILTER before adding to buffer
        try:
            self.vad = get_vad(threshold=0.5)
            logger.info("🎤 Silero VAD initialized (VACOnlineProcessor pattern - pre-filters silence)")
        except Exception as e:
            logger.warning(f"⚠️ VAD initialization failed: {e}, continuing without VAD")
            self.vad = None

        # Phase 3: Initialize Stability Trackers for draft/final emission
        # Per-session trackers for token stability detection
        self.session_stability_trackers = {}  # session_id -> StabilityTracker
        self.stability_config = StabilityConfig(
            stability_threshold=self.config.get("stability_threshold", 0.85),
            min_stable_words=self.config.get("min_stable_words", 2),
            min_hold_time=self.config.get("min_hold_time", 0.3),
            max_latency=self.config.get("max_latency", 2.0)
        )

        if not self.orchestration_mode:
            logger.info("🎤 Per-session audio buffering enabled (SimulStreaming-style)")
            logger.info(f"📊 Stability tracking enabled (threshold={self.stability_config.stability_threshold}, "
                       f"max_latency={self.stability_config.max_latency}s)")
        else:
            logger.info("🎯 Orchestration mode enabled - internal chunking disabled")

        self.session_manager = SessionManager(self.config.get("session_dir"))
        
        # Streaming settings
        self.streaming_active = False
        self.streaming_thread = None
        self.inference_interval = self.config.get("inference_interval", 3.0)
        
        # Enhanced statistics for orchestration mode
        self.stats = {
            "requests_processed": 0,
            "orchestration_chunks_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "errors": 0,
            "active_sessions": 0
        }
        
        logger.info(f"WhisperService initialized successfully (orchestration_mode: {self.orchestration_mode})")

    def _segments_len(self, session_id: str) -> float:
        """
        Calculate total duration of audio segments in session buffer (in seconds)
        Following SimulStreaming reference pattern
        """
        with self.session_buffers_lock:
            if session_id not in self.session_audio_buffers:
                return 0.0

            segments = self.session_audio_buffers[session_id]
            if not segments or len(segments) == 0:
                return 0.0

            total_samples = sum(len(seg) for seg in segments)
            return total_samples / 16000.0  # Assuming 16kHz sample rate

    # Phase 2 Day 8: Helper methods extracted to transcription/text_analysis.py
    # - detect_hallucination()
    # - find_stable_word_prefix()
    # - calculate_text_stability_score()

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
                # Load audio from bytes (handles stereo to mono conversion)
                audio_data, sr = load_audio_from_bytes(request.audio_data)
            else:
                audio_data = request.audio_data
                sr = request.sample_rate

            # Ensure correct sample rate (resample if needed)
            audio_data = ensure_sample_rate(audio_data, sr, request.sample_rate)

            # Note: VAD is handled internally by AlignAtt/BeamSearch decoders
            # No need for separate VAD processing in SimulStreaming mode

            # Prepare domain-specific prompt and context carryover
            initial_prompt = None
            if request.domain or request.custom_terms or request.previous_context or request.initial_prompt:
                try:
                    from domain_prompt_manager import DomainPromptManager

                    # Create domain prompt manager
                    domain_mgr = DomainPromptManager()

                    # Use provided initial_prompt or generate from domain/terms
                    if request.initial_prompt:
                        initial_prompt = request.initial_prompt
                        logger.info(f"[DOMAIN] Using provided initial prompt")
                    else:
                        initial_prompt = domain_mgr.create_domain_prompt(
                            domain=request.domain,
                            custom_terms=request.custom_terms,
                            previous_context=request.previous_context
                        )
                        logger.info(f"[DOMAIN] Generated prompt: {len(initial_prompt)} chars, domain={request.domain}")

                except Exception as e:
                    logger.warning(f"[DOMAIN] Failed to create prompt: {e}")
                    # Fall back to basic initial_prompt if provided
                    initial_prompt = request.initial_prompt

            # Perform inference with beam search, domain prompts, and streaming policy
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.model_manager.safe_inference,
                request.model_name,
                audio_data,
                request.beam_size,
                initial_prompt,
                request.language,
                request.temperature,
                request.streaming_policy,
                request.task,
                request.target_language,
                request.session_id,  # Pass session_id for per-session rolling context
                request.enable_code_switching  # Pass code-switching flag
            )

            logger.info(f"[INFERENCE] Complete: model={request.model_name}, beam_size={request.beam_size}, "
                       f"domain={request.domain}, streaming={request.streaming_policy}")
            
            processing_time = time.time() - start_time
            
            # Parse OpenVINO WhisperDecodedResults properly
            logger.info(f"[WHISPER] 🔍 Result type: {type(result)}")
            logger.info(f"[WHISPER] 🔍 Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

            # Initialize confidence score - will be extracted from model output
            confidence_score = 0.8  # Default for successful transcription

            # CRITICAL FIX: Check for dict FIRST before checking hasattr
            if isinstance(result, dict):
                # PyTorch Whisper returns dict with 'text' and 'segments' keys
                text = result.get('text', '')
                segments = result.get('segments', [])
                language = result.get('language', 'unknown')
                logger.info(f"[WHISPER] 📝 Dict result - text: '{text[:60]}', segments: {len(segments)}, lang: {language}")

                # Extract confidence from segments
                if segments:
                    avg_logprobs = [seg.get('avg_logprob', -1.0) for seg in segments if 'avg_logprob' in seg]
                    if avg_logprobs:
                        avg_logprob = sum(avg_logprobs) / len(avg_logprobs)
                        confidence_score = min(1.0, max(0.0, (avg_logprob + 1.0)))
                        logger.info(f"[WHISPER] 🎯 Calculated confidence from {len(avg_logprobs)} segments: {confidence_score:.3f}")

            # Handle OpenVINO WhisperDecodedResults structure
            elif hasattr(result, 'texts') and result.texts:
                # OpenVINO returns 'texts' (plural) - get the first text
                text = result.texts[0] if result.texts else ""
                logger.info(f"[WHISPER] 📝 Text extracted from 'texts': '{text}'")
                
                # Try to get segments from chunks and extract confidence
                segments = []
                chunk_confidences = []
                
                if hasattr(result, 'chunks') and result.chunks:
                    segments = result.chunks
                    logger.info(f"[WHISPER] 📋 Chunks/segments count: {len(segments)}")
                    
                    # Extract confidence from all chunks
                    for i, chunk in enumerate(segments):
                        chunk_confidence = None
                        
                        # Try different confidence attributes
                        if hasattr(chunk, 'confidence'):
                            chunk_confidence = chunk.confidence
                        elif hasattr(chunk, 'score'):
                            chunk_confidence = chunk.score
                        elif hasattr(chunk, 'probability'):
                            chunk_confidence = chunk.probability
                        elif hasattr(chunk, 'prob'):
                            chunk_confidence = chunk.prob
                        elif hasattr(chunk, 'avg_logprob'):
                            # Convert log probability to confidence (0-1 range)
                            # avg_logprob is typically negative, closer to 0 is better
                            chunk_confidence = min(1.0, max(0.0, (chunk.avg_logprob + 1.0)))
                        elif hasattr(chunk, 'no_speech_prob'):
                            # Convert no-speech probability to confidence
                            chunk_confidence = 1.0 - chunk.no_speech_prob
                        
                        if chunk_confidence is not None:
                            # Ensure confidence is in valid range
                            chunk_confidence = max(0.0, min(1.0, chunk_confidence))
                            chunk_confidences.append(chunk_confidence)
                            if i == 0:  # Log first chunk for debugging
                                logger.info(f"[WHISPER] 🎯 Chunk {i} confidence: {chunk_confidence:.3f}")
                
                # Calculate overall confidence from chunks
                if chunk_confidences:
                    # Use weighted average of chunk confidences
                    confidence_score = sum(chunk_confidences) / len(chunk_confidences)
                    logger.info(f"[WHISPER] 🎯 Calculated confidence from {len(chunk_confidences)} chunks: {confidence_score:.3f}")
                
                # Try to extract overall confidence from result object if no chunk confidence
                elif hasattr(result, 'confidence'):
                    confidence_score = result.confidence
                    logger.info(f"[WHISPER] 🎯 Found result confidence: {confidence_score:.3f}")
                elif hasattr(result, 'avg_logprob'):
                    # Convert log probability to confidence
                    confidence_score = min(1.0, max(0.0, (result.avg_logprob + 1.0)))
                    logger.info(f"[WHISPER] 🎯 Calculated confidence from result avg_logprob: {confidence_score:.3f}")
                elif hasattr(result, 'no_speech_prob'):
                    confidence_score = 1.0 - result.no_speech_prob
                    logger.info(f"[WHISPER] 🎯 Calculated confidence from no_speech_prob: {confidence_score:.3f}")
                elif hasattr(result, 'scores') and result.scores:
                    try:
                        # Get average score
                        avg_score = sum(result.scores) / len(result.scores)
                        confidence_score = max(0.0, min(1.0, avg_score))
                        logger.info(f"[WHISPER] 🎯 Average result score: {confidence_score:.3f}")
                    except:
                        logger.info(f"[WHISPER] ⚠️ Failed to calculate average score - using default")
                else:
                    logger.info(f"[WHISPER] ⚠️ No confidence attributes found - using default: {confidence_score:.3f}")
                
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
            # Only detect if not already set from dict
            if 'language' not in locals() or language == 'unknown':
                language = 'unknown'

            # Method 1: Check result attributes (for non-dict results)
            if language == 'unknown' and hasattr(result, 'language'):
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
            
            # Improved hallucination detection - only flag obvious cases
            is_likely_hallucination = detect_hallucination(text, confidence_score)
            
            if is_likely_hallucination:
                # Reduce confidence but don't make it too low if the model was confident
                confidence_score = max(0.3, confidence_score * 0.7)
                logger.info(f"[WHISPER] ⚠️ Possible hallucination detected: '{text[:50]}...' - adjusted confidence to {confidence_score:.3f}")
            
            # Ensure confidence is in valid range
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            transcription_result = TranscriptionResult(
                text=text,
                segments=segments,
                language=language,
                confidence_score=confidence_score,  # Now using extracted confidence
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
        Stream transcription results in real-time using SimulStreaming pattern

        Following SimulStreaming reference (simulstreaming_whisper.py):
        - Feed ENTIRE buffer to model each time
        - AlignAtt decoder tracks what's already been decoded internally
        - Simple list-based buffer with rolling window
        - Per-session buffers to prevent cross-contamination

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

        # Get or create session ID
        session_id = request.session_id or f"stream-{time.time()}"

        # Initialize session buffer if needed
        with self.session_buffers_lock:
            if session_id not in self.session_audio_buffers:
                self.session_audio_buffers[session_id] = []
                logger.info(f"[STREAM] Created new buffer for session {session_id}")

        # Audio buffer configuration (SimulStreaming-style)
        audio_max_len = 30.0  # Maximum buffer duration in seconds
        audio_min_len = 1.0   # Minimum audio before processing

        # Start streaming transcription
        try:
            # Convert audio to tensor and add to SESSION-SPECIFIC buffer
            if isinstance(request.audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(request.audio_data).float()
                if not self.orchestration_mode:
                    with self.session_buffers_lock:
                        self.session_audio_buffers[session_id].append(audio_tensor)
                        buffer_count = len(self.session_audio_buffers[session_id])
                    logger.debug(f"[STREAM] Session {session_id}: Added audio chunk, buffer has {buffer_count} segments")

            # Start periodic inference
            if not self.streaming_active:
                await self.start_streaming(request)

            # Track last emission for deduplication (SimulStreaming pattern)
            # SimulStreaming only emits when there's NEW content, not on every cycle
            last_emitted_text = None
            last_emitted_segments = None

            # Phase 3: Initialize stability tracker for this session
            if session_id not in self.session_stability_trackers:
                # Create tokenizer from model (if available)
                try:
                    tokenizer = self.model_manager.current_model.tokenizer if hasattr(self.model_manager.current_model, 'tokenizer') else None
                except:
                    tokenizer = None

                self.session_stability_trackers[session_id] = StabilityTracker(
                    config=self.stability_config,
                    tokenizer=tokenizer
                )
                logger.info(f"[STABILITY] Created tracker for session {session_id}")

            tracker = self.session_stability_trackers[session_id]

            # Track text history for word-based stability detection
            text_history = []  # List of (text, timestamp) tuples
            last_stable_prefix = ""

            # Yield results as they become available
            while self.streaming_active:
                await asyncio.sleep(self.inference_interval)

                if not self.orchestration_mode:
                    # Calculate current buffer length for THIS SESSION
                    segments_len = self._segments_len(session_id)

                    # Maintain rolling window (remove old segments when buffer full)
                    with self.session_buffers_lock:
                        while segments_len > audio_max_len and len(self.session_audio_buffers[session_id]) > 1:
                            self.session_audio_buffers[session_id].pop(0)
                            segments_len = self._segments_len(session_id)
                            logger.debug(f"[STREAM] Session {session_id}: Removed old segment, buffer now {segments_len:.2f}s")

                    # Process if we have enough audio
                    if segments_len >= audio_min_len:
                        try:
                            logger.info(f"[STREAM] Session {session_id}: Processing buffer with {segments_len:.2f}s audio")

                            # Concatenate ENTIRE buffer for THIS SESSION (SimulStreaming pattern)
                            with self.session_buffers_lock:
                                full_audio = torch.cat(self.session_audio_buffers[session_id], dim=0).numpy()

                            # NOTE: VAD filtering is now done at ingestion time (add_audio_chunk)
                            # using VACOnlineProcessor pattern - only speech chunks reach this buffer
                            logger.debug(f"[STREAM] Session {session_id}: Processing {len(full_audio)} audio samples (all pre-filtered by VAD)")

                            # Create request with full buffer
                            # AlignAtt will track internally what's already been decoded
                            stream_request = TranscriptionRequest(
                                audio_data=full_audio,
                                model_name=request.model_name,
                                language=request.language,
                                session_id=session_id,
                                sample_rate=request.sample_rate,
                                enable_vad=False,  # VAD handled by AlignAtt
                                beam_size=request.beam_size,
                                temperature=request.temperature,
                                streaming_policy=request.streaming_policy,
                                frame_threshold_offset=request.frame_threshold_offset,
                                task=request.task,
                                target_language=request.target_language
                            )

                            # Transcribe full buffer
                            result = await self.transcribe(stream_request)

                            # Phase 3: Word-based stability detection
                            # Since OpenVINO doesn't expose token-level data, use word-level analysis
                            current_time = time.time()
                            current_text = result.text.strip()

                            # Add to text history
                            text_history.append((current_text, current_time))

                            # Keep only recent history (last max_latency window)
                            cutoff_time = current_time - self.stability_config.max_latency
                            text_history = [(txt, ts) for txt, ts in text_history if ts >= cutoff_time]

                            # Find stable word prefix (words that appear consistently across recent history)
                            stable_prefix = find_stable_word_prefix(text_history, current_text)
                            unstable_tail = current_text[len(stable_prefix):].strip()

                            # Determine emission type
                            is_draft = len(stable_prefix) > len(last_stable_prefix)
                            is_final = False  # Will be set to True on segment boundaries
                            is_forced = (current_time - text_history[0][1]) >= self.stability_config.max_latency if text_history else False
                            should_translate = len(stable_prefix.split()) >= self.stability_config.min_stable_words

                            # Update result with stability information
                            result.stable_text = stable_prefix
                            result.unstable_text = unstable_tail
                            result.is_draft = is_draft
                            result.is_final = is_final
                            result.is_forced = is_forced
                            result.should_translate = should_translate
                            result.translation_mode = "draft" if (is_draft and should_translate) else ("final" if is_final else "none")
                            result.stable_end_time = current_time

                            # Calculate stability score (based on text consistency)
                            result.stability_score = calculate_text_stability_score(text_history, stable_prefix)

                            last_stable_prefix = stable_prefix

                            logger.info(f"[STABILITY] Session {session_id}: stable='{stable_prefix[:30]}...' unstable='{unstable_tail[:20]}...' "
                                       f"(draft={is_draft}, should_translate={should_translate}, score={result.stability_score:.2f})")

                            # DEDUPLICATION: Only emit if content changed (SimulStreaming pattern)
                            # SimulStreaming returns {} when no update - we check text/segments instead
                            content_changed = False

                            # Check if text changed
                            if result.text != last_emitted_text:
                                content_changed = True
                                logger.info(f"[STREAM] Session {session_id}: 📝 Text changed: '{last_emitted_text}' → '{result.text[:50]}...'")

                            # Check if segments changed (important for preserving semantic boundaries)
                            # Convert segments to comparable format (list of dicts with timing)
                            current_segments_comparable = [
                                {'start': seg.get('start'), 'end': seg.get('end'), 'text': seg.get('text')}
                                for seg in result.segments
                            ] if result.segments else []

                            if current_segments_comparable != last_emitted_segments:
                                content_changed = True
                                logger.info(f"[STREAM] Session {session_id}: 🎯 Segments changed: {len(last_emitted_segments or [])} → {len(current_segments_comparable)} segments")

                            # Only emit if content changed
                            if content_changed:
                                # Update tracking
                                last_emitted_text = result.text
                                last_emitted_segments = current_segments_comparable

                                # Yield result with segment boundary information preserved
                                yield result

                                logger.info(f"[STREAM] Session {session_id}: ✅ Emitted update: '{result.text[:50]}...' (Lang: {result.language}, Segments: {len(result.segments)})")
                            else:
                                # No change - skip emission (like SimulStreaming returning {})
                                logger.info(f"[STREAM] Session {session_id}: ⏸️  No change detected, skipping emission")

                        except Exception as e:
                            logger.warning(f"Streaming transcription error for session {session_id}: {e}")
                            continue

        except Exception as e:
            logger.error(f"Streaming transcription failed for session {session_id}: {e}")
            raise
        finally:
            # Cleanup session buffer on stream end
            with self.session_buffers_lock:
                if session_id in self.session_audio_buffers:
                    del self.session_audio_buffers[session_id]
                    logger.info(f"[STREAM] Cleaned up buffer for session {session_id}")

            # Cleanup stability tracker
            if session_id in self.session_stability_trackers:
                del self.session_stability_trackers[session_id]
                logger.info(f"[STABILITY] Cleaned up tracker for session {session_id}")

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
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, session_id: str = "default", enable_vad_prefilter: bool = False) -> int:
        """
        Add audio chunk to the session-specific streaming buffer.

        CRITICAL: VAD pre-filtering is OPTIONAL and controlled by enable_vad_prefilter flag.

        Use Cases:
        - enable_vad_prefilter=False: For file playback, testing, or when you want ALL audio
        - enable_vad_prefilter=True: For live microphone input to filter background noise/silence

        NOTE: SimulStreaming alignment still works without VAD pre-filtering.
        AlignAtt handles silence internally - VAD pre-filtering is an optimization, not a requirement.

        Args:
            audio_chunk: Audio data as numpy array
            session_id: Session identifier for buffer isolation
            enable_vad_prefilter: If True, use VAD to filter silence chunks before adding to buffer

        Returns:
            Number of chunks in the session buffer
        """
        if self.orchestration_mode:
            logger.warning("add_audio_chunk called in orchestration mode - use process_orchestration_chunk instead")
            return 0

        # VAD PRE-FILTER (OPTIONAL - controlled by enable_vad_prefilter flag)
        # When enabled, filters silence chunks before adding to buffer
        # When disabled, ALL audio chunks are added (better for file playback/testing)
        if enable_vad_prefilter and self.vad is not None:
            vad_result = self.vad.check_speech(audio_chunk)

            # Initialize session VAD state if needed
            if session_id not in self.session_vad_states:
                self.session_vad_states[session_id] = None
                logger.info(f"[VAD] Session {session_id}: Initialized VAD state tracking")

                # CRITICAL FIX: Preload VAD with silence to build analysis window
                # This "primes" the VAD so it's ready to detect speech immediately
                # SimulStreaming does this implicitly - we need to do it explicitly
                silence_frames = 3  # Preload 3 frames of silence (3 * 512 samples = 1536 samples = ~96ms)
                silence_chunk = np.zeros(512, dtype=np.float32)
                for _ in range(silence_frames):
                    self.vad.check_speech(silence_chunk)
                logger.info(f"[VAD] Session {session_id}: 🔧 Preloaded VAD with {silence_frames} silence frames for immediate readiness")

            # Update VAD state based on result
            if vad_result is not None:
                if 'start' in vad_result:
                    self.session_vad_states[session_id] = 'voice'
                    logger.info(f"[VAD] Session {session_id}: 🎤 Speech started at {vad_result['start']:.2f}s")
                elif 'end' in vad_result:
                    self.session_vad_states[session_id] = 'nonvoice'
                    logger.info(f"[VAD] Session {session_id}: 🔇 Speech ended at {vad_result['end']:.2f}s")
            else:
                # VAD returned None - this could mean:
                # 1. ONGOING speech (between start and end events)
                # 2. VAD is still accumulating data (first few chunks)
                # 3. Silence after speech has ended
                current_state = self.session_vad_states.get(session_id)
                if current_state == 'voice':
                    # Still in speech - vad_result is None but we're between start and end
                    logger.debug(f"[VAD] Session {session_id}: ✅ Ongoing speech (vad_result=None, state=voice)")
                elif current_state is None:
                    # CRITICAL FIX: First chunks - VAD needs data to accumulate
                    # ACCEPT these chunks so VAD can build up its analysis window
                    logger.debug(f"[VAD] Session {session_id}: ✅ Accepting initial chunk for VAD analysis (state=None)")
                elif current_state == 'nonvoice':
                    # Confirmed non-voice state and still no speech detected - discard
                    logger.info(f"[VAD] Session {session_id}: ❌ Discarding chunk (state: nonvoice, vad_result: None)")
                    with self.session_buffers_lock:
                        return len(self.session_audio_buffers.get(session_id, []))

            # If we got here with an explicit start/end event, handle it
            if vad_result is not None:
                current_state = self.session_vad_states.get(session_id)
                if current_state != 'voice':
                    logger.info(f"[VAD] Session {session_id}: ❌ Discarding chunk (state: {current_state}, vad_result: {vad_result})")
                    # Return current buffer size without adding chunk
                    with self.session_buffers_lock:
                        return len(self.session_audio_buffers.get(session_id, []))

        # Only reached if VAD detected speech or VAD is disabled
        audio_tensor = torch.from_numpy(audio_chunk).float()

        with self.session_buffers_lock:
            if session_id not in self.session_audio_buffers:
                self.session_audio_buffers[session_id] = []
                logger.info(f"[BUFFER] Created new buffer for session {session_id}")

            self.session_audio_buffers[session_id].append(audio_tensor)
            buffer_size = len(self.session_audio_buffers[session_id])

        logger.debug(f"[BUFFER] Session {session_id}: Added speech chunk, total {buffer_size} chunks")
        return buffer_size
    
    async def process_orchestration_chunk(self, 
                                        chunk_id: str,
                                        session_id: str,
                                        audio_data: bytes,
                                        chunk_metadata: Dict[str, Any],
                                        model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single audio chunk from orchestration service.
        This bypasses internal buffering and processes chunks directly.
        """
        if not self.orchestration_mode:
            logger.warning("process_orchestration_chunk called in legacy mode")
        
        start_time = time.time()
        
        try:
            logger.info(f"[ORCHESTRATION] 🎯 Processing chunk {chunk_id} for session {session_id}")
            logger.debug(f"[ORCHESTRATION] Chunk metadata: {chunk_metadata}")
            
            # Create transcription request for the chunk
            transcription_request = TranscriptionRequest(
                audio_data=audio_data,
                model_name=model_name or self.config.get("default_model", "whisper-tiny"),
                session_id=session_id,
                streaming=False,  # Single chunk processing
                enhanced=chunk_metadata.get('enable_enhancement', False),
                sample_rate=chunk_metadata.get('sample_rate', 16000),
                enable_vad=False,  # VAD already applied by orchestration
                timestamp_mode=chunk_metadata.get('timestamp_mode', 'word')
            )
            
            # Process the chunk directly (bypass buffering)
            result = await self.transcribe(transcription_request)
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["orchestration_chunks_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            # Prepare orchestration-compatible response
            response = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "status": "success",
                "transcription": {
                    "text": result.text,
                    "language": result.language,
                    "confidence_score": result.confidence_score,
                    "segments": result.segments,
                    "timestamp": result.timestamp
                },
                "processing_info": {
                    "model_used": result.model_used,
                    "device_used": result.device_used,
                    "processing_time": processing_time,
                    "chunk_metadata": chunk_metadata,
                    "service_mode": "orchestration"
                },
                "chunk_sequence": chunk_metadata.get('sequence_number', 0),
                "chunk_timing": {
                    "start_time": chunk_metadata.get('start_time', 0.0),
                    "end_time": chunk_metadata.get('end_time', 0.0),
                    "duration": chunk_metadata.get('duration', 0.0),
                    "overlap_start": chunk_metadata.get('overlap_start', 0.0),
                    "overlap_end": chunk_metadata.get('overlap_end', 0.0)
                }
            }
            
            logger.info(f"[ORCHESTRATION] ✅ Chunk {chunk_id} processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["errors"] += 1
            
            logger.error(f"[ORCHESTRATION] ❌ Failed to process chunk {chunk_id}: {e}")
            
            return {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "error_type": "orchestration_processing_error",
                "processing_time": processing_time,
                "chunk_metadata": chunk_metadata
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_manager.list_models()
    
    def get_service_status(self) -> Dict:
        """Get service status information"""
        # Calculate total buffer info across all sessions
        with self.session_buffers_lock:
            total_buffers = len(self.session_audio_buffers)
            total_segments = sum(len(buffer) for buffer in self.session_audio_buffers.values())

        return {
            "device": self.model_manager.device,
            "loaded_models": list(self.model_manager.models.keys()),
            "available_models": self.get_available_models(),
            "streaming_active": self.streaming_active,
            "active_stream_sessions": total_buffers,
            "total_buffer_segments": total_segments,
            "orchestration_mode": self.orchestration_mode,
            "sessions": len(self.session_manager.sessions)
        }
    
    def create_session(self, session_id: str, config: Optional[Dict] = None) -> Dict:
        """Create a new transcription session"""
        return self.session_manager.create_session(session_id, config)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.session_manager.get_session(session_id)
    
    def close_session(self, session_id: str) -> Optional[Dict]:
        """Close a transcription session and clean up per-session resources"""
        # Clean up per-session rolling context and tokenizer
        self.model_manager.cleanup_session_context(session_id)

        # Clean up audio buffers and stability trackers
        with self.session_buffers_lock:
            if session_id in self.session_audio_buffers:
                del self.session_audio_buffers[session_id]
                logger.info(f"[CLEANUP] Cleared audio buffer for session {session_id}")

            if session_id in self.session_vad_states:
                del self.session_vad_states[session_id]
                logger.info(f"[CLEANUP] Cleared VAD state for session {session_id}")

            if session_id in self.session_stability_trackers:
                del self.session_stability_trackers[session_id]
                logger.info(f"[CLEANUP] Cleared stability tracker for session {session_id}")

        return self.session_manager.close_session(session_id)
    
    def get_transcription_history(self, limit: int = 50) -> List[Dict]:
        """Get recent transcription history"""
        return self.session_manager.get_transcription_history(limit)
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_manager.clear_cache()

    # Phase 2 Day 9: Configuration loading extracted to config/config_loader.py
    # - load_whisper_config()

    async def shutdown(self):
        """Shutdown the whisper service and cleanup resources"""
        try:
            # Stop streaming
            await self.stop_streaming()

            # Clear all session buffers
            with self.session_buffers_lock:
                session_count = len(self.session_audio_buffers)
                self.session_audio_buffers.clear()
                logger.info(f"[SHUTDOWN] Cleared {session_count} session buffers")

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
        parser.add_argument("--model", default="whisper-tiny", help="Model to use")
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