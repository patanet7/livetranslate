#!/usr/bin/env python3
"""
Continuous Stream Processor for Whisper Service

Handles continuous audio streaming with sliding window inference and text deduplication.
Processes 1-second audio chunks into 4-second sliding windows every 3 seconds.
Removes duplicate text segments and forwards clean transcriptions to translation service.

Key Features:
- Continuous audio buffering with sliding window processing
- Text-level deduplication using character overlap detection  
- Complete transcript storage for session management
- Integration with translation service for clean text forwarding
- Session-based processing for multiple concurrent conversations
"""

import os
import time
import logging
import asyncio
import httpx
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousStreamProcessor:
    """
    Processes continuous audio streams with sliding window inference and text deduplication.
    
    Architecture:
    - Receives 1s audio chunks from frontend
    - Maintains 6s rolling audio buffer
    - Runs inference on 4s sliding windows every 3s
    - Deduplicates transcribed text using character overlap detection
    - Forwards clean text to translation service
    """
    
    def __init__(self, sample_rate: int = 16000, session_id: str = None):
        """
        Initialize continuous stream processor.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            session_id: Session identifier for this stream
        """
        self.sample_rate = sample_rate
        self.session_id = session_id or f"stream_{int(time.time())}"
        
        # Audio processing configuration
        self.audio_buffer = deque(maxlen=sample_rate * 6)  # 6 second audio buffer
        self.inference_interval = 3.0  # Run inference every 3 seconds
        self.inference_window = 4.0    # Use 4 second windows for inference
        self.last_inference_time = 0
        
        # Text deduplication tracking
        self.previous_transcriptions = deque(maxlen=5)  # Last 5 transcriptions for overlap detection
        self.processed_text_segments = []  # Complete processed text history
        
        # Integration components (set by whisper service)
        self.transcription_callback = None  # Function to call for transcription
        self.translation_client = None     # Client for sending to translation service
        self.transcript_manager = None     # Manager for complete transcript storage
        
        # Performance tracking
        self.total_chunks_received = 0
        self.total_inferences_run = 0
        self.total_text_segments_sent = 0
        
        logger.info(f"ContinuousStreamProcessor initialized for session {self.session_id}")
    
    def add_chunk(self, audio_chunk: np.ndarray, chunk_metadata: Dict = None) -> Dict:
        """
        Add 1-second audio chunk to the continuous stream and potentially run inference.
        
        Args:
            audio_chunk: 1-second audio data as numpy array
            chunk_metadata: Optional metadata about the chunk
            
        Returns:
            Dict with processing status and any transcription results
        """
        try:
            self.total_chunks_received += 1
            chunk_metadata = chunk_metadata or {}
            
            # Validate audio chunk
            if len(audio_chunk) == 0:
                logger.warning(f"[{self.session_id}] Empty audio chunk received, skipping")
                return {"status": "empty_chunk", "chunk_number": self.total_chunks_received}
            
            # Add to rolling buffer
            self.audio_buffer.extend(audio_chunk.flatten())
            
            logger.debug(f"[{self.session_id}] Added chunk {self.total_chunks_received} "
                        f"({len(audio_chunk)} samples, buffer: {len(self.audio_buffer)} samples)")
            
            # Check if we should run inference
            if self._should_run_inference():
                return self._run_inference_with_deduplication(chunk_metadata)
            
            return {
                "status": "buffered",
                "chunk_number": self.total_chunks_received,
                "buffer_length": len(self.audio_buffer),
                "next_inference_in": self._time_until_next_inference()
            }
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Error adding chunk: {e}")
            return {"status": "error", "error": str(e)}
    
    def _should_run_inference(self) -> bool:
        """
        Determine if we should run inference now based on timing and buffer readiness.
        
        Returns:
            True if inference should be run
        """
        current_time = time.time()
        time_ready = current_time - self.last_inference_time >= self.inference_interval
        
        # Need at least 4 seconds of audio for inference window
        min_samples = int(self.sample_rate * self.inference_window)
        buffer_ready = len(self.audio_buffer) >= min_samples
        
        return time_ready and buffer_ready
    
    def _time_until_next_inference(self) -> float:
        """Calculate seconds until next inference should run."""
        elapsed = time.time() - self.last_inference_time
        return max(0, self.inference_interval - elapsed)
    
    def _run_inference_with_deduplication(self, chunk_metadata: Dict) -> Dict:
        """
        Run inference on sliding 4-second window and deduplicate text output.
        
        Args:
            chunk_metadata: Metadata about the triggering chunk
            
        Returns:
            Dict with inference results and deduplication status
        """
        try:
            self.total_inferences_run += 1
            current_time = time.time()
            
            logger.info(f"[{self.session_id}] Running inference #{self.total_inferences_run} "
                       f"(buffer: {len(self.audio_buffer)} samples)")
            
            # Extract 4-second window for inference
            window_samples = int(self.sample_rate * self.inference_window)
            inference_audio = np.array(list(self.audio_buffer)[-window_samples:])
            
            # Run transcription using callback (provided by whisper service)
            if not self.transcription_callback:
                logger.error(f"[{self.session_id}] No transcription callback configured")
                return {"status": "error", "error": "No transcription callback"}
            
            # Perform transcription
            transcription_start = time.time()
            result = self.transcription_callback(inference_audio)
            transcription_time = time.time() - transcription_start
            
            raw_text = result.text.strip() if hasattr(result, 'text') and result.text else ""
            language = result.language if hasattr(result, 'language') else 'unknown'
            
            logger.info(f"[{self.session_id}] Raw transcription: \"{raw_text}\" "
                       f"(lang: {language}, time: {transcription_time:.2f}s)")
            
            # Deduplicate text
            clean_text = self._deduplicate_text(raw_text)
            
            # Update timing
            self.last_inference_time = current_time
            
            result_data = {
                "status": "inference_complete",
                "inference_number": self.total_inferences_run,
                "raw_text": raw_text,
                "clean_text": clean_text,
                "language": language,
                "transcription_time": transcription_time,
                "window_start_time": current_time - self.inference_window,
                "window_end_time": current_time
            }
            
            if clean_text:
                # Store in complete transcript
                if self.transcript_manager:
                    self.transcript_manager.store_segment(
                        clean_text, self.session_id, {
                            "inference_number": self.total_inferences_run,
                            "language": language,
                            "timestamp": current_time
                        }
                    )
                
                # Send to translation service
                asyncio.create_task(self._send_to_translation_service(clean_text, language))
                
                self.total_text_segments_sent += 1
                result_data["status"] = "transcribed_and_sent"
                result_data["segments_sent"] = self.total_text_segments_sent
                
                logger.info(f"[{self.session_id}] Clean text sent: \"{clean_text}\"")
            else:
                result_data["status"] = "duplicate_removed"
                logger.info(f"[{self.session_id}] Text was duplicate, removed")
            
            return result_data
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Inference error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _deduplicate_text(self, new_text: str) -> str:
        """
        Remove text overlap with previous transcriptions using character-level matching.
        
        Args:
            new_text: Newly transcribed text
            
        Returns:
            Clean text with overlaps removed, or empty string if completely duplicate
        """
        if not new_text:
            return ""
        
        # If no previous transcriptions, this is clean
        if not self.previous_transcriptions:
            self.previous_transcriptions.append(new_text)
            logger.debug(f"[{self.session_id}] First transcription, accepting: \"{new_text}\"")
            return new_text
        
        # Check for overlap with recent transcriptions
        best_clean_text = new_text
        max_overlap_removed = 0
        
        for i, prev_text in enumerate(reversed(list(self.previous_transcriptions))):
            overlap_length = self._find_text_overlap(prev_text, new_text)
            
            if overlap_length > 0:
                # Remove overlapping portion from start of new text
                potential_clean = new_text[overlap_length:].strip()
                
                logger.debug(f"[{self.session_id}] Found {overlap_length}-char overlap with previous text #{i}: "
                           f"\"{new_text[:overlap_length]}\" -> clean: \"{potential_clean}\"")
                
                # Use the result with the most overlap removed (most likely correct)
                if overlap_length > max_overlap_removed:
                    max_overlap_removed = overlap_length
                    best_clean_text = potential_clean
        
        # Store new transcription for future overlap detection
        self.previous_transcriptions.append(new_text)
        
        if max_overlap_removed > 0:
            logger.info(f"[{self.session_id}] Removed {max_overlap_removed} overlapping characters")
        
        return best_clean_text
    
    def _find_text_overlap(self, prev_text: str, new_text: str) -> int:
        """
        Find character-level overlap between previous and new text.
        
        Args:
            prev_text: Previous transcription
            new_text: New transcription
            
        Returns:
            Number of overlapping characters at the beginning of new_text
        """
        if not prev_text or not new_text:
            return 0
        
        # Check if new_text starts with the end of prev_text
        max_check = min(len(prev_text), len(new_text), 150)  # Check up to 150 characters
        
        for overlap_len in range(max_check, 2, -1):  # Minimum 3 character overlap
            if prev_text[-overlap_len:] == new_text[:overlap_len]:
                return overlap_len
        
        return 0
    
    async def _send_to_translation_service(self, clean_text: str, language: str):
        """
        Send clean, deduplicated text to the translation service.
        
        Args:
            clean_text: Clean text to translate
            language: Detected language code
        """
        try:
            if not self.translation_client:
                logger.warning(f"[{self.session_id}] No translation client configured")
                return
            
            # Send to translation service using the clean text endpoint
            await self.translation_client.send_clean_text(
                text=clean_text,
                session_id=self.session_id,
                metadata={
                    'language': language,
                    'inference_number': self.total_inferences_run,
                    'timestamp': time.time()
                }
            )
            
            logger.debug(f"[{self.session_id}] Sent clean text to translation service: '{clean_text[:30]}...'")
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to send to translation service: {e}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics for this session."""
        return {
            "session_id": self.session_id,
            "chunks_received": self.total_chunks_received,
            "inferences_run": self.total_inferences_run,
            "text_segments_sent": self.total_text_segments_sent,
            "buffer_length": len(self.audio_buffer),
            "previous_transcriptions": len(self.previous_transcriptions),
            "processed_segments": len(self.processed_text_segments),
            "last_inference_time": self.last_inference_time,
            "time_until_next_inference": self._time_until_next_inference()
        }
    
    def clear_session(self):
        """Clear all session data for cleanup."""
        self.audio_buffer.clear()
        self.previous_transcriptions.clear()
        self.processed_text_segments.clear()
        logger.info(f"[{self.session_id}] Session data cleared")


class TranslationServiceClient:
    """
    Client for sending clean text to the translation service.
    """
    
    def __init__(self, translation_service_url: str = "http://localhost:5003"):
        """
        Initialize translation service client.
        
        Args:
            translation_service_url: Base URL for translation service
        """
        self.base_url = translation_service_url.rstrip('/')
        
    async def send_clean_text(self, text: str, session_id: str, metadata: Dict = None):
        """
        Send clean, deduplicated text to translation service.
        
        Args:
            text: Clean text to translate
            session_id: Session identifier
            metadata: Additional metadata about the text
        """
        try:
            metadata = metadata or {}
            
            payload = {
                'text': text,
                'session_id': session_id,
                'source_language': metadata.get('language', 'auto'),
                'target_language': 'en',  # Default to English
                'metadata': metadata
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/process_clean_text",
                    json=payload,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Translation service response: {result.get('status', 'unknown')}")
                    return result
                else:
                    logger.error(f"Translation service error: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to send to translation service: {e}")
            return None