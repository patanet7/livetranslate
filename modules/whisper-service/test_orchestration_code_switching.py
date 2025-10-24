#!/usr/bin/env python3
"""
Orchestration Code-Switching Test - Real-World Mixed Language Streaming

Tests the complete pipeline with realistic code-switching:
- Client → Orchestration Service → Whisper Service
- Mixed English (JFK) + Chinese audio in realistic pattern
- Hybrid tracking through orchestration layer
- Performance statistics collection

Pattern: 2 JFK chunks → Chinese chunk → JFK chunk → 3 Chinese chunks
"""

import sys
import os
import time
import base64
import socketio
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import librosa

# Service URLs
ORCHESTRATION_URL = 'http://127.0.0.1:3000'
WHISPER_URL = 'http://127.0.0.1:5001'

# Audio files
JFK_AUDIO = 'tests/audio/jfk.wav'
CHINESE_AUDIO = 'tests/audio/OSR_cn_000_0072_8k.wav'


@dataclass
class PerformanceMetrics:
    """Track performance statistics for code-switching"""
    chunk_times: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    language_switches: List[Tuple[str, str, float]] = field(default_factory=list)  # (from_lang, to_lang, time)
    result_lags: List[float] = field(default_factory=list)  # Time from chunk sent to result received

    total_chunks_sent: int = 0
    total_results_received: int = 0
    english_chunks: int = 0
    chinese_chunks: int = 0
    english_results: int = 0
    chinese_results: int = 0

    def add_chunk(self, language: str, sent_at: float):
        """Track sent chunk"""
        self.chunk_times.append(sent_at)
        self.total_chunks_sent += 1
        if language == 'en':
            self.english_chunks += 1
        elif language == 'zh':
            self.chinese_chunks += 1

    def add_result(self, language: str, received_at: float, processed_through_time: float):
        """Track received result"""
        self.response_times.append(received_at)
        self.total_results_received += 1

        if language == 'en':
            self.english_results += 1
        elif language == 'zh':
            self.chinese_results += 1

        # Calculate lag from last chunk to result
        if self.chunk_times:
            last_chunk_time = self.chunk_times[-1]
            lag = received_at - last_chunk_time
            self.result_lags.append(lag)

    def add_language_switch(self, from_lang: str, to_lang: str, time: float):
        """Track language switch"""
        if from_lang != to_lang:
            self.language_switches.append((from_lang, to_lang, time))

    def print_summary(self):
        """Print performance statistics"""
        print("\n" + "="*80)
        print("PERFORMANCE STATISTICS")
        print("="*80)

        print(f"\n📊 Chunk Statistics:")
        print(f"  Total chunks sent: {self.total_chunks_sent}")
        print(f"  English chunks: {self.english_chunks} ({self.english_chunks/self.total_chunks_sent*100:.1f}%)")
        print(f"  Chinese chunks: {self.chinese_chunks} ({self.chinese_chunks/self.total_chunks_sent*100:.1f}%)")

        print(f"\n📝 Result Statistics:")
        print(f"  Total results received: {self.total_results_received}")
        if self.total_results_received > 0:
            print(f"  English results: {self.english_results} ({self.english_results/self.total_results_received*100:.1f}%)")
            print(f"  Chinese results: {self.chinese_results} ({self.chinese_results/self.total_results_received*100:.1f}%)")
        else:
            print(f"  ⚠️  No results received!")

        print(f"\n🌍 Language Switching:")
        print(f"  Total switches: {len(self.language_switches)}")
        for i, (from_lang, to_lang, t) in enumerate(self.language_switches, 1):
            print(f"    Switch {i}: {from_lang} → {to_lang} at {t:.2f}s")

        if self.result_lags:
            print(f"\n⏱️  Response Latency:")
            print(f"  Mean lag: {np.mean(self.result_lags):.3f}s")
            print(f"  Median lag: {np.median(self.result_lags):.3f}s")
            print(f"  Min lag: {np.min(self.result_lags):.3f}s")
            print(f"  Max lag: {np.max(self.result_lags):.3f}s")

        print()


@dataclass
class ChunkTracker:
    """
    Hybrid tracking for sent chunks and received results

    Combines:
    - vexa-style timestamp tracking for correlation
    - SimulStreaming-style progress monitoring
    """
    chunks_sent: List[Dict[str, Any]] = field(default_factory=list)
    results_by_abs_time: Dict[float, Dict[str, Any]] = field(default_factory=dict)
    latest_processed_time: float = 0.0
    total_audio_sent: float = 0.0
    results_received_count: int = 0
    last_language: str = None

    def track_sent_chunk(self, chunk_index: int, audio_data: np.ndarray, sample_rate: int,
                        sent_at: float, language_label: str) -> Dict[str, Any]:
        """Track a sent chunk with timestamp metadata"""
        chunk_duration = len(audio_data) / sample_rate
        chunk_start = self.total_audio_sent
        chunk_end = chunk_start + chunk_duration

        chunk_metadata = {
            'chunk_index': chunk_index,
            'audio_start_time': chunk_start,
            'audio_end_time': chunk_end,
            'chunk_duration': chunk_duration,
            'sent_at': sent_at,
            'language_label': language_label,  # Expected language
        }
        self.chunks_sent.append(chunk_metadata)
        self.total_audio_sent = chunk_end
        return chunk_metadata

    def track_received_result(self, result: Dict[str, Any]):
        """Track received result with hybrid metadata"""
        self.results_received_count += 1

        # vexa-style deduplication
        abs_start = result.get('absolute_start_time')
        if abs_start is not None:
            existing = self.results_by_abs_time.get(abs_start)
            updated_at_new = result.get('updated_at', 0)
            updated_at_old = existing.get('updated_at', 0) if existing else 0
            if updated_at_new >= updated_at_old:
                self.results_by_abs_time[abs_start] = result

        # Update latest processed time from hybrid tracking
        timestamp_tracking = result.get('timestamp_tracking', {})
        processed_through = timestamp_tracking.get('processed_through_time', 0)
        if processed_through > 0:
            self.latest_processed_time = max(self.latest_processed_time, processed_through)

        # Track language
        current_language = result.get('detected_language')
        if current_language and current_language != self.last_language:
            self.last_language = current_language

    def is_complete(self) -> bool:
        """Check if all sent audio has been processed"""
        tolerance = 0.1  # 100ms tolerance
        return (self.total_audio_sent - self.latest_processed_time) < tolerance

    def get_processing_progress(self) -> float:
        """Get processing progress percentage"""
        if self.total_audio_sent == 0:
            return 0.0
        return (self.latest_processed_time / self.total_audio_sent) * 100.0

    def get_unprocessed_chunks(self) -> int:
        """Estimate number of unprocessed chunks"""
        unprocessed_duration = self.total_audio_sent - self.latest_processed_time
        if unprocessed_duration <= 0:
            return 0
        # Assume ~1 second per chunk
        return int(np.ceil(unprocessed_duration))


def load_wav_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load WAV file and return int16 PCM data"""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    audio_int16 = (audio * 32767).astype(np.int16)
    print(f"  Loaded: {file_path}")
    print(f"  Sample rate: {sr}Hz, Duration: {len(audio)/sr:.2f}s")
    return audio_int16, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio

    print(f"  Resampling {orig_sr}Hz → {target_sr}Hz...")
    audio_float = audio.astype(np.float32) / 32767.0
    audio_resampled = librosa.resample(audio_float, orig_sr=orig_sr, target_sr=target_sr)
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    return audio_int16


def create_mixed_stream(jfk_audio: np.ndarray, chinese_audio: np.ndarray,
                       chunk_size_samples: int = 640) -> List[Tuple[np.ndarray, str]]:
    """
    Create mixed audio stream with realistic code-switching pattern

    Default chunk size: 640 samples = 0.04s @ 16kHz (matches SimulStreaming)
    Tiny chunks for fine-grained VAD, server processes every 1.2s internally.

    Pattern: Realistic code-switching blocks throughout full audio

    Returns:
        List of (chunk, language_label) tuples
    """
    stream = []

    # Calculate number of chunks available from each audio
    jfk_chunks_available = len(jfk_audio) // chunk_size_samples
    chinese_chunks_available = len(chinese_audio) // chunk_size_samples

    print(f"\n📦 Audio Inventory:")
    print(f"  JFK chunks available: {jfk_chunks_available} ({jfk_chunks_available * chunk_size_samples / 16000:.1f}s)")
    print(f"  Chinese chunks available: {chinese_chunks_available} ({chinese_chunks_available * chunk_size_samples / 16000:.1f}s)")

    # Pattern: Realistic code-switching throughout FULL audio duration
    # Blocks of ~2s EN, ~1s ZH, ~2s EN, ~3s ZH, then remaining audio
    # With 0.04s chunks: 25 chunks per second
    jfk_idx = 0
    chinese_idx = 0

    # Block 1: 2 seconds JFK (50 chunks @ 0.04s)
    for _ in range(min(50, jfk_chunks_available - jfk_idx)):
        start = jfk_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = jfk_audio[start:end]
        stream.append((chunk, 'en'))
        jfk_idx += 1

    # Block 2: 1 second Chinese (25 chunks)
    for _ in range(min(25, chinese_chunks_available - chinese_idx)):
        start = chinese_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = chinese_audio[start:end]
        stream.append((chunk, 'zh'))
        chinese_idx += 1

    # Block 3: 2 seconds JFK (50 chunks)
    for _ in range(min(50, jfk_chunks_available - jfk_idx)):
        start = jfk_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = jfk_audio[start:end]
        stream.append((chunk, 'en'))
        jfk_idx += 1

    # Block 4: 3 seconds Chinese (75 chunks)
    for _ in range(min(75, chinese_chunks_available - chinese_idx)):
        start = chinese_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = chinese_audio[start:end]
        stream.append((chunk, 'zh'))
        chinese_idx += 1

    # Block 5: Remaining JFK (all remaining)
    while jfk_idx < jfk_chunks_available:
        start = jfk_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = jfk_audio[start:end]
        stream.append((chunk, 'en'))
        jfk_idx += 1

    # Block 6: Remaining Chinese (all remaining)
    while chinese_idx < chinese_chunks_available:
        start = chinese_idx * chunk_size_samples
        end = start + chunk_size_samples
        chunk = chinese_audio[start:end]
        stream.append((chunk, 'zh'))
        chinese_idx += 1

    print(f"\n📊 Stream Pattern Created: {len(stream)} total chunks ({len(stream)*chunk_size_samples/16000:.1f}s total)")
    print(f"  Block 1: 50 EN chunks (2.0s)")
    print(f"  Block 2: 25 ZH chunks (1.0s)")
    print(f"  Block 3: 50 EN chunks (2.0s)")
    print(f"  Block 4: 75 ZH chunks (3.0s)")
    print(f"  Block 5: {jfk_idx - 100} EN chunks ({(jfk_idx-100)*chunk_size_samples/16000:.1f}s)")
    print(f"  Block 6: {chinese_idx - 100} ZH chunks ({(chinese_idx-100)*chunk_size_samples/16000:.1f}s)")

    return stream


def test_orchestration_code_switching():
    """Test code-switching through orchestration service"""
    print("="*80)
    print("ORCHESTRATION CODE-SWITCHING TEST")
    print("="*80)

    # Load audio files
    jfk_audio, jfk_sr = load_wav_file(JFK_AUDIO)
    chinese_audio, chinese_sr = load_wav_file(CHINESE_AUDIO)

    # Resample to 16kHz if needed
    jfk_audio = resample_audio(jfk_audio, jfk_sr, 16000)
    chinese_audio = resample_audio(chinese_audio, chinese_sr, 16000)

    # Create mixed stream with 0.25s chunks (4000 samples @ 16kHz)
    # Balance between processing efficiency and code-switching responsiveness
    stream = create_mixed_stream(jfk_audio, chinese_audio, chunk_size_samples=640)

    # Initialize trackers
    tracker = ChunkTracker()
    metrics = PerformanceMetrics()

    # Setup SocketIO client to Whisper service (for now - will use orchestration later)
    sio = socketio.Client()
    results_received = []
    last_language = None

    @sio.on('connect')
    def on_connect():
        print("\n✓ Connected to Whisper service")

    @sio.on('transcription_result')
    def on_result(data):
        nonlocal last_language
        try:
            results_received.append(data)

            # Track result
            tracker.track_received_result(data)

            # Extract metadata
            detected_lang = data.get('detected_language')
            text = data.get('text', '')
            is_final = data.get('is_final', False)

            attention_tracking = data.get('attention_tracking', {})
            timestamp_tracking = data.get('timestamp_tracking', {})

            is_session_complete = timestamp_tracking.get('is_session_complete', False)
            processed_through = timestamp_tracking.get('processed_through_time', 0)
            most_attended_frame = attention_tracking.get('most_attended_frame', 0)

            # Track language switch
            if detected_lang and detected_lang != last_language and last_language is not None:
                metrics.add_language_switch(last_language, detected_lang, time.time())
                print(f"\n  🌍 LANGUAGE SWITCH: {last_language} → {detected_lang}")
            last_language = detected_lang

            # Track performance
            metrics.add_result(detected_lang, time.time(), processed_through)

            # Print result
            print(f"\n  Result #{tracker.results_received_count}:")
            print(f"    language={detected_lang}, is_final={is_final}")
            print(f"    text='{text[:60]}...'")
            print(f"    processed_through={processed_through:.2f}s, frame={most_attended_frame}")
            print(f"    progress={tracker.get_processing_progress():.1f}%")

        except Exception as e:
            print(f"  ⚠️  Error in on_result callback: {e}")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        # Connect to service
        sio.connect(WHISPER_URL)

        # Generate session ID
        session_id = f"test-codeswitching-{int(time.time())}"

        # Join session
        sio.emit('join_session', {
            'session_id': session_id,
            'model_name': 'large-v3-turbo',
        })
        time.sleep(0.5)

        print(f"\n📡 Streaming {len(stream)} chunks with code-switching enabled...")

        # Stream chunks
        for i, (chunk, expected_lang) in enumerate(stream):
            # Track chunk
            chunk_metadata = tracker.track_sent_chunk(i, chunk, 16000, time.time(), expected_lang)
            metrics.add_chunk(expected_lang, time.time())

            # Convert to base64
            chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')

            # Prepare request with hybrid tracking metadata
            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "large-v3-turbo",
                "sample_rate": 16000,
                "enable_code_switching": True,  # CRITICAL: Enable code-switching

                # Hybrid tracking metadata
                "chunk_index": chunk_metadata['chunk_index'],
                "audio_start_time": chunk_metadata['audio_start_time'],
                "audio_end_time": chunk_metadata['audio_end_time'],
                "chunk_duration": chunk_metadata['chunk_duration'],
                "is_last_chunk": (i == len(stream) - 1),

                # Configuration
                "config": {
                    "enable_code_switching": True,
                    "sustained_lang_duration": 3.0,
                }
            }

            print(f"  Chunk {i+1}/{len(stream)}: {expected_lang} ({chunk_metadata['audio_start_time']:.2f}s - {chunk_metadata['audio_end_time']:.2f}s)")

            sio.emit('transcribe_stream', request_data)
            time.sleep(0.1)  # 100ms between chunks (slightly faster than realtime for 0.25s chunks)

        print(f"\n  All {len(stream)} chunks sent (total: {tracker.total_audio_sent:.2f}s)")
        print(f"  ⚠️  IMPORTANT: is_final=True means 'sentence complete', NOT 'session done'!")

        # Intelligent wait using hybrid tracking
        print(f"  Waiting for is_session_complete=True OR tracker.is_complete()...")

        max_wait_seconds = 60
        check_interval = 5

        for i in range(int(max_wait_seconds / check_interval)):
            time.sleep(check_interval)

            # Check hybrid completion criteria
            if tracker.is_complete():
                print(f"  ✅ Tracker reports complete: {tracker.latest_processed_time:.2f}s / {tracker.total_audio_sent:.2f}s")
                break

            # Check server-reported completion
            if results_received:
                last_result = results_received[-1]
                timestamp_tracking = last_result.get('timestamp_tracking', {})
                if timestamp_tracking.get('is_session_complete', False):
                    print(f"  ✅ Server reports is_session_complete=True")
                    break

            # Print progress
            progress = tracker.get_processing_progress()
            unprocessed = tracker.get_unprocessed_chunks()
            print(f"    Waiting... {progress:.1f}% processed, {unprocessed} chunks pending")
        else:
            print(f"\n  ⚠️  Timeout after {max_wait_seconds}s")
            print(f"    Final progress: {tracker.get_processing_progress():.1f}%")
            print(f"    Unprocessed chunks: {tracker.get_unprocessed_chunks()}")

        # Leave session
        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)

        # Print summary
        print(f"\n  Received {len(results_received)} total results")
        print(f"  Tracker summary: {len(tracker.chunks_sent)} chunks sent, {tracker.latest_processed_time:.2f}s/{tracker.total_audio_sent:.2f}s processed")

        for i, result in enumerate(results_received, 1):
            text = result.get('text', '')
            is_final = result.get('is_final', False)
            detected_lang = result.get('detected_language')
            print(f"    Result {i}: lang={detected_lang}, is_final={is_final}, text='{text[:60]}...'")

        # Print performance statistics
        metrics.print_summary()

        # Check for language detection
        languages_detected = set()
        for result in results_received:
            lang = result.get('detected_language')
            if lang:
                languages_detected.add(lang)

        print(f"\n✅ Languages detected: {languages_detected}")

        # Verify code-switching worked
        if 'en' in languages_detected and 'zh' in languages_detected:
            print("✅ Code-switching detected! Both English and Chinese recognized.")
        else:
            print("⚠️  Code-switching may not have worked properly.")

        return len(languages_detected) >= 2

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sio.disconnect()


if __name__ == '__main__':
    success = test_orchestration_code_switching()
    sys.exit(0 if success else 1)
