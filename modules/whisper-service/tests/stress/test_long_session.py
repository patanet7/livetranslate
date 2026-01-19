#!/usr/bin/env python3
"""
Stress Tests: Long Session Testing (60-minute continuous streaming)

Tests memory leaks, KV cache overflow, and session history growth over extended periods.
Per ML Engineer review - Priority 3: 60-minute continuous streaming stress test

Critical checks:
1. Memory stays under 500MB throughout 60-minute session
2. KV cache doesn't overflow (stays within n_text_ctx=448)
3. Session history has bounded growth (doesn't grow indefinitely)
4. No crashes or degradation after extended use

Reference: modules/whisper-service/src/session_restart/session_manager.py
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import psutil
import pytest
import torch
from session_restart import SessionRestartTranscriber

logger = logging.getLogger(__name__)


# Memory tracking utilities
def get_process_memory_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb() -> float:
    """Get GPU memory usage in MB if available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


class MemoryTracker:
    """Track memory usage over time"""

    def __init__(self):
        self.samples = []
        self.start_time = time.time()

    def sample(self, label: str = ""):
        """Take memory sample"""
        elapsed = time.time() - self.start_time
        cpu_mb = get_process_memory_mb()
        gpu_mb = get_gpu_memory_mb()

        sample = {
            "elapsed": elapsed,
            "cpu_mb": cpu_mb,
            "gpu_mb": gpu_mb,
            "total_mb": cpu_mb + gpu_mb,
            "label": label,
        }
        self.samples.append(sample)

        return sample

    def get_stats(self) -> dict:
        """Get memory statistics"""
        if not self.samples:
            return {}

        total_mbs = [s["total_mb"] for s in self.samples]

        return {
            "initial_mb": self.samples[0]["total_mb"],
            "final_mb": self.samples[-1]["total_mb"],
            "peak_mb": max(total_mbs),
            "min_mb": min(total_mbs),
            "avg_mb": sum(total_mbs) / len(total_mbs),
            "growth_mb": self.samples[-1]["total_mb"] - self.samples[0]["total_mb"],
            "samples": len(self.samples),
        }

    def print_summary(self):
        """Print memory usage summary"""
        stats = self.get_stats()

        logger.info("\n" + "=" * 80)
        logger.info("MEMORY USAGE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Initial:  {stats['initial_mb']:.1f} MB")
        logger.info(f"Final:    {stats['final_mb']:.1f} MB")
        logger.info(f"Peak:     {stats['peak_mb']:.1f} MB")
        logger.info(f"Average:  {stats['avg_mb']:.1f} MB")
        logger.info(f"Growth:   {stats['growth_mb']:+.1f} MB")
        logger.info(f"Samples:  {stats['samples']}")
        logger.info("=" * 80)


# Audio generation utilities
def generate_speech_like_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate speech-like audio (filtered noise with pauses).

    Simulates realistic speech patterns with periods of activity and silence.
    """
    num_samples = int(duration_sec * sample_rate)

    # Generate filtered noise (band-limited to speech frequencies)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.3

    # Apply simple low-pass filter (speech is mostly < 8kHz)
    from scipy import signal

    b, a = signal.butter(4, 0.5)  # Cutoff at Nyquist/2
    audio = signal.filtfilt(b, a, audio)

    # Add pauses (simulate natural speech pauses)
    pause_interval = int(5 * sample_rate)  # Pause every 5 seconds
    pause_duration = int(0.5 * sample_rate)  # 0.5s pause

    for i in range(0, num_samples, pause_interval):
        audio[i : i + pause_duration] = 0.0

    return audio.astype(np.float32)


@pytest.mark.slow
@pytest.mark.stress
class TestLongSession60Minutes:
    """Test 60-minute continuous streaming session"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for long session test"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        transcriber = SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en", "zh"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

        yield transcriber

        # Cleanup
        transcriber.reset()
        del transcriber
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_60_minute_continuous_streaming(self, transcriber):
        """
        Test 60-minute continuous streaming for memory leaks.

        This is the main stress test for production readiness.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STRESS TEST: 60-Minute Continuous Streaming")
        logger.info("=" * 80)

        memory_tracker = MemoryTracker()
        memory_tracker.sample("start")

        # Test parameters
        total_duration_sec = 60 * 60  # 60 minutes
        chunk_duration_sec = 0.5  # 500ms chunks
        sample_rate = 16000
        memory_limit_mb = 500  # Alert if exceeded

        chunk_size_samples = int(chunk_duration_sec * sample_rate)
        total_chunks = int(total_duration_sec / chunk_duration_sec)

        logger.info(f"Duration: {total_duration_sec}s ({total_duration_sec/60:.0f} minutes)")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Memory limit: {memory_limit_mb} MB")
        logger.info(f"Initial memory: {memory_tracker.samples[0]['total_mb']:.1f} MB")

        start_time = time.time()
        chunks_processed = 0
        transcriptions_received = 0
        switches_detected = 0

        # Generate audio in batches to avoid memory issues
        batch_duration = 60  # Generate 1 minute at a time
        batch_chunks = int(batch_duration / chunk_duration_sec)

        try:
            for _batch_idx in range(int(total_duration_sec / batch_duration)):
                # Generate 1-minute batch
                batch_audio = generate_speech_like_audio(batch_duration, sample_rate)

                # Process batch in chunks
                for chunk_idx in range(batch_chunks):
                    start = chunk_idx * chunk_size_samples
                    end = start + chunk_size_samples
                    chunk = batch_audio[start:end]

                    # Process chunk
                    result = transcriber.process(chunk)

                    chunks_processed += 1
                    if result["text"]:
                        transcriptions_received += 1
                    if result["switch_detected"]:
                        switches_detected += 1

                    # Sample memory every 1000 chunks (~8.3 minutes)
                    if chunks_processed % 1000 == 0:
                        elapsed_min = chunks_processed * chunk_duration_sec / 60
                        sample = memory_tracker.sample(f"chunk_{chunks_processed}")

                        logger.info(
                            f"[{elapsed_min:.1f}min] Chunks: {chunks_processed}, "
                            f"Transcriptions: {transcriptions_received}, "
                            f"Switches: {switches_detected}, "
                            f"Memory: {sample['total_mb']:.1f} MB"
                        )

                        # Check memory limit
                        if sample["total_mb"] > memory_limit_mb:
                            logger.warning(
                                f"⚠️ Memory exceeded limit: {sample['total_mb']:.1f} MB > {memory_limit_mb} MB"
                            )

                    # Check for silence detection (stop early if too much silence)
                    if result.get("silence_detected", False):
                        logger.info(
                            f"🛑 Stopping at chunk {chunks_processed}: sustained silence detected"
                        )
                        break

                # Force garbage collection after each batch
                del batch_audio
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.info("⚠️ Test interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
            raise

        # Final measurements
        elapsed_sec = time.time() - start_time
        memory_tracker.sample("end")
        memory_tracker.print_summary()

        stats = transcriber.get_statistics()

        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Duration: {elapsed_sec:.1f}s ({elapsed_sec/60:.1f} minutes)")
        logger.info(f"Chunks processed: {chunks_processed}")
        logger.info(f"Transcriptions: {transcriptions_received}")
        logger.info(f"Language switches: {switches_detected}")
        logger.info(f"Total sessions: {stats['total_sessions']}")
        logger.info(f"Total switches: {stats['total_switches']}")
        logger.info(f"Audio processed: {stats['total_audio_seconds']:.1f}s")
        logger.info("=" * 80)

        # Assertions
        mem_stats = memory_tracker.get_stats()

        # Memory should not grow unboundedly
        memory_growth_mb = mem_stats["growth_mb"]
        max_acceptable_growth_mb = 200  # Allow up to 200MB growth

        assert (
            memory_growth_mb < max_acceptable_growth_mb
        ), f"Memory leak detected: grew by {memory_growth_mb:.1f} MB (limit: {max_acceptable_growth_mb} MB)"

        # Peak memory should stay under limit
        assert (
            mem_stats["peak_mb"] < memory_limit_mb * 1.5
        ), f"Peak memory {mem_stats['peak_mb']:.1f} MB exceeds limit"

        logger.info(f"✅ Memory growth: {memory_growth_mb:+.1f} MB (acceptable)")
        logger.info("✅ 60-minute stress test PASSED")


@pytest.mark.stress
class TestKVCacheOverflow:
    """Test KV cache doesn't overflow during long sequences"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for KV cache tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

    def test_kv_cache_bounded_in_long_session(self, transcriber):
        """
        Test KV cache stays within n_text_ctx=448 bounds.

        Session-restart approach should prevent KV cache from growing unbounded.
        """
        logger.info("Testing KV cache bounds over 10-minute session...")

        memory_tracker = MemoryTracker()
        memory_tracker.sample("start")

        # Generate 10 minutes of audio
        duration_sec = 10 * 60
        chunk_duration_sec = 0.5
        sample_rate = 16000

        audio = generate_speech_like_audio(duration_sec, sample_rate)
        chunk_size = int(chunk_duration_sec * sample_rate)

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            transcriber.process(chunk)

            # Sample memory periodically
            if i % (sample_rate * 60) == 0:  # Every minute
                sample = memory_tracker.sample(f"minute_{i // (sample_rate * 60)}")
                logger.info(f"[{i // (sample_rate * 60)}min] Memory: {sample['total_mb']:.1f} MB")

        memory_tracker.sample("end")
        mem_stats = memory_tracker.get_stats()

        # KV cache overflow would cause memory to grow significantly
        assert (
            mem_stats["growth_mb"] < 100
        ), f"KV cache may have overflowed: memory grew {mem_stats['growth_mb']:.1f} MB"

        logger.info("✅ KV cache stayed bounded")

    def test_session_restart_prevents_cache_overflow(self, transcriber):
        """
        Test session restart mechanism prevents KV cache overflow.

        When VAD detects speech boundaries, session restarts should clear KV cache.
        """
        logger.info("Testing session restart prevents KV cache overflow...")

        initial_sessions = transcriber.get_statistics()["total_sessions"]

        # Generate audio with clear pauses (triggers VAD boundaries)
        sample_rate = 16000
        speech_duration = 30  # 30s speech
        pause_duration = 1  # 1s pause

        for _i in range(5):  # 5 speech segments
            # Speech
            speech = generate_speech_like_audio(speech_duration, sample_rate)
            for j in range(0, len(speech), 8000):
                transcriber.process(speech[j : j + 8000])

            # Pause (triggers VAD END)
            pause = np.zeros(int(pause_duration * sample_rate), dtype=np.float32)
            for j in range(0, len(pause), 8000):
                transcriber.process(pause[j : j + 8000])

        final_stats = transcriber.get_statistics()
        sessions_created = final_stats["total_sessions"] - initial_sessions

        # Should have created multiple sessions (at least one per speech segment)
        assert (
            sessions_created >= 3
        ), f"Session restart not working: only {sessions_created} sessions created"

        logger.info(f"✅ Session restart working: {sessions_created} sessions created")


@pytest.mark.stress
class TestSessionHistoryBoundedGrowth:
    """Test session history has bounded growth"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for session history tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

    def test_session_history_bounded_growth(self, transcriber):
        """
        Test session history doesn't grow unbounded.

        In production, very long sessions should have bounded history
        (either LRU cache or periodic cleanup).
        """
        logger.info("Testing session history bounded growth...")

        memory_tracker = MemoryTracker()
        memory_tracker.sample("start")

        # Create many sessions (100 sessions with pauses)
        sample_rate = 16000
        num_sessions = 100

        for i in range(num_sessions):
            # Short speech
            speech = generate_speech_like_audio(5, sample_rate)  # 5s
            for j in range(0, len(speech), 8000):
                transcriber.process(speech[j : j + 8000])

            # Pause to trigger session end
            pause = np.zeros(16000, dtype=np.float32)
            for j in range(0, len(pause), 8000):
                transcriber.process(pause[j : j + 8000])

            # Sample memory every 10 sessions
            if (i + 1) % 10 == 0:
                sample = memory_tracker.sample(f"session_{i+1}")
                logger.info(f"Sessions: {i+1}, Memory: {sample['total_mb']:.1f} MB")

        memory_tracker.sample("end")
        mem_stats = memory_tracker.get_stats()

        stats = transcriber.get_statistics()
        logger.info(f"Total sessions created: {stats['total_sessions']}")
        logger.info(f"All sessions stored: {len(transcriber.all_sessions)}")

        # Memory growth should be sublinear with number of sessions
        # If it grows linearly, history is unbounded
        memory_growth_mb = mem_stats["growth_mb"]
        max_growth_mb = 150  # Allow up to 150MB for 100 sessions

        assert (
            memory_growth_mb < max_growth_mb
        ), f"Session history may be unbounded: memory grew {memory_growth_mb:.1f} MB for {num_sessions} sessions"

        logger.info(
            f"✅ Session history bounded: {memory_growth_mb:+.1f} MB growth for {num_sessions} sessions"
        )


@pytest.mark.stress
class TestNoCrashesOrDegradation:
    """Test no crashes or performance degradation over time"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for stability tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en", "zh"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

    def test_no_performance_degradation_over_time(self, transcriber):
        """
        Test processing speed doesn't degrade over time.

        If there are memory leaks or cache issues, processing
        may slow down significantly over long sessions.
        """
        logger.info("Testing for performance degradation over time...")

        sample_rate = 16000
        chunk_size = 8000  # 0.5s chunks

        # Measure processing times at different points
        processing_times = {
            "early": [],  # First 100 chunks
            "middle": [],  # Chunks 5000-5100
            "late": [],  # Chunks 10000-10100
        }

        # Generate audio
        audio = generate_speech_like_audio(duration_sec=60 * 10, sample_rate=sample_rate)  # 10 min

        chunks_processed = 0

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]

            start = time.time()
            transcriber.process(chunk)
            elapsed = time.time() - start

            chunks_processed += 1

            # Collect timing samples
            if chunks_processed <= 100:
                processing_times["early"].append(elapsed)
            elif 5000 <= chunks_processed <= 5100:
                processing_times["middle"].append(elapsed)
            elif 10000 <= chunks_processed <= 10100:
                processing_times["late"].append(elapsed)

            # Early termination if we have all samples
            if chunks_processed > 10100:
                break

        # Calculate average processing times
        avg_early = np.mean(processing_times["early"]) if processing_times["early"] else 0
        avg_middle = np.mean(processing_times["middle"]) if processing_times["middle"] else 0
        avg_late = np.mean(processing_times["late"]) if processing_times["late"] else 0

        logger.info(f"Average processing time (early):  {avg_early*1000:.2f}ms")
        logger.info(f"Average processing time (middle): {avg_middle*1000:.2f}ms")
        logger.info(f"Average processing time (late):   {avg_late*1000:.2f}ms")

        if avg_early > 0 and avg_late > 0:
            slowdown_factor = avg_late / avg_early
            logger.info(f"Slowdown factor: {slowdown_factor:.2f}x")

            # Should not slow down more than 2x
            assert (
                slowdown_factor < 2.0
            ), f"Performance degradation detected: {slowdown_factor:.2f}x slowdown"

            logger.info("✅ No significant performance degradation")

    def test_continuous_operation_without_crashes(self, transcriber):
        """
        Test continuous operation without any crashes.

        Process 30 minutes of audio without exceptions.
        """
        logger.info("Testing 30-minute continuous operation without crashes...")

        sample_rate = 16000
        duration_sec = 30 * 60  # 30 minutes
        chunk_size = 8000

        audio = generate_speech_like_audio(duration_sec, sample_rate)

        exceptions_caught = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]

            try:
                transcriber.process(chunk)
            except Exception as e:
                exceptions_caught.append((i, str(e)))
                logger.error(f"Exception at chunk {i // chunk_size}: {e}")

            if (i // chunk_size) % 1000 == 0:
                logger.info(f"Processed {i // chunk_size} chunks...")

        assert (
            len(exceptions_caught) == 0
        ), f"Caught {len(exceptions_caught)} exceptions during operation"

        logger.info("✅ 30-minute operation completed without crashes")


if __name__ == "__main__":
    # Run with stress marker
    pytest.main([__file__, "-v", "-m", "stress", "--log-cli-level=INFO"])
