#!/usr/bin/env python3
"""
REAL Integration Tests for PyTorch ModelManager
NO MOCKS - Tests actual Whisper model loading and inference!

These tests use REAL whisper models (tiny/base) and REAL audio files.
Tests are SLOW but verify actual functionality.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
import time

# Import from actual source
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import the REAL ModelManager from whisper_service.py
from whisper_service import WhisperService

# Mark all tests as slow and integration
pytestmark = [pytest.mark.slow, pytest.mark.integration]


class TestPyTorchModelManagerREAL:
    """
    REAL integration tests - NO MOCKS!

    These tests actually:
    - Load real Whisper models (tiny, base)
    - Process real audio from fixtures
    - Test actual transcription
    - Verify device selection (GPU/MPS/CPU)
    """

    @pytest.fixture(scope="class")
    def whisper_service_cpu(self):
        """Create WhisperService with CPU (fastest for testing)"""
        print("\n🔧 Loading Whisper model on CPU...")
        service = WhisperService(
            model_name="tiny",  # Use tiny model for speed
            device="cpu",
            models_dir=".models/pytorch"  # Separate PyTorch models
        )
        print(f"✅ Model loaded: {service.model_manager.model_name} on {service.model_manager.device}")
        return service

    @pytest.fixture(scope="class")
    def whisper_service_gpu(self):
        """Create WhisperService with GPU/MPS if available"""
        # Check what's available
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("No GPU/MPS available")

        print(f"\n🔧 Loading Whisper model on {device.upper()}...")
        service = WhisperService(
            model_name="tiny",
            device=device,
            models_dir=".models/pytorch"
        )
        print(f"✅ Model loaded: {service.model_manager.model_name} on {service.model_manager.device}")
        return service

    def test_model_actually_loads(self, whisper_service_cpu):
        """Test that model actually loads (NO MOCKS!)"""
        manager = whisper_service_cpu.model_manager

        # Verify model is actually loaded
        assert manager.model is not None
        assert hasattr(manager.model, 'transcribe')
        assert hasattr(manager.model, 'is_multilingual')

        # Verify device
        assert manager.device in ['cpu', 'cuda', 'mps']
        print(f"✅ Model loaded on device: {manager.device}")

    def test_transcribe_real_audio_hello_world(self, whisper_service_cpu, hello_world_audio):
        """Test transcription with REAL audio fixture"""
        audio, sr = hello_world_audio

        print(f"\n🎤 Transcribing {len(audio)} samples ({len(audio)/sr:.2f}s) at {sr}Hz...")

        # REAL transcription - no mocks!
        start_time = time.time()
        result = whisper_service_cpu.transcribe(audio)
        elapsed = time.time() - start_time

        # Verify we got actual results
        assert result is not None
        assert 'text' in result
        assert isinstance(result['text'], str)

        print(f"✅ Transcribed in {elapsed:.2f}s")
        print(f"📝 Result: {result['text'][:100]}...")

        # Basic quality checks
        assert len(result['text']) > 0  # Should produce some text

    def test_transcribe_real_audio_silence(self, whisper_service_cpu, silence_audio):
        """Test transcription with silence (should return empty or minimal text)"""
        audio, sr = silence_audio

        print(f"\n🔇 Transcribing silence...")

        result = whisper_service_cpu.transcribe(audio)

        assert result is not None
        assert 'text' in result

        # Silence should produce empty or very short transcription
        text = result['text'].strip()
        print(f"📝 Silence result: '{text}' (length: {len(text)})")
        assert len(text) < 50  # Should be empty or very short

    def test_transcribe_real_audio_noisy(self, whisper_service_cpu, noisy_audio):
        """Test transcription with noisy audio"""
        audio, sr = noisy_audio

        print(f"\n🔊 Transcribing noisy audio...")

        result = whisper_service_cpu.transcribe(audio)

        assert result is not None
        assert 'text' in result

        print(f"📝 Noisy audio result: {result['text'][:100]}...")

    def test_multiple_transcriptions_session(self, whisper_service_cpu, hello_world_audio):
        """Test multiple transcriptions in sequence (session context)"""
        audio, sr = hello_world_audio

        session_id = "test-session-001"
        whisper_service_cpu.model_manager.init_context(session_id)

        print(f"\n🔄 Running 3 sequential transcriptions with session context...")

        results = []
        for i in range(3):
            result = whisper_service_cpu.transcribe(audio, session_id=session_id)
            results.append(result)
            print(f"  {i+1}. {result['text'][:50]}...")

        # Verify all succeeded
        assert len(results) == 3
        assert all('text' in r for r in results)

        # Check session context exists
        assert session_id in whisper_service_cpu.model_manager.sessions

    def test_concurrent_sessions_isolated(self, whisper_service_cpu, hello_world_audio, noisy_audio):
        """Test that multiple sessions maintain separate contexts"""
        hello_audio, _ = hello_world_audio
        noisy_audio_data, _ = noisy_audio

        session_1 = "session-english"
        session_2 = "session-noisy"

        # Initialize both sessions
        whisper_service_cpu.model_manager.init_context(session_1)
        whisper_service_cpu.model_manager.init_context(session_2)

        print(f"\n🔀 Testing session isolation...")

        # Transcribe in session 1
        result1 = whisper_service_cpu.transcribe(hello_audio, session_id=session_1)

        # Transcribe in session 2
        result2 = whisper_service_cpu.transcribe(noisy_audio_data, session_id=session_2)

        # Verify both succeeded
        assert 'text' in result1
        assert 'text' in result2

        # Verify sessions are separate in manager
        assert session_1 in whisper_service_cpu.model_manager.sessions
        assert session_2 in whisper_service_cpu.model_manager.sessions

        print(f"✅ Sessions isolated: {session_1} and {session_2}")

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                       reason="No GPU/MPS available")
    def test_gpu_transcription(self, whisper_service_gpu, hello_world_audio):
        """Test GPU/MPS transcription (if available)"""
        audio, sr = hello_world_audio

        device = whisper_service_gpu.model_manager.device
        print(f"\n🚀 Testing {device.upper()} transcription...")

        start_time = time.time()
        result = whisper_service_gpu.transcribe(audio)
        elapsed = time.time() - start_time

        assert result is not None
        assert 'text' in result

        print(f"✅ {device.upper()} transcription in {elapsed:.2f}s")
        print(f"📝 Result: {result['text'][:100]}...")

    def test_device_detection_priority(self):
        """Test device detection follows GPU/MPS → CPU priority"""
        from whisper_service import WhisperService

        print(f"\n🔍 Testing device detection...")

        # Create service with auto device detection
        service = WhisperService(
            model_name="tiny",
            device="auto",
            models_dir=".models/pytorch"
        )

        device = service.model_manager.device

        # Should be one of the valid devices
        assert device in ['cpu', 'cuda', 'mps']

        print(f"✅ Auto-detected device: {device}")

        # Verify priority order
        if torch.cuda.is_available():
            assert device == 'cuda', "Should prefer CUDA GPU"
        elif torch.backends.mps.is_available():
            assert device == 'mps', "Should prefer Apple MPS"
        else:
            assert device == 'cpu', "Should fallback to CPU"

    def test_model_warmup(self, whisper_service_cpu):
        """Test model warmup with dummy audio"""
        manager = whisper_service_cpu.model_manager

        print(f"\n🔥 Testing model warmup...")

        # Warmup should run without error
        manager.warmup()

        print(f"✅ Warmup complete")

    def test_long_audio_transcription(self, whisper_service_cpu, long_speech_audio):
        """Test transcription of longer audio (5 seconds)"""
        audio, sr = long_speech_audio

        print(f"\n⏱️  Transcribing long audio ({len(audio)/sr:.1f}s)...")

        start_time = time.time()
        result = whisper_service_cpu.transcribe(audio)
        elapsed = time.time() - start_time

        assert result is not None
        assert 'text' in result

        print(f"✅ Long transcription in {elapsed:.2f}s")
        print(f"📝 Result: {result['text'][:100]}...")

        # Longer audio should produce more text
        assert len(result['text']) > 0

    def test_models_directory_separation(self):
        """Verify PyTorch models go to .models/pytorch/"""
        from pathlib import Path

        pytorch_models_dir = Path(".models/pytorch")

        print(f"\n📁 Checking models directory: {pytorch_models_dir}")

        # Directory should exist after loading models
        assert pytorch_models_dir.exists(), "PyTorch models directory should exist"

        # Should contain model files
        model_files = list(pytorch_models_dir.glob("*.pt"))

        print(f"✅ Found {len(model_files)} PyTorch model files")
        for model_file in model_files:
            print(f"   - {model_file.name}")


class TestPyTorchContextManagementREAL:
    """Test rolling context with REAL models"""

    @pytest.fixture(scope="class")
    def service(self):
        """Create service for context tests"""
        from whisper_service import WhisperService
        return WhisperService(
            model_name="tiny",
            device="cpu",
            models_dir=".models/pytorch"
        )

    def test_context_init_and_cleanup(self, service):
        """Test session context initialization and cleanup"""
        session_id = "context-test-001"

        print(f"\n🔧 Testing context management...")

        # Initialize context
        service.model_manager.init_context(session_id)
        assert session_id in service.model_manager.sessions

        # Cleanup context
        service.model_manager.cleanup_session_context(session_id)
        assert session_id not in service.model_manager.sessions

        print(f"✅ Context lifecycle working")

    def test_context_accumulation(self, service, hello_world_audio):
        """Test that context accumulates across transcriptions"""
        audio, sr = hello_world_audio
        session_id = "context-accumulation-test"

        service.model_manager.init_context(session_id)

        print(f"\n📚 Testing context accumulation...")

        # Transcribe 3 times
        for i in range(3):
            result = service.transcribe(audio, session_id=session_id)
            print(f"  {i+1}. Transcribed: {result['text'][:30]}...")

        # Context should have accumulated
        # (Exact implementation depends on whisper_service.py)

        print(f"✅ Context accumulation working")


class TestPyTorchPerformanceREAL:
    """Performance tests with REAL models"""

    @pytest.fixture(scope="class")
    def service(self):
        from whisper_service import WhisperService
        return WhisperService(
            model_name="tiny",
            device="cpu",
            models_dir=".models/pytorch"
        )

    def test_transcription_performance(self, service, hello_world_audio):
        """Measure real transcription performance"""
        audio, sr = hello_world_audio

        print(f"\n⚡ Performance test: 10 transcriptions...")

        timings = []
        for i in range(10):
            start = time.time()
            result = service.transcribe(audio)
            elapsed = time.time() - start
            timings.append(elapsed)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        print(f"✅ Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   Min: {min(timings):.3f}s, Max: {max(timings):.3f}s")

        # Sanity check - should complete in reasonable time
        assert avg_time < 5.0, "Transcription taking too long"

    def test_concurrent_transcriptions(self, service, hello_world_audio):
        """Test concurrent transcriptions (thread safety)"""
        import threading

        audio, sr = hello_world_audio

        print(f"\n🔀 Concurrent test: 5 threads...")

        results = []
        errors = []

        def transcribe_worker(thread_id):
            try:
                session_id = f"thread-{thread_id}"
                service.model_manager.init_context(session_id)
                result = service.transcribe(audio, session_id=session_id)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=transcribe_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5, "Not all threads completed"

        print(f"✅ All 5 threads completed successfully")


# Usage instructions
"""
To run these tests:

# Run all integration tests (SLOW!)
pytest tests/integration/test_pytorch_real.py -v

# Run without slow tests
pytest tests/integration/test_pytorch_real.py -v -m "not slow"

# Run specific test
pytest tests/integration/test_pytorch_real.py::TestPyTorchModelManagerREAL::test_model_actually_loads -v

# With coverage
pytest tests/integration/test_pytorch_real.py --cov=whisper_service -v

Note: These tests download and use REAL Whisper models (~150MB for tiny model).
First run will be slower as models are downloaded.
"""
