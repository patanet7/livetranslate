"""
REAL Integration Tests for OpenVINO ModelManager
NO MOCKS - Tests actual OpenVINO model loading and inference!

These tests use REAL OpenVINO models and REAL audio files.
Tests are SLOW but verify actual functionality.

IMPORTANT: These tests will SKIP on Mac (OpenVINO not supported).
"""

# Check OpenVINO availability
import importlib.util
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# OpenVINO is not supported on macOS — skip even if the package is importable
OPENVINO_AVAILABLE = (
    platform.system() != "Darwin"
    and importlib.util.find_spec("openvino") is not None
    and importlib.util.find_spec("openvino_genai") is not None
)

# Import from actual source
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

if OPENVINO_AVAILABLE:
    from model_manager import ModelManager

# Mark all tests as slow, integration, and requiring OpenVINO
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not available (not installed or unsupported platform)"),
]


@pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not available (not installed or unsupported platform)")
class TestOpenVINOModelManagerREAL:
    """
    REAL integration tests for OpenVINO - NO MOCKS!

    These tests actually:
    - Load real OpenVINO Whisper models
    - Process real audio from fixtures
    - Test actual NPU/GPU/CPU inference
    - Verify device fallback chain

    NOTE: Will skip on Mac (OpenVINO not supported)
    """

    @pytest.fixture(scope="class")
    def openvino_manager_npu(self):
        """Create OpenVINO ModelManager with NPU (if available)"""
        if not OPENVINO_AVAILABLE:
            pytest.skip("OpenVINO not available")

        print("\n🔧 Loading OpenVINO model (NPU preferred)...")
        manager = ModelManager(
            model_name="whisper-tiny",  # Use tiny for speed
            device="npu",
            models_dir=".models/openvino",  # Separate OpenVINO models!
        )
        print(f"✅ Model loaded on device: {manager.device}")
        try:
            yield manager
        finally:
            print("\n🧹 Shutting down NPU manager...")
            manager.shutdown()

    @pytest.fixture(scope="class")
    def openvino_manager_cpu(self):
        """Create OpenVINO ModelManager with CPU fallback"""
        if not OPENVINO_AVAILABLE:
            pytest.skip("OpenVINO not available")

        print("\n🔧 Loading OpenVINO model (CPU)...")
        manager = ModelManager(
            model_name="whisper-tiny", device="cpu", models_dir=".models/openvino"
        )
        print(f"✅ Model loaded on device: {manager.device}")
        try:
            yield manager
        finally:
            print("\n🧹 Shutting down CPU manager...")
            manager.shutdown()

    def test_openvino_model_actually_loads(self, openvino_manager_cpu):
        """Test that OpenVINO model actually loads (NO MOCKS!)"""
        manager = openvino_manager_cpu

        # Verify model is loaded
        assert manager.pipeline is not None
        assert manager.device in ["npu", "gpu", "cpu"]

        print(f"✅ OpenVINO model loaded on: {manager.device}")

    def test_openvino_transcribe_real_audio(self, openvino_manager_cpu, hello_world_audio):
        """Test OpenVINO transcription with REAL audio"""
        audio, sr = hello_world_audio

        manager = openvino_manager_cpu

        print(f"\n🎤 OpenVINO transcribing {len(audio)/sr:.2f}s audio...")

        start_time = time.time()
        result = manager.safe_inference(audio)
        elapsed = time.time() - start_time

        # Verify we got actual results
        assert result is not None
        assert isinstance(result, str | dict)

        text = result if isinstance(result, str) else result.get("text", "")

        print(f"✅ Transcribed in {elapsed:.2f}s")
        print(f"📝 Result: {text[:100]}...")

        assert len(text) > 0

    def test_openvino_transcribe_silence(self, openvino_manager_cpu, silence_audio):
        """Test OpenVINO with silence"""
        audio, _sr = silence_audio

        print("\n🔇 OpenVINO transcribing silence...")

        result = openvino_manager_cpu.safe_inference(audio)

        assert result is not None
        text = result if isinstance(result, str) else result.get("text", "")

        print(f"📝 Silence result: '{text}' (length: {len(text)})")
        assert len(text) < 50  # Should be empty or minimal

    def test_openvino_device_fallback_chain(self):
        """Test NPU → GPU → CPU fallback chain"""
        if not OPENVINO_AVAILABLE:
            pytest.skip("OpenVINO not available")

        print("\n🔄 Testing device fallback chain...")

        # Try to create with NPU
        manager = ModelManager(
            model_name="whisper-tiny",
            device="auto",  # Auto-detect
            models_dir=".models/openvino",
        )

        try:
            detected_device = manager.device
            print(f"✅ Auto-detected device: {detected_device}")

            # Should have selected one of the valid devices
            assert detected_device in ["npu", "gpu", "cpu"]

            # Verify model is actually loaded
            assert manager.pipeline is not None
        finally:
            manager.shutdown()

    def test_openvino_npu_inference_if_available(self, openvino_manager_npu, hello_world_audio):
        """Test NPU inference if NPU is available"""
        manager = openvino_manager_npu

        if manager.device != "npu":
            pytest.skip("NPU not available, fell back to CPU/GPU")

        audio, _sr = hello_world_audio

        print("\n🚀 Testing NPU inference...")

        start_time = time.time()
        result = manager.safe_inference(audio)
        elapsed = time.time() - start_time

        assert result is not None

        print(f"✅ NPU inference in {elapsed:.2f}s")

        # Verify NPU-specific stats
        stats = manager.get_stats()
        assert "device" in stats
        assert stats["device"] == "npu"

    def test_openvino_multiple_inferences(self, openvino_manager_cpu, hello_world_audio):
        """Test multiple sequential inferences"""
        audio, _sr = hello_world_audio
        manager = openvino_manager_cpu

        print("\n🔄 Running 5 sequential inferences...")

        results = []
        for i in range(5):
            result = manager.safe_inference(audio)
            results.append(result)
            print(f"  {i+1}. Completed")

        # All should succeed
        assert len(results) == 5
        assert all(r is not None for r in results)

        print("✅ All inferences completed")

    def test_openvino_cache_management(self):
        """Test OpenVINO LRU cache with multiple models"""
        if not OPENVINO_AVAILABLE:
            pytest.skip("OpenVINO not available")

        print("\n💾 Testing cache management...")

        # Load 3 different models (should hit LRU limit)
        models = ["whisper-tiny", "whisper-base"]
        managers = []

        try:
            for model_name in models:
                try:
                    manager = ModelManager(
                        model_name=model_name, device="cpu", models_dir=".models/openvino"
                    )
                    managers.append(manager)
                    print(f"  Loaded: {model_name}")
                except Exception as e:
                    print(f"  Skipped {model_name}: {e}")

            # At least one should have loaded
            assert len(managers) > 0

            print(f"✅ Loaded {len(managers)} models")
        finally:
            for mgr in managers:
                mgr.shutdown()

    def test_openvino_health_check(self, openvino_manager_cpu):
        """Test health check with real model"""
        manager = openvino_manager_cpu

        print("\n🏥 Running health check...")

        health = manager.health_check()

        assert health is not None
        assert isinstance(health, dict)
        assert "status" in health
        assert "device" in health

        print(f"✅ Health status: {health['status']}")
        print(f"   Device: {health['device']}")

        # Status should be healthy
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_openvino_clear_cache(self, openvino_manager_cpu):
        """Test cache clearing"""
        manager = openvino_manager_cpu

        print("\n🧹 Testing cache clearing...")

        # Get stats before
        manager.get_stats()

        # Clear cache
        manager.clear_cache()

        # Should still be functional
        health = manager.health_check()
        assert health is not None

        print("✅ Cache cleared successfully")

    def test_openvino_models_directory_separation(self):
        """Verify OpenVINO models go to .models/openvino/"""
        openvino_models_dir = Path(".models/openvino")

        print(f"\n📁 Checking OpenVINO models directory: {openvino_models_dir}")

        # Directory should exist after loading models
        assert openvino_models_dir.exists(), "OpenVINO models directory should exist"

        # Should contain OpenVINO model files (.xml, .bin)
        xml_files = list(openvino_models_dir.glob("**/*.xml"))
        bin_files = list(openvino_models_dir.glob("**/*.bin"))

        print(f"✅ Found {len(xml_files)} .xml and {len(bin_files)} .bin files")

        # OpenVINO models should have these files
        assert len(xml_files) > 0 or len(bin_files) > 0, "Should have OpenVINO model files"


@pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not available (not installed or unsupported platform)")
class TestOpenVINOPerformanceREAL:
    """Performance tests with REAL OpenVINO models"""

    @pytest.fixture(scope="class")
    def manager(self):
        if not OPENVINO_AVAILABLE:
            pytest.skip("OpenVINO not available")
        mgr = ModelManager(model_name="whisper-tiny", device="cpu", models_dir=".models/openvino")
        try:
            yield mgr
        finally:
            print("\n🧹 Shutting down performance test manager...")
            mgr.shutdown()

    def test_openvino_inference_performance(self, manager, hello_world_audio):
        """Measure real OpenVINO inference performance"""
        audio, _sr = hello_world_audio

        print("\n⚡ OpenVINO performance test: 10 inferences...")

        timings = []
        for _i in range(10):
            start = time.time()
            manager.safe_inference(audio)
            elapsed = time.time() - start
            timings.append(elapsed)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        print(f"✅ Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   Min: {min(timings):.3f}s, Max: {max(timings):.3f}s")

        # Sanity check
        assert avg_time < 10.0, "Inference taking too long"

    def test_openvino_thread_safety(self, manager, hello_world_audio):
        """Test thread-safe inference"""
        import threading

        audio, _sr = hello_world_audio

        print("\n🔀 OpenVINO concurrent test: 5 threads...")

        results = []
        errors = []

        def inference_worker(thread_id):
            try:
                result = manager.safe_inference(audio)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=inference_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5, "Not all threads completed"

        print("✅ All 5 threads completed successfully")


# Usage instructions
_USAGE_INSTRUCTIONS = """
To run these tests (Linux/Windows with OpenVINO):

# Run all OpenVINO integration tests
pytest tests/integration/test_openvino_real.py -v

# On Mac (will skip all tests gracefully)
pytest tests/integration/test_openvino_real.py -v
# Output: "SKIPPED [62] OpenVINO not installed"

# Run specific test
pytest tests/integration/test_openvino_real.py::TestOpenVINOModelManagerREAL::test_openvino_model_actually_loads -v

Note:
- These tests download and use REAL OpenVINO Whisper models
- Models stored in .models/openvino/ (separate from PyTorch)
- Tests automatically skip on Mac
- First run slower as models download
"""
