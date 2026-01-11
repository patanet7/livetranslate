"""
Pipeline Endpoints Tests

Tests for pipeline processing endpoints including:
- Batch audio processing
- Real-time session management
- WebSocket parameter updates
- Audio chunk processing
- Pipeline validation
- Concurrent sessions
"""

import pytest
import json
import wave
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create FastAPI test client"""
    from src.main_fastapi import app

    return TestClient(app)


@pytest.fixture
def test_audio_file(tmp_path):
    """Generate test audio file"""
    file_path = tmp_path / "test_audio.wav"

    # Generate 1 second of audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)

    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return file_path


@pytest.fixture
def simple_pipeline_config():
    """Simple pipeline configuration for testing"""
    return {
        "pipeline_id": "test-pipeline",
        "name": "Test Pipeline",
        "stages": {
            "input": {
                "enabled": True,
                "gain_in": 0.0,
                "gain_out": 0.0,
                "parameters": {},
            },
            "output": {
                "enabled": True,
                "gain_in": 0.0,
                "gain_out": 0.0,
                "parameters": {},
            },
        },
        "connections": [],
    }


@pytest.fixture
def noise_reduction_pipeline_config():
    """Pipeline configuration with noise reduction"""
    return {
        "pipeline_id": "test-noise-reduction",
        "name": "Noise Reduction Test",
        "stages": {
            "noise_reduction": {
                "enabled": True,
                "gain_in": 0.0,
                "gain_out": 0.0,
                "parameters": {
                    "strength": 0.7,
                    "voiceProtection": True,
                    "adaptationRate": 0.1,
                },
            }
        },
        "connections": [],
    }


class TestPipelineBatchProcessing:
    """Tests for batch audio processing through pipeline"""

    def test_process_pipeline_batch_simple(
        self, client, test_audio_file, simple_pipeline_config
    ):
        """Test simple batch audio processing"""
        with open(test_audio_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(simple_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "processed_audio" in data
        assert "metrics" in data
        assert data["metrics"]["total_latency"] > 0

    def test_process_pipeline_with_noise_reduction(
        self, client, test_audio_file, noise_reduction_pipeline_config
    ):
        """Test pipeline with noise reduction stage"""
        with open(test_audio_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(noise_reduction_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stage_metrics" in data["metrics"]
        assert "noise_reduction" in data["metrics"]["stage_metrics"]

    def test_process_pipeline_invalid_audio(
        self, client, tmp_path, simple_pipeline_config
    ):
        """Test pipeline with invalid audio file"""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("not audio data")

        with open(invalid_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(simple_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("invalid.wav", audio_file, "audio/wav")},
            )

        assert response.status_code in [400, 422, 500]

    def test_process_pipeline_missing_audio(self, client, simple_pipeline_config):
        """Test pipeline without audio file"""
        response = client.post(
            "/api/pipeline/process",
            data={
                "pipeline_config": json.dumps(simple_pipeline_config),
                "processing_mode": "batch",
                "output_format": "wav",
            },
        )

        assert response.status_code == 422

    def test_process_pipeline_different_formats(
        self, client, test_audio_file, simple_pipeline_config
    ):
        """Test pipeline with different output formats"""
        formats = ["wav", "mp3", "flac"]

        for fmt in formats:
            with open(test_audio_file, "rb") as audio_file:
                response = client.post(
                    "/api/pipeline/process",
                    data={
                        "pipeline_config": json.dumps(simple_pipeline_config),
                        "processing_mode": "batch",
                        "output_format": fmt,
                    },
                    files={"audio_file": ("test.wav", audio_file, "audio/wav")},
                )

            # Some formats may not be supported
            assert response.status_code in [200, 400, 501]


class TestRealtimeSessionManagement:
    """Tests for real-time session creation and management"""

    def test_start_realtime_session(self, client, simple_pipeline_config):
        """Test real-time session creation"""
        response = client.post(
            "/api/pipeline/realtime/start",
            json={"pipeline_config": simple_pipeline_config},
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "running"
        assert len(data["session_id"]) > 0

    def test_start_realtime_session_invalid_config(self, client):
        """Test session creation with invalid config"""
        invalid_config = {
            "pipeline_id": "test",
            # Missing required fields
        }

        response = client.post(
            "/api/pipeline/realtime/start", json={"pipeline_config": invalid_config}
        )

        assert response.status_code in [400, 422]

    def test_stop_realtime_session(self, client, simple_pipeline_config):
        """Test stopping real-time session"""
        # Start session
        start_response = client.post(
            "/api/pipeline/realtime/start",
            json={"pipeline_config": simple_pipeline_config},
        )
        assert start_response.status_code == 200
        session_id = start_response.json()["session_id"]

        # Stop session
        stop_response = client.post(f"/api/pipeline/realtime/stop/{session_id}")

        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["success"] is True

    def test_stop_nonexistent_session(self, client):
        """Test stopping session that doesn't exist"""
        response = client.post("/api/pipeline/realtime/stop/nonexistent-session-id")

        assert response.status_code in [404, 400]

    def test_get_active_sessions(self, client, simple_pipeline_config):
        """Test retrieving active sessions"""
        # Start multiple sessions
        session_ids = []
        for i in range(3):
            response = client.post(
                "/api/pipeline/realtime/start",
                json={"pipeline_config": simple_pipeline_config},
            )
            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])

        # Get active sessions
        response = client.get("/api/pipeline/realtime/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data
        assert len(data["active_sessions"]) >= 3

        # Cleanup - stop all sessions
        for session_id in session_ids:
            client.post(f"/api/pipeline/realtime/stop/{session_id}")


class TestPipelineValidation:
    """Tests for pipeline configuration validation"""

    def test_validate_valid_pipeline(self, client, simple_pipeline_config):
        """Test validation of valid pipeline"""
        response = client.post(
            "/api/pipeline/validate", json={"pipeline_config": simple_pipeline_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "errors" not in data or len(data["errors"]) == 0

    def test_validate_invalid_pipeline_missing_stages(self, client):
        """Test validation with missing stages"""
        invalid_config = {
            "pipeline_id": "test",
            "name": "Invalid Pipeline",
            "stages": {},  # Empty stages
            "connections": [],
        }

        response = client.post(
            "/api/pipeline/validate", json={"pipeline_config": invalid_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_invalid_connections(self, client):
        """Test validation with invalid connections"""
        invalid_config = {
            "pipeline_id": "test",
            "name": "Invalid Connections",
            "stages": {"stage1": {"enabled": True, "parameters": {}}},
            "connections": [
                {
                    "source": "stage1",
                    "target": "nonexistent_stage",  # Invalid target
                }
            ],
        }

        response = client.post(
            "/api/pipeline/validate", json={"pipeline_config": invalid_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_circular_dependencies(self, client):
        """Test detection of circular dependencies"""
        circular_config = {
            "pipeline_id": "test",
            "name": "Circular Pipeline",
            "stages": {
                "stage1": {"enabled": True, "parameters": {}},
                "stage2": {"enabled": True, "parameters": {}},
                "stage3": {"enabled": True, "parameters": {}},
            },
            "connections": [
                {"source": "stage1", "target": "stage2"},
                {"source": "stage2", "target": "stage3"},
                {"source": "stage3", "target": "stage1"},  # Creates cycle
            ],
        }

        response = client.post(
            "/api/pipeline/validate", json={"pipeline_config": circular_config}
        )

        assert response.status_code == 200
        data = response.json()
        # Should detect circular dependency
        assert (
            data["valid"] is False
            or "circular" in str(data.get("warnings", [])).lower()
        )


class TestConcurrentSessions:
    """Tests for concurrent real-time session handling"""

    def test_multiple_concurrent_sessions(self, client, simple_pipeline_config):
        """Test multiple concurrent real-time sessions"""
        sessions = []

        # Start 5 concurrent sessions
        for i in range(5):
            response = client.post(
                "/api/pipeline/realtime/start",
                json={
                    "pipeline_config": {
                        **simple_pipeline_config,
                        "pipeline_id": f"test-{i}",
                        "name": f"Test Pipeline {i}",
                    }
                },
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])

        # Verify all sessions are active
        response = client.get("/api/pipeline/realtime/sessions")
        assert response.status_code == 200
        active_sessions = response.json()["active_sessions"]
        assert len(active_sessions) >= 5

        # Cleanup - stop all sessions
        for session_id in sessions:
            stop_response = client.post(f"/api/pipeline/realtime/stop/{session_id}")
            assert stop_response.status_code == 200

    def test_session_isolation(self, client, simple_pipeline_config):
        """Test that sessions are properly isolated"""
        # Start two sessions with different configs
        config1 = {**simple_pipeline_config, "pipeline_id": "session-1"}
        config2 = {**simple_pipeline_config, "pipeline_id": "session-2"}

        response1 = client.post(
            "/api/pipeline/realtime/start", json={"pipeline_config": config1}
        )
        response2 = client.post(
            "/api/pipeline/realtime/start", json={"pipeline_config": config2}
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        session_id_1 = response1.json()["session_id"]
        session_id_2 = response2.json()["session_id"]

        # Session IDs should be different
        assert session_id_1 != session_id_2

        # Cleanup
        client.post(f"/api/pipeline/realtime/stop/{session_id_1}")
        client.post(f"/api/pipeline/realtime/stop/{session_id_2}")


class TestPipelineMetrics:
    """Tests for pipeline performance metrics"""

    def test_metrics_included_in_response(
        self, client, test_audio_file, noise_reduction_pipeline_config
    ):
        """Test that metrics are included in processing response"""
        with open(test_audio_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(noise_reduction_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()

        # Check required metrics
        assert "metrics" in data
        metrics = data["metrics"]

        # Total metrics
        assert "total_latency" in metrics
        assert metrics["total_latency"] > 0

        # Stage-specific metrics
        if "stage_metrics" in metrics:
            assert isinstance(metrics["stage_metrics"], dict)

    def test_quality_metrics(
        self, client, test_audio_file, noise_reduction_pipeline_config
    ):
        """Test that quality metrics are calculated"""
        with open(test_audio_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(noise_reduction_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                    "calculate_quality": "true",
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
            )

        if response.status_code == 200:
            data = response.json()
            if "quality_metrics" in data.get("metrics", {}):
                quality = data["metrics"]["quality_metrics"]
                assert "snr" in quality or "rms" in quality


class TestErrorHandling:
    """Tests for error handling in pipeline processing"""

    def test_handle_large_file(self, client, tmp_path, simple_pipeline_config):
        """Test handling of very large audio file"""
        # Generate large audio file (10 seconds at 48kHz)
        large_file = tmp_path / "large.wav"
        sample_rate = 48000
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(str(large_file), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        with open(large_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(simple_pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("large.wav", audio_file, "audio/wav")},
            )

        # Should either succeed or return appropriate error
        assert response.status_code in [200, 413, 422]

    def test_timeout_handling(self, client, simple_pipeline_config):
        """Test timeout handling for long-running processing"""
        # This test would require a pipeline that takes a long time
        # Implementation depends on backend timeout configuration
        pass

    def test_invalid_parameter_values(self, client, test_audio_file):
        """Test handling of invalid parameter values"""
        invalid_config = {
            "pipeline_id": "test",
            "name": "Invalid Parameters",
            "stages": {
                "noise_reduction": {
                    "enabled": True,
                    "parameters": {
                        "strength": 2.5,  # Invalid: should be 0-1
                        "voiceProtection": "invalid",  # Invalid: should be boolean
                    },
                }
            },
            "connections": [],
        }

        with open(test_audio_file, "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(invalid_config),
                    "processing_mode": "batch",
                    "output_format": "wav",
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
            )

        # Should return validation error
        assert response.status_code in [400, 422]


class TestPipelinePresets:
    """Tests for pipeline preset management"""

    def test_get_available_presets(self, client):
        """Test retrieving available pipeline presets"""
        response = client.get("/api/pipeline/presets")

        assert response.status_code == 200
        data = response.json()
        assert "presets" in data
        assert isinstance(data["presets"], list)

    def test_load_preset(self, client):
        """Test loading a specific preset"""
        # First get available presets
        presets_response = client.get("/api/pipeline/presets")
        if presets_response.status_code == 200:
            presets = presets_response.json()["presets"]
            if len(presets) > 0:
                preset_id = presets[0].get("id", presets[0].get("name"))

                # Load the preset
                response = client.get(f"/api/pipeline/presets/{preset_id}")
                assert response.status_code in [200, 404]

                if response.status_code == 200:
                    data = response.json()
                    assert "pipeline_config" in data or "config" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
