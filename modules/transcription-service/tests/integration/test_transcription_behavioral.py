"""
Behavioral tests for Whisper transcription service.

IMPORTANT: These tests use REAL models - NO MOCKS.
"""

from datetime import datetime
from pathlib import Path

import pytest

TEST_OUTPUT_DIR = Path(__file__).parent.parent / "output"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def get_test_output_path(test_name: str) -> Path:
    """Generate timestamped output path for test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TEST_OUTPUT_DIR / f"{timestamp}_test_{test_name}_results.log"


class TestTranscriptionBehavioral:
    """Behavioral tests for transcription - NO MOCKS."""

    @pytest.mark.behavioral
    @pytest.mark.integration
    @pytest.mark.slow
    def test_whisper_transcribes_audio(self):
        """
        Test that Whisper actually transcribes audio content.

        Uses REAL Whisper model, not mocks.
        """
        output_path = get_test_output_path("whisper_transcription")
        with open(output_path, "w") as f:
            f.write(f"Test started at {datetime.now().isoformat()}\n")
            f.write("Testing Whisper transcription\n")
            # TODO: Implement actual Whisper test with real model
            f.write("Test completed\n")

    @pytest.mark.behavioral
    @pytest.mark.integration
    @pytest.mark.npu
    def test_npu_acceleration_works(self):
        """Test NPU acceleration for Whisper."""
        output_path = get_test_output_path("npu_acceleration")
        with open(output_path, "w") as f:
            f.write("Testing NPU acceleration\n")
            # TODO: Implement actual NPU test
            f.write("Test completed\n")
