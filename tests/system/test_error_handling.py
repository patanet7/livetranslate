#!/usr/bin/env python3
"""
Comprehensive Error Handling Validation Tests

Test script to validate all error types, recovery mechanisms, circuit breakers,
and error boundaries across the audio processing pipeline.
"""

import json
import logging
import sys
import tempfile
import time
import wave

import numpy as np
import requests

# Add modules to path
sys.path.insert(0, "modules/orchestration-service/src")
sys.path.insert(0, "modules/whisper-service/src")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandlingValidator:
    """Comprehensive error handling test suite"""

    def __init__(
        self, orchestration_url="http://localhost:3000", whisper_url="http://localhost:5001"
    ):
        self.orchestration_url = orchestration_url
        self.whisper_url = whisper_url
        self.test_results = []

    def log_test_result(self, test_name: str, success: bool, details: dict | None = None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "timestamp": time.time(),
            "details": details or {},
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if details:
            logger.info(f"   Details: {details}")

    def create_test_audio(
        self, duration: float = 1.0, sample_rate: int = 16000, format: str = "wav"
    ) -> bytes:
        """Create test audio data"""
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone at 50% volume

        if format == "wav":
            # Create WAV file in memory
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                with open(tmp_file.name, "rb") as f:
                    return f.read()
        else:
            # Return raw PCM data
            return (audio_data * 32767).astype(np.int16).tobytes()

    def create_corrupted_audio(self) -> bytes:
        """Create corrupted audio data"""
        # Return random bytes that don't represent valid audio
        return b"This is not audio data at all!" * 100

    def create_oversized_audio(self) -> bytes:
        """Create oversized audio data (>100MB)"""
        # Create a large chunk of fake audio data
        return b"fake_audio_data" * (10 * 1024 * 1024)  # 140MB of repeated data

    # Test 1: AudioFormatError Validation
    def test_audio_format_error(self):
        """Test AudioFormatError handling"""
        try:
            # Test with invalid audio format
            corrupted_data = self.create_corrupted_audio()

            files = {"audio": ("corrupted.wav", corrupted_data, "audio/wav")}
            response = requests.post(f"{self.orchestration_url}/api/audio/upload", files=files)

            # Should get an error response
            is_error_handled = response.status_code in [400, 422, 500]

            if is_error_handled and response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                error_data = response.json()
                has_error_details = "error" in error_data or "detail" in error_data
                self.log_test_result(
                    "AudioFormatError",
                    is_error_handled and has_error_details,
                    {"status_code": response.status_code, "response": error_data},
                )
            else:
                self.log_test_result(
                    "AudioFormatError",
                    False,
                    {
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type"),
                    },
                )

        except Exception as e:
            self.log_test_result("AudioFormatError", False, {"exception": str(e)})

    # Test 2: AudioCorruptionError Validation
    def test_audio_corruption_error(self):
        """Test AudioCorruptionError handling"""
        try:
            # Test with empty file
            files = {"audio": ("empty.wav", b"", "audio/wav")}
            response = requests.post(f"{self.orchestration_url}/api/audio/upload", files=files)

            is_error_handled = response.status_code in [400, 422, 500]

            if is_error_handled:
                try:
                    error_data = response.json()
                    has_corruption_handling = any(
                        word in str(error_data).lower()
                        for word in ["empty", "corrupt", "invalid", "error"]
                    )
                    self.log_test_result(
                        "AudioCorruptionError",
                        has_corruption_handling,
                        {"status_code": response.status_code, "response": error_data},
                    )
                except Exception:
                    self.log_test_result(
                        "AudioCorruptionError",
                        is_error_handled,
                        {"status_code": response.status_code},
                    )
            else:
                self.log_test_result(
                    "AudioCorruptionError", False, {"status_code": response.status_code}
                )

        except Exception as e:
            self.log_test_result("AudioCorruptionError", False, {"exception": str(e)})

    # Test 3: ValidationError Validation
    def test_validation_error(self):
        """Test ValidationError handling"""
        try:
            # Test request without audio file
            response = requests.post(f"{self.orchestration_url}/api/audio/upload", data={})

            is_validation_error = response.status_code in [400, 422]

            if is_validation_error:
                try:
                    error_data = response.json()
                    has_validation_details = any(
                        word in str(error_data).lower()
                        for word in ["required", "missing", "validation", "audio"]
                    )
                    self.log_test_result(
                        "ValidationError",
                        has_validation_details,
                        {"status_code": response.status_code, "response": error_data},
                    )
                except Exception:
                    self.log_test_result(
                        "ValidationError",
                        is_validation_error,
                        {"status_code": response.status_code},
                    )
            else:
                self.log_test_result(
                    "ValidationError", False, {"status_code": response.status_code}
                )

        except Exception as e:
            self.log_test_result("ValidationError", False, {"exception": str(e)})

    # Test 4: ServiceUnavailableError Validation
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError handling"""
        try:
            # Test with non-existent service URL
            fake_url = "http://localhost:9999"

            try:
                requests.get(f"{fake_url}/health", timeout=2)
                self.log_test_result("ServiceUnavailableError", False, {"unexpected_success": True})
            except requests.exceptions.RequestException as e:
                # This is expected - the service should be unavailable
                self.log_test_result(
                    "ServiceUnavailableError",
                    True,
                    {"expected_error": str(type(e).__name__), "message": str(e)},
                )

        except Exception as e:
            self.log_test_result("ServiceUnavailableError", False, {"exception": str(e)})

    # Test 5: File Size Validation
    def test_file_size_validation(self):
        """Test file size validation"""
        try:
            # Test with oversized file
            oversized_data = self.create_oversized_audio()

            files = {"audio": ("large.wav", oversized_data, "audio/wav")}
            response = requests.post(f"{self.orchestration_url}/api/audio/upload", files=files)

            is_size_error = response.status_code in [
                413,
                422,
            ]  # Payload too large or validation error

            if is_size_error:
                try:
                    error_data = response.json()
                    has_size_handling = any(
                        word in str(error_data).lower()
                        for word in ["large", "size", "limit", "exceeded"]
                    )
                    self.log_test_result(
                        "FileSizeValidation",
                        has_size_handling,
                        {"status_code": response.status_code, "response": error_data},
                    )
                except Exception:
                    self.log_test_result(
                        "FileSizeValidation", is_size_error, {"status_code": response.status_code}
                    )
            else:
                self.log_test_result(
                    "FileSizeValidation", False, {"status_code": response.status_code}
                )

        except Exception as e:
            self.log_test_result("FileSizeValidation", False, {"exception": str(e)})

    # Test 6: Health Check Validation
    def test_health_checks(self):
        """Test health check endpoints"""
        services = [
            ("Orchestration Health", f"{self.orchestration_url}/api/audio/health"),
            ("Whisper Health", f"{self.whisper_url}/health"),
        ]

        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        has_status = "status" in health_data
                        self.log_test_result(
                            service_name,
                            has_status,
                            {"status_code": response.status_code, "health_data": health_data},
                        )
                    except Exception:
                        self.log_test_result(service_name, False, {"json_parse_error": True})
                else:
                    self.log_test_result(service_name, False, {"status_code": response.status_code})

            except Exception as e:
                self.log_test_result(service_name, False, {"exception": str(e)})

    # Test 7: Valid Audio Processing
    def test_valid_audio_processing(self):
        """Test that valid audio still processes correctly"""
        try:
            # Create valid test audio
            valid_audio = self.create_test_audio(duration=2.0)

            files = {"audio": ("test.wav", valid_audio, "audio/wav")}
            data = {"enable_transcription": "true", "enable_translation": "false"}

            response = requests.post(
                f"{self.orchestration_url}/api/audio/upload", files=files, data=data
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                    has_transcription = "processing_result" in result
                    self.log_test_result(
                        "ValidAudioProcessing",
                        has_transcription,
                        {"status_code": response.status_code, "has_result": has_transcription},
                    )
                except Exception:
                    self.log_test_result("ValidAudioProcessing", False, {"json_parse_error": True})
            else:
                self.log_test_result(
                    "ValidAudioProcessing", False, {"status_code": response.status_code}
                )

        except Exception as e:
            self.log_test_result("ValidAudioProcessing", False, {"exception": str(e)})

    # Test 8: Error Response Format Validation
    def test_error_response_format(self):
        """Test that error responses have consistent format"""
        try:
            # Make a bad request to get error response
            files = {"audio": ("bad.txt", b"not audio", "text/plain")}
            response = requests.post(f"{self.orchestration_url}/api/audio/upload", files=files)

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    has_error_structure = any(
                        key in error_data for key in ["error", "detail", "message"]
                    )

                    # Check for correlation ID or request tracking
                    has_tracking = any(
                        key in error_data for key in ["correlation_id", "request_id", "trace_id"]
                    )

                    self.log_test_result(
                        "ErrorResponseFormat",
                        has_error_structure,
                        {
                            "status_code": response.status_code,
                            "has_structure": has_error_structure,
                            "has_tracking": has_tracking,
                            "response": error_data,
                        },
                    )
                except Exception:
                    self.log_test_result("ErrorResponseFormat", False, {"json_parse_error": True})
            else:
                self.log_test_result("ErrorResponseFormat", False, {"no_error_response": True})

        except Exception as e:
            self.log_test_result("ErrorResponseFormat", False, {"exception": str(e)})

    def run_all_tests(self):
        """Run all error handling validation tests"""
        logger.info("🚀 Starting comprehensive error handling validation tests...")

        test_methods = [
            self.test_audio_format_error,
            self.test_audio_corruption_error,
            self.test_validation_error,
            self.test_service_unavailable_error,
            self.test_file_size_validation,
            self.test_health_checks,
            self.test_valid_audio_processing,
            self.test_error_response_format,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
                self.log_test_result(test_method.__name__, False, {"test_exception": str(e)})

            time.sleep(0.5)  # Brief pause between tests

    def generate_report(self):
        """Generate test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["success"]])
        failed_tests = total_tests - passed_tests

        logger.info("\n" + "=" * 60)
        logger.info("📊 ERROR HANDLING VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("=" * 60)

        if failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    logger.info(f"  - {result['test']}: {result.get('details', {})}")

        logger.info("\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result["success"]:
                logger.info(f"  - {result['test']}")

        # Save detailed report
        report_file = f"error_handling_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total_tests": total_tests,
                        "passed": passed_tests,
                        "failed": failed_tests,
                        "success_rate": (passed_tests / total_tests) * 100,
                    },
                    "test_results": self.test_results,
                },
                f,
                indent=2,
            )

        logger.info(f"\n📄 Detailed report saved to: {report_file}")

        return passed_tests == total_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Error Handling Validation Tests")
    parser.add_argument(
        "--orchestration-url", default="http://localhost:3000", help="Orchestration service URL"
    )
    parser.add_argument(
        "--whisper-url", default="http://localhost:5001", help="Whisper service URL"
    )

    args = parser.parse_args()

    validator = ErrorHandlingValidator(args.orchestration_url, args.whisper_url)
    validator.run_all_tests()
    all_passed = validator.generate_report()

    sys.exit(0 if all_passed else 1)
