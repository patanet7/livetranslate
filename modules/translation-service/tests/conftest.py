"""
Test Configuration - Config-Driven Test Settings

All test configuration is centralized here and driven by:
1. Environment variables (highest priority)
2. .env.test file (if exists)
3. Sensible defaults (localhost)

To customize for your environment:
- Set environment variables: OLLAMA_BASE_URL, TRANSLATION_SERVICE_URL, etc.
- Or create a .env.test file in the tests/ directory
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load test-specific .env file if it exists
test_env_file = Path(__file__).parent / ".env.test"
if test_env_file.exists():
    load_dotenv(test_env_file)


class TestConfig:
    """Centralized test configuration - all URLs and settings in one place"""

    def __init__(self) -> None:
        # Service URLs - override via environment variables
        self.translation_service_url = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.whisper_service_url = os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001")
        self.orchestration_service_url = os.getenv(
            "ORCHESTRATION_SERVICE_URL", "http://localhost:3000"
        )

        # Timeouts (in seconds)
        self.default_timeout = int(os.getenv("TEST_DEFAULT_TIMEOUT", "30"))
        self.long_timeout = int(os.getenv("TEST_LONG_TIMEOUT", "120"))

        # Test output directory
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
test_config = TestConfig()


@pytest.fixture(scope="session")
def config() -> TestConfig:
    """Pytest fixture providing test configuration"""
    return test_config


@pytest.fixture(scope="session")
def translation_service_url() -> str:
    """URL for the translation service"""
    return test_config.translation_service_url


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    """URL for Ollama service"""
    return test_config.ollama_base_url


@pytest.fixture(scope="session")
def whisper_service_url() -> str:
    """URL for Whisper service"""
    return test_config.whisper_service_url


@pytest.fixture(scope="session")
def orchestration_service_url() -> str:
    """URL for Orchestration service"""
    return test_config.orchestration_service_url


def get_test_output_path(test_name: str) -> Path:
    """
    Get a timestamped output path for test results.

    Usage:
        output_path = get_test_output_path("model_switching")
        # Returns: tests/output/20260119_model_switching_results.log
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}_{test_name}_results.log"
    return test_config.output_dir / filename


# Export for easy import in tests
__all__ = [
    "TestConfig",
    "get_test_output_path",
    "test_config",
]
