#!/usr/bin/env python3
"""
Dependency validation utility for orchestration service.

Validates that all required and optional dependencies are available
and provides informative warnings about missing optional features.
"""

import logging
import sys

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Validates dependencies and provides feature availability information."""

    def __init__(self):
        self.missing_required: list[str] = []
        self.missing_optional: dict[str, str] = {}
        self.available_features: dict[str, bool] = {}

    def check_required_dependencies(self) -> bool:
        """Check all required dependencies are available."""
        required_deps = [
            ("fastapi", "FastAPI web framework"),
            ("uvicorn", "ASGI server"),
            ("httpx", "HTTP client for service communication"),
            ("pydantic", "Data validation"),
            ("numpy", "Audio processing"),
            ("asyncpg", "PostgreSQL async driver (optional in dev)"),
        ]

        all_available = True
        for dep, description in required_deps:
            try:
                __import__(dep)
                logger.debug(f"✅ {dep}: {description}")
            except ImportError:
                if dep == "asyncpg":
                    # asyncpg is optional in development
                    logger.warning(f"⚠️  {dep}: {description} - optional for development")
                    self.missing_optional[dep] = description
                else:
                    logger.error(f"❌ {dep}: {description} - REQUIRED")
                    self.missing_required.append(dep)
                    all_available = False

        return all_available

    def check_optional_dependencies(self) -> dict[str, bool]:
        """Check optional dependencies and return feature availability."""
        optional_deps = [
            (
                "h2",
                "HTTP/2 support for improved service communication",
                "http2_support",
            ),
            ("scipy", "Advanced audio processing algorithms", "advanced_audio"),
            ("librosa", "Audio analysis and feature extraction", "audio_analysis"),
            ("soundfile", "Audio file I/O operations", "audio_io"),
            ("psutil", "System monitoring and resource tracking", "system_monitoring"),
        ]

        for dep, description, feature in optional_deps:
            try:
                __import__(dep)
                logger.info(f"✅ {dep}: {description}")
                self.available_features[feature] = True
            except ImportError:
                logger.warning(f"⚠️  {dep}: {description} - feature disabled")
                self.missing_optional[dep] = description
                self.available_features[feature] = False

        return self.available_features

    def generate_install_commands(self) -> list[str]:
        """Generate pip install commands for missing dependencies."""
        commands = []

        if self.missing_required:
            commands.append(f"pip install {' '.join(self.missing_required)}")

        # Group optional dependencies by common install patterns
        http2_deps = [dep for dep in self.missing_optional if dep == "h2"]
        audio_deps = [
            dep for dep in self.missing_optional if dep in ["scipy", "librosa", "soundfile"]
        ]
        system_deps = [dep for dep in self.missing_optional if dep in ["psutil"]]

        if http2_deps:
            commands.append("pip install 'httpx[http2]'  # For HTTP/2 support")
        if audio_deps:
            commands.append(f"pip install {' '.join(audio_deps)}  # For advanced audio processing")
        if system_deps:
            commands.append(f"pip install {' '.join(system_deps)}  # For system monitoring")

        return commands

    def print_summary(self):
        """Print a comprehensive dependency summary."""
        logger.info("🔍 Dependency Validation Summary")
        logger.info("=" * 50)

        if not self.missing_required:
            logger.info("✅ All required dependencies available")
        else:
            logger.error("❌ Missing required dependencies:")
            for dep in self.missing_required:
                logger.error(f"   - {dep}")

        if self.available_features:
            logger.info("📋 Feature Availability:")
            for feature, available in self.available_features.items():
                status = "✅ Enabled" if available else "⚠️  Disabled"
                logger.info(f"   - {feature}: {status}")

        if self.missing_optional:
            logger.info("📦 Optional Dependencies:")
            for dep, desc in self.missing_optional.items():
                logger.info(f"   - {dep}: {desc}")

        install_commands = self.generate_install_commands()
        if install_commands:
            logger.info("🔧 To install missing dependencies:")
            for cmd in install_commands:
                logger.info(f"   {cmd}")

        logger.info("=" * 50)

    def validate_environment(self) -> tuple[bool, dict[str, bool]]:
        """
        Complete environment validation.

        Returns:
            Tuple of (all_required_available, feature_availability_dict)
        """
        logger.info("🔍 Validating orchestration service environment...")

        required_ok = self.check_required_dependencies()
        features = self.check_optional_dependencies()

        self.print_summary()

        if not required_ok:
            logger.error("❌ Cannot start service due to missing required dependencies")
            sys.exit(1)

        return required_ok, features


def validate_startup_environment() -> dict[str, bool]:
    """
    Validate environment at startup and return feature availability.

    This function should be called during application startup to ensure
    all dependencies are available and to determine which features can be enabled.
    """
    checker = DependencyChecker()
    required_ok, features = checker.validate_environment()

    if required_ok:
        logger.info("🚀 Environment validation successful - service can start")

    return features


if __name__ == "__main__":
    # Allow running this module directly for dependency checking
    validate_startup_environment()
