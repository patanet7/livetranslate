#!/usr/bin/env python3
"""
Environment Variable Validation Script for LiveTranslate

This script validates that required environment variables are present and properly
formatted before deployment or during pre-commit hooks.

Usage:
    python scripts/validate_env.py                    # Validate all services
    python scripts/validate_env.py --service=orchestration  # Validate specific service
    python scripts/validate_env.py --strict           # Fail on warnings
    python scripts/validate_env.py --env-file=.env    # Check specific .env file

Exit codes:
    0 - All validations passed
    1 - Required variables missing or invalid
    2 - Warnings present (only with --strict)
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar


class ValidationLevel(Enum):
    """Validation severity levels."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ValidationType(Enum):
    """Types of validation checks."""

    PRESENCE = "presence"
    FORMAT = "format"
    RANGE = "range"
    ENUM = "enum"
    URL = "url"
    PORT = "port"
    PATH = "path"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    variable: str
    valid: bool
    level: ValidationLevel
    message: str
    value: str | None = None


@dataclass
class EnvVarSpec:
    """Specification for an environment variable."""

    name: str
    level: ValidationLevel
    validation_type: ValidationType
    description: str
    default: str | None = None
    pattern: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    custom_validator: Callable[[str], bool] | None = None


class EnvironmentValidator:
    """Validates environment variables against specifications."""

    # URL pattern for validation
    URL_PATTERN = re.compile(
        r"^(https?|wss?|redis|postgresql|postgres)://"
        r"[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?"
        r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*"
        r"(:\d{1,5})?"
        r"(/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?$"
    )

    # Common environment variable specifications by service
    ORCHESTRATION_SPECS: ClassVar[list[EnvVarSpec]] = [
        # Server Configuration
        EnvVarSpec(
            name="HOST",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.FORMAT,
            description="Server host address",
            default="0.0.0.0",
            pattern=r"^(\d{1,3}\.){3}\d{1,3}$|^localhost$|^0\.0\.0\.0$",
        ),
        EnvVarSpec(
            name="PORT",
            level=ValidationLevel.RECOMMENDED,
            validation_type=ValidationType.PORT,
            description="Server port",
            default="3000",
        ),
        EnvVarSpec(
            name="WORKERS",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.INTEGER,
            description="Number of worker processes",
            default="4",
            min_value=1,
            max_value=32,
        ),
        EnvVarSpec(
            name="DEBUG",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Debug mode flag",
            default="false",
        ),
        EnvVarSpec(
            name="ENVIRONMENT",
            level=ValidationLevel.RECOMMENDED,
            validation_type=ValidationType.ENUM,
            description="Deployment environment",
            default="development",
            allowed_values=["development", "staging", "production", "testing"],
        ),
        # Database Configuration
        EnvVarSpec(
            name="DATABASE_URL",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.URL,
            description="PostgreSQL connection URL",
        ),
        EnvVarSpec(
            name="DATABASE_POOL_SIZE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.INTEGER,
            description="Database connection pool size",
            default="10",
            min_value=1,
            max_value=100,
        ),
        # Redis Configuration
        EnvVarSpec(
            name="REDIS_URL",
            level=ValidationLevel.RECOMMENDED,
            validation_type=ValidationType.URL,
            description="Redis connection URL",
            default="redis://localhost:6379/0",
        ),
        # Service URLs
        EnvVarSpec(
            name="WHISPER_SERVICE_URL",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.URL,
            description="Whisper service endpoint URL",
        ),
        EnvVarSpec(
            name="TRANSLATION_SERVICE_URL",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.URL,
            description="Translation service endpoint URL",
        ),
        # Security Configuration
        EnvVarSpec(
            name="JWT_SECRET",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.FORMAT,
            description="JWT signing secret (min 32 chars)",
            pattern=r"^.{32,}$",
        ),
        EnvVarSpec(
            name="JWT_ALGORITHM",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="JWT signing algorithm",
            default="HS256",
            allowed_values=["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"],
        ),
        EnvVarSpec(
            name="CORS_ORIGINS",
            level=ValidationLevel.RECOMMENDED,
            validation_type=ValidationType.FORMAT,
            description="Allowed CORS origins (comma-separated)",
            pattern=r"^(https?://[^\s,]+)(,\s*https?://[^\s,]+)*$|^\*$",
        ),
        # Logging Configuration
        EnvVarSpec(
            name="LOG_LEVEL",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="Logging level",
            default="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
        # Monitoring
        EnvVarSpec(
            name="ENABLE_METRICS",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable Prometheus metrics",
            default="true",
        ),
    ]

    WHISPER_SPECS: ClassVar[list[EnvVarSpec]] = [
        EnvVarSpec(
            name="WHISPER_MODEL_PATH",
            level=ValidationLevel.RECOMMENDED,
            validation_type=ValidationType.PATH,
            description="Path to Whisper model file",
        ),
        EnvVarSpec(
            name="WHISPER_DECODER_TYPE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="Whisper decoder type",
            default="greedy",
            allowed_values=["greedy", "beam"],
        ),
        EnvVarSpec(
            name="WHISPER_SAMPLING_RATE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="Audio sampling rate",
            default="16000",
            allowed_values=["8000", "16000"],
        ),
        EnvVarSpec(
            name="VAD_THRESHOLD",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.FLOAT,
            description="VAD speech probability threshold",
            default="0.5",
            min_value=0.0,
            max_value=1.0,
        ),
        EnvVarSpec(
            name="LOG_LEVEL",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="Logging level",
            default="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
    ]

    TRANSLATION_SPECS: ClassVar[list[EnvVarSpec]] = [
        EnvVarSpec(
            name="PORT",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.PORT,
            description="Translation service port",
            default="5003",
        ),
        EnvVarSpec(
            name="OLLAMA_ENABLE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable Ollama backend",
            default="true",
        ),
        EnvVarSpec(
            name="OLLAMA_BASE_URL",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.URL,
            description="Ollama API base URL",
            default="http://localhost:11434/v1",
        ),
        EnvVarSpec(
            name="GROQ_ENABLE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable Groq backend",
            default="false",
        ),
        EnvVarSpec(
            name="GROQ_API_KEY",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.FORMAT,
            description="Groq API key",
            pattern=r"^gsk_[a-zA-Z0-9]{50,}$|^.{20,}$",
        ),
        EnvVarSpec(
            name="OPENAI_ENABLE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable OpenAI backend",
            default="false",
        ),
        EnvVarSpec(
            name="OPENAI_API_KEY",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.FORMAT,
            description="OpenAI API key",
            pattern=r"^sk-[a-zA-Z0-9]{40,}$|^.{20,}$",
        ),
        EnvVarSpec(
            name="GPU_ENABLE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable GPU acceleration",
            default="true",
        ),
        EnvVarSpec(
            name="MAX_BATCH_SIZE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.INTEGER,
            description="Maximum batch size for translation",
            default="32",
            min_value=1,
            max_value=256,
        ),
        EnvVarSpec(
            name="TEMPERATURE",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.FLOAT,
            description="LLM temperature setting",
            default="0.3",
            min_value=0.0,
            max_value=2.0,
        ),
        EnvVarSpec(
            name="LOG_LEVEL",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.ENUM,
            description="Logging level",
            default="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
    ]

    FRONTEND_SPECS: ClassVar[list[EnvVarSpec]] = [
        EnvVarSpec(
            name="VITE_API_BASE_URL",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.URL,
            description="Backend API base URL",
        ),
        EnvVarSpec(
            name="VITE_WS_BASE_URL",
            level=ValidationLevel.REQUIRED,
            validation_type=ValidationType.URL,
            description="WebSocket base URL",
        ),
        EnvVarSpec(
            name="VITE_ENABLE_DEBUG",
            level=ValidationLevel.OPTIONAL,
            validation_type=ValidationType.BOOLEAN,
            description="Enable debug mode in frontend",
            default="false",
        ),
    ]

    def __init__(self, env_file: str | None = None):
        """Initialize validator with optional .env file path."""
        self.env_file = env_file
        self.env_vars: dict[str, str] = {}
        self._load_environment()

    def _load_environment(self) -> None:
        """Load environment variables from file and/or system."""
        # Start with system environment
        self.env_vars = dict(os.environ)

        # Override with .env file if specified
        if self.env_file:
            env_path = Path(self.env_file)
            if env_path.exists():
                self._parse_env_file(env_path)

    def _parse_env_file(self, path: Path) -> None:
        """Parse a .env file and add variables to env_vars."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    self.env_vars[key] = value

    def validate_spec(self, spec: EnvVarSpec) -> ValidationResult:
        """Validate a single environment variable against its specification."""
        value = self.env_vars.get(spec.name)

        # Check presence
        if value is None or value == "":
            if spec.level == ValidationLevel.REQUIRED:
                return ValidationResult(
                    variable=spec.name,
                    valid=False,
                    level=spec.level,
                    message=f"Required variable '{spec.name}' is not set. {spec.description}",
                )
            elif spec.level == ValidationLevel.RECOMMENDED:
                return ValidationResult(
                    variable=spec.name,
                    valid=True,
                    level=spec.level,
                    message=f"Recommended variable '{spec.name}' is not set. "
                    f"Using default: {spec.default}",
                    value=spec.default,
                )
            else:
                return ValidationResult(
                    variable=spec.name,
                    valid=True,
                    level=spec.level,
                    message=f"Optional variable '{spec.name}' using default: {spec.default}",
                    value=spec.default,
                )

        # Validate based on type
        validation_method = getattr(self, f"_validate_{spec.validation_type.value}", None)
        if validation_method:
            valid, msg = validation_method(value, spec)
            return ValidationResult(
                variable=spec.name,
                valid=valid,
                level=spec.level,
                message=msg if not valid else f"'{spec.name}' is valid",
                value=value,
            )

        return ValidationResult(
            variable=spec.name,
            valid=True,
            level=spec.level,
            message=f"'{spec.name}' is set (no specific validation)",
            value=value,
        )

    def _validate_url(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate URL format."""
        if self.URL_PATTERN.match(value):
            return True, ""
        return False, f"'{spec.name}' is not a valid URL: {value}"

    def _validate_port(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate port number."""
        try:
            port = int(value)
            if 1 <= port <= 65535:
                return True, ""
            return False, f"'{spec.name}' port must be between 1 and 65535: {value}"
        except ValueError:
            return False, f"'{spec.name}' must be a valid port number: {value}"

    def _validate_boolean(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate boolean value."""
        if value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return True, ""
        return False, f"'{spec.name}' must be a boolean value: {value}"

    def _validate_integer(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate integer value with optional range."""
        try:
            int_val = int(value)
            if spec.min_value is not None and int_val < spec.min_value:
                return (
                    False,
                    f"'{spec.name}' must be >= {spec.min_value}: {value}",
                )
            if spec.max_value is not None and int_val > spec.max_value:
                return (
                    False,
                    f"'{spec.name}' must be <= {spec.max_value}: {value}",
                )
            return True, ""
        except ValueError:
            return False, f"'{spec.name}' must be an integer: {value}"

    def _validate_float(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate float value with optional range."""
        try:
            float_val = float(value)
            if spec.min_value is not None and float_val < spec.min_value:
                return (
                    False,
                    f"'{spec.name}' must be >= {spec.min_value}: {value}",
                )
            if spec.max_value is not None and float_val > spec.max_value:
                return (
                    False,
                    f"'{spec.name}' must be <= {spec.max_value}: {value}",
                )
            return True, ""
        except ValueError:
            return False, f"'{spec.name}' must be a float: {value}"

    def _validate_enum(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate value is in allowed list."""
        if spec.allowed_values and value not in spec.allowed_values:
            return (
                False,
                f"'{spec.name}' must be one of {spec.allowed_values}: {value}",
            )
        return True, ""

    def _validate_format(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate value matches regex pattern."""
        if spec.pattern:
            if re.match(spec.pattern, value):
                return True, ""
            return (
                False,
                f"'{spec.name}' does not match required format: {value}",
            )
        return True, ""

    def _validate_path(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate file/directory path exists."""
        path = Path(value)
        if path.exists():
            return True, ""
        return (
            False,
            f"'{spec.name}' path does not exist: {value}",
        )

    def _validate_presence(self, value: str, spec: EnvVarSpec) -> tuple[bool, str]:
        """Validate value is present (non-empty)."""
        if value and value.strip():
            return True, ""
        return False, f"'{spec.name}' cannot be empty"

    def validate_service(self, service: str) -> list[ValidationResult]:
        """Validate all environment variables for a specific service."""
        specs_map = {
            "orchestration": self.ORCHESTRATION_SPECS,
            "whisper": self.WHISPER_SPECS,
            "translation": self.TRANSLATION_SPECS,
            "frontend": self.FRONTEND_SPECS,
        }

        specs = specs_map.get(service.lower(), [])
        if not specs:
            return []

        return [self.validate_spec(spec) for spec in specs]

    def validate_all(self) -> dict[str, list[ValidationResult]]:
        """Validate environment variables for all services."""
        return {
            "orchestration": self.validate_service("orchestration"),
            "whisper": self.validate_service("whisper"),
            "translation": self.validate_service("translation"),
            "frontend": self.validate_service("frontend"),
        }


class ValidationReporter:
    """Reports validation results in various formats."""

    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, use_colors: bool = True):
        """Initialize reporter with color preference."""
        self.use_colors = use_colors and sys.stdout.isatty()

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def report_results(
        self, results: dict[str, list[ValidationResult]], verbose: bool = False
    ) -> tuple[int, int, int]:
        """
        Report validation results and return counts.

        Returns:
            Tuple of (errors, warnings, passed)
        """
        total_errors = 0
        total_warnings = 0
        total_passed = 0

        print(self._colorize("\n=== LiveTranslate Environment Validation ===\n", self.BOLD))

        for service, service_results in results.items():
            if not service_results:
                continue

            errors = [r for r in service_results if not r.valid]
            warnings = [
                r
                for r in service_results
                if r.valid and r.level == ValidationLevel.RECOMMENDED and r.value is None
            ]
            passed = [r for r in service_results if r.valid and r not in warnings]

            total_errors += len(errors)
            total_warnings += len(warnings)
            total_passed += len(passed)

            # Service header
            status_icon = (
                self._colorize("[FAIL]", self.RED)
                if errors
                else self._colorize("[PASS]", self.GREEN)
            )
            print(f"{status_icon} {self._colorize(service.upper(), self.BOLD)}")

            # Report errors
            for result in errors:
                print(f"  {self._colorize('ERROR', self.RED)}: {result.message}")

            # Report warnings
            for result in warnings:
                print(f"  {self._colorize('WARN', self.YELLOW)}: {result.message}")

            # Report passed (verbose only)
            if verbose:
                for result in passed:
                    print(f"  {self._colorize('OK', self.GREEN)}: {result.message}")

            print()

        # Summary
        print(self._colorize("=== Summary ===", self.BOLD))
        print(f"  {self._colorize('Errors:', self.RED)} {total_errors}")
        print(f"  {self._colorize('Warnings:', self.YELLOW)} {total_warnings}")
        print(f"  {self._colorize('Passed:', self.GREEN)} {total_passed}")

        return total_errors, total_warnings, total_passed


def main() -> int:
    """Main entry point for environment validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate environment variables for LiveTranslate services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--service",
        "-s",
        choices=["orchestration", "whisper", "translation", "frontend", "all"],
        default="all",
        help="Service to validate (default: all)",
    )
    parser.add_argument(
        "--env-file",
        "-e",
        type=str,
        help="Path to .env file to validate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (recommended variables not set)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all validation results including passed",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show errors (no summary)",
    )

    args = parser.parse_args()

    # Initialize validator
    validator = EnvironmentValidator(env_file=args.env_file)

    # Run validation
    if args.service == "all":
        results = validator.validate_all()
    else:
        results = {args.service: validator.validate_service(args.service)}

    # Report results
    if not args.quiet:
        reporter = ValidationReporter(use_colors=not args.no_color)
        errors, warnings, _passed = reporter.report_results(results, verbose=args.verbose)
    else:
        # Quiet mode - just count
        errors = sum(1 for svc in results.values() for r in svc if not r.valid)
        warnings = sum(
            1
            for svc in results.values()
            for r in svc
            if r.valid and r.level == ValidationLevel.RECOMMENDED and r.value is None
        )

    # Determine exit code
    if errors > 0:
        return 1
    if args.strict and warnings > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
