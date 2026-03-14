#!/usr/bin/env python3
"""
Configuration Validation Script

Validates configuration before starting services.
Helps catch configuration errors early in development.

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --env-file .env.production
    python scripts/validate_config.py --model-path /path/to/model.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from service_config import LIDConfig, SessionConfig, VADConfig, WhisperConfig


def load_env_file(env_file: str):
    """Load environment variables from file"""
    if not os.path.exists(env_file):
        print(f"❌ Environment file not found: {env_file}")
        return False

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

    print(f"✅ Loaded environment from: {env_file}")
    return True


def validate_vad_config():
    """Validate VAD configuration"""
    print("\n=== Validating VAD Configuration ===")

    try:
        config = VADConfig.from_env()
        print("✅ VAD Config Valid")
        print(f"   - Threshold: {config.threshold}")
        print(f"   - Sampling Rate: {config.sampling_rate} Hz")
        print(f"   - Min Silence: {config.min_silence_duration_ms} ms")
        print(f"   - Silence Threshold: {config.silence_threshold_chunks} chunks")
        return True
    except Exception as e:
        print(f"❌ VAD Config Invalid: {e}")
        return False


def validate_lid_config():
    """Validate LID configuration"""
    print("\n=== Validating LID Configuration ===")

    try:
        config = LIDConfig.from_env()
        print("✅ LID Config Valid")
        print(f"   - LID Hop: {config.lid_hop_ms} ms")
        print(f"   - Confidence Margin: {config.confidence_margin}")
        print(f"   - Min Dwell: {config.min_dwell_ms} ms ({config.min_dwell_frames} frames)")
        print(f"   - Smoothing: {config.smoothing_enabled}")
        return True
    except Exception as e:
        print(f"❌ LID Config Invalid: {e}")
        return False


def validate_whisper_config(model_path: str):
    """Validate Whisper configuration"""
    print("\n=== Validating Whisper Configuration ===")

    try:
        config = WhisperConfig.from_env(model_path)
        print("✅ Whisper Config Valid")
        print(f"   - Model Path: {config.model_path}")
        print(f"   - Models Dir: {config.models_dir}")
        print(f"   - Decoder: {config.decoder_type}")
        print(f"   - Beam Size: {config.beam_size}")
        print(f"   - Chunk Size: {config.online_chunk_size}s")
        print(f"   - Languages: {', '.join(config.target_languages)}")

        # Check if model file exists
        if not os.path.exists(config.model_path):
            print(f"⚠️  Warning: Model file not found: {config.model_path}")
            print("   (This is OK if model will be downloaded)")

        return True
    except Exception as e:
        print(f"❌ Whisper Config Invalid: {e}")
        return False


def validate_session_config(model_path: str):
    """Validate complete session configuration"""
    print("\n=== Validating Session Configuration ===")

    try:
        config = SessionConfig.from_env(model_path)
        print("✅ Session Config Valid")
        print(f"   - Log Level: {config.log_level}")
        print(f"   - Performance Logging: {config.enable_performance_logging}")
        print(f"   - Debug Audio Stats: {config.enable_debug_audio_stats}")
        return True
    except Exception as e:
        print(f"❌ Session Config Invalid: {e}")
        return False


def check_environment_variables():
    """Check for common environment variables"""
    print("\n=== Environment Variables ===")

    env_vars = {
        "VAD_THRESHOLD": "VAD speech threshold",
        "LID_CONFIDENCE_MARGIN": "LID confidence margin",
        "WHISPER_DECODER_TYPE": "Whisper decoder type",
        "WHISPER_LANGUAGES": "Target languages",
        "LOG_LEVEL": "Logging level",
    }

    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value} ({description})")
        else:
            print(f"[i] {var} not set (using default) - {description}")


def main():
    parser = argparse.ArgumentParser(description="Validate Whisper Service Configuration")
    parser.add_argument("--env-file", type=str, help="Path to .env file to load")
    parser.add_argument(
        "--model-path", type=str, default="/path/to/model.pt", help="Path to Whisper model file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Whisper Service Configuration Validator")
    print("=" * 60)

    # Load environment file if specified
    if args.env_file and not load_env_file(args.env_file):
        sys.exit(1)

    # Check environment variables
    check_environment_variables()

    # Validate each configuration component
    all_valid = True

    all_valid &= validate_vad_config()
    all_valid &= validate_lid_config()
    all_valid &= validate_whisper_config(args.model_path)
    all_valid &= validate_session_config(args.model_path)

    # Final result
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL CONFIGURATIONS VALID")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ CONFIGURATION VALIDATION FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
