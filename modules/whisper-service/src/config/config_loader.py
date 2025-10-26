#!/usr/bin/env python3
"""
Configuration Loader

Loads Whisper service configuration from environment variables and config files.
Extracted from whisper_service.py for better modularity and testability.
"""

import os
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def load_whisper_config(config_file_path: str = None) -> Dict:
    """
    Load Whisper service configuration from environment variables and config files.

    Args:
        config_file_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary containing configuration settings
    """
    # Default config from environment variables
    config = {
        # Model settings - use local .models directory first
        "models_dir": os.getenv("WHISPER_MODELS_DIR",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models"))
            else os.path.expanduser("~/.whisper/models")),
        "default_model": os.getenv("WHISPER_DEFAULT_MODEL", "large-v3-turbo"),

        # Audio settings - optimized for reduced duplicates
        "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
        "buffer_duration": float(os.getenv("BUFFER_DURATION", "4.0")),  # Reduced from 6.0
        "inference_interval": float(os.getenv("INFERENCE_INTERVAL", "3.0")),
        "overlap_duration": float(os.getenv("OVERLAP_DURATION", "0.2")),  # Minimal overlap
        "enable_vad": os.getenv("ENABLE_VAD", "true").lower() == "true",

        # Device settings
        "device": os.getenv("OPENVINO_DEVICE"),

        # Session settings
        "session_dir": os.getenv("SESSION_DIR"),

        # Performance settings
        "min_inference_interval": float(os.getenv("MIN_INFERENCE_INTERVAL", "0.2")),
        "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),

        # Orchestration integration settings
        "orchestration_mode": os.getenv("ORCHESTRATION_MODE", "false").lower() == "true",
        "orchestration_endpoint": os.getenv("ORCHESTRATION_ENDPOINT", "http://localhost:3000/api/audio"),
    }

    # Determine config file path
    if config_file_path is None:
        config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

    # Load from config file if exists
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    return config
