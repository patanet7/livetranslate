#!/usr/bin/env python3
"""
Start Whisper Service with NPU acceleration
"""

import os
import sys

# Force NPU usage
os.environ["OPENVINO_DEVICE"] = "NPU"
os.environ["WHISPER_DEFAULT_MODEL"] = "whisper-base"
os.environ["LOG_LEVEL"] = "INFO"

# Set models directory to use local OpenVINO models
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.environ["WHISPER_MODELS_DIR"] = models_dir

# Import and run main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from main import main

if __name__ == "__main__":
    print("🚀 Starting Whisper Service with NPU acceleration...")
    print("   Device: NPU (Intel® AI Boost)")
    print("   Default Model: whisper-base")
    print(f"   Models Directory: {models_dir}")
    print("")
    main()