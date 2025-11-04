#!/usr/bin/env python3
"""
Quick debug script to identify where test hangs
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("1. Imports starting...")
from session_restart import SessionRestartTranscriber
print("2. SessionRestartTranscriber imported")

print("3. Creating transcriber...")
models_dir = Path.home() / ".whisper" / "models"
model_path = str(models_dir / "large-v3-turbo.pt")

transcriber = SessionRestartTranscriber(
    model_path=model_path,
    models_dir=str(models_dir),
    target_languages=['en', 'zh'],
    online_chunk_size=1.2,
    vad_threshold=0.5,
    sampling_rate=16000,
    lid_hop_ms=100,
    confidence_margin=0.2,
    min_dwell_frames=6,
    min_dwell_ms=250.0
)
print("4. Transcriber created successfully!")

print("5. All steps completed - test would start here")
