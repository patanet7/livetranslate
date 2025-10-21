#!/usr/bin/env python3
"""
Create warmup audio file for Whisper service

Creates a 1-second silent WAV file at 16kHz mono
This is used to warm up the model on startup to avoid cold start delays
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# 1 second of silence at 16kHz
SAMPLE_RATE = 16000
DURATION = 1.0

# Create silent audio
audio = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)

# Save as WAV file
output_path = Path(__file__).parent / "warmup.wav"
sf.write(output_path, audio, SAMPLE_RATE)

print(f"✅ Created warmup audio file: {output_path}")
print(f"   Sample rate: {SAMPLE_RATE}Hz")
print(f"   Duration: {DURATION}s")
print(f"   Samples: {len(audio)}")
print(f"   File size: {output_path.stat().st_size} bytes")
