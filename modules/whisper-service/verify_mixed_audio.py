#!/usr/bin/env python3
"""Quick verification: What's actually in test_clean_mixed_en_zh.wav?"""
import soundfile as sf
import numpy as np
from pathlib import Path

# Load the mixed audio file
audio_path = Path("tests/fixtures/audio/test_clean_mixed_en_zh.wav")
audio, sr = sf.read(str(audio_path))

print(f"Total duration: {len(audio)/sr:.2f}s @ {sr}Hz")
print(f"Total samples: {len(audio):,}")

# Check RMS levels in different sections
def check_section(start_sec, end_sec, label):
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    section = audio[start_idx:end_idx]
    rms = np.sqrt(np.mean(section**2))
    max_amp = np.max(np.abs(section))
    print(f"{label:20s} ({start_sec:5.1f}s-{end_sec:5.1f}s): RMS={rms:.4f}, Max={max_amp:.4f}")
    return rms, max_amp

print("\nSection analysis:")
check_section(0, 11, "JFK (English)")
check_section(11, 20, "Chinese segment 1")
check_section(20, 30, "Chinese segment 1")
check_section(30, 40, "Chinese segment 2")
check_section(40, 53, "Chinese segment 2")
check_section(53, 67, "Chinese segment 3")

# Check if sections after 11s are silent
rms_chinese, _ = check_section(15, 25, "Test Chinese section")
if rms_chinese < 0.001:
    print("\n⚠️  WARNING: Chinese section appears to be SILENT!")
    print("The mixed audio file might only contain the English (JFK) portion.")
else:
    print(f"\n✅ Chinese section has audio (RMS={rms_chinese:.4f})")
