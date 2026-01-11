#!/usr/bin/env python3
"""
Create mixed language audio file for testing code-switching through orchestration.
Pattern: JFK (EN) → Chinese → JFK (EN) → Chinese → JFK (EN)
"""

import librosa
import soundfile as sf
import numpy as np

# Load audio files
jfk_audio, jfk_sr = librosa.load(
    "../whisper-service/tests/fixtures/audio/jfk.wav", sr=16000, mono=True
)
chinese_audio, cn_sr = librosa.load(
    "../whisper-service/tests/fixtures/audio/OSR_cn_000_0072_8k.wav", sr=None, mono=True
)

# Resample Chinese if needed
if cn_sr != 16000:
    chinese_audio = librosa.resample(chinese_audio, orig_sr=cn_sr, target_sr=16000)

print(f"JFK: {len(jfk_audio) / 16000:.2f}s ({len(jfk_audio)} samples)")
print(f"Chinese: {len(chinese_audio) / 16000:.2f}s ({len(chinese_audio)} samples)")

# Split JFK into 3 parts
jfk_third = len(jfk_audio) // 3
jfk_part1 = jfk_audio[:jfk_third]
jfk_part2 = jfk_audio[jfk_third : jfk_third * 2]
jfk_part3 = jfk_audio[jfk_third * 2 :]

# Split Chinese into 2 parts
chinese_half = len(chinese_audio) // 2
chinese_part1 = chinese_audio[:chinese_half]
chinese_part2 = chinese_audio[chinese_half:]

# Create pattern: JFK1 (EN) → Chinese1 (ZH) → JFK2 (EN) → Chinese2 (ZH) → JFK3 (EN)
mixed_audio = np.concatenate(
    [jfk_part1, chinese_part1, jfk_part2, chinese_part2, jfk_part3]
)

print("\nMixed audio pattern:")
print(f"  0-{len(jfk_part1) / 16000:.2f}s: JFK (EN)")
print(
    f"  {len(jfk_part1) / 16000:.2f}-{(len(jfk_part1) + len(chinese_part1)) / 16000:.2f}s: Chinese (ZH)"
)
print(
    f"  {(len(jfk_part1) + len(chinese_part1)) / 16000:.2f}-{(len(jfk_part1) + len(chinese_part1) + len(jfk_part2)) / 16000:.2f}s: JFK (EN)"
)
print(
    f"  {(len(jfk_part1) + len(chinese_part1) + len(jfk_part2)) / 16000:.2f}-{(len(jfk_part1) + len(chinese_part1) + len(jfk_part2) + len(chinese_part2)) / 16000:.2f}s: Chinese (ZH)"
)
print(
    f"  {(len(jfk_part1) + len(chinese_part1) + len(jfk_part2) + len(chinese_part2)) / 16000:.2f}-{len(mixed_audio) / 16000:.2f}s: JFK (EN)"
)
print(f"\nTotal: {len(mixed_audio) / 16000:.2f}s")

# Save
output_file = "test_mixed_en_zh.wav"
sf.write(output_file, mixed_audio, 16000)
print(f"\n✅ Created {output_file}")
