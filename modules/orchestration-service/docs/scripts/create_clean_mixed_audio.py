#!/usr/bin/env python3
"""
Create clean mixed language audio file with COMPLETE utterances.
Pattern: Full JFK (EN) → Full Chinese sentence → Full JFK (EN) → Full Chinese sentence
This avoids mid-sentence switching confusion.
"""

import librosa
import soundfile as sf
import numpy as np

# Load COMPLETE audio files
jfk_audio, jfk_sr = librosa.load(
    "../whisper-service/tests/fixtures/audio/jfk.wav", sr=16000, mono=True
)
chinese_audio, cn_sr = librosa.load(
    "../whisper-service/tests/fixtures/audio/OSR_cn_000_0072_8k.wav", sr=None, mono=True
)

# Resample Chinese if needed
if cn_sr != 16000:
    chinese_audio = librosa.resample(chinese_audio, orig_sr=cn_sr, target_sr=16000)

print(f"JFK (full): {len(jfk_audio) / 16000:.2f}s ({len(jfk_audio)} samples)")
print(
    f"Chinese #1 (full): {len(chinese_audio) / 16000:.2f}s ({len(chinese_audio)} samples)"
)

# Add 1 second of silence between sections to make boundaries clear
silence = np.zeros(16000, dtype=np.float32)

# Load additional Chinese files for variety
chinese_audio2, cn_sr2 = librosa.load(
    "../whisper-service/tests/fixtures/audio/OSR_cn_000_0073_8k.wav", sr=None, mono=True
)
if cn_sr2 != 16000:
    chinese_audio2 = librosa.resample(chinese_audio2, orig_sr=cn_sr2, target_sr=16000)

print(
    f"Chinese #2 (full): {len(chinese_audio2) / 16000:.2f}s ({len(chinese_audio2)} samples)"
)

# Create pattern with COMPLETE utterances and clear boundaries:
# JFK (full) → silence → Chinese #1 (full) → silence → JFK (full) → silence → Chinese #2 (full)
mixed_audio = np.concatenate(
    [
        jfk_audio,  # Complete JFK speech in English
        silence,  # 1 second gap
        chinese_audio,  # Complete Chinese sentence #1: "院子门口不远处就是一个地铁站"
        silence,  # 1 second gap
        jfk_audio,  # Complete JFK speech again in English
        silence,  # 1 second gap
        chinese_audio2,  # Complete Chinese sentence #2
    ]
)

# Calculate timestamps
t1 = len(jfk_audio) / 16000
t2 = t1 + 1.0
t3 = t2 + len(chinese_audio) / 16000
t4 = t3 + 1.0
t5 = t4 + len(jfk_audio) / 16000
t6 = t5 + 1.0
t7 = t6 + len(chinese_audio2) / 16000

print("\nMixed audio pattern (COMPLETE utterances):")
print(f"  0.00-{t1:.2f}s: JFK FULL (EN)")
print(f"  {t1:.2f}-{t2:.2f}s: Silence")
print(f"  {t2:.2f}-{t3:.2f}s: Chinese #1 FULL (ZH): '院子门口不远处就是一个地铁站'")
print(f"  {t3:.2f}-{t4:.2f}s: Silence")
print(f"  {t4:.2f}-{t5:.2f}s: JFK FULL (EN)")
print(f"  {t5:.2f}-{t6:.2f}s: Silence")
print(f"  {t6:.2f}-{t7:.2f}s: Chinese #2 FULL (ZH): '这是一个美丽而神奇的景象'")
print(f"\nTotal: {len(mixed_audio) / 16000:.2f}s")

# Save
output_file = "test_clean_mixed_en_zh.wav"
sf.write(output_file, mixed_audio, 16000)
print(f"\n✅ Created {output_file}")
print("\nExpected transcription:")
print(
    "  EN: 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'"
)
print("  ZH: '院子门口不远处就是一个地铁站'")
print(
    "  EN: 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'"
)
print("  ZH: '这是一个美丽而神奇的景象'")
