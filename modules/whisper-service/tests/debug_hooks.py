#!/usr/bin/env python3
"""
Debug script to inspect Whisper model structure and hook behavior
"""

import whisper
import torch
import numpy as np
from pathlib import Path

# Load model
models_dir = Path(__file__).parent.parent / ".models"
print(f"Loading model from: {models_dir}")

model = whisper.load_model("large-v3", download_root=str(models_dir))
print(f"Model loaded on device: {next(model.parameters()).device}")

# Inspect decoder structure
print("\n=== Decoder Structure ===")
print(f"Number of decoder blocks: {len(model.decoder.blocks)}")

# Check first block structure
first_block = model.decoder.blocks[0]
print(f"\nFirst block attributes:")
for attr in dir(first_block):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Check cross_attn specifically
if hasattr(first_block, 'cross_attn'):
    print(f"\nCross-attention module: {first_block.cross_attn}")
    print(f"Cross-attention type: {type(first_block.cross_attn)}")

# Install test hook
captured_outputs = []

def test_hook(module, input, output):
    print(f"\n🎯 Hook triggered!")
    print(f"   Input type: {type(input)}, len: {len(input) if isinstance(input, tuple) else 'N/A'}")
    print(f"   Output type: {type(output)}, len: {len(output) if isinstance(output, tuple) else 'N/A'}")

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if out is not None:
                print(f"   Output[{i}] shape: {out.shape if hasattr(out, 'shape') else 'no shape'}")
            else:
                print(f"   Output[{i}]: None ⚠️")

    captured_outputs.append(output)

# Install hook
first_block.cross_attn.register_forward_hook(test_hook)
print("\n✅ Hook installed on first decoder block")

# Run inference
print("\n=== Running Inference ===")
audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence

result = model.transcribe(audio, beam_size=1, temperature=0.0)

print(f"\n=== Results ===")
print(f"Transcription: {result['text']}")
print(f"Hooks triggered: {len(captured_outputs)} times")

if len(captured_outputs) > 0:
    print(f"\n✅ SUCCESS: Hooks were triggered!")
    print(f"First output structure:")
    first_out = captured_outputs[0]
    if isinstance(first_out, tuple):
        for i, out in enumerate(first_out):
            if out is not None:
                print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}")
else:
    print(f"\n❌ PROBLEM: Hooks were NOT triggered during transcription!")
    print("This means model.transcribe() doesn't use the cross_attn modules as expected")
