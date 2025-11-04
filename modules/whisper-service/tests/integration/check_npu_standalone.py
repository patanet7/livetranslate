#!/usr/bin/env python3
"""
Test NPU detection and OpenVINO setup
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== NPU Detection Test ===")
print()

# Test 1: Check OpenVINO import
try:
    import openvino as ov
    print("✓ OpenVINO imported successfully")
    print(f"  Version: {ov.__version__}")
except Exception as e:
    print(f"✗ Failed to import OpenVINO: {e}")
    sys.exit(1)

# Test 2: Check OpenVINO GenAI
try:
    import openvino_genai
    print("✓ OpenVINO GenAI imported successfully")
except Exception as e:
    print(f"✗ Failed to import OpenVINO GenAI: {e}")

print()

# Test 3: Check available devices
try:
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    
    for device in devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {device_name}")
        except:
            print(f"  {device}: (no details available)")
            
except Exception as e:
    print(f"✗ Failed to get devices: {e}")

print()

# Test 4: Check environment variables
print("Environment variables:")
print(f"  OPENVINO_DEVICE: {os.getenv('OPENVINO_DEVICE', '(not set)')}")
print(f"  PATH includes OpenVINO: {'openvino' in os.environ.get('PATH', '').lower()}")

print()

# Test 5: Try to load a simple model on NPU
if "NPU" in devices:
    print("Testing NPU with a simple model...")
    try:
        # Try to create a simple model
        from openvino.runtime import Core, Model, op
        import numpy as np
        
        core = Core()
        
        # Create a minimal model
        param = op.Parameter(np.float32, [1, 3, 224, 224])
        relu = op.v0.Relu(param)
        model = Model(relu, [param])
        
        # Try to compile on NPU
        compiled = core.compile_model(model, "NPU")
        print("✓ Successfully compiled test model on NPU!")
        
    except Exception as e:
        print(f"✗ NPU test failed: {e}")
else:
    print("⚠ NPU not detected in available devices")

print()

# Test 6: Check models directory
models_dir = os.path.join(os.path.dirname(__file__), "models")
if os.path.exists(models_dir):
    print(f"✓ Models directory found: {models_dir}")
    models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"  Available models: {models}")
else:
    print(f"✗ Models directory not found: {models_dir}")

print()

# Test 7: Try loading Whisper model
print("Testing Whisper model loading...")
try:
    model_path = os.path.join(models_dir, "whisper-base")
    if os.path.exists(model_path):
        print(f"  Model path: {model_path}")
        
        # Check for NPU
        device = "NPU" if "NPU" in devices else "CPU"
        print(f"  Target device: {device}")
        
        # Try to load
        pipeline = openvino_genai.WhisperPipeline(str(model_path), device=device)
        print(f"✓ Successfully loaded whisper-base on {device}!")
        
    else:
        print(f"✗ Model path not found: {model_path}")
        
except Exception as e:
    print(f"✗ Failed to load Whisper model: {e}")

print()
print("=== Test Complete ===")