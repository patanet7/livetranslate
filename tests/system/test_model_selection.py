#!/usr/bin/env python3
"""
Test script to verify model selection functionality
"""

import requests
import json
import time

# Configuration
ORCHESTRATION_URL = "http://localhost:3000"
WHISPER_URL = "http://localhost:5001"

def test_model_selection():
    """Test model selection through orchestration service"""
    
    print("🧪 Testing Model Selection Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Get available models through orchestration
        print("1. Testing model list through orchestration gateway...")
        response = requests.get(f"{ORCHESTRATION_URL}/api/whisper/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('available_models', data.get('models', []))
            print(f"   ✅ Found {len(models)} models: {models}")
        else:
            print(f"   ❌ Failed to get models: {response.status_code}")
            return False
            
        # Test 2: Direct whisper service models
        print("2. Testing direct whisper service models...")
        try:
            response = requests.get(f"{WHISPER_URL}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                direct_models = data.get('available_models', data.get('models', []))
                print(f"   ✅ Direct service found {len(direct_models)} models: {direct_models}")
            else:
                print(f"   ⚠️ Direct service returned: {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ Direct service not accessible: {e}")
            
        # Test 3: Health check to see current model
        print("3. Testing health check for current model...")
        response = requests.get(f"{WHISPER_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            current_device = health_data.get('hardware', {}).get('device', 'Unknown')
            loaded_models = health_data.get('models', {}).get('loaded', [])
            print(f"   ✅ Current device: {current_device}")
            print(f"   ✅ Loaded models: {loaded_models}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            
        print("\n🎉 Model selection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_selection()
    exit(0 if success else 1)