#!/usr/bin/env python3
"""
Simple test script for Triton-based Translation Service
"""

import asyncio
import json
import aiohttp
import requests

TRITON_BASE_URL = "http://localhost:8000"
TRANSLATION_SERVICE_URL = "http://localhost:5003"

async def test_triton_health():
    """Test if Triton server is healthy"""
    print("Testing Triton server health...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{TRITON_BASE_URL}/v2/health") as response:
                if response.status == 200:
                    print("SUCCESS: Triton server is healthy")
                    return True
                else:
                    print(f"FAILED: Triton server health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"FAILED: Failed to connect to Triton server: {e}")
        return False

async def test_translation_service():
    """Test translation service"""
    print("Testing Translation Service...")
    
    payload = {
        "text": "Hello, how are you today?",
        "target_language": "Spanish"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{TRANSLATION_SERVICE_URL}/translate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"SUCCESS: Translation successful: {result.get('translated_text', '')}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"FAILED: Translation failed: {response.status} - {error_text}")
                    return False
    except Exception as e:
        print(f"FAILED: Translation error: {e}")
        return False

async def main():
    """Run basic tests"""
    print("=== Triton Translation Service Test ===\n")
    
    triton_ok = await test_triton_health()
    translation_ok = await test_translation_service()
    
    print(f"\nResults:")
    print(f"Triton Health: {'PASS' if triton_ok else 'FAIL'}")
    print(f"Translation: {'PASS' if translation_ok else 'FAIL'}")
    
    if triton_ok and translation_ok:
        print("\nSUCCESS: All tests passed!")
        return True
    else:
        print("\nWARNING: Some tests failed.")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)