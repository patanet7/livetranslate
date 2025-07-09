#!/usr/bin/env python3
"""
Simple test script to test multipart form data forwarding through the API gateway
"""

import requests
import tempfile
import os

def create_test_file():
    """Create a simple test file"""
    content = b"This is a test audio file content that should be forwarded properly"
    return content

def test_direct_whisper():
    """Test direct connection to whisper service"""
    print("Testing direct whisper service...")
    
    content = create_test_file()
    files = {'audio': ('test.mp4', content, 'audio/mp4')}
    
    try:
        response = requests.post('http://localhost:5001/transcribe/whisper-base', files=files, timeout=10)
        print(f"Direct whisper - Status: {response.status_code}")
        print(f"Direct whisper - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Direct whisper failed: {e}")

def test_gateway_forwarding():
    """Test forwarding through API gateway"""
    print("\nTesting API gateway forwarding...")
    
    content = create_test_file()
    files = {'audio': ('test.mp4', content, 'audio/mp4')}
    
    try:
        response = requests.post('http://localhost:3000/api/whisper/transcribe/whisper-base', files=files, timeout=10)
        print(f"Gateway - Status: {response.status_code}")
        print(f"Gateway - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Gateway failed: {e}")

def test_debug_endpoint():
    """Test the debug endpoint to see if Flask receives multipart data"""
    print("\nTesting orchestration debug endpoint...")
    
    content = create_test_file()
    files = {'audio': ('test.mp4', content, 'audio/mp4')}
    
    try:
        response = requests.post('http://localhost:3000/api/debug/multipart', files=files, timeout=10)
        print(f"Debug - Status: {response.status_code}")
        if response.status_code == 200:
            import json
            data = response.json()
            print(f"Debug - Files received: {len(data.get('files_received', []))}")
            for file_info in data.get('files_received', []):
                print(f"  File: {file_info.get('filename')} ({file_info.get('size')} bytes, {file_info.get('content_type')})")
        else:
            print(f"Debug - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Debug endpoint failed: {e}")

if __name__ == "__main__":
    print("Testing multipart form data forwarding...")
    test_direct_whisper()
    test_debug_endpoint()
    test_gateway_forwarding()