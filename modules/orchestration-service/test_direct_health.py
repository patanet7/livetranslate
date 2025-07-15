#!/usr/bin/env python3
"""Test direct health check to whisper service"""

import requests
import json

print("Testing Direct Health Check to Whisper Service")
print("="*50)

# Test from Python (not aiohttp)
try:
    response = requests.get("http://localhost:5001/health", timeout=5)
    
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"\nResponse Text: {response.text}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"\nParsed JSON:")
            print(json.dumps(data, indent=2))
            
            # Check if it has the expected fields
            print(f"\nHas 'status' field: {'status' in data}")
            print(f"Status value: {data.get('status')}")
        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {e}")
    else:
        print(f"\nNon-200 status code!")
        
except requests.exceptions.ConnectionError as e:
    print(f"\nConnection Error: Cannot connect to localhost:5001")
    print(f"Error: {e}")
except requests.exceptions.Timeout:
    print(f"\nTimeout: Request took longer than 5 seconds")
except Exception as e:
    print(f"\nUnexpected Error: {e}")
    
print("\n" + "="*50)
print("If this works, the issue is with aiohttp in orchestration service")