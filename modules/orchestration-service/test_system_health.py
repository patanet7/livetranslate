#!/usr/bin/env python3
"""Test system health endpoint response"""

import requests
import json

print("Testing System Health Endpoint")
print("="*50)

try:
    # Test the system health endpoint
    response = requests.get("http://localhost:3000/api/system/health")
    
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse Data:")
        print(json.dumps(data, indent=2))
        
        # Check services section
        if 'services' in data:
            print("\nServices Status:")
            for service_name, service_data in data['services'].items():
                status = service_data.get('status', 'unknown')
                print(f"  {service_name}: {status}")
        else:
            print("\nNo 'services' key in response!")
    else:
        print(f"\nError Response: {response.text}")
        
except Exception as e:
    print(f"\nError: {e}")
    
# Also test the audio health endpoint
print("\n" + "="*50)
print("Testing Audio Health Endpoint")
print("="*50)

try:
    response = requests.get("http://localhost:3000/api/audio/health")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse Data:")
        print(json.dumps(data, indent=2))
    else:
        print(f"\nError Response: {response.text}")
        
except Exception as e:
    print(f"\nError: {e}")