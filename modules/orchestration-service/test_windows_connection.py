#!/usr/bin/env python3
"""Test connection from Windows Python (not WSL)"""

print("""
To test if this is a WSL issue, please run this script from a Windows command prompt or PowerShell:

1. Open a Windows PowerShell or Command Prompt (NOT WSL)
2. Navigate to: C:\\Users\\patan\\Projects\\livetranslate\\modules\\orchestration-service
3. Run: python test_windows_connection.py

This will test if the connection works from Windows Python.
""")

import asyncio
import aiohttp
import time

async def test_connection():
    url = "http://localhost:5001/health"
    print(f"Testing connection to: {url}")
    
    try:
        # Simple connection without any SSL configuration
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                print(f"✅ Success! Status: {response.status}")
                print(f"Response: {data}")
                return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import platform
    print(f"Running on: {platform.system()} - {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    result = asyncio.run(test_connection())
    
    if not result and "Linux" in platform.system():
        print("\n⚠️  You're running this from WSL. Please run from Windows to test properly.")