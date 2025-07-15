#!/usr/bin/env python3
"""Test script to debug whisper service connection issue"""

import asyncio
import aiohttp
import ssl
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_whisper_connection():
    """Test connection to whisper service"""
    
    # Test URLs
    urls_to_test = [
        "http://localhost:5001/health",
        "http://127.0.0.1:5001/health",
        "http://0.0.0.0:5001/health"
    ]
    
    # Create SSL context (even though we're using HTTP)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for url in urls_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {url}")
        print('='*50)
        
        try:
            start_time = time.time()
            
            # Determine SSL configuration based on URL scheme
            ssl_setting = False if url.startswith("http://") else ssl_context
            
            # Test 1: With TCPConnector (explicit no SSL)
            print("\nTest 1: With TCPConnector (explicit no SSL)")
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    data = await response.json()
                    print(f"✅ Success! Status: {response.status}")
                    print(f"Response time: {response_time:.2f}ms")
                    print(f"Response: {data}")
            
            # Test 2: Without explicit connector
            print("\nTest 2: Without explicit connector")
            start_time = time.time()
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    data = await response.json()
                    print(f"✅ Success! Status: {response.status}")
                    print(f"Response time: {response_time:.2f}ms")
                    print(f"Response: {data}")
                    
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   OS error: {e.os_error if hasattr(e, 'os_error') else 'N/A'}")
            
        except asyncio.TimeoutError:
            print(f"❌ Timeout error (5 seconds)")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing Whisper Service Connection")
    print("==================================")
    
    # First test with curl to make sure service is running
    import subprocess
    print("\nTesting with curl first:")
    try:
        result = subprocess.run(["curl", "-s", "http://127.0.0.1:5001/health"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ curl successful: {result.stdout}")
        else:
            print(f"❌ curl failed: {result.stderr}")
    except Exception as e:
        print(f"❌ curl test failed: {e}")
    
    print("\nNow testing with aiohttp:")
    asyncio.run(test_whisper_connection())