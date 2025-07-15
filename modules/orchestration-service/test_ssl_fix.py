#!/usr/bin/env python3
"""Test script to verify SSL fix for whisper service connection"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_health_monitor():
    """Test the health monitor with our SSL fix"""
    print("Testing Health Monitor with SSL Fix")
    print("="*50)
    
    try:
        # Import after path setup
        from managers.health_monitor import HealthMonitor
        from config import get_settings
        
        settings = get_settings()
        print(f"Audio Service URL from settings: {settings.services.audio_service_url}")
        
        # Create health monitor
        health_monitor = HealthMonitor(settings=settings)
        
        print("\nChecking whisper service health...")
        
        # Check whisper health directly
        await health_monitor._check_service_health("whisper")
        
        service = health_monitor.services.get("whisper")
        if service:
            print(f"\nWhisper service status: {service.status}")
            print(f"Last error: {service.last_error}")
            print(f"Error count: {service.error_count}")
            print(f"Response time: {service.response_time}")
        else:
            print("Whisper service not found in health monitor")
        
        print("\n" + "="*50)
        print("Test completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

async def test_audio_client():
    """Test the audio service client with our SSL fix"""
    print("\n\nTesting Audio Service Client with SSL Fix")
    print("="*50)
    
    try:
        # Import after path setup
        from clients.audio_service_client import AudioServiceClient
        from config import get_settings
        
        settings = get_settings()
        
        # Create audio client
        audio_client = AudioServiceClient(settings=settings)
        print(f"Audio Service URL: {audio_client.base_url}")
        
        print("\nChecking audio service health...")
        
        # Check health
        health_result = await audio_client.health_check()
        
        print(f"\nHealth check result:")
        print(f"  Status: {health_result.get('status')}")
        print(f"  URL: {health_result.get('url')}")
        print(f"  Error: {health_result.get('error', 'None')}")
        
        # Close the client
        await audio_client.close()
        
        print("\n" + "="*50)
        print("Test completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("SSL Fix Test Script")
    print("==================\n")
    
    # Run both tests
    asyncio.run(test_health_monitor())
    asyncio.run(test_audio_client())