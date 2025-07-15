#!/usr/bin/env python3
"""Test environment and configuration loading"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing Configuration Loading")
print("="*50)

# Check environment variables
print("\nEnvironment Variables:")
print(f"AUDIO_SERVICE_URL: {os.getenv('AUDIO_SERVICE_URL', 'Not set')}")
print(f"WHISPER_SERVICE_URL: {os.getenv('WHISPER_SERVICE_URL', 'Not set')}")
print(f"TRANSLATION_SERVICE_URL: {os.getenv('TRANSLATION_SERVICE_URL', 'Not set')}")

# Load configuration
try:
    from config import get_settings
    settings = get_settings()
    
    print("\nLoaded Settings:")
    print(f"Audio Service URL: {settings.services.audio_service_url}")
    print(f"Translation Service URL: {settings.services.translation_service_url}")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    
except Exception as e:
    print(f"\nError loading settings: {e}")
    import traceback
    traceback.print_exc()

# Test health monitor initialization
try:
    from managers.health_monitor import HealthMonitor
    
    health_monitor = HealthMonitor(settings=settings)
    
    print("\nHealth Monitor Service Configs:")
    for name, config in health_monitor.service_configs.items():
        print(f"{name}: {config['url']}{config['health_endpoint']}")
        
except Exception as e:
    print(f"\nError initializing health monitor: {e}")
    import traceback
    traceback.print_exc()