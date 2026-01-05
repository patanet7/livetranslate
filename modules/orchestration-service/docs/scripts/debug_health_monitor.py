#!/usr/bin/env python3
"""Debug health monitor to see what's happening"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_health_monitor():
    """Test the health monitor directly"""
    print("Testing Health Monitor")
    print("=" * 50)

    try:
        # Import after path setup
        from managers.health_monitor import HealthMonitor
        from config import get_settings

        settings = get_settings()
        print(
            f"Loaded settings - Audio Service URL: {settings.services.audio_service_url}"
        )

        # Create health monitor
        health_monitor = HealthMonitor(settings=settings)

        print("\nService configs:")
        for name, config in health_monitor.service_configs.items():
            print(f"  {name}: {config['url']}{config['health_endpoint']}")

        print("\nChecking whisper service health...")

        # Check whisper health directly
        await health_monitor._check_service_health("whisper")

        service = health_monitor.services.get("whisper")
        if service:
            print(f"\nWhisper service status: {service.status}")
            print(f"Last error: {service.last_error}")
            print(f"Error count: {service.error_count}")
            print(f"Response time: {service.response_time}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_health_monitor())
