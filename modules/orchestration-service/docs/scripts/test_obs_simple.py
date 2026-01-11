#!/usr/bin/env python3
"""
Simple OBS Connection Test

Minimal test script for OBS WebSocket connectivity.
No complex imports - tests basic connection only.

Prerequisites:
1. OBS Studio running
2. WebSocket server enabled (Tools > WebSocket Server Settings)
3. pip install obsws-python

Usage:
    python test_obs_simple.py
    python test_obs_simple.py --password YOUR_PASSWORD
"""

import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load .env from orchestration-service root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("Note: python-dotenv not installed, using environment only")

try:
    import obsws_python as obsws
except ImportError:
    print("ERROR: obsws-python not installed. Run: pip install obsws-python")
    exit(1)


@dataclass
class OBSConfig:
    """OBS connection configuration."""
    host: str = "localhost"
    port: int = 4455
    password: Optional[str] = None
    caption_source: str = "LiveTranslate Caption"


def get_config_from_env() -> OBSConfig:
    """Load config from environment variables."""
    password = os.getenv("OBS_PASSWORD", "")
    return OBSConfig(
        host=os.getenv("OBS_HOST", "localhost"),
        port=int(os.getenv("OBS_PORT", "4455")),
        password=password if password else None,
        caption_source=os.getenv("OBS_CAPTION_SOURCE", "LiveTranslate Caption"),
    )


def test_connection(config: OBSConfig) -> bool:
    """Test OBS WebSocket connection."""
    print(f"\n🔌 Testing connection to OBS at {config.host}:{config.port}...")

    try:
        # Create client
        client = obsws.ReqClient(
            host=config.host,
            port=config.port,
            password=config.password,
            timeout=10,
        )

        # Get version info
        version = client.get_version()
        print(f"✅ Connected to OBS!")
        print(f"   OBS Version: {version.obs_version}")
        print(f"   WebSocket Version: {version.obs_web_socket_version}")
        print(f"   Platform: {version.platform}")

        # Get input list
        inputs = client.get_input_list()
        text_sources = [
            inp for inp in inputs.inputs
            if "text" in inp.get("inputKind", "").lower()
        ]

        print(f"\n📋 Text Sources Found ({len(text_sources)}):")
        for src in text_sources:
            marker = "✓" if src["inputName"] == config.caption_source else " "
            print(f"   [{marker}] {src['inputName']} ({src['inputKind']})")

        if config.caption_source not in [s["inputName"] for s in text_sources]:
            print(f"\n⚠️ Caption source '{config.caption_source}' not found!")
            print("   Please create a text source with this name in OBS.")

        # Disconnect
        client.disconnect()
        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def send_test_caption(config: OBSConfig, text: str) -> bool:
    """Send a test caption to OBS."""
    print(f"\n📝 Sending test caption to '{config.caption_source}'...")

    try:
        client = obsws.ReqClient(
            host=config.host,
            port=config.port,
            password=config.password,
            timeout=10,
        )

        # Update text source
        client.set_input_settings(
            name=config.caption_source,
            settings={"text": text},
        )

        print(f"✅ Caption sent: \"{text}\"")
        print("   Check your OBS text source!")

        client.disconnect()
        return True

    except Exception as e:
        print(f"❌ Failed to send caption: {e}")
        return False


def run_demo(config: OBSConfig, duration: int = 10) -> None:
    """Run a live demo showing captions."""
    import time

    print(f"\n🎬 Running caption demo for {duration} seconds...")
    print("   Press Ctrl+C to stop")

    demo_lines = [
        ("Alice", "Hello, welcome to the meeting."),
        ("Bob", "Thanks for joining us today."),
        ("Alice", "Let's discuss the agenda."),
        ("Charlie", "I have a question about the budget."),
        ("Bob", "Good question, let me explain."),
    ]

    try:
        client = obsws.ReqClient(
            host=config.host,
            port=config.port,
            password=config.password,
            timeout=10,
        )

        start_time = time.time()
        index = 0

        while time.time() - start_time < duration:
            speaker, text = demo_lines[index % len(demo_lines)]
            caption = f"[{speaker}] {text}"

            client.set_input_settings(
                name=config.caption_source,
                settings={"text": caption},
            )

            print(f"   {caption}")
            index += 1
            time.sleep(2)

        # Clear caption
        client.set_input_settings(
            name=config.caption_source,
            settings={"text": ""},
        )

        client.disconnect()
        print("\n✅ Demo complete!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted")
    except Exception as e:
        print(f"❌ Demo error: {e}")


def main():
    # Load defaults from environment
    default_config = get_config_from_env()

    parser = argparse.ArgumentParser(
        description="Simple OBS WebSocket connection test"
    )

    parser.add_argument("--host", default=default_config.host, help="OBS WebSocket host")
    parser.add_argument("--port", type=int, default=default_config.port, help="OBS WebSocket port")
    parser.add_argument("--password", default=default_config.password, help="OBS WebSocket password")
    parser.add_argument("--source", default=default_config.caption_source, help="Caption text source name")

    parser.add_argument("--test", action="store_true", help="Test connection only")
    parser.add_argument("--send", metavar="TEXT", help="Send a test caption")
    parser.add_argument("--demo", type=int, metavar="SECONDS", help="Run demo for N seconds")

    args = parser.parse_args()

    config = OBSConfig(
        host=args.host,
        port=args.port,
        password=args.password,
        caption_source=args.source,
    )

    print("=" * 60)
    print("  OBS WebSocket Test - LiveTranslate")
    print("=" * 60)
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Password: {'***' if config.password else '(none)'}")
    print(f"  Source: {config.caption_source}")
    print("=" * 60)

    # Default to test if no action specified
    if not any([args.test, args.send, args.demo]):
        args.test = True

    if args.test:
        test_connection(config)

    if args.send:
        send_test_caption(config, args.send)

    if args.demo:
        run_demo(config, args.demo)


if __name__ == "__main__":
    main()
