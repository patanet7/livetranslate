#!/usr/bin/env python3
"""
OBS Integration Manual Test Script

This script tests the OBS WebSocket output integration with a real OBS instance.

Prerequisites:
1. OBS Studio installed (https://obsproject.com/)
2. obs-websocket plugin enabled (built-in since OBS 28+)
3. WebSocket server enabled in OBS: Tools > WebSocket Server Settings
4. obs-websocket-py installed: pip install obsws-python

Setup in OBS:
1. Open OBS Studio
2. Go to Tools > WebSocket Server Settings
3. Enable WebSocket server
4. Set port (default: 4455)
5. Optionally set password
6. Create text sources:
   - "LiveTranslate Caption" (GDI+ Text or Text (FreeType 2))
   - "LiveTranslate Speaker" (optional, for speaker name)

Environment Variables (.env):
    OBS_HOST=localhost
    OBS_PORT=4455
    OBS_PASSWORD=your_password
    OBS_CAPTION_SOURCE=LiveTranslate Caption
    OBS_SPEAKER_SOURCE=LiveTranslate Speaker

Usage:
    # Test connection only
    python test_obs_integration.py --test-connection

    # Send test caption
    python test_obs_integration.py --send-caption "Hello World"

    # Run full integration test
    python test_obs_integration.py --full-test

    # With custom settings (overrides .env)
    python test_obs_integration.py --host 192.168.1.100 --port 4455 --password secret
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path / "services"))
sys.path.insert(0, str(src_path / "models"))

# Load environment variables from .env
load_dotenv(orchestration_root / ".env")

# Now import after path setup
from fireflies import CaptionEntry
from obs_output import OBSConnectionError, OBSOutput

# Try to get settings (may fail if DB not configured)
try:
    from config import get_settings
except Exception:
    get_settings = None


def get_obs_config_from_env():
    """Get OBS configuration from environment or settings."""
    # Use raw env vars (most reliable, avoids import issues)
    password = os.getenv("OBS_PASSWORD", "")
    return {
        "host": os.getenv("OBS_HOST", "localhost"),
        "port": int(os.getenv("OBS_PORT", "4455")),
        "password": password if password else None,
        "caption_source": os.getenv("OBS_CAPTION_SOURCE", "LiveTranslate Caption"),
        "speaker_source": os.getenv("OBS_SPEAKER_SOURCE"),
    }


async def test_connection(host: str, port: int, password: str | None = None) -> bool:
    """Test basic OBS connection."""
    print(f"\n🔌 Testing connection to OBS at {host}:{port}...")

    obs = OBSOutput(host=host, port=port, password=password)

    try:
        await obs.connect()
        print("✅ Successfully connected to OBS!")

        # Get connection info
        info = obs.get_connection_info()
        print(f"   Host: {info['host']}")
        print(f"   Port: {info['port']}")
        print(f"   Connected: {info['connected']}")

        await obs.disconnect()
        print("✅ Disconnected successfully")
        return True

    except OBSConnectionError as e:
        print(f"❌ Connection failed: {e}")
        return False


async def test_source_validation(
    host: str, port: int, password: str | None = None, caption_source: str = "LiveTranslate Caption"
) -> bool:
    """Test that text sources exist in OBS."""
    print("\n🔍 Validating OBS sources...")

    obs = OBSOutput(host=host, port=port, password=password, caption_source=caption_source)

    try:
        await obs.connect()

        exists = await obs.validate_sources()
        if exists:
            print(f"✅ Caption source '{caption_source}' found!")
        else:
            print(f"⚠️ Caption source '{caption_source}' NOT found")
            print("   Please create a text source in OBS with this name")

        await obs.disconnect()
        return exists

    except OBSConnectionError as e:
        print(f"❌ Connection failed: {e}")
        return False


async def send_test_caption(
    host: str,
    port: int,
    password: str | None = None,
    caption_source: str = "LiveTranslate Caption",
    text: str = "Hello, this is a test caption!",
) -> bool:
    """Send a test caption to OBS."""
    print("\n📝 Sending test caption to OBS...")

    obs = OBSOutput(host=host, port=port, password=password, caption_source=caption_source)

    try:
        await obs.connect()

        # Create test caption
        caption = CaptionEntry(
            id="test-001",
            original_text="Test original text",
            translated_text=text,
            speaker_name="Test Speaker",
            speaker_color="#4CAF50",
            target_language="es",
            timestamp=datetime.now(UTC),
            duration_seconds=5.0,
            confidence=0.95,
        )

        result = await obs.update_caption(caption)
        if result:
            print("✅ Caption sent successfully!")
            print(f"   Text: {text}")
            print("   Check your OBS text source for the caption")
        else:
            print("❌ Failed to send caption")

        await obs.disconnect()
        return result

    except OBSConnectionError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_full_test(
    host: str,
    port: int,
    password: str | None = None,
    caption_source: str = "LiveTranslate Caption",
) -> bool:
    """Run full integration test with multiple captions."""
    print("\n🧪 Running full OBS integration test...")
    print("=" * 50)

    obs = OBSOutput(
        host=host,
        port=port,
        password=password,
        caption_source=caption_source,
        show_original=True,
    )

    try:
        # Test 1: Connection
        print("\n[Test 1] Connecting to OBS...")
        await obs.connect()
        print("✅ Connected")

        # Test 2: Source validation
        print("\n[Test 2] Validating sources...")
        exists = await obs.validate_sources()
        if not exists:
            print(f"⚠️ Source '{caption_source}' not found - creating...")
            # Can't auto-create without scene info, warn user
            print("   Please create the text source manually and run again")
            await obs.disconnect()
            return False
        print("✅ Sources validated")

        # Test 3: Send captions
        print("\n[Test 3] Sending test captions (watch OBS)...")

        test_captions = [
            ("Alice", "Hello, how are you?", "Hola, ¿cómo estás?", "es"),
            ("Bob", "I'm doing great, thanks!", "¡Estoy muy bien, gracias!", "es"),
            ("Alice", "Let's discuss the project", "Discutamos el proyecto", "es"),
            ("Charlie", "Good morning everyone", "Buenos días a todos", "es"),
            ("Bob", "The meeting starts now", "La reunión comienza ahora", "es"),
        ]

        for i, (speaker, original, translated, lang) in enumerate(test_captions, 1):
            caption = CaptionEntry(
                id=f"test-{i:03d}",
                original_text=original,
                translated_text=translated,
                speaker_name=speaker,
                speaker_color=["#4CAF50", "#2196F3", "#FF9800"][i % 3],
                target_language=lang,
                timestamp=datetime.now(UTC),
                duration_seconds=3.0,
                confidence=0.95,
            )

            result = await obs.update_caption(caption)
            print(f"   [{i}/5] {speaker}: {translated} {'✅' if result else '❌'}")

            await asyncio.sleep(2)  # Display each caption for 2 seconds

        # Test 4: Clear caption
        print("\n[Test 4] Clearing caption...")
        await obs.clear_caption()
        print("✅ Caption cleared")

        # Test 5: Statistics
        print("\n[Test 5] Checking statistics...")
        stats = obs.get_stats()
        print(f"   Updates sent: {stats['updates_sent']}")
        print(f"   Errors: {stats['errors']}")
        print("✅ Statistics tracked")

        await obs.disconnect()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! OBS integration is working.")
        return True

    except OBSConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def demo_live_captions(
    host: str,
    port: int,
    password: str | None = None,
    caption_source: str = "LiveTranslate Caption",
    duration_seconds: int = 30,
) -> None:
    """Run a live demo with simulated captions."""
    print(f"\n🎬 Running live caption demo for {duration_seconds} seconds...")
    print("   Press Ctrl+C to stop")

    obs = OBSOutput(
        host=host,
        port=port,
        password=password,
        caption_source=caption_source,
        show_original=True,
    )

    try:
        await obs.connect()

        demo_sentences = [
            ("Alice", "Welcome to the meeting", "Bienvenidos a la reunión"),
            ("Bob", "Thank you for joining us today", "Gracias por unirse hoy"),
            ("Alice", "Let's start with the agenda", "Empecemos con la agenda"),
            (
                "Charlie",
                "I have a question about the budget",
                "Tengo una pregunta sobre el presupuesto",
            ),
            ("Bob", "Good question, let me explain", "Buena pregunta, déjame explicar"),
            ("Alice", "The project is on track", "El proyecto va por buen camino"),
            ("Charlie", "When is the deadline?", "¿Cuándo es la fecha límite?"),
            ("Bob", "Next Friday at noon", "El próximo viernes al mediodía"),
            ("Alice", "Any other questions?", "¿Alguna otra pregunta?"),
            ("Charlie", "No, I think we're good", "No, creo que estamos bien"),
        ]

        start_time = datetime.now(UTC)
        caption_index = 0

        while True:
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > duration_seconds:
                break

            speaker, original, translated = demo_sentences[caption_index % len(demo_sentences)]

            caption = CaptionEntry(
                id=f"demo-{caption_index:03d}",
                original_text=original,
                translated_text=translated,
                speaker_name=speaker,
                speaker_color=["#4CAF50", "#2196F3", "#FF9800"][caption_index % 3],
                target_language="es",
                timestamp=datetime.now(UTC),
                duration_seconds=3.0,
                confidence=0.92 + (caption_index % 8) * 0.01,
            )

            await obs.update_caption(caption)
            print(f"   [{speaker}] {translated}")

            caption_index += 1
            await asyncio.sleep(3)

        await obs.clear_caption()
        await obs.disconnect()
        print("\n✅ Demo complete!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted")
        await obs.clear_caption()
        await obs.disconnect()
    except Exception as e:
        print(f"❌ Demo error: {e}")


def main():
    # Load default config from environment
    env_config = get_obs_config_from_env()

    parser = argparse.ArgumentParser(
        description="Test OBS WebSocket integration for LiveTranslate captions"
    )

    parser.add_argument("--host", default=env_config["host"], help="OBS WebSocket host")
    parser.add_argument("--port", type=int, default=env_config["port"], help="OBS WebSocket port")
    parser.add_argument("--password", default=env_config["password"], help="OBS WebSocket password")
    parser.add_argument(
        "--caption-source",
        default=env_config["caption_source"],
        help="Name of OBS text source for captions",
    )

    # Test modes
    parser.add_argument("--test-connection", action="store_true", help="Test OBS connection only")
    parser.add_argument(
        "--validate-sources", action="store_true", help="Validate OBS sources exist"
    )
    parser.add_argument("--send-caption", metavar="TEXT", help="Send a test caption")
    parser.add_argument("--full-test", action="store_true", help="Run full integration test")
    parser.add_argument(
        "--demo",
        type=int,
        metavar="SECONDS",
        help="Run live demo for N seconds",
    )

    args = parser.parse_args()

    # Default to full test if no mode specified
    if not any(
        [
            args.test_connection,
            args.validate_sources,
            args.send_caption,
            args.full_test,
            args.demo,
        ]
    ):
        args.full_test = True

    print("=" * 60)
    print("  OBS WebSocket Integration Test - LiveTranslate")
    print("=" * 60)

    if args.test_connection:
        asyncio.run(test_connection(args.host, args.port, args.password))

    if args.validate_sources:
        asyncio.run(
            test_source_validation(args.host, args.port, args.password, args.caption_source)
        )

    if args.send_caption:
        asyncio.run(
            send_test_caption(
                args.host, args.port, args.password, args.caption_source, args.send_caption
            )
        )

    if args.full_test:
        asyncio.run(run_full_test(args.host, args.port, args.password, args.caption_source))

    if args.demo:
        asyncio.run(
            demo_live_captions(args.host, args.port, args.password, args.caption_source, args.demo)
        )


if __name__ == "__main__":
    main()
