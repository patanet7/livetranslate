#!/usr/bin/env python3
"""
Caption Overlay Test Script

Tests the HTML caption overlay by sending captions via HTTP POST.

Prerequisites:
1. Orchestration service running: poetry run python -m main_fastapi
2. Open http://localhost:3000/static/captions.html?session=test&showStatus=true in browser

Usage:
    python test_caption_overlay.py
    python test_caption_overlay.py --session my-session
    python test_caption_overlay.py --demo 30  # Run 30 second demo
"""

import argparse
import asyncio

import aiohttp


async def post_caption(
    session: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
    text: str,
    original: str | None = None,
    speaker: str = "Speaker",
    color: str = "#4CAF50",
    duration: float = 8.0,
):
    """Post a caption via HTTP."""
    url = f"{base_url}/api/captions/{session_id}"
    payload = {
        "text": text,
        "original_text": original,
        "speaker_name": speaker,
        "speaker_color": color,
        "target_language": "es",
        "duration_seconds": duration,
        "confidence": 0.95,
    }

    async with session.post(url, json=payload) as response:
        if response.status == 201:
            print(f"   [{speaker}] {text}")
            return True
        else:
            error = await response.text()
            print(f"   Error: {response.status} - {error}")
            return False


async def test_basic(session_id: str, host: str, port: int):
    """Test basic caption display."""
    print("\n🧪 Testing basic caption display...")
    print(f"   Session: {session_id}")
    print("\n   Open this URL in your browser:")
    print(f"   http://{host}:{port}/static/captions.html?session={session_id}&showStatus=true")
    print()

    base_url = f"http://{host}:{port}"

    input("\nPress Enter to send test captions...")

    async with aiohttp.ClientSession() as session:
        # Send test captions
        await post_caption(
            session,
            base_url,
            session_id,
            text="Hola, bienvenidos a la reunión.",
            original="Hello, welcome to the meeting.",
            speaker="Alice",
            color="#4CAF50",
            duration=5.0,
        )

        await asyncio.sleep(2)

        await post_caption(
            session,
            base_url,
            session_id,
            text="Gracias por unirse hoy.",
            original="Thank you for joining today.",
            speaker="Bob",
            color="#2196F3",
            duration=5.0,
        )

        await asyncio.sleep(2)

        await post_caption(
            session,
            base_url,
            session_id,
            text="Vamos a empezar con la agenda.",
            original="Let's start with the agenda.",
            speaker="Alice",
            color="#4CAF50",
            duration=5.0,
        )

        print("\n✅ Test captions sent! Check your browser.")
        input("\nPress Enter to exit...")


async def run_demo(session_id: str, host: str, port: int, duration: int):
    """Run a live demo with simulated captions."""
    print(f"\n🎬 Running caption demo for {duration} seconds...")
    print(f"   Session: {session_id}")
    print("\n   Open this URL in your browser:")
    print(f"   http://{host}:{port}/static/captions.html?session={session_id}&showStatus=true")
    print("\n   Press Ctrl+C to stop\n")

    demo_conversations = [
        (
            "Alice",
            "#4CAF50",
            "Hello everyone, welcome to the meeting.",
            "Hola a todos, bienvenidos a la reunión.",
        ),
        ("Bob", "#2196F3", "Thanks for joining us today.", "Gracias por unirse hoy."),
        ("Alice", "#4CAF50", "Let's start with the agenda.", "Comencemos con la agenda."),
        (
            "Charlie",
            "#FF9800",
            "I have a question about the budget.",
            "Tengo una pregunta sobre el presupuesto.",
        ),
        ("Bob", "#2196F3", "Good question, let me explain.", "Buena pregunta, déjame explicar."),
        ("Alice", "#4CAF50", "The project is on track.", "El proyecto va por buen camino."),
        ("Charlie", "#FF9800", "When is the deadline?", "¿Cuándo es la fecha límite?"),
        ("Bob", "#2196F3", "Next Friday at noon.", "El próximo viernes al mediodía."),
        ("Alice", "#4CAF50", "Any other questions?", "¿Alguna otra pregunta?"),
        ("Charlie", "#FF9800", "No, I think we're good.", "No, creo que estamos bien."),
    ]

    base_url = f"http://{host}:{port}"

    try:
        async with aiohttp.ClientSession() as session:
            start_time = asyncio.get_event_loop().time()
            index = 0

            while asyncio.get_event_loop().time() - start_time < duration:
                speaker, color, original, translated = demo_conversations[
                    index % len(demo_conversations)
                ]

                await post_caption(
                    session,
                    base_url,
                    session_id,
                    text=translated,
                    original=original,
                    speaker=speaker,
                    color=color,
                    duration=4.0,
                )

                index += 1
                await asyncio.sleep(3)

            print("\n✅ Demo complete!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_multiple_speakers(session_id: str, host: str, port: int):
    """Test multiple speakers appearing simultaneously."""
    print("\n🎭 Testing multiple speakers...")
    print("\n   Open this URL in your browser:")
    print(f"   http://{host}:{port}/static/captions.html?session={session_id}&showStatus=true")

    base_url = f"http://{host}:{port}"

    input("\nPress Enter to send rapid captions from multiple speakers...")

    speakers = [
        ("Alice", "#4CAF50"),
        ("Bob", "#2196F3"),
        ("Charlie", "#FF9800"),
        ("Diana", "#E91E63"),
    ]

    async with aiohttp.ClientSession() as session:
        for i in range(8):
            speaker, color = speakers[i % len(speakers)]
            await post_caption(
                session,
                base_url,
                session_id,
                text=f"Message {i+1} from {speaker}",
                speaker=speaker,
                color=color,
                duration=6.0,
            )
            await asyncio.sleep(0.5)

        print("\n✅ Rapid captions sent!")
        input("\nPress Enter to exit...")


def main():
    parser = argparse.ArgumentParser(description="Test the HTML caption overlay")

    parser.add_argument("--session", default="test", help="Session ID")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=3000, help="Server port")

    parser.add_argument("--test", action="store_true", help="Run basic test")
    parser.add_argument("--demo", type=int, metavar="SECONDS", help="Run demo for N seconds")
    parser.add_argument("--multi", action="store_true", help="Test multiple speakers")

    args = parser.parse_args()

    print("=" * 60)
    print("  Caption Overlay Test - LiveTranslate")
    print("=" * 60)

    # Default to basic test
    if not any([args.test, args.demo, args.multi]):
        args.test = True

    if args.test:
        asyncio.run(test_basic(args.session, args.host, args.port))

    if args.demo:
        asyncio.run(run_demo(args.session, args.host, args.port, args.demo))

    if args.multi:
        asyncio.run(test_multiple_speakers(args.session, args.host, args.port))


if __name__ == "__main__":
    main()
