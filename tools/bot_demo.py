#!/usr/bin/env python3
"""Bot Demo — replay captured meeting data through the subtitle pipeline.

Usage:
    uv run python tools/bot_demo.py
    just bot-demo

Connects to the orchestration WebSocket, starts a demo replay of real
captured Fireflies data (983 events, 2 speakers), and prints captions
as they flow. Opens the overlay URL for browser viewing.
"""

import asyncio
import json
import signal
import sys

import websockets


async def run():
    uri = "ws://localhost:3000/api/audio/stream"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        sid = msg["session_id"]

        await ws.send(json.dumps({"type": "start_session", "sample_rate": 16000, "channels": 1}))
        await asyncio.sleep(1)
        await ws.send(json.dumps({"type": "chat_command", "command": "/demo replay", "sender": "Demo"}))

        print()
        print("Demo running!")
        print(f"  Overlay: http://localhost:5173/captions?session={sid}")
        print(f"  Press Ctrl+C to stop")
        print()

        try:
            count = 0
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if m.get("event") == "caption_added":
                    cap = m["caption"]
                    count += 1
                    speaker = cap.get("speaker_name", "?")
                    text = cap.get("text", "")[:70]
                    print(f"  [{speaker}] {text}")
                elif m.get("type") == "chat_response":
                    print(f"  Bot: {m.get('text', '')}")
        except asyncio.TimeoutError:
            print(f"\nDemo finished ({count} captions).")
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\nStopping demo ({count} captions)...")
            await ws.send(json.dumps({"type": "chat_command", "command": "/demo stop", "sender": "Demo"}))
            print("Demo stopped.")


def main():
    # Handle Ctrl+C gracefully
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
