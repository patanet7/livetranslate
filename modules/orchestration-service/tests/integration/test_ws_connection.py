#!/usr/bin/env python3
"""
Simple WebSocket connection test to diagnose issues
"""
import asyncio
import httpx
import websockets
import json
import time

BASE_URL = "http://localhost:3000"
WS_BASE_URL = "ws://localhost:3000"

async def test_connection():
    """Test basic WebSocket connection"""
    print("🔍 Testing WebSocket connection...")

    # 1. Start a session
    print("\n1. Creating realtime session...")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        response = await client.post("/api/pipeline/realtime/start", json={
            "pipeline_config": {
                "pipeline_id": "test-pipeline",
                "name": "Test Pipeline",
                "stages": {
                    "vad": {
                        "enabled": True,
                        "gain_in": 0.0,
                        "gain_out": 0.0,
                        "parameters": {"aggressiveness": 2}
                    }
                },
                "connections": []
            }
        })
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        session_data = response.json()
        session_id = session_data.get("session_id") or session_data.get("data", {}).get("session_id")

        if not session_id:
            print(f"❌ No session_id in response: {session_data}")
            return

        print(f"   ✅ Session created: {session_id}")

    # 2. Connect WebSocket
    print(f"\n2. Connecting to WebSocket...")
    ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"
    print(f"   URL: {ws_url}")

    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"   ✅ WebSocket connected!")

            # 3. Send ping
            print("\n3. Sending ping...")
            await websocket.send(json.dumps({"type": "ping"}))
            print("   ✅ Ping sent")

            # 4. Receive pong
            print("\n4. Waiting for pong...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            message = json.loads(response)
            print(f"   ✅ Received: {message}")

            if message.get("type") == "pong":
                print("\n✅ SUCCESS: WebSocket connection is working!")
            elif message.get("type") == "error":
                print(f"\n❌ ERROR from backend: {message.get('error')}")
            else:
                print(f"\n⚠️  Unexpected response: {message}")

    except websockets.exceptions.ConnectionClosedOK as e:
        print(f"\n❌ Connection closed: {e}")
    except asyncio.TimeoutError:
        print(f"\n❌ Timeout waiting for response")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

    # 5. Cleanup
    print("\n5. Cleaning up session...")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        response = await client.delete(f"/api/pipeline/realtime/{session_id}")
        print(f"   Status: {response.status_code}")

if __name__ == "__main__":
    asyncio.run(test_connection())
