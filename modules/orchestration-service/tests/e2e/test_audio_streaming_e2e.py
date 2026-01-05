#!/usr/bin/env python3
"""
End-to-End Audio Streaming Test

Tests the complete audio pipeline:
1. Bot joins Google Meet
2. Captures audio-only
3. Streams to orchestration service
4. Forwards to Whisper service
5. Receives transcriptions

Usage:
    python test_audio_streaming_e2e.py https://meet.google.com/your-meeting-code
"""

import asyncio
import sys
import httpx
from datetime import datetime

# Test configuration
MEETING_BOT_SERVICE_URL = "http://localhost:5005"
ORCHESTRATION_SERVICE_URL = "http://localhost:3000"
TEST_MEETING_URL = (
    sys.argv[1] if len(sys.argv) > 1 else "https://meet.google.com/oss-kqzr-ztg"
)


async def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/api/health", timeout=5.0)
            if response.status_code == 200:
                print(f"✅ {service_name} is running")
                return True
            else:
                print(f"❌ {service_name} returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ {service_name} is not reachable: {e}")
        return False


async def join_meeting_with_audio_streaming(meeting_url: str) -> dict:
    """Join a Google Meet meeting with audio streaming enabled"""
    print(f"\n🎯 Joining meeting: {meeting_url}")

    bot_id = f"audio-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    request_data = {
        "meetingUrl": meeting_url,
        "botName": "LiveTranslate Audio Test",
        "botId": bot_id,
        "userId": "test-user-001",
        "teamId": "livetranslate-team",
        "orchestrationUrl": "ws://localhost:3000/api/audio/stream",  # Enable audio streaming
    }

    print(f"📤 Sending join request with audio streaming enabled...")
    print(f"   Bot ID: {bot_id}")
    print(f"   Orchestration URL: {request_data['orchestrationUrl']}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MEETING_BOT_SERVICE_URL}/api/bot/join", json=request_data
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Bot join request successful!")
                print(f"   Correlation ID: {result.get('correlationId')}")
                print(f"   Message: {result.get('message')}")
                return {"success": True, "botId": bot_id, "data": result}
            else:
                print(f"❌ Bot join failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return {"success": False, "error": response.text}

    except Exception as e:
        print(f"❌ Error joining meeting: {e}")
        return {"success": False, "error": str(e)}


async def check_bot_status(bot_id: str) -> dict:
    """Check the status of a bot"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEETING_BOT_SERVICE_URL}/api/bot/status/{bot_id}", timeout=5.0
            )

            if response.status_code == 200:
                status = response.json()
                return {"success": True, "status": status}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def monitor_audio_streaming(bot_id: str, duration_seconds: int = 30):
    """Monitor bot status and log audio streaming state"""
    print(f"\n🔍 Monitoring bot for {duration_seconds} seconds...")
    print("   Checking bot status every 5 seconds...")

    for i in range(duration_seconds // 5):
        await asyncio.sleep(5)

        status_result = await check_bot_status(bot_id)

        if status_result["success"]:
            bot_status = status_result["status"]
            state = bot_status.get("state", "unknown")
            print(f"   [{i * 5:2d}s] Bot state: {state}")

            if state == "streaming":
                print(f"      ✅ Audio streaming is ACTIVE!")
        else:
            print(
                f"   [{i * 5:2d}s] Could not get bot status: {status_result.get('error')}"
            )


async def leave_meeting(bot_id: str):
    """Tell the bot to leave the meeting"""
    print(f"\n🚪 Leaving meeting...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEETING_BOT_SERVICE_URL}/api/bot/leave/{bot_id}", timeout=10.0
            )

            if response.status_code == 200:
                print(f"✅ Bot left successfully")
                return True
            else:
                print(f"❌ Leave failed with status {response.status_code}")
                return False

    except Exception as e:
        print(f"❌ Error leaving meeting: {e}")
        return False


async def main():
    """Run the end-to-end audio streaming test"""
    print("=" * 70)
    print("🎙️  AUDIO STREAMING END-TO-END TEST")
    print("=" * 70)

    # Step 1: Check services are running
    print("\n📡 Checking services...")

    bot_service_ok = await check_service_health(
        "Meeting Bot Service", MEETING_BOT_SERVICE_URL
    )
    orch_service_ok = await check_service_health(
        "Orchestration Service", ORCHESTRATION_SERVICE_URL
    )

    if not bot_service_ok:
        print("\n❌ Meeting Bot Service is not running!")
        print("   Please start it with: cd modules/meeting-bot-service && npm run api")
        return

    if not orch_service_ok:
        print("\n⚠️  Orchestration Service is not running")
        print("   Audio will be streamed but may not be processed")
        print(
            "   You can start it with: cd modules/orchestration-service && poetry run python src/main_fastapi.py"
        )

    # Step 2: Join meeting with audio streaming
    join_result = await join_meeting_with_audio_streaming(TEST_MEETING_URL)

    if not join_result["success"]:
        print("\n❌ Test failed: Could not join meeting")
        return

    bot_id = join_result["botId"]

    # Step 3: Wait for bot to fully join
    print("\n⏳ Waiting 15 seconds for bot to fully join and start audio capture...")
    await asyncio.sleep(15)

    # Step 4: Monitor audio streaming
    await monitor_audio_streaming(bot_id, duration_seconds=30)

    # Step 5: Cleanup
    await leave_meeting(bot_id)

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print("\nWhat to check:")
    print("1. Bot should have joined the meeting")
    print("2. Bot state should show 'streaming' (indicates audio capture is active)")
    print(
        "3. Check meeting-bot-service logs for '[LiveTranslate] Audio streaming started successfully'"
    )
    print("4. Check orchestration-service logs for incoming audio chunks")
    print("\nTo view logs:")
    print("  - Bot service: Check the terminal where 'npm run api' is running")
    print("  - Orchestration: Check the terminal where main_fastapi.py is running")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  No meeting URL provided, using default test meeting")
        print(f"   Meeting: {TEST_MEETING_URL}")
        print(f"\nUsage: python {sys.argv[0]} <meeting-url>")
        print()

    asyncio.run(main())
