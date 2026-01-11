#!/usr/bin/env python3
"""
Quick Google Meet Bot Test

A simplified script to quickly test if a bot can log into Google Meet.
This script:
1. Checks if services are running
2. Starts a bot with a test meeting URL
3. Shows real-time status updates
4. Provides clear success/failure feedback

Usage:
    python quick_bot_test.py

Or provide a custom meeting URL:
    python quick_bot_test.py --url https://meet.google.com/your-meeting-code
"""

import asyncio
import argparse
import httpx
import sys
import time

# ANSI colors for better output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


async def check_service(url: str) -> bool:
    """Check if orchestration service is available"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{url}/api/health")
            return response.status_code == 200
    except:
        return False


async def start_bot(base_url: str, meeting_url: str) -> str:
    """Start a bot and return connection ID"""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url}/api/bots/start",
            json={
                "meeting_url": meeting_url,
                "user_token": "quick-test-token",
                "user_id": "quick-test-user",
                "language": "en",
                "task": "transcribe",
                "enable_virtual_webcam": False,
                "metadata": {"test": True, "quick_test": True},
            },
        )

        if response.status_code == 200:
            return response.json()["connection_id"]
        else:
            raise Exception(f"Failed to start bot: {response.text}")


async def get_status(base_url: str, connection_id: str) -> dict:
    """Get bot status"""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url}/api/bots/status/{connection_id}")
        if response.status_code == 200:
            return response.json()
        return None


async def monitor_bot(base_url: str, connection_id: str, max_time: int = 60):
    """Monitor bot status and show progress"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}Monitoring Bot: {connection_id}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_time:
        status = await get_status(base_url, connection_id)

        if not status:
            print(f"{RED}❌ Lost connection to bot{RESET}")
            return False

        current_status = status["status"]

        # Show status changes
        if current_status != last_status:
            timestamp = time.strftime("%H:%M:%S")

            if current_status == "spawning":
                print(
                    f"{timestamp} | {YELLOW}🔄 SPAWNING{RESET} - Bot container starting..."
                )
            elif current_status == "starting":
                print(f"{timestamp} | {YELLOW}🚀 STARTING{RESET} - Bot initializing...")
            elif current_status == "joining":
                print(
                    f"{timestamp} | {YELLOW}🚪 JOINING{RESET} - Bot joining Google Meet..."
                )
            elif current_status == "active":
                print(f"{timestamp} | {GREEN}✅ ACTIVE{RESET} - Bot is in the meeting!")
                print(
                    f"\n{GREEN}{BOLD}SUCCESS! Bot successfully joined Google Meet!{RESET}\n"
                )

                # Show additional info
                print(f"{BLUE}Bot Details:{RESET}")
                print(f"  • Container: {status.get('container_name', 'N/A')}")
                print(f"  • Uptime: {status.get('uptime_seconds', 0):.1f}s")
                print(f"  • Healthy: {status.get('is_healthy', False)}")
                return True
            elif current_status == "failed":
                error = status.get("error_message", "Unknown error")
                print(f"{timestamp} | {RED}❌ FAILED{RESET} - {error}")
                return False
            elif current_status == "completed":
                print(f"{timestamp} | {GREEN}✅ COMPLETED{RESET} - Bot finished")
                return True

            last_status = current_status

        await asyncio.sleep(2)

    print(f"\n{YELLOW}⏱️  Timeout reached ({max_time}s){RESET}")
    print(f"Final status: {last_status}")
    return last_status == "active"


async def main():
    parser = argparse.ArgumentParser(description="Quick Google Meet bot test")
    parser.add_argument(
        "--url",
        default="https://meet.google.com/test-bot-meeting",
        help="Google Meet URL (default: test URL)",
    )
    parser.add_argument(
        "--service",
        default="http://localhost:3000",
        help="Orchestration service URL (default: localhost:3000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Maximum time to wait in seconds (default: 90)",
    )

    args = parser.parse_args()

    print(f"{BLUE}{BOLD}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           GOOGLE MEET BOT - QUICK LOGIN TEST                      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")

    # Step 1: Check service
    print(f"{BLUE}🔍 Checking orchestration service...{RESET}")
    if not await check_service(args.service):
        print(f"{RED}❌ Orchestration service not available at {args.service}{RESET}")
        print(f"{YELLOW}💡 Tip: Make sure the service is running with:{RESET}")
        print("   cd modules/orchestration-service")
        print("   python src/main.py")
        sys.exit(1)

    print(f"{GREEN}✅ Orchestration service is running{RESET}\n")

    # Step 2: Start bot
    print(f"{BLUE}🚀 Starting bot for meeting: {args.url}{RESET}\n")
    try:
        connection_id = await start_bot(args.service, args.url)
        print(f"{GREEN}✅ Bot started with ID: {connection_id}{RESET}")
    except Exception as e:
        print(f"{RED}❌ Failed to start bot: {e}{RESET}")
        sys.exit(1)

    # Step 3: Monitor
    success = await monitor_bot(args.service, connection_id, args.timeout)

    # Final result
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    if success:
        print(f"{GREEN}{BOLD}✅ TEST PASSED{RESET}")
        print(f"{GREEN}The bot successfully logged into Google Meet!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}⚠️  TEST INCOMPLETE{RESET}")
        print(f"{YELLOW}The bot started but didn't reach active state.{RESET}")
        print(f"{YELLOW}This might be normal if the meeting requires approval.{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Test interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}❌ Unexpected error: {e}{RESET}")
        sys.exit(1)
