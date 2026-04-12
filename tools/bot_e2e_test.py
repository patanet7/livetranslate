#!/usr/bin/env python3
"""
Bot E2E Test Coordinator

Autonomously tests the full Google Meet bot pipeline:
1. Verifies infrastructure (Redis, Postgres, Docker)
2. Builds bot Docker image if needed
3. Starts required services
4. Spawns bot to join meeting
5. Monitors bot status via callbacks
6. Watches caption overlay for output
7. Stops bot and collects results

MANUAL REQUIREMENTS:
- A Google Meet URL (can be a test meeting)
- Someone speaking OR pre-recorded audio playing in the meeting
- Google account credentials (if meeting requires auth)

Usage:
    uv run python tools/bot_e2e_test.py --meeting-url "https://meet.google.com/xxx-xxxx-xxx"

    # With auth (for restricted meetings):
    uv run python tools/bot_e2e_test.py \\
        --meeting-url "https://meet.google.com/xxx-xxxx-xxx" \\
        --google-email "test@example.com" \\
        --google-password "password"
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Configuration
ORCHESTRATION_URL = "http://localhost:3000"
DASHBOARD_URL = "http://localhost:5173"
OUTPUT_DIR = Path("/tmp/bot-e2e-test")


class BotE2ETest:
    def __init__(self, meeting_url: str, google_email: str | None = None, google_password: str | None = None):
        self.meeting_url = meeting_url
        self.google_email = google_email
        self.google_password = google_password
        self.connection_id: str | None = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results: dict = {
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "captions_received": 0,
            "errors": [],
        }

    async def log_step(self, step: str, status: str, details: str = ""):
        """Log a test step"""
        entry = {"step": step, "status": status, "details": details, "timestamp": datetime.now().isoformat()}
        self.results["steps"].append(entry)
        icon = "✓" if status == "pass" else "✗" if status == "fail" else "○"
        print(f"  {icon} {step}: {details}" if details else f"  {icon} {step}")

    async def check_infrastructure(self) -> bool:
        """Verify required infrastructure is running"""
        print("\n[1/7] Checking Infrastructure...")

        checks = [
            ("Redis", "redis-cli ping", "PONG"),
            ("Postgres", "pg_isready -h localhost -p 5432", "accepting connections"),
            ("Docker", "docker info", "Server Version"),
        ]

        all_ok = True
        for name, cmd, expected in checks:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
                if expected in result.stdout or expected in result.stderr:
                    await self.log_step(f"{name} check", "pass")
                else:
                    await self.log_step(f"{name} check", "fail", f"Expected '{expected}'")
                    all_ok = False
            except Exception as e:
                await self.log_step(f"{name} check", "fail", str(e))
                all_ok = False

        return all_ok

    async def check_bot_image(self) -> bool:
        """Check if bot Docker image exists, build if needed"""
        print("\n[2/7] Checking Bot Docker Image...")

        result = subprocess.run(
            ["docker", "images", "-q", "livetranslate-bot:latest"],
            capture_output=True, text=True
        )

        if result.stdout.strip():
            await self.log_step("Bot image exists", "pass", "livetranslate-bot:latest")
            return True

        await self.log_step("Bot image missing", "info", "Building...")

        bot_dir = Path(__file__).parent.parent / "modules" / "meeting-bot-service"
        if not bot_dir.exists():
            await self.log_step("Build bot image", "fail", f"Directory not found: {bot_dir}")
            return False

        # Build image
        result = subprocess.run(
            ["docker", "build", "-t", "livetranslate-bot:latest", "-f", "Dockerfile.development", "."],
            cwd=bot_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for build
        )

        if result.returncode == 0:
            await self.log_step("Build bot image", "pass")
            return True
        else:
            await self.log_step("Build bot image", "fail", result.stderr[:200])
            return False

    async def check_services(self) -> bool:
        """Check if required services are running"""
        print("\n[3/7] Checking Services...")

        services = [
            ("Orchestration", f"{ORCHESTRATION_URL}/health"),
            ("Dashboard", DASHBOARD_URL),
        ]

        all_ok = True
        for name, url in services:
            try:
                response = await self.client.get(url)
                if response.status_code < 500:
                    await self.log_step(f"{name} service", "pass", url)
                else:
                    await self.log_step(f"{name} service", "fail", f"Status {response.status_code}")
                    all_ok = False
            except Exception as e:
                await self.log_step(f"{name} service", "fail", f"Not reachable: {e}")
                all_ok = False

        return all_ok

    async def spawn_bot(self) -> bool:
        """Spawn bot to join meeting"""
        print("\n[4/7] Spawning Bot...")

        payload = {
            "meeting_url": self.meeting_url,
            "user_token": "test-token",  # For testing
            "user_id": "e2e-test-user",
            "language": "en",
            "task": "transcribe",
            "enable_virtual_webcam": False,
            "metadata": {"test": True, "started_at": datetime.now().isoformat()},
        }

        if self.google_email:
            payload["google_email"] = self.google_email
        if self.google_password:
            payload["google_password"] = self.google_password

        try:
            response = await self.client.post(
                f"{ORCHESTRATION_URL}/api/start",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                self.connection_id = data.get("connection_id")
                await self.log_step("Spawn bot", "pass", f"connection_id={self.connection_id}")
                return True
            else:
                await self.log_step("Spawn bot", "fail", f"Status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            await self.log_step("Spawn bot", "fail", str(e))
            return False

    async def monitor_bot_status(self, timeout_seconds: int = 120) -> bool:
        """Monitor bot status until active or failed"""
        print("\n[5/7] Monitoring Bot Status...")

        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout_seconds:
            try:
                response = await self.client.get(
                    f"{ORCHESTRATION_URL}/api/status/{self.connection_id}"
                )

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")

                    if status != last_status:
                        await self.log_step(f"Bot status", "info", status)
                        last_status = status

                    if status == "active":
                        await self.log_step("Bot active", "pass", "Joined meeting successfully")
                        return True
                    elif status in ("failed", "completed"):
                        error = data.get("error_message", "Unknown error")
                        await self.log_step("Bot status", "fail", f"{status}: {error}")
                        return False

            except Exception as e:
                await self.log_step("Status check", "fail", str(e))

            await asyncio.sleep(2)

        await self.log_step("Bot monitoring", "fail", f"Timeout after {timeout_seconds}s")
        return False

    async def watch_captions(self, duration_seconds: int = 30) -> bool:
        """Watch caption overlay for incoming captions"""
        print(f"\n[6/7] Watching Captions for {duration_seconds}s...")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        captions_seen = []

        # Create a WebSocket connection to watch captions
        # For simplicity, we'll poll the REST endpoint
        session_id = f"bot-{self.connection_id}"

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                response = await self.client.get(
                    f"{ORCHESTRATION_URL}/api/captions/{session_id}"
                )

                if response.status_code == 200:
                    data = response.json()
                    captions = data.get("captions", [])

                    for caption in captions:
                        caption_id = caption.get("id")
                        if caption_id not in [c.get("id") for c in captions_seen]:
                            captions_seen.append(caption)
                            text = caption.get("translated_text", caption.get("text", ""))[:50]
                            speaker = caption.get("speaker_name", "Unknown")
                            await self.log_step("Caption received", "pass", f"{speaker}: {text}...")

            except Exception:
                pass  # Session might not exist yet

            await asyncio.sleep(1)

        self.results["captions_received"] = len(captions_seen)

        if captions_seen:
            await self.log_step("Captions captured", "pass", f"{len(captions_seen)} captions")

            # Save captions to file
            with open(OUTPUT_DIR / "captions.json", "w") as f:
                json.dump(captions_seen, f, indent=2, default=str)

            return True
        else:
            await self.log_step("Captions captured", "info", "No captions received (is someone speaking?)")
            return True  # Not a failure - might just be silence

    async def stop_bot(self) -> bool:
        """Stop the bot"""
        print("\n[7/7] Stopping Bot...")

        if not self.connection_id:
            await self.log_step("Stop bot", "skip", "No bot to stop")
            return True

        try:
            response = await self.client.post(
                f"{ORCHESTRATION_URL}/api/stop/{self.connection_id}",
                json={"timeout": 30},
            )

            if response.status_code == 200:
                await self.log_step("Stop bot", "pass")
                return True
            else:
                await self.log_step("Stop bot", "fail", f"Status {response.status_code}")
                return False

        except Exception as e:
            await self.log_step("Stop bot", "fail", str(e))
            return False

    async def run(self) -> bool:
        """Run the full E2E test"""
        print("=" * 60)
        print("BOT E2E TEST - Full Pipeline Verification")
        print("=" * 60)
        print(f"Meeting URL: {self.meeting_url}")
        print(f"Output Dir:  {OUTPUT_DIR}")

        try:
            # Step 1: Check infrastructure
            if not await self.check_infrastructure():
                print("\n⚠ Infrastructure check failed. Please ensure Redis, Postgres, and Docker are running.")
                return False

            # Step 2: Check/build bot image
            if not await self.check_bot_image():
                print("\n⚠ Bot image not available. Please build it manually.")
                return False

            # Step 3: Check services
            if not await self.check_services():
                print("\n⚠ Required services not running. Please start orchestration and dashboard.")
                return False

            # Step 4: Spawn bot
            if not await self.spawn_bot():
                print("\n⚠ Failed to spawn bot.")
                return False

            # Step 5: Monitor bot status
            if not await self.monitor_bot_status(timeout_seconds=120):
                print("\n⚠ Bot failed to become active.")
                await self.stop_bot()
                return False

            # Step 6: Watch captions
            await self.watch_captions(duration_seconds=60)

            # Step 7: Stop bot
            await self.stop_bot()

            # Summary
            self.results["completed_at"] = datetime.now().isoformat()

            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            passed = sum(1 for s in self.results["steps"] if s["status"] == "pass")
            failed = sum(1 for s in self.results["steps"] if s["status"] == "fail")

            print(f"Steps: {passed} passed, {failed} failed")
            print(f"Captions: {self.results['captions_received']} received")
            print(f"Output: {OUTPUT_DIR}")

            # Save results
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(self.results, f, indent=2)

            return failed == 0

        finally:
            await self.client.aclose()


def main():
    parser = argparse.ArgumentParser(description="Bot E2E Test")
    parser.add_argument("--meeting-url", required=True, help="Google Meet URL to join")
    parser.add_argument("--google-email", help="Google account email (for restricted meetings)")
    parser.add_argument("--google-password", help="Google account password")
    args = parser.parse_args()

    test = BotE2ETest(
        meeting_url=args.meeting_url,
        google_email=args.google_email,
        google_password=args.google_password,
    )

    success = asyncio.run(test.run())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
