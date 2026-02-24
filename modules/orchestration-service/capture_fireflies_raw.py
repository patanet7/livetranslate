#!/usr/bin/env python3
"""
Fireflies Raw Data Capture Script

Connects to a live Fireflies meeting and dumps ALL raw Socket.IO events
to both console and a timestamped JSON log file.

Purpose: See exactly what data Fireflies sends in real-time before any
deduplication or processing.

Usage:
    uv run python capture_fireflies_raw.py
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import socketio

# ── Config ───────────────────────────────────────────────────────────
API_KEY = os.getenv("FIREFLIES_API_KEY", "***REDACTED***")
GRAPHQL_ENDPOINT = "https://api.fireflies.ai/graphql"
SOCKETIO_ENDPOINT = "wss://api.fireflies.ai"
SOCKETIO_PATH = "/ws/realtime"

# Output directory
OUTPUT_DIR = Path("captured_data")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
RAW_LOG_FILE = OUTPUT_DIR / f"{timestamp}_fireflies_raw_capture.jsonl"
SUMMARY_FILE = OUTPUT_DIR / f"{timestamp}_fireflies_capture_summary.json"

# ── Counters ─────────────────────────────────────────────────────────
stats = {
    "started_at": datetime.now(UTC).isoformat(),
    "total_events": 0,
    "events_by_type": {},
    "unique_chunk_ids": set(),
    "unique_speakers": set(),
    "chunks_by_speaker": {},
    "first_chunk_at": None,
    "last_chunk_at": None,
}


def log_raw(event_name: str, data, *, is_transcript: bool = False):
    """Log a raw event to console and JSONL file."""
    now = datetime.now(UTC)
    stats["total_events"] += 1
    stats["events_by_type"][event_name] = stats["events_by_type"].get(event_name, 0) + 1

    record = {
        "timestamp": now.isoformat(),
        "event": event_name,
        "data": data,
    }

    # Write to JSONL file (one JSON object per line)
    with open(RAW_LOG_FILE, "a") as f:
        # Convert sets to lists for JSON serialization
        f.write(json.dumps(record, default=str) + "\n")

    # Console output
    if is_transcript:
        # Parse transcript data for readable display
        chunk_data = data
        if isinstance(data, dict):
            chunk_data = data.get("payload", data.get("data", data))

        chunk_id = "?"
        text = ""
        speaker = "?"
        start_time = 0
        end_time = 0

        if isinstance(chunk_data, dict):
            chunk_id = chunk_data.get("chunk_id", chunk_data.get("id", "?"))
            text = chunk_data.get("text", chunk_data.get("content", ""))
            speaker = chunk_data.get("speaker_name", chunk_data.get("speaker", "?"))
            start_time = chunk_data.get("start_time", chunk_data.get("startTime", 0))
            end_time = chunk_data.get("end_time", chunk_data.get("endTime", 0))

            # Track stats
            stats["unique_chunk_ids"].add(str(chunk_id))
            stats["unique_speakers"].add(str(speaker))
            stats["chunks_by_speaker"][str(speaker)] = (
                stats["chunks_by_speaker"].get(str(speaker), 0) + 1
            )
            if stats["first_chunk_at"] is None:
                stats["first_chunk_at"] = now.isoformat()
            stats["last_chunk_at"] = now.isoformat()

        print(
            f"\n{'='*70}\n"
            f"[{now.strftime('%H:%M:%S.%f')[:-3]}] TRANSCRIPT EVENT: {event_name}\n"
            f"  chunk_id:    {chunk_id}\n"
            f"  speaker:     {speaker}\n"
            f"  time:        {start_time} -> {end_time}\n"
            f"  text:        \"{text}\"\n"
            f"  RAW KEYS:    {list(chunk_data.keys()) if isinstance(chunk_data, dict) else type(chunk_data).__name__}\n"
            f"{'='*70}"
        )
        # Also dump the full raw data structure (for first few)
        if stats["total_events"] <= 10:
            print(f"  FULL RAW DATA:\n{json.dumps(data, indent=2, default=str)}")
    else:
        print(f"[{now.strftime('%H:%M:%S.%f')[:-3]}] EVENT: {event_name} -> {json.dumps(data, default=str)[:200]}")


def save_summary():
    """Save capture summary."""
    summary = {
        **stats,
        "unique_chunk_ids": list(stats["unique_chunk_ids"]),
        "unique_speakers": list(stats["unique_speakers"]),
        "ended_at": datetime.now(UTC).isoformat(),
        "raw_log_file": str(RAW_LOG_FILE),
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\nSummary saved to: {SUMMARY_FILE}")
    print(f"Raw log saved to:  {RAW_LOG_FILE}")
    print(f"Total events:      {stats['total_events']}")
    print(f"Unique chunks:     {len(stats['unique_chunk_ids'])}")
    print(f"Unique speakers:   {list(stats['unique_speakers'])}")
    print(f"Events by type:    {stats['events_by_type']}")


# ── Step 1: Find active meeting ──────────────────────────────────────
async def find_active_meeting() -> str | None:
    """Query Fireflies GraphQL for active meetings, return first transcript ID."""
    query = """
    query ActiveMeetings {
      active_meetings(input: {}) {
        id
        title
        organizer_email
        meeting_link
        start_time
        end_time
        privacy
        state
      }
    }
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            GRAPHQL_ENDPOINT,
            json={"query": query},
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        ) as resp:
            result = await resp.json()
            print(f"\n--- GraphQL Response (active_meetings) ---")
            print(json.dumps(result, indent=2, default=str))
            log_raw("graphql_active_meetings", result)

            if "errors" in result:
                print(f"GraphQL error: {result['errors']}")
                return None

            meetings = result.get("data", {}).get("active_meetings", [])
            if not meetings:
                print("No active meetings found!")
                return None

            # Show all meetings and pick first
            for i, m in enumerate(meetings):
                print(f"\n  [{i}] Meeting: {m.get('title', 'Untitled')}")
                print(f"      ID: {m['id']}")
                print(f"      State: {m.get('state')}")
                print(f"      Link: {m.get('meeting_link')}")
                print(f"      Organizer: {m.get('organizer_email')}")
                print(f"      Start: {m.get('start_time')}")

            transcript_id = meetings[0]["id"]
            print(f"\n>>> Connecting to meeting: {meetings[0].get('title', transcript_id)}")
            return transcript_id


# ── Step 2: Connect Socket.IO and capture everything ─────────────────
async def capture_meeting(transcript_id: str):
    """Connect to Fireflies Socket.IO and log ALL events."""
    sio = socketio.AsyncClient(
        reconnection=True,
        reconnection_attempts=5,
        logger=False,
        engineio_logger=False,
    )

    # ── Catch-all handler: log EVERY event ──
    @sio.on("*")
    async def catch_all(event, data=None):
        is_transcript = event in ("transcription.broadcast", "transcript")
        log_raw(event, data, is_transcript=is_transcript)

    # ── Named handlers for known events ──
    @sio.event
    async def connect():
        log_raw("connect", {"status": "connected"})
        print("\n>>> Socket.IO CONNECTED to Fireflies!")

    @sio.event
    async def disconnect():
        log_raw("disconnect", {"status": "disconnected"})
        print("\n>>> Socket.IO DISCONNECTED")

    @sio.event
    async def connect_error(data):
        log_raw("connect_error", data)
        print(f"\n>>> CONNECTION ERROR: {data}")

    @sio.on("auth.success")
    async def on_auth_success(data=None):
        log_raw("auth.success", data)
        print("\n>>> AUTH SUCCESS!")

    @sio.on("auth.failed")
    async def on_auth_failed(data=None):
        log_raw("auth.failed", data)
        print(f"\n>>> AUTH FAILED: {data}")

    @sio.on("connection.established")
    async def on_conn_established(data=None):
        log_raw("connection.established", data)
        print("\n>>> CONNECTION ESTABLISHED - Ready for transcripts!")

    @sio.on("connection.error")
    async def on_conn_error(data=None):
        log_raw("connection.error", data)
        print(f"\n>>> CONNECTION ERROR: {data}")

    @sio.on("transcription.broadcast")
    async def on_transcript(data):
        log_raw("transcription.broadcast", data, is_transcript=True)

    @sio.on("transcript")
    async def on_transcript_alt(data):
        log_raw("transcript", data, is_transcript=True)

    # Auth payload
    auth = {
        "token": f"Bearer {API_KEY}",
        "transcriptId": transcript_id,
    }

    print(f"\n>>> Connecting to Socket.IO...")
    print(f"    Endpoint:      {SOCKETIO_ENDPOINT}")
    print(f"    Path:          {SOCKETIO_PATH}")
    print(f"    Transcript ID: {transcript_id}")
    print(f"    Logging to:    {RAW_LOG_FILE}")
    print(f"\n>>> Press Ctrl+C to stop capture and save summary.\n")

    try:
        await sio.connect(
            SOCKETIO_ENDPOINT,
            socketio_path=SOCKETIO_PATH,
            auth=auth,
            transports=["websocket"],
            wait_timeout=30,
        )

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            # Print periodic stats every 30 seconds
            if stats["total_events"] > 0 and stats["total_events"] % 50 == 0:
                print(
                    f"\n--- Stats: {stats['total_events']} events, "
                    f"{len(stats['unique_chunk_ids'])} chunks, "
                    f"speakers: {list(stats['unique_speakers'])} ---"
                )

    except KeyboardInterrupt:
        print("\n\n>>> Stopping capture...")
    except Exception as e:
        print(f"\n>>> Error: {e}")
        log_raw("error", {"error": str(e), "type": type(e).__name__})
    finally:
        if sio.connected:
            await sio.disconnect()
        save_summary()


# ── Main ─────────────────────────────────────────────────────────────
async def main():
    print("=" * 70)
    print("  FIREFLIES RAW DATA CAPTURE")
    print(f"  Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Output: {RAW_LOG_FILE}")
    print("=" * 70)

    # Allow passing transcript_id directly as argument
    if len(sys.argv) > 1:
        transcript_id = sys.argv[1]
        print(f"\n>>> Using provided transcript ID: {transcript_id}")
    else:
        # Query for active meetings
        transcript_id = await find_active_meeting()

    if not transcript_id:
        print("\nNo meeting to capture. You can also pass a transcript ID as argument:")
        print(f"  uv run python {sys.argv[0]} <transcript_id>")
        return

    await capture_meeting(transcript_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        save_summary()
        print("\nCapture ended.")
