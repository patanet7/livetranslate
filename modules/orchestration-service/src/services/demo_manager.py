"""
Demo Manager

Singleton managing the Fireflies demo server lifecycle.
Starts a mock Fireflies server in-process so the full pipeline
(Socket.IO → orchestration → captions WebSocket → overlay) runs
without a real Fireflies account.

Supports three modes:
- passthrough: full pipeline with canned conversation templates
- pretranslated: pre-translated Spanish captions injected directly into CaptionBuffer
- replay: replay a real captured JSONL file with original ASR timing + dedup
"""

import asyncio
import contextlib
from pathlib import Path

from livetranslate_common.logging import get_logger
from services.demo_server import FirefliesDemoServer, MockTranscriptScenario

logger = get_logger()

DEMO_PORT = 8090
DEMO_API_KEY = "demo-api-key"  # pragma: allowlist secret

# Captured meeting data for replay mode
# __file__ is src/services/demo_manager.py → parent.parent.parent = module root
_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent
CAPTURED_DATA_DIR = _MODULE_ROOT / "captured_data"
DEFAULT_CAPTURE_FILE = "20260224_190411_fireflies_raw_capture.jsonl"


class DemoManager:
    """Manages the demo Fireflies server lifecycle."""

    def __init__(self):
        self.server: FirefliesDemoServer | None = None
        self.session_id: str | None = None
        self.transcript_id: str | None = None
        self.mode: str = "passthrough"
        self.active: bool = False
        self._speakers: list[str] = []
        self._scenario: MockTranscriptScenario | None = None
        self._pretranslated_task: asyncio.Task | None = None
        self._injection_delay_ms: float = 2000.0
        self._lock = asyncio.Lock()

    async def start(
        self,
        speakers: list[str] | None = None,
        num_exchanges: int = 30,
        chunk_delay_ms: float = 2000.0,
        mode: str = "passthrough",
        speed_multiplier: float = 1.0,
    ) -> dict:
        """
        Start the demo server with a conversation scenario.

        Args:
            mode: "passthrough", "pretranslated", or "replay"
            speed_multiplier: For replay mode, speed up playback (2.0 = 2x faster)

        Returns dict with transcript_id, speakers, base_url, and mode.
        """
        async with self._lock:
            if self.active:
                raise RuntimeError("Demo is already running")

            if speakers is None:
                speakers = ["Alice Chen", "Bob Martinez", "Charlie Kim"]

            self.mode = mode

            self.server = FirefliesDemoServer(
                host="localhost",
                port=DEMO_PORT,
                valid_api_keys={DEMO_API_KEY},
            )

            if mode == "replay":
                # Replay a real captured meeting
                jsonl_path = CAPTURED_DATA_DIR / DEFAULT_CAPTURE_FILE
                if not jsonl_path.exists():
                    raise RuntimeError(
                        f"Captured data not found: {jsonl_path}. "
                        "Record a meeting first with capture_fireflies_raw.py"
                    )
                scenario = MockTranscriptScenario.replay_scenario(
                    jsonl_path=jsonl_path,
                    speed_multiplier=speed_multiplier,
                )
                # Extract speakers from the captured data
                speakers = sorted(
                    {c.speaker_name for c in scenario.chunks}
                )
            else:
                scenario = MockTranscriptScenario.conversation_scenario(
                    speakers=speakers,
                    num_exchanges=num_exchanges,
                    chunk_delay_ms=chunk_delay_ms,
                    include_translations=(mode == "pretranslated"),
                )

            self._scenario = scenario
            self._injection_delay_ms = chunk_delay_ms

            if mode == "pretranslated":
                # Only register the meeting for GraphQL — don't stream chunks
                # via Socket.IO. The injection task handles caption delivery.
                self.server._meetings.append(scenario.meeting)
            else:
                self.server.add_scenario(scenario)

            try:
                await self.server.start()
            except OSError as e:
                self.server = None
                raise RuntimeError(
                    f"Cannot start demo server on port {DEMO_PORT}: {e}. "
                    "Is another demo or process already using this port?"
                ) from e

            self.transcript_id = scenario.transcript_id
            self._speakers = speakers
            self.active = True

            num_events = len(scenario.chunks)
            logger.info(
                f"Demo started: mode={mode}, transcript_id={self.transcript_id}, "
                f"speakers={speakers}, events={num_events}"
            )

            return {
                "transcript_id": self.transcript_id,
                "speakers": speakers,
                "base_url": self.server.base_url,
                "num_exchanges": num_events,
                "chunk_delay_ms": chunk_delay_ms,
                "mode": mode,
            }

    def start_pretranslated_injection(self, caption_buffer, ws_manager) -> None:
        """Start background task to inject pre-translated captions.

        Called by the router after session creation when mode == "pretranslated".
        Reads chunks from the scenario and injects captions directly into the
        CaptionBuffer, bypassing the translation pipeline entirely.
        """
        if not self._scenario or self.mode != "pretranslated":
            return

        async def _inject():
            try:
                logger.info(
                    f"Pre-translated injection starting: "
                    f"session={self.session_id}, chunks={len(self._scenario.chunks)}, "
                    f"delay={self._scenario.chunk_delay_ms}ms"
                )
                for i, chunk in enumerate(self._scenario.chunks):
                    if not self.active:
                        logger.info(
                            f"Pre-translated injection stopped: demo no longer active (after {i} chunks)"
                        )
                        break

                    if chunk.translated_text:
                        caption, was_updated = caption_buffer.add_caption(
                            translated_text=chunk.translated_text,
                            speaker_name=chunk.speaker_name,
                            original_text=chunk.text,
                            target_language="es",
                            confidence=0.95,
                        )

                        # Broadcast to WebSocket clients (no language filter — send to all)
                        event_type = "caption_updated" if was_updated else "caption_added"
                        await ws_manager.broadcast_to_session(
                            self.session_id,
                            {
                                "event": event_type,
                                "caption": caption.to_dict(),
                            },
                        )

                        if i == 0:
                            logger.info(f"First caption injected: {chunk.speaker_name}")
                        elif i % 10 == 0:
                            logger.debug(f"Injected {i+1}/{len(self._scenario.chunks)} captions")

                    await asyncio.sleep(self._injection_delay_ms / 1000.0)

                logger.info("Pre-translated caption injection complete")
            except asyncio.CancelledError:
                logger.debug("Pre-translated injection cancelled")
            except Exception as e:
                logger.warning(f"Pre-translated injection error: {e}", exc_info=True)

        self._pretranslated_task = asyncio.create_task(_inject())

    def start_direct_replay(self, on_chunk) -> None:
        """Replay scenario chunks directly through a callback.

        Unlike pretranslated injection (which uses CaptionBuffer), this sends
        raw chunks to the provided callback — typically the SourceOrchestrator's
        on_caption or a WebSocket send function.

        Args:
            on_chunk: async callable(chunk_dict) receiving each replay chunk.
        """
        if not self._scenario:
            return

        async def _replay():
            try:
                chunks = self._scenario.chunks
                logger.info(
                    "direct_replay_starting",
                    chunks=len(chunks),
                    mode=self.mode,
                )
                last_delay = 0.0
                for i, chunk in enumerate(chunks):
                    if not self.active:
                        logger.info("direct_replay_stopped", after_chunks=i)
                        break

                    # Use replay timing if available
                    target_ms = getattr(chunk, "_replay_delay_ms", 0.0)
                    if target_ms > last_delay:
                        await asyncio.sleep((target_ms - last_delay) / 1000.0)
                    last_delay = target_ms

                    await on_chunk({
                        "text": chunk.text,
                        "speaker_name": chunk.speaker_name,
                        "chunk_id": chunk.chunk_id,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "translated_text": chunk.translated_text,
                    })

                    if i == 0:
                        logger.info("direct_replay_first_chunk", speaker=chunk.speaker_name)
                    elif i % 50 == 0:
                        logger.debug("direct_replay_progress", chunk=i, total=len(chunks))

                logger.info("direct_replay_complete", total_chunks=len(chunks))
            except asyncio.CancelledError:
                logger.debug("direct_replay_cancelled")
            except Exception as e:
                logger.warning("direct_replay_error", error=str(e), exc_info=True)

        self._pretranslated_task = asyncio.create_task(_replay())

    async def stop(self):
        """Stop the demo server and reset state."""
        async with self._lock:
            # Cancel pretranslated injection
            if self._pretranslated_task and not self._pretranslated_task.done():
                self._pretranslated_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._pretranslated_task
                self._pretranslated_task = None

            if self.server:
                try:
                    await self.server.stop()
                except BaseException as e:
                    logger.warning(f"Demo server stop error (ignored): {e}")
                self.server = None

            self.session_id = None
            self.transcript_id = None
            self._speakers = []
            self._scenario = None
            self._injection_delay_ms = 2000.0
            self.mode = "passthrough"
            self.active = False

            logger.info("Demo stopped")

    def get_status(self) -> dict:
        """Return current demo status."""
        return {
            "active": self.active,
            "session_id": self.session_id,
            "transcript_id": self.transcript_id,
            "speakers": self._speakers,
            "mode": self.mode,
            "server_stats": self.server.stats if self.server else None,
        }


# Module-level singleton
_demo_manager: DemoManager | None = None


def get_demo_manager() -> DemoManager:
    """Get or create the DemoManager singleton."""
    global _demo_manager
    if _demo_manager is None:
        _demo_manager = DemoManager()
    return _demo_manager
