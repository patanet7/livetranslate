"""
Demo Manager

Singleton managing the Fireflies demo server lifecycle.
Starts a mock Fireflies server in-process so the full pipeline
(Socket.IO → orchestration → captions WebSocket → overlay) runs
without a real Fireflies account.
"""

import asyncio
import logging

from services.demo_server import FirefliesDemoServer, MockTranscriptScenario

logger = logging.getLogger(__name__)

DEMO_PORT = 8090
DEMO_API_KEY = "demo-api-key"  # pragma: allowlist secret


class DemoManager:
    """Manages the demo Fireflies server lifecycle."""

    def __init__(self):
        self.server: FirefliesDemoServer | None = None
        self.session_id: str | None = None
        self.transcript_id: str | None = None
        self.active: bool = False
        self._speakers: list[str] = []
        self._lock = asyncio.Lock()

    async def start(
        self,
        speakers: list[str] | None = None,
        num_exchanges: int = 30,
        chunk_delay_ms: float = 2000.0,
    ) -> dict:
        """
        Start the demo server with a conversation scenario.

        Returns dict with transcript_id, speakers, and base_url for
        the router to create a real Fireflies session against.
        """
        async with self._lock:
            if self.active:
                raise RuntimeError("Demo is already running")

            if speakers is None:
                speakers = ["Alice Chen", "Bob Martinez", "Charlie Kim"]

            self.server = FirefliesDemoServer(
                host="localhost",
                port=DEMO_PORT,
                valid_api_keys={DEMO_API_KEY},
            )

            scenario = MockTranscriptScenario.conversation_scenario(
                speakers=speakers,
                num_exchanges=num_exchanges,
                chunk_delay_ms=chunk_delay_ms,
            )

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

            logger.info(
                f"Demo started: transcript_id={self.transcript_id}, "
                f"speakers={speakers}, exchanges={num_exchanges}"
            )

            return {
                "transcript_id": self.transcript_id,
                "speakers": speakers,
                "base_url": self.server.base_url,
                "num_exchanges": num_exchanges,
                "chunk_delay_ms": chunk_delay_ms,
            }

    async def stop(self):
        """Stop the demo server and reset state."""
        async with self._lock:
            if self.server:
                try:
                    await self.server.stop()
                except BaseException as e:
                    logger.warning(f"Demo server stop error (ignored): {e}")
                self.server = None

            self.session_id = None
            self.transcript_id = None
            self._speakers = []
            self.active = False

            logger.info("Demo stopped")

    def get_status(self) -> dict:
        """Return current demo status."""
        return {
            "active": self.active,
            "session_id": self.session_id,
            "transcript_id": self.transcript_id,
            "speakers": self._speakers,
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
