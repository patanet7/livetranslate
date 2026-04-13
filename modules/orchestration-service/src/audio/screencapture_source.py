"""ScreenCaptureAudioSource — subprocess wrapper for livetranslate-capture CLI.

Spawns the ``livetranslate-capture`` binary, reads raw f32le PCM from its
stdout, converts to numpy float32 arrays, and delivers chunks via the
``on_audio`` callback.  A watchdog timer detects hung processes and triggers
``on_error``.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import struct
import threading
import time
from collections.abc import Callable
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPTURE_BINARY = "livetranslate-capture"
CHUNK_SIZE = 16384          # 4096 samples × 4 bytes (f32le)
WATCHDOG_TIMEOUT = 5.0      # seconds before a silent process is considered hung


class ScreenCaptureAudioSource:
    """Wraps ``livetranslate-capture`` as an audio source.

    Parameters
    ----------
    on_audio:
        Callback invoked with each ``numpy.ndarray`` chunk (float32, 1-D).
    on_error:
        Optional callback invoked with an ``Exception`` when the process
        exits unexpectedly or a watchdog timeout fires.
    sample_rate:
        Expected sample rate of the captured audio (informational; not
        validated against the binary's output).
    """

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        sample_rate: int = 16000,
    ) -> None:
        self._on_audio = on_audio
        self._on_error = on_error
        self.sample_rate = sample_rate

        self._process: Optional[asyncio.subprocess.Process] = None
        self._read_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._last_read_time: float = 0.0
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if ``livetranslate-capture`` is found in PATH."""
        return shutil.which(CAPTURE_BINARY) is not None

    async def start(self) -> None:
        """Spawn the capture process and begin reading audio.

        Raises
        ------
        RuntimeError
            If the binary is not found in PATH or the process fails to start.
        """
        if not self.is_available():
            exc = RuntimeError(
                f"'{CAPTURE_BINARY}' binary not found in PATH. "
                "Install livetranslate-capture and ensure it is on PATH."
            )
            if self._on_error:
                self._on_error(exc)
            raise exc

        self._process = await asyncio.create_subprocess_exec(
            CAPTURE_BINARY,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._running = True
        self._last_read_time = time.monotonic()
        self._stop_event.clear()

        # Start watchdog on a daemon thread so it doesn't block shutdown.
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="screencapture-watchdog"
        )
        self._watchdog_thread.start()

        # Schedule read loop as an asyncio task.
        self._read_task = asyncio.get_event_loop().create_task(
            self._read_loop(), name="screencapture-read"
        )

    async def stop(self) -> None:
        """Gracefully shut down the capture process.

        Idempotent — safe to call multiple times.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Cancel and drain the read task.
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._read_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Terminate the subprocess.
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()

        # Wait for watchdog thread to exit.
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

        self._process = None
        self._read_task = None
        self._watchdog_thread = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Continuously read PCM bytes from the process stdout."""
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            while self._running:
                raw = await self._process.stdout.read(CHUNK_SIZE)
                if not raw:
                    # EOF — process exited.
                    if self._running:
                        exc = RuntimeError(
                            f"'{CAPTURE_BINARY}' exited unexpectedly (EOF on stdout)."
                        )
                        if self._on_error:
                            self._on_error(exc)
                    break

                self._last_read_time = time.monotonic()

                # Parse f32le samples.
                n_samples = len(raw) // 4
                if n_samples == 0:
                    continue

                samples = struct.unpack_from(f"{n_samples}f", raw, 0)
                chunk = np.array(samples, dtype=np.float32)
                self._on_audio(chunk)

        except asyncio.CancelledError:
            pass  # Normal shutdown path.
        except Exception as exc:  # noqa: BLE001
            if self._running and self._on_error:
                self._on_error(exc)

    def _watchdog_loop(self) -> None:
        """Run on a background thread; fires ``on_error`` if stdout goes silent."""
        while not self._stop_event.wait(timeout=1.0):
            if not self._running:
                break
            elapsed = time.monotonic() - self._last_read_time
            if elapsed > WATCHDOG_TIMEOUT:
                exc = RuntimeError(
                    f"'{CAPTURE_BINARY}' watchdog timeout: no audio received for "
                    f"{elapsed:.1f}s (threshold: {WATCHDOG_TIMEOUT}s)."
                )
                if self._on_error:
                    self._on_error(exc)
                # Only fire once; stop the source to avoid repeated callbacks.
                self._running = False
                break
