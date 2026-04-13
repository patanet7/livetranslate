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
from livetranslate_common.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPTURE_BINARY_NAME_NAME = "livetranslate-capture"
CHUNK_SIZE = 16384          # 4096 samples × 4 bytes (f32le)
WATCHDOG_TIMEOUT = 5.0      # seconds before a silent process is considered hung


def _find_capture_binary() -> str | None:
    """Find capture binary - check project bin/ first, then PATH."""
    from pathlib import Path

    # Check project-local bin/ (relative to this file's location)
    # This file: modules/orchestration-service/src/audio/screencapture_source.py
    # Project root: 4 levels up
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[4]
    local_binary = project_root / "bin" / CAPTURE_BINARY_NAME_NAME

    if local_binary.exists() and local_binary.is_file():
        logger.debug(
            "screencapture_binary_found",
            path=str(local_binary),
            source="project_bin",
        )
        return str(local_binary)

    # Fall back to PATH
    path_binary = shutil.which(CAPTURE_BINARY_NAME_NAME)
    if path_binary:
        logger.debug(
            "screencapture_binary_found",
            path=path_binary,
            source="system_path",
        )
    else:
        logger.warning(
            "screencapture_binary_not_found",
            checked_local=str(local_binary),
        )
    return path_binary


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
        """Return ``True`` if ``livetranslate-capture`` is found."""
        return _find_capture_binary() is not None

    async def start(self) -> None:
        """Spawn the capture process and begin reading audio.

        Raises
        ------
        RuntimeError
            If the binary is not found or the process fails to start.
        """
        binary_path = _find_capture_binary()
        if binary_path is None:
            logger.error(
                "screencapture_start_failed",
                reason="binary_not_found",
                binary_name=CAPTURE_BINARY_NAME_NAME,
            )
            exc = RuntimeError(
                f"'{CAPTURE_BINARY_NAME_NAME}' binary not found. "
                "Run 'just build-screencapture' to build and install it."
            )
            if self._on_error:
                self._on_error(exc)
            raise exc

        logger.info(
            "screencapture_starting",
            binary_path=binary_path,
            sample_rate=self.sample_rate,
        )

        self._process = await asyncio.create_subprocess_exec(
            binary_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._running = True
        self._last_read_time = time.monotonic()
        self._stop_event.clear()

        logger.info(
            "screencapture_started",
            pid=self._process.pid,
            watchdog_timeout_s=WATCHDOG_TIMEOUT,
        )

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
            logger.debug("screencapture_stop_skipped", reason="not_running")
            return

        pid = self._process.pid if self._process else None
        logger.info("screencapture_stopping", pid=pid)

        self._running = False
        self._stop_event.set()

        # Cancel and drain the read task.
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._read_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("screencapture_read_task_cancelled")

        # Terminate the subprocess.
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
                logger.debug("screencapture_process_terminated", pid=pid)
            except (ProcessLookupError, asyncio.TimeoutError):
                logger.warning("screencapture_process_kill_required", pid=pid)
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()

        # Wait for watchdog thread to exit.
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

        self._process = None
        self._read_task = None
        self._watchdog_thread = None

        logger.info("screencapture_stopped", pid=pid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Continuously read PCM bytes from the process stdout."""
        assert self._process is not None
        assert self._process.stdout is not None

        chunks_read = 0
        try:
            while self._running:
                raw = await self._process.stdout.read(CHUNK_SIZE)
                if not raw:
                    # EOF — process exited.
                    if self._running:
                        logger.error(
                            "screencapture_unexpected_eof",
                            chunks_read=chunks_read,
                            pid=self._process.pid,
                        )
                        exc = RuntimeError(
                            f"'{CAPTURE_BINARY_NAME_NAME}' exited unexpectedly (EOF on stdout)."
                        )
                        if self._on_error:
                            self._on_error(exc)
                    break

                self._last_read_time = time.monotonic()

                # Parse f32le samples.
                n_samples = len(raw) // 4
                if n_samples == 0:
                    continue

                chunks_read += 1
                samples = struct.unpack_from(f"{n_samples}f", raw, 0)
                chunk = np.array(samples, dtype=np.float32)
                self._on_audio(chunk)

        except asyncio.CancelledError:
            logger.debug("screencapture_read_loop_cancelled", chunks_read=chunks_read)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "screencapture_read_loop_error",
                chunks_read=chunks_read,
                error=str(exc),
            )
            if self._running and self._on_error:
                self._on_error(exc)

    def _watchdog_loop(self) -> None:
        """Run on a background thread; fires ``on_error`` if stdout goes silent."""
        while not self._stop_event.wait(timeout=1.0):
            if not self._running:
                break
            elapsed = time.monotonic() - self._last_read_time
            if elapsed > WATCHDOG_TIMEOUT:
                logger.error(
                    "screencapture_watchdog_timeout",
                    elapsed_s=round(elapsed, 1),
                    threshold_s=WATCHDOG_TIMEOUT,
                )
                exc = RuntimeError(
                    f"'{CAPTURE_BINARY_NAME_NAME}' watchdog timeout: no audio received for "
                    f"{elapsed:.1f}s (threshold: {WATCHDOG_TIMEOUT}s)."
                )
                if self._on_error:
                    self._on_error(exc)
                # Only fire once; stop the source to avoid repeated callbacks.
                self._running = False
                break
