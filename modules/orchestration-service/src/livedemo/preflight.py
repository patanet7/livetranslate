"""Preflight check registry — runs every dependency before joining a Meet.

Each check is a small function returning :class:`CheckResult`. Checks declare
which sources they apply to via the registry; `check_all(config)` filters to
the relevant subset for the current source.

Adding a check is one call to :func:`register_check`. The CLI (Phase 6) renders
results as a table and exits non-zero on any failure (B1).
"""
from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import LiveDemoConfig

CheckFn = Callable[[LiveDemoConfig], "CheckResult"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    hint: str | None = None
    detail: str | None = None


@dataclass
class _Registered:
    name: str
    fn: CheckFn
    sources: frozenset[str]  # empty == applies to all sources


_REGISTRY: list[_Registered] = []


def register_check(
    name: str,
    fn: CheckFn,
    *,
    sources: set[str] | frozenset[str] | None = None,
) -> None:
    """Register a check. `sources=None` means run for every source kind."""
    _REGISTRY.append(
        _Registered(
            name=name,
            fn=fn,
            sources=frozenset(sources) if sources else frozenset(),
        )
    )


def _registry_snapshot() -> list[_Registered]:
    return list(_REGISTRY)


def check_all(config: LiveDemoConfig) -> list[CheckResult]:
    """Run every check whose `sources` matches (or is universal)."""
    results: list[CheckResult] = []
    for entry in _registry_snapshot():
        if entry.sources and config.source not in entry.sources:
            continue
        try:
            results.append(entry.fn(config))
        except Exception as exc:  # surface the failure as a check result, not a crash
            results.append(
                CheckResult(
                    name=entry.name,
                    ok=False,
                    hint=f"check raised: {type(exc).__name__}: {exc}",
                )
            )
    return results


# ───────────────────────── built-in checks ─────────────────────────


def _check_chrome_profile(cfg: LiveDemoConfig) -> CheckResult:
    p = cfg.chrome_profile_dir
    if not p.exists():
        return CheckResult(
            name="chrome_profile",
            ok=False,
            hint=f"Create profile at {p} (open Chrome with --user-data-dir and sign in)",
        )
    default_subdir = p / "Default"
    if not default_subdir.exists():
        return CheckResult(
            name="chrome_profile",
            ok=False,
            hint=f"{p} exists but has no Default/ — sign into Chrome with this profile once",
        )
    return CheckResult(name="chrome_profile", ok=True, detail=str(p))


def _check_canvas_ws_port(cfg: LiveDemoConfig) -> CheckResult:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", cfg.canvas_ws_port))
    except OSError:
        return CheckResult(
            name="canvas_ws_port",
            ok=False,
            hint=f"Port {cfg.canvas_ws_port} is bound — kill the existing livedemo run "
            "or change `canvas_ws_port`",
        )
    finally:
        s.close()
    return CheckResult(name="canvas_ws_port", ok=True, detail=f"{cfg.canvas_ws_port} free")


def _check_replay_jsonl(cfg: LiveDemoConfig) -> CheckResult:
    p = cfg.replay_jsonl
    if p is None or not Path(p).exists():
        return CheckResult(
            name="replay_jsonl",
            ok=False,
            hint=f"Recorder JSONL not found at {p}. Record one first via `livedemo run --source=mic`",
        )
    if Path(p).stat().st_size == 0:
        return CheckResult(
            name="replay_jsonl",
            ok=False,
            hint=f"{p} is empty — recorder didn't capture any events",
        )
    return CheckResult(name="replay_jsonl", ok=True, detail=str(p))


def _check_orchestration_ws(cfg: LiveDemoConfig) -> CheckResult:
    """Reachability probe — full WS handshake done in mic source itself."""
    url = cfg.orchestration_ws_url
    # Parse host:port out of ws://host:port/path
    try:
        rest = url.split("://", 1)[1]
        host_port = rest.split("/", 1)[0]
        host, _, port_s = host_port.partition(":")
        port = int(port_s) if port_s else 80
    except Exception:
        return CheckResult(
            name="orchestration_ws",
            ok=False,
            hint=f"Cannot parse orchestration_ws_url={url!r}",
        )
    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
    except OSError as exc:
        return CheckResult(
            name="orchestration_ws",
            ok=False,
            hint=f"{host}:{port} unreachable ({exc}). Start orchestration: "
            "`uv run python modules/orchestration-service/src/main_fastapi.py`",
        )
    return CheckResult(name="orchestration_ws", ok=True, detail=f"{host}:{port} reachable")


def _check_transcription_service(cfg: LiveDemoConfig) -> CheckResult:
    try:
        with socket.create_connection(("127.0.0.1", 5001), timeout=1.5):
            pass
    except OSError:
        return CheckResult(
            name="transcription_service",
            ok=False,
            hint="Transcription service not on :5001. Start it: "
            "`uv run python modules/transcription-service/src/main.py`",
        )
    return CheckResult(name="transcription_service", ok=True, detail="127.0.0.1:5001 reachable")


def _check_mic_device(cfg: LiveDemoConfig) -> CheckResult:
    try:
        import sounddevice  # type: ignore
    except Exception as exc:
        return CheckResult(
            name="mic_device",
            ok=False,
            hint=f"sounddevice import failed: {exc}",
        )
    try:
        devices = sounddevice.query_devices()
    except Exception as exc:
        return CheckResult(
            name="mic_device",
            ok=False,
            hint=f"sounddevice.query_devices failed: {exc} — check macOS mic permission",
        )
    if cfg.mic_device:
        names = [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
        if cfg.mic_device not in names:
            return CheckResult(
                name="mic_device",
                ok=False,
                hint=f"mic_device={cfg.mic_device!r} not found. Available: {names}",
            )
        return CheckResult(name="mic_device", ok=True, detail=cfg.mic_device)
    has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
    if not has_input:
        return CheckResult(
            name="mic_device",
            ok=False,
            hint="No input devices found — check OS mic permissions for the terminal",
        )
    return CheckResult(name="mic_device", ok=True, detail="default input present")


def _check_playwright_chromium(cfg: LiveDemoConfig) -> CheckResult:
    """Verify Playwright's chromium binary is installed (used by bot harness)."""
    import os

    cache_root = Path(
        os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH",
            str(Path.home() / "Library/Caches/ms-playwright"),
        )
    )
    if not cache_root.exists():
        return CheckResult(
            name="playwright_chromium",
            ok=False,
            hint=f"{cache_root} missing — run: `npx playwright install chromium`",
        )
    chromium_dirs = list(cache_root.glob("chromium-*"))
    if not chromium_dirs:
        return CheckResult(
            name="playwright_chromium",
            ok=False,
            hint="No chromium-* in cache — run: `npx playwright install chromium`",
        )
    return CheckResult(
        name="playwright_chromium",
        ok=True,
        detail=f"{len(chromium_dirs)} chromium build(s) in {cache_root}",
    )


def _check_fireflies_api(cfg: LiveDemoConfig) -> CheckResult:
    import os

    api_key = os.environ.get("FIREFLIES_API_KEY")
    if not api_key:
        return CheckResult(
            name="fireflies_api",
            ok=False,
            hint="FIREFLIES_API_KEY not set",
        )
    return CheckResult(
        name="fireflies_api",
        ok=True,
        detail="API key present (request validated by source on first call)",
    )


def _register_builtins() -> None:
    # Universal — needed for any sink that uses the bot (canvas) or any source.
    register_check("playwright_chromium", _check_playwright_chromium)
    register_check("canvas_ws_port", _check_canvas_ws_port)
    # Bot-bound (skip for png-only file replays in CI).
    register_check("chrome_profile", _check_chrome_profile, sources={"mic", "fireflies"})
    # Source-specific.
    register_check("replay_jsonl", _check_replay_jsonl, sources={"file"})
    register_check("orchestration_ws", _check_orchestration_ws, sources={"mic"})
    register_check("transcription_service", _check_transcription_service, sources={"mic"})
    register_check("mic_device", _check_mic_device, sources={"mic"})
    register_check("fireflies_api", _check_fireflies_api, sources={"fireflies"})


_register_builtins()
