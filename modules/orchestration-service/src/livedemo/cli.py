"""livedemo CLI — `livedemo doctor|run|smoke|replay`.

The CLI is the only place that resolves config, runs preflight, builds
sources/sinks/recorder, and runs the pipeline. Library callers can use
`livedemo.pipeline.run_once` directly.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import LiveDemoConfig
from .pipeline import run_once
from .preflight import CheckResult, check_all
from .recorder import WSRecorder

app = typer.Typer(
    add_completion=False,
    help="Run the LiveTranslate end-to-end demo pipeline (mic / fireflies / file).",
)


# ──────────────────── helpers ────────────────────


def _load_config(
    config_path: Optional[Path],
    cli_overrides: dict,
) -> LiveDemoConfig:
    if config_path is not None:
        cfg = LiveDemoConfig.from_yaml(config_path)
    else:
        cfg = LiveDemoConfig(**cli_overrides)
        return cfg
    # Apply CLI overrides on top of the loaded config (CLI > env > yaml > defaults).
    if cli_overrides:
        cfg = cfg.model_copy(update={k: v for k, v in cli_overrides.items() if v is not None})
    return cfg


def _print_results(results: list[CheckResult]) -> int:
    failed = 0
    typer.echo("livedemo doctor")
    typer.echo("─" * 60)
    for r in results:
        mark = "✓" if r.ok else "✗"
        line = f"{mark} {r.name:<24}"
        if r.detail:
            line += f"  {r.detail}"
        typer.echo(line)
        if not r.ok and r.hint:
            typer.echo(f"   hint: {r.hint}")
            failed += 1
    typer.echo("─" * 60)
    if failed == 0:
        typer.echo("All checks passed.")
    else:
        typer.echo(f"{failed} check(s) failed.")
    return failed


def _make_run_dir(runs_dir: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    p = runs_dir / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_source(cfg: LiveDemoConfig):
    if cfg.source == "file":
        from .sources.file import FileSource

        return FileSource(jsonl_path=cfg.replay_jsonl, replay_speed=cfg.fireflies_replay_speed or 1.0)
    if cfg.source == "fireflies":
        from .sources.fireflies import FirefliesSource

        return FirefliesSource.from_config(cfg)
    if cfg.source == "mic":
        from .sources.mic import MicSource

        return MicSource(
            ws_url=cfg.orchestration_ws_url,
            target_language=cfg.target_language,
            source_language=cfg.source_language,
        )
    raise ValueError(f"unknown source: {cfg.source}")


def _build_sink(cfg: LiveDemoConfig, *, out_dir: Path | None):
    if cfg.sink == "png":
        from .sinks.png import PngSink

        return PngSink(
            out_dir=out_dir or Path("frames"),
            display_mode=cfg.bot_display_mode,
            show_diarization_ids=cfg.bot_show_diarization_ids,
        )
    if cfg.sink == "pyvirtualcam":
        from .sinks.pyvirtualcam import PyVirtualCamSink

        return PyVirtualCamSink(
            display_mode=cfg.bot_display_mode,
            show_diarization_ids=cfg.bot_show_diarization_ids,
        )
    if cfg.sink == "canvas":
        # Real bot harness wired in Phase 6.5/6.75; here we expose the surface.
        from .bot_harness import BotHarness
        from .sinks.canvas_ws import CanvasWsSink

        # CanvasWsSink expects a harness implementing push_frame; harness must be
        # run as a context manager elsewhere. For now CLI canvas mode requires
        # the caller to pre-spawn the harness — Phase 6.75 wires this fully.
        raise NotImplementedError(
            "Canvas sink via CLI lands in Phase 6.75 (full live demo flow)."
        )
    raise ValueError(f"unknown sink: {cfg.sink}")


# ──────────────────── commands ────────────────────


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Override source kind"),
):
    """Run preflight checks and print a status table."""
    overrides = {}
    if source:
        overrides["source"] = source
    if config is None and "source" not in overrides:
        typer.echo("doctor requires --config or --source")
        raise typer.Exit(2)
    # For doctor, allow constructing a minimal config without a meeting URL or replay file.
    minimal = {
        "meeting_url": "https://meet.google.com/aaa-bbbb-ccc",
        "source": overrides.get("source", "file"),
    }
    if minimal["source"] == "file":
        minimal["replay_jsonl"] = "/tmp/_unused.jsonl"
    if minimal["source"] == "fireflies":
        minimal["fireflies_meeting_id"] = "stub"
    if config is not None:
        cfg = LiveDemoConfig.from_yaml(config)
    else:
        cfg = LiveDemoConfig(**minimal)
    if source:
        cfg = cfg.model_copy(update={"source": source})
    results = check_all(cfg)
    failed = _print_results(results)
    raise typer.Exit(code=1 if failed else 0)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    source: Optional[str] = typer.Option(None, "--source"),
    sink: Optional[str] = typer.Option(None, "--sink"),
    replay_jsonl: Optional[Path] = typer.Option(None, "--replay-jsonl"),
    fireflies_meeting_id: Optional[str] = typer.Option(None, "--meeting-id"),
    target_language: Optional[str] = typer.Option(None, "--target", "-t"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
    skip_doctor: bool = typer.Option(False, "--skip-doctor"),
):
    """Run the demo pipeline end-to-end."""
    overrides = {
        k: v
        for k, v in {
            "source": source,
            "sink": sink,
            "replay_jsonl": replay_jsonl,
            "fireflies_meeting_id": fireflies_meeting_id,
            "target_language": target_language,
        }.items()
        if v is not None
    }
    cfg = _load_config(config, overrides)
    if runs_dir is not None:
        cfg = cfg.model_copy(update={"runs_dir": runs_dir})

    if not skip_doctor:
        results = check_all(cfg)
        failed = sum(1 for r in results if not r.ok)
        if failed:
            _print_results(results)
            raise typer.Exit(code=1)

    run_dir = _make_run_dir(cfg.runs_dir)
    cfg.write_snapshot(run_dir / "config.snapshot.yaml")

    src = _build_source(cfg)
    if cfg.sink == "png":
        snk = _build_sink(cfg, out_dir=out_dir or run_dir / "frames")
    else:
        snk = _build_sink(cfg, out_dir=None)
    rec = WSRecorder(run_dir=run_dir, enabled=cfg.record_messages)

    n = asyncio.run(run_once(source=src, sink=snk, recorder=rec))
    rec.close()
    typer.echo(f"OK — {n} captions processed → {run_dir}")


@app.command()
def smoke(
    replay_jsonl: Path = typer.Option(..., "--replay-jsonl"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    runs_dir: Path = typer.Option(..., "--runs-dir"),
):
    """Offline smoke run — file source → png sink. Used in CI."""
    cfg = LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        sink="png",
        replay_jsonl=replay_jsonl,
        runs_dir=runs_dir,
    )
    run_dir = _make_run_dir(cfg.runs_dir)
    cfg.write_snapshot(run_dir / "config.snapshot.yaml")
    from .sources.file import FileSource
    from .sinks.png import PngSink

    src = FileSource(jsonl_path=cfg.replay_jsonl, replay_speed=0.0)
    snk = PngSink(out_dir=out_dir)
    rec = WSRecorder(run_dir=run_dir, enabled=True)
    n = asyncio.run(run_once(source=src, sink=snk, recorder=rec))
    rec.close()
    typer.echo(f"smoke OK — {n} captions → {out_dir}")


@app.command()
def replay(
    run_dir: Path = typer.Argument(..., help="Path to a previous run directory"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
):
    """Replay a previous run's messages.jsonl through the png sink."""
    msg_path = run_dir / "messages.jsonl"
    if not msg_path.exists():
        typer.echo(f"No messages.jsonl in {run_dir}", err=True)
        raise typer.Exit(2)
    out_dir = out_dir or (run_dir / "replay-frames")
    from .sources.file import FileSource
    from .sinks.png import PngSink

    src = FileSource(jsonl_path=msg_path, replay_speed=0.0)
    snk = PngSink(out_dir=out_dir)
    n = asyncio.run(run_once(source=src, sink=snk, recorder=None))
    typer.echo(f"replay OK — {n} captions → {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
