"""LiveDemoConfig — single source of truth for the demo pipeline.

Precedence (highest wins): CLI flags → env vars (LIVEDEMO_*) → YAML → defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # pragma: no cover - import guard
    from livetranslate_common.theme import DisplayMode
except ImportError:  # fallback for isolated unit tests
    from enum import Enum

    class DisplayMode(str, Enum):
        SUBTITLE = "subtitle"
        SPLIT = "split"
        INTERPRETER = "interpreter"


SourceKind = Literal["mic", "fireflies", "file"]
SinkKind = Literal["canvas", "png", "pyvirtualcam"]


class LiveDemoConfig(BaseSettings):
    """Resolved configuration for one livedemo run.

    Loaded from env (`LIVEDEMO_*`) and optionally a YAML file. CLI flag overrides
    are applied by `cli.py` via `model_copy(update=...)`.
    """

    model_config = SettingsConfigDict(
        env_prefix="LIVEDEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Bot ───────────────────────────────────────────────
    meeting_url: str
    chrome_profile_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config/livetranslate/chrome-profile"
    )
    bot_display_mode: DisplayMode = DisplayMode.SUBTITLE
    bot_show_diarization_ids: bool = False

    # ── Bridge ────────────────────────────────────────────
    canvas_ws_port: int = 7081
    frame_fps: int = 10

    # ── Source / sink selection ───────────────────────────
    source: SourceKind
    sink: SinkKind = "canvas"

    # ── Source: mic ───────────────────────────────────────
    mic_device: str | None = None
    orchestration_ws_url: str = "ws://localhost:3000/api/audio/stream"

    # ── Source: fireflies ─────────────────────────────────
    fireflies_meeting_id: str | None = None
    fireflies_replay_speed: float = Field(default=1.0, gt=0.0)

    # ── Source: file ──────────────────────────────────────
    replay_jsonl: Path | None = None

    # ── Translation ───────────────────────────────────────
    source_language: str = "auto"
    target_language: str = "en"

    # ── Recording ─────────────────────────────────────────
    runs_dir: Path = Path("runs/livedemo")
    record_messages: bool = True

    @model_validator(mode="after")
    def _validate_source_requirements(self) -> "LiveDemoConfig":
        if self.source == "file" and self.replay_jsonl is None:
            raise ValueError("source=file requires replay_jsonl")
        if self.source == "fireflies" and not self.fireflies_meeting_id:
            raise ValueError("source=fireflies requires fireflies_meeting_id")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LiveDemoConfig":
        """Load config from YAML — env vars override yaml (env > yaml > defaults).

        Implementation: drop yaml keys that have a corresponding LIVEDEMO_* env var
        set, so BaseSettings' env loader supplies them instead. Init kwargs would
        otherwise win over env per Pydantic Settings' precedence.
        """
        import os

        path = Path(path)
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        env_prefix = (cls.model_config.get("env_prefix") or "").upper()
        filtered = {
            k: v
            for k, v in data.items()
            if f"{env_prefix}{k.upper()}" not in os.environ
        }
        return cls(**filtered)

    def write_snapshot(self, path: Path | str) -> None:
        """Dump fully-resolved config (B10).

        Path/Enum are normalized to strings so YAML round-trips cleanly.
        """
        path = Path(path)
        data = self.model_dump(mode="json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=True))
