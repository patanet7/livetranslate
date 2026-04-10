"""CommandDispatcher — routes chat commands to MeetingSessionConfig.

Receives raw command strings from the bot WebSocket, parses them,
applies config changes, and returns response text for the bot to
type in meeting chat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()

VALID_MODES = {"subtitle", "split", "interpreter"}
VALID_THEMES = {
    "dark": "dark",
    "light": "light",
    "contrast": "high_contrast",
    "high_contrast": "high_contrast",
    "minimal": "minimal",
    "corporate": "corporate",
}
VALID_SOURCES = {"bot": "bot_audio", "fireflies": "fireflies"}

VALID_DEMO_MODES = {"replay", "passthrough", "pretranslated", "stop"}

HELP_TEXT = (
    "Commands: /lang <code>, /font up|down|<size>, "
    "/mode subtitle|split|interpreter, /theme dark|light|contrast, "
    "/speakers on|off, /original on|off, /source bot|fireflies, "
    "/translate on|off, /demo [replay|passthrough|stop], /status, /help"
)


@dataclass
class DispatchResult:
    """Result of dispatching a chat command."""

    response_text: str
    changed_fields: set[str]
    demo_action: str | None = None


def _parse_toggle(value: str) -> bool | None:
    if value in ("on", "true", "yes"):
        return True
    if value in ("off", "false", "no"):
        return False
    return None


class CommandDispatcher:
    """Routes chat commands to MeetingSessionConfig mutations."""

    def __init__(self, config: Any, demo_manager: Any = None):
        self._config = config
        self._demo_manager = demo_manager

    def dispatch(self, text: str, sender: str = "") -> DispatchResult | None:
        """Parse and execute a command. Returns None for non-commands."""
        trimmed = text.strip()
        if not trimmed.startswith("/"):
            return None

        parts = trimmed.split()
        cmd = parts[0].lower()
        arg = parts[1].lower() if len(parts) > 1 else ""

        logger.info("chat_command_received", command=cmd, arg=arg, sender=sender)

        if cmd == "/lang":
            return self._handle_lang(arg, trimmed)
        elif cmd == "/font":
            return self._handle_font(arg, trimmed)
        elif cmd == "/mode":
            return self._handle_mode(arg, trimmed)
        elif cmd == "/theme":
            return self._handle_theme(arg, trimmed)
        elif cmd == "/speakers":
            return self._handle_toggle("show_speakers", "Speaker names", arg, trimmed)
        elif cmd == "/original":
            return self._handle_toggle("show_original", "Original text", arg, trimmed)
        elif cmd == "/translate":
            return self._handle_toggle("translation_enabled", "Translation", arg, trimmed)
        elif cmd == "/source":
            return self._handle_source(arg, trimmed)
        elif cmd == "/demo":
            return self._handle_demo(arg, trimmed)
        elif cmd == "/status":
            return self._handle_status()
        elif cmd == "/help":
            return DispatchResult(response_text=HELP_TEXT, changed_fields=set())
        else:
            return DispatchResult(
                response_text=f"Unknown command: {cmd}. Type /help for commands.",
                changed_fields=set(),
            )

    def _handle_lang(self, arg: str, raw: str) -> DispatchResult:
        if not arg:
            return DispatchResult(response_text="Usage: /lang <code> or /lang <src>-<tgt>", changed_fields=set())
        if "-" in arg:
            source, target = arg.split("-", 1)
            changed = self._config.update(source_lang=source, target_lang=target)
            return DispatchResult(response_text=f"✓ Translating: {source} → {target}", changed_fields=changed)
        changed = self._config.update(source_lang="auto", target_lang=arg)
        return DispatchResult(response_text=f"✓ Translating: auto-detect → {arg}", changed_fields=changed)

    def _handle_font(self, arg: str, raw: str) -> DispatchResult:
        if arg == "up":
            new_size = self._config.font_size + 4
            changed = self._config.update(font_size=new_size)
            return DispatchResult(response_text=f"✓ Font size: {new_size}", changed_fields=changed)
        elif arg == "down":
            new_size = max(8, self._config.font_size - 4)
            changed = self._config.update(font_size=new_size)
            return DispatchResult(response_text=f"✓ Font size: {new_size}", changed_fields=changed)
        else:
            try:
                size = int(arg)
                if size > 0:
                    changed = self._config.update(font_size=size)
                    return DispatchResult(response_text=f"✓ Font size: {size}", changed_fields=changed)
            except ValueError:
                pass
            return DispatchResult(response_text="Usage: /font up|down|<size>", changed_fields=set())

    def _handle_mode(self, arg: str, raw: str) -> DispatchResult:
        if arg in VALID_MODES:
            changed = self._config.update(display_mode=arg)
            return DispatchResult(response_text=f"✓ Display mode: {arg}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /mode subtitle|split|interpreter", changed_fields=set())

    def _handle_theme(self, arg: str, raw: str) -> DispatchResult:
        theme = VALID_THEMES.get(arg)
        if theme:
            changed = self._config.update(theme=theme)
            return DispatchResult(response_text=f"✓ Theme: {theme}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /theme dark|light|contrast|minimal|corporate", changed_fields=set())

    def _handle_toggle(self, field: str, label: str, arg: str, raw: str) -> DispatchResult:
        val = _parse_toggle(arg)
        if val is not None:
            changed = self._config.update(**{field: val})
            state = "on" if val else "off"
            return DispatchResult(response_text=f"✓ {label}: {state}", changed_fields=changed)
        return DispatchResult(response_text=f"Usage: /{raw.split()[0][1:]} on|off", changed_fields=set())

    def _handle_source(self, arg: str, raw: str) -> DispatchResult:
        source = VALID_SOURCES.get(arg)
        if source:
            changed = self._config.update(caption_source=source)
            return DispatchResult(response_text=f"✓ Caption source: {source}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /source bot|fireflies", changed_fields=set())

    def _handle_demo(self, arg: str, raw: str) -> DispatchResult:
        if not self._demo_manager:
            return DispatchResult(
                response_text="Demo mode not available (no demo_manager configured)",
                changed_fields=set(),
            )
        mode = arg if arg in VALID_DEMO_MODES else "replay"
        if mode == "stop":
            return DispatchResult(
                response_text="⏹ Stopping demo...",
                changed_fields=set(),
                demo_action="stop",
            )
        return DispatchResult(
            response_text=f"▶ Starting demo ({mode})...",
            changed_fields=set(),
            demo_action=mode,
        )

    def _handle_status(self) -> DispatchResult:
        snap = self._config.snapshot()
        lines = [
            f"Lang: {snap['source_lang']} → {snap['target_lang']}",
            f"Mode: {snap['display_mode']} | Theme: {snap['theme']}",
            f"Font: {snap['font_size']} | Speakers: {'on' if snap['show_speakers'] else 'off'}",
            f"Source: {snap['caption_source']} | Translation: {'on' if snap['translation_enabled'] else 'off'}",
        ]
        return DispatchResult(response_text=" | ".join(lines), changed_fields=set())
