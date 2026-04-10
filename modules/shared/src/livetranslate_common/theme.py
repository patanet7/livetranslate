"""Canonical theme definitions for LiveTranslate.

This is the SINGLE source of truth for speaker colors, display modes,
and theme color schemes. Both PIL (Python) and SvelteKit (browser)
renderers consume these definitions.

DO NOT define colors, display modes, or themes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Speaker Colors — canonical palette (hex strings)
# =============================================================================

SPEAKER_COLORS: list[str] = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#F44336",  # Red
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#FFEB3B",  # Yellow
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple. For PIL rendering."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# =============================================================================
# Display Modes — canonical enum
# =============================================================================


class DisplayMode(str, Enum):
    """Display modes for subtitle rendering.

    These match the dashboard's display mode names exactly.
    Both PIL and SvelteKit renderers use these values.
    """

    SUBTITLE = "subtitle"
    SPLIT = "split"
    INTERPRETER = "interpreter"


# =============================================================================
# Theme Colors
# =============================================================================


@dataclass(frozen=True)
class ThemeColors:
    """Color scheme for a visual theme."""

    background: tuple[int, int, int]
    text_primary: tuple[int, int, int]
    text_secondary: tuple[int, int, int]
    accent: tuple[int, int, int]
    border: tuple[int, int, int]


_THEMES: dict[str, ThemeColors] = {
    "dark": ThemeColors(
        background=(20, 20, 20),
        text_primary=(255, 255, 255),
        text_secondary=(180, 180, 180),
        accent=(0, 150, 255),
        border=(60, 60, 60),
    ),
    "light": ThemeColors(
        background=(240, 240, 240),
        text_primary=(20, 20, 20),
        text_secondary=(80, 80, 80),
        accent=(0, 120, 200),
        border=(200, 200, 200),
    ),
    "high_contrast": ThemeColors(
        background=(0, 0, 0),
        text_primary=(255, 255, 255),
        text_secondary=(255, 255, 0),
        accent=(255, 0, 255),
        border=(255, 255, 255),
    ),
    "minimal": ThemeColors(
        background=(250, 250, 250),
        text_primary=(40, 40, 40),
        text_secondary=(120, 120, 120),
        accent=(100, 100, 100),
        border=(220, 220, 220),
    ),
    "corporate": ThemeColors(
        background=(245, 245, 245),
        text_primary=(30, 30, 30),
        text_secondary=(100, 100, 100),
        accent=(0, 100, 180),
        border=(210, 210, 210),
    ),
}


def get_theme_colors(theme_name: str) -> ThemeColors:
    """Get colors for a theme by name. Raises KeyError for unknown themes."""
    return _THEMES[theme_name]
