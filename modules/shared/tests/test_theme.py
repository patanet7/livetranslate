"""Tests for canonical theme definitions."""

import re

import pytest

from livetranslate_common.theme import (
    SPEAKER_COLORS,
    DisplayMode,
    ThemeColors,
    get_theme_colors,
    hex_to_rgb,
)


class TestSpeakerColors:
    def test_has_at_least_10_colors(self):
        assert len(SPEAKER_COLORS) >= 10

    def test_all_hex_format(self):
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for color in SPEAKER_COLORS:
            assert hex_pattern.match(color), f"Invalid hex color: {color}"

    def test_no_duplicates(self):
        assert len(SPEAKER_COLORS) == len(set(SPEAKER_COLORS))


class TestHexToRgb:
    def test_converts_hex_to_rgb(self):
        assert hex_to_rgb("#4CAF50") == (76, 175, 80)

    def test_converts_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_converts_white(self):
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)


class TestDisplayMode:
    def test_canonical_modes(self):
        assert DisplayMode.SUBTITLE == "subtitle"
        assert DisplayMode.SPLIT == "split"
        assert DisplayMode.INTERPRETER == "interpreter"

    def test_all_values_are_strings(self):
        for mode in DisplayMode:
            assert isinstance(mode.value, str)


class TestThemeColors:
    def test_dark_theme_exists(self):
        colors = get_theme_colors("dark")
        assert isinstance(colors, ThemeColors)
        assert colors.background is not None
        assert colors.text_primary is not None

    def test_all_themes_loadable(self):
        for theme_name in ("dark", "light", "high_contrast", "minimal", "corporate"):
            colors = get_theme_colors(theme_name)
            assert isinstance(colors, ThemeColors)

    def test_invalid_theme_raises(self):
        with pytest.raises(KeyError):
            get_theme_colors("nonexistent")
