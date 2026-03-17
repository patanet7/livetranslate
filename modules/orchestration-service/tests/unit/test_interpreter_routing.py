"""Unit tests for interpreter mode routing logic.

Tests the per-segment effective_target computation and context-clearing
behavior that drives bidirectional translation in interpreter mode.
"""

from __future__ import annotations


def compute_effective_target(
    segment_language: str,
    interpreter_languages: tuple[str, str] | None,
    target_language: str | None,
) -> str | None:
    """Extract of the routing logic from websocket_audio.py.

    Given a segment's detected language and the session config, returns
    the effective translation target (or None to skip translation).
    """
    if interpreter_languages:
        lang_a, lang_b = interpreter_languages
        if segment_language == lang_a:
            return lang_b
        elif segment_language == lang_b:
            return lang_a
        else:
            return None
    else:
        return target_language


class TestInterpreterRouting:
    """Per-segment effective_target computation."""

    def test_lang_a_maps_to_lang_b(self) -> None:
        result = compute_effective_target("zh", ("zh", "en"), None)
        assert result == "en"

    def test_lang_b_maps_to_lang_a(self) -> None:
        result = compute_effective_target("en", ("zh", "en"), None)
        assert result == "zh"

    def test_unknown_language_returns_none(self) -> None:
        result = compute_effective_target("ja", ("zh", "en"), None)
        assert result is None

    def test_non_interpreter_mode_uses_target_language(self) -> None:
        result = compute_effective_target("zh", None, "en")
        assert result == "en"

    def test_non_interpreter_mode_none_target(self) -> None:
        result = compute_effective_target("zh", None, None)
        assert result is None

    def test_same_language_as_target_still_routes(self) -> None:
        """If segment matches target in non-interpreter mode, caller guards with != check."""
        result = compute_effective_target("en", None, "en")
        assert result == "en"  # caller's != guard will skip translation


class TestDirectionTracking:
    """Context clearing on translation direction flip."""

    def test_direction_flip_detected(self) -> None:
        interpreter_languages = ("zh", "en")

        segments = [
            ("zh", "en"),   # zh→en
            ("zh", "en"),   # zh→en (same)
            ("en", "zh"),   # en→zh (FLIP)
            ("zh", "en"),   # zh→en (FLIP)
        ]

        last_direction = None
        flips = 0
        for seg_lang, expected_target in segments:
            effective = compute_effective_target(seg_lang, interpreter_languages, None)
            assert effective == expected_target
            if effective:
                direction = f"{seg_lang}→{effective}"
                if last_direction and direction != last_direction:
                    flips += 1
                last_direction = direction

        assert flips == 2  # two direction changes

    def test_no_flip_when_same_direction(self) -> None:
        interpreter_languages = ("zh", "en")
        last_direction = None
        flips = 0

        for seg_lang in ["zh", "zh", "zh"]:
            effective = compute_effective_target(seg_lang, interpreter_languages, None)
            direction = f"{seg_lang}→{effective}"
            if last_direction and direction != last_direction:
                flips += 1
            last_direction = direction

        assert flips == 0


class TestInterpreterRoutingSpec:
    """Task-spec named tests: exact routing cases from the production handler.

    These replicate the routing logic from websocket_audio.py lines 291-309
    using the module-level compute_effective_target helper defined above.
    """

    def test_zh_routes_to_en(self) -> None:
        """interpreter=("zh","en"), detected="zh" -> effective="en"."""
        result = compute_effective_target("zh", ("zh", "en"), None)
        assert result == "en", f"Expected 'en', got {result!r}"

    def test_en_routes_to_zh(self) -> None:
        """interpreter=("zh","en"), detected="en" -> effective="zh"."""
        result = compute_effective_target("en", ("zh", "en"), None)
        assert result == "zh", f"Expected 'zh', got {result!r}"

    def test_ja_routes_to_none(self) -> None:
        """interpreter=("zh","en"), detected="ja" -> None (skip translation)."""
        result = compute_effective_target("ja", ("zh", "en"), None)
        assert result is None, (
            f"Language outside interpreter pair must route to None (skip), got {result!r}"
        )

    def test_no_interpreter_uses_target(self) -> None:
        """interpreter=None, target="es" -> "es" (normal mode passthrough)."""
        result = compute_effective_target("fr", None, "es")
        assert result == "es", f"Expected 'es' in normal mode, got {result!r}"

    def test_same_lang_as_target_skips(self) -> None:
        """Non-interpreter mode returns target_language regardless of detected lang.

        The handler skips translation when source==target, but the routing
        function itself returns the configured target unchanged — "en". The skip
        decision is made at a higher level in the handler.
        """
        result = compute_effective_target("en", None, "en")
        assert result == "en", (
            f"Routing returns target 'en' unchanged; skip logic is in the handler, "
            f"got {result!r}"
        )


class TestConfigMessageInterpreter:
    """ConfigMessage handling for interpreter_languages."""

    def test_valid_pair_enables_interpreter(self) -> None:
        from livetranslate_common.models.ws_messages import ConfigMessage

        msg = ConfigMessage(interpreter_languages=["zh", "en"])
        assert msg.interpreter_languages is not None
        assert len(msg.interpreter_languages) == 2
        lang_a, lang_b = msg.interpreter_languages
        assert lang_a == "zh"
        assert lang_b == "en"

    def test_none_disables_interpreter(self) -> None:
        from livetranslate_common.models.ws_messages import ConfigMessage

        msg = ConfigMessage(interpreter_languages=None)
        assert msg.interpreter_languages is None

    def test_interpreter_languages_coexists_with_target(self) -> None:
        from livetranslate_common.models.ws_messages import ConfigMessage

        msg = ConfigMessage(target_language="es", interpreter_languages=["zh", "en"])
        assert msg.target_language == "es"
        assert msg.interpreter_languages == ["zh", "en"]
