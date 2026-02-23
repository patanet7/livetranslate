"""
Behavioral tests for CommandInterceptor and LiveCaptionManager.

These test REAL behavior — real PipelineConfig, real coordinator,
real async broadcast capture. No mocks.
"""

import asyncio

import pytest

from services.pipeline.adapters.base import ChunkAdapter, TranscriptChunk
from services.pipeline.command_interceptor import CommandInterceptor, LANGUAGE_ALIASES
from services.pipeline.config import PipelineConfig
from services.pipeline.coordinator import TranscriptionPipelineCoordinator
from services.pipeline.live_caption_manager import LiveCaptionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TestAdapter(ChunkAdapter):
    """Minimal concrete adapter to satisfy coordinator's required argument."""

    @property
    def source_type(self) -> str:
        return "test"

    def adapt(self, raw_chunk):
        return TranscriptChunk(
            text=raw_chunk.get("text", ""),
            speaker_name=raw_chunk.get("speaker_name"),
            timestamp_ms=0,
            chunk_id=raw_chunk.get("chunk_id", "test-chunk"),
        )

    def extract_speaker(self, raw_chunk):
        return raw_chunk.get("speaker_name")


def _make_config(**overrides) -> PipelineConfig:
    """Build a real PipelineConfig with sensible test defaults."""
    defaults = {
        "session_id": "test-session-001",
        "source_type": "fireflies",
        "transcript_id": "transcript-xyz",
        "target_languages": ["es"],
        "voice_commands_enabled": True,
        "voice_command_prefix": "LiveTranslate",
        "display_mode": "both",
        "enable_interim_captions": True,
        "enable_persistence": False,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_coordinator(config: PipelineConfig) -> TranscriptionPipelineCoordinator:
    """Build a real coordinator with a minimal test adapter."""
    return TranscriptionPipelineCoordinator(config=config, adapter=_TestAdapter())


class BroadcastRecorder:
    """Async callable that records every broadcast for assertion.

    This is NOT a mock — it's a real async function that stores results.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, session_id: str, payload: dict) -> None:
        self.calls.append((session_id, payload))

    @property
    def count(self) -> int:
        return len(self.calls)

    def last_payload(self) -> dict:
        assert self.calls, "No broadcasts recorded"
        return self.calls[-1][1]

    def payloads(self) -> list[dict]:
        return [c[1] for c in self.calls]


class FakeChunk:
    """Minimal chunk object matching FirefliesChunk interface for caption tests."""

    def __init__(self, chunk_id: str = "c1", text: str = "hello", speaker_name: str = "Alice"):
        self.chunk_id = chunk_id
        self.text = text
        self.speaker_name = speaker_name


class FakeCaption:
    """Minimal caption object with to_dict() and id for caption event tests."""

    def __init__(
        self,
        caption_id: str = "cap-1",
        original_text: str = "Hello",
        translated_text: str = "Hola",
    ):
        self.id = caption_id
        self._original = original_text
        self._translated = translated_text

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_text": self._original,
            "translated_text": self._translated,
            "speaker_name": "Alice",
        }


# ===========================================================================
# CommandInterceptor Tests
# ===========================================================================


class TestCommandInterceptorCheck:
    """Test the check() method — does it correctly identify voice commands?"""

    def test_check_returns_true_for_matching_prefix(self):
        config = _make_config(voice_commands_enabled=True, voice_command_prefix="LiveTranslate")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("LiveTranslate pause") is True

    def test_check_is_case_insensitive(self):
        config = _make_config(voice_commands_enabled=True, voice_command_prefix="LiveTranslate")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("livetranslate RESUME") is True
        assert interceptor.check("LIVETRANSLATE language spanish") is True

    def test_check_returns_false_for_normal_speech(self):
        config = _make_config(voice_commands_enabled=True, voice_command_prefix="LiveTranslate")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("Hello everyone, welcome to the meeting.") is False

    def test_check_returns_false_when_disabled(self):
        config = _make_config(voice_commands_enabled=False)
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("LiveTranslate pause") is False

    def test_check_handles_leading_whitespace(self):
        config = _make_config(voice_commands_enabled=True, voice_command_prefix="LiveTranslate")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("  LiveTranslate pause  ") is True

    def test_check_with_custom_prefix(self):
        config = _make_config(voice_commands_enabled=True, voice_command_prefix="Hey Jarvis")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.check("Hey Jarvis pause") is True
        assert interceptor.check("LiveTranslate pause") is False


class TestCommandInterceptorExecute:
    """Test the execute() method — does it correctly dispatch commands?"""

    @pytest.mark.asyncio
    async def test_pause_command_pauses_coordinator(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert coordinator.paused is False
        result = await interceptor.execute("LiveTranslate pause")

        assert result["command"] == "pause"
        assert result["executed"] is True
        assert coordinator.paused is True

    @pytest.mark.asyncio
    async def test_resume_command_resumes_coordinator(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        coordinator.pause()
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate resume")

        assert result["command"] == "resume"
        assert result["executed"] is True
        assert coordinator.paused is False

    @pytest.mark.asyncio
    async def test_stop_is_alias_for_pause(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate stop")

        assert result["command"] == "pause"
        assert result["executed"] is True
        assert coordinator.paused is True

    @pytest.mark.asyncio
    async def test_continue_is_alias_for_resume(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        coordinator.pause()
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate continue")

        assert result["command"] == "resume"
        assert result["executed"] is True
        assert coordinator.paused is False

    @pytest.mark.asyncio
    async def test_language_command_with_name(self):
        config = _make_config(target_languages=["es"])
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate language french")

        assert result["command"] == "language"
        assert result["language"] == "fr"
        assert result["executed"] is True
        assert coordinator.config.target_languages == ["fr"]

    @pytest.mark.asyncio
    async def test_language_command_with_iso_code(self):
        config = _make_config(target_languages=["es"])
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate language de")

        assert result["language"] == "de"
        assert coordinator.config.target_languages == ["de"]

    @pytest.mark.asyncio
    async def test_lang_is_alias_for_language(self):
        config = _make_config(target_languages=["es"])
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate lang japanese")

        assert result["language"] == "ja"
        assert result["executed"] is True

    @pytest.mark.asyncio
    async def test_display_command_changes_mode(self):
        config = _make_config(display_mode="both")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate display english")

        assert result["command"] == "display"
        assert result["mode"] == "english"
        assert result["executed"] is True
        assert coordinator.config.display_mode == "english"

    @pytest.mark.asyncio
    async def test_display_command_rejects_invalid_mode(self):
        config = _make_config(display_mode="both")
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate display rainbow")

        assert result["executed"] is False
        assert "unknown mode" in result.get("error", "")
        # Config should be unchanged
        assert coordinator.config.display_mode == "both"

    @pytest.mark.asyncio
    async def test_unrecognized_command_returns_error(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate fly to the moon")

        assert result["executed"] is False
        assert result.get("error") == "unrecognized"

    @pytest.mark.asyncio
    async def test_trailing_punctuation_stripped(self):
        """ASR often adds trailing punctuation — should be stripped."""
        config = _make_config()
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        result = await interceptor.execute("LiveTranslate pause.")

        assert result["command"] == "pause"
        assert result["executed"] is True

    @pytest.mark.asyncio
    async def test_commands_executed_counter_increments(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        assert interceptor.commands_executed == 0
        await interceptor.execute("LiveTranslate pause")
        assert interceptor.commands_executed == 1
        await interceptor.execute("LiveTranslate resume")
        assert interceptor.commands_executed == 2
        # Failed command should NOT increment
        await interceptor.execute("LiveTranslate gibberish")
        assert interceptor.commands_executed == 2

    @pytest.mark.asyncio
    async def test_broadcast_fires_on_executed_command(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        recorder = BroadcastRecorder()
        interceptor = CommandInterceptor(
            config=config,
            coordinator=coordinator,
            ws_broadcast=recorder,
            session_id="sess-123",
        )

        await interceptor.execute("LiveTranslate pause")

        assert recorder.count == 1
        payload = recorder.last_payload()
        assert payload["event"] == "voice_command"
        assert payload["command"] == "pause"

    @pytest.mark.asyncio
    async def test_broadcast_does_not_fire_on_failed_command(self):
        config = _make_config()
        coordinator = _make_coordinator(config)
        recorder = BroadcastRecorder()
        interceptor = CommandInterceptor(
            config=config,
            coordinator=coordinator,
            ws_broadcast=recorder,
        )

        await interceptor.execute("LiveTranslate xyzzy")

        assert recorder.count == 0


class TestCommandInterceptorLanguageAliases:
    """Verify all supported language aliases resolve correctly."""

    def test_all_language_aliases_have_two_letter_codes(self):
        for name, code in LANGUAGE_ALIASES.items():
            assert len(code) == 2, f"Language '{name}' has non-2-letter code: {code}"

    def test_common_languages_present(self):
        expected = {"spanish", "french", "german", "italian", "japanese", "chinese", "korean"}
        assert expected.issubset(set(LANGUAGE_ALIASES.keys()))


# ===========================================================================
# LiveCaptionManager Tests
# ===========================================================================


class TestLiveCaptionManagerInterim:
    """Test interim caption filtering based on config."""

    @pytest.mark.asyncio
    async def test_interim_sent_when_enabled_and_mode_both(self):
        config = _make_config(display_mode="both", enable_interim_captions=True)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk(chunk_id="c1", text="Hello world")
        await mgr.handle_interim_update(chunk, is_final=False)

        assert recorder.count == 1
        assert recorder.last_payload()["event"] == "interim_caption"
        assert recorder.last_payload()["is_final"] is False

    @pytest.mark.asyncio
    async def test_interim_filtered_when_disabled(self):
        config = _make_config(display_mode="both", enable_interim_captions=False)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=False)

        assert recorder.count == 0

    @pytest.mark.asyncio
    async def test_final_sent_even_when_interim_disabled(self):
        config = _make_config(display_mode="both", enable_interim_captions=False)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=True)

        assert recorder.count == 1
        assert recorder.last_payload()["is_final"] is True

    @pytest.mark.asyncio
    async def test_interim_filtered_in_translated_mode(self):
        """In 'translated' mode, interim ASR text is useless (it's source language)."""
        config = _make_config(display_mode="translated", enable_interim_captions=True)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=False)

        assert recorder.count == 0

    @pytest.mark.asyncio
    async def test_final_sent_in_translated_mode(self):
        config = _make_config(display_mode="translated", enable_interim_captions=True)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=True)

        assert recorder.count == 1

    @pytest.mark.asyncio
    async def test_interim_sent_in_english_mode(self):
        config = _make_config(display_mode="english", enable_interim_captions=True)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=False)

        assert recorder.count == 1

    @pytest.mark.asyncio
    async def test_stats_track_sent_and_filtered(self):
        config = _make_config(display_mode="both", enable_interim_captions=False)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        # 3 interim (filtered) + 1 final (sent)
        await mgr.handle_interim_update(chunk, is_final=False)
        await mgr.handle_interim_update(chunk, is_final=False)
        await mgr.handle_interim_update(chunk, is_final=False)
        await mgr.handle_interim_update(chunk, is_final=True)

        assert mgr.stats["interim_updates_filtered"] == 3
        assert mgr.stats["interim_updates_sent"] == 1


class TestLiveCaptionManagerCaptionEvent:
    """Test caption event filtering based on display_mode."""

    @pytest.mark.asyncio
    async def test_both_mode_sends_all_fields(self):
        config = _make_config(display_mode="both")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        caption = FakeCaption(original_text="Hello", translated_text="Hola")
        await mgr.handle_caption_event("caption_added", caption)

        payload = recorder.last_payload()
        assert payload["event"] == "caption_added"
        assert payload["caption"]["original_text"] == "Hello"
        assert payload["caption"]["translated_text"] == "Hola"

    @pytest.mark.asyncio
    async def test_english_mode_clears_translated_text(self):
        config = _make_config(display_mode="english")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        caption = FakeCaption(original_text="Hello", translated_text="Hola")
        await mgr.handle_caption_event("caption_added", caption)

        payload = recorder.last_payload()
        assert payload["caption"]["original_text"] == "Hello"
        assert payload["caption"]["translated_text"] == ""

    @pytest.mark.asyncio
    async def test_translated_mode_clears_original_text(self):
        config = _make_config(display_mode="translated")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        caption = FakeCaption(original_text="Hello", translated_text="Hola")
        await mgr.handle_caption_event("caption_added", caption)

        payload = recorder.last_payload()
        assert payload["caption"]["original_text"] == ""
        assert payload["caption"]["translated_text"] == "Hola"

    @pytest.mark.asyncio
    async def test_caption_expired_event_does_not_filter(self):
        config = _make_config(display_mode="english")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        caption = FakeCaption(caption_id="cap-99")
        await mgr.handle_caption_event("caption_expired", caption)

        payload = recorder.last_payload()
        assert payload["event"] == "caption_expired"
        assert payload["caption_id"] == "cap-99"
        # caption_expired should NOT increment captions_sent (it's a removal)
        assert mgr.stats["captions_sent"] == 0

    @pytest.mark.asyncio
    async def test_captions_sent_counter_increments(self):
        config = _make_config(display_mode="both")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        caption = FakeCaption()
        await mgr.handle_caption_event("caption_added", caption)
        await mgr.handle_caption_event("caption_updated", caption)

        assert mgr.stats["captions_sent"] == 2


class TestLiveCaptionManagerConfigDriven:
    """Test that LiveCaptionManager reads LIVE config values (not snapshot)."""

    @pytest.mark.asyncio
    async def test_display_mode_change_takes_effect_immediately(self):
        """If config.display_mode changes mid-session, manager should reflect it."""
        config = _make_config(display_mode="both")
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        # First caption with "both" mode
        caption = FakeCaption(original_text="Hi", translated_text="Hola")
        await mgr.handle_caption_event("caption_added", caption)
        p1 = recorder.payloads()[-1]
        assert p1["caption"]["original_text"] == "Hi"
        assert p1["caption"]["translated_text"] == "Hola"

        # Change config live (as a voice command would)
        config.display_mode = "english"

        # Next caption should reflect new mode
        caption2 = FakeCaption(original_text="Bye", translated_text="Adios")
        await mgr.handle_caption_event("caption_added", caption2)
        p2 = recorder.payloads()[-1]
        assert p2["caption"]["original_text"] == "Bye"
        assert p2["caption"]["translated_text"] == ""

    @pytest.mark.asyncio
    async def test_interim_toggle_takes_effect_immediately(self):
        """Toggling enable_interim_captions mid-session should take effect."""
        config = _make_config(enable_interim_captions=True)
        recorder = BroadcastRecorder()
        mgr = LiveCaptionManager(config=config, broadcast=recorder, session_id="s1")

        chunk = FakeChunk()
        await mgr.handle_interim_update(chunk, is_final=False)
        assert recorder.count == 1  # sent

        # Disable mid-session
        config.enable_interim_captions = False

        await mgr.handle_interim_update(chunk, is_final=False)
        assert recorder.count == 1  # still 1, second was filtered


# ===========================================================================
# Integration: CommandInterceptor + LiveCaptionManager + Coordinator
# ===========================================================================


class TestCommandAndCaptionIntegration:
    """Test that CommandInterceptor changes actually affect LiveCaptionManager."""

    @pytest.mark.asyncio
    async def test_voice_command_changes_display_mode_for_captions(self):
        """End-to-end: voice command changes display_mode, captions reflect it."""
        config = _make_config(
            voice_commands_enabled=True,
            display_mode="both",
        )
        coordinator = _make_coordinator(config)
        caption_recorder = BroadcastRecorder()
        command_recorder = BroadcastRecorder()

        interceptor = CommandInterceptor(
            config=config,
            coordinator=coordinator,
            ws_broadcast=command_recorder,
        )
        mgr = LiveCaptionManager(
            config=config,
            broadcast=caption_recorder,
            session_id=config.session_id,
        )

        # Verify both texts come through initially
        caption = FakeCaption(original_text="Hi", translated_text="Hola")
        await mgr.handle_caption_event("caption_added", caption)
        p1 = caption_recorder.payloads()[-1]
        assert p1["caption"]["original_text"] == "Hi"
        assert p1["caption"]["translated_text"] == "Hola"

        # Issue voice command to switch to "translated" mode
        result = await interceptor.execute("LiveTranslate display translated")
        assert result["executed"] is True

        # Config should be updated
        assert config.display_mode == "translated"

        # Next caption should only show translated text
        caption2 = FakeCaption(original_text="Bye", translated_text="Adios")
        await mgr.handle_caption_event("caption_added", caption2)
        p2 = caption_recorder.payloads()[-1]
        assert p2["caption"]["original_text"] == ""
        assert p2["caption"]["translated_text"] == "Adios"

    @pytest.mark.asyncio
    async def test_pause_command_pauses_coordinator_which_drops_chunks(self):
        """Voice command pauses coordinator; coordinator's paused flag is set."""
        config = _make_config(voice_commands_enabled=True)
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        # Pause via voice command
        result = await interceptor.execute("LiveTranslate pause")
        assert result["executed"] is True
        assert coordinator.paused is True

        # Resume via voice command
        result = await interceptor.execute("LiveTranslate go")
        assert result["executed"] is True
        assert coordinator.paused is False

    @pytest.mark.asyncio
    async def test_language_command_updates_shared_config(self):
        """Voice command updates target_languages on the shared config object."""
        config = _make_config(
            voice_commands_enabled=True,
            target_languages=["es"],
        )
        coordinator = _make_coordinator(config)
        interceptor = CommandInterceptor(config=config, coordinator=coordinator)

        # Both interceptor and coordinator see the same config
        assert coordinator.config.target_languages == ["es"]

        await interceptor.execute("LiveTranslate language german")

        # Config is shared — coordinator sees the change
        assert coordinator.config.target_languages == ["de"]
        assert config.target_languages == ["de"]
