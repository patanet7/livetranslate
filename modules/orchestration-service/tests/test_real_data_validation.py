"""End-to-end pipeline validation with real meeting data fixtures."""

import json
from pathlib import Path

import pytest

from bot.text_wrapper import wrap_text
from services.meeting_session_config import MeetingSessionConfig
from services.pipeline.adapters.source_adapter import BotAudioCaptionSource, CaptionEvent

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "meeting_data"


def load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    return json.loads(path.read_text())


class TestRealDataValidation:
    def test_fixture_directory_exists(self):
        assert FIXTURES_DIR.exists()

    def test_sample_bilingual_fixture_loads(self):
        data = load_fixture("sample_bilingual.json")
        assert data["source"] == "bot_audio"
        assert len(data["segments"]) == 6

    @pytest.mark.asyncio
    async def test_bot_audio_source_processes_fixture(self):
        data = load_fixture("sample_bilingual.json")
        source = BotAudioCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        for seg in data["segments"]:
            await source.handle_transcription(
                text=seg["text"],
                speaker_name=seg["speaker_name"],
                source_lang="auto",
                is_final=seg["is_final"],
            )

        assert len(events) == 6
        alice_colors = {e.speaker_color for e in events if e.speaker_name == "Alice"}
        bob_colors = {e.speaker_color for e in events if e.speaker_name == "Bob"}
        assert len(alice_colors) == 1
        assert len(bob_colors) == 1
        assert alice_colors != bob_colors

    def test_text_wrapper_handles_cjk_fixture_segments(self):
        data = load_fixture("sample_bilingual.json")
        for seg in data["segments"]:
            lines = wrap_text(seg["text"], max_chars=20, max_lines=3)
            assert len(lines) >= 1
            for line in lines:
                assert len(line) <= 20

    def test_meeting_config_snapshot_for_fixture(self):
        data = load_fixture("sample_bilingual.json")
        config = MeetingSessionConfig(
            session_id=data["meeting_id"],
            source_lang="zh",
            target_lang="en",
        )
        snap = config.snapshot()
        assert snap["session_id"] == "fixture-bilingual-001"
        assert snap["source_lang"] == "zh"
        assert snap["target_lang"] == "en"
