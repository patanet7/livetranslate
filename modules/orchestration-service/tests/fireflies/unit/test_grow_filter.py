#!/usr/bin/env python3
"""Tests for LiveCaptionManager grow filter.

The grow filter ensures interim captions only flow forward (grow or correct-to-longer).
Shrinks are suppressed because Fireflies' ASR frequently shortens text mid-correction
then grows it back. This eliminates visual jitter.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pytest

os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

import importlib.util

_lcm_spec = importlib.util.spec_from_file_location(
    "live_caption_manager", src_path / "services" / "pipeline" / "live_caption_manager.py"
)
_lcm_module = importlib.util.module_from_spec(_lcm_spec)
_lcm_spec.loader.exec_module(_lcm_module)

LiveCaptionManager = _lcm_module.LiveCaptionManager


# Minimal stubs
@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    text: str = "hello"
    speaker_name: str = "Alice"

@dataclass
class FakeConfig:
    session_id: str = "s1"
    display_mode: str = "both"
    enable_interim_captions: bool = True


class TestGrowFilter:
    """Test the grow-only interim caption filter."""

    @pytest.fixture
    def broadcasts(self):
        return []

    @pytest.fixture
    def manager(self, broadcasts):
        async def fake_broadcast(session_id, msg):
            broadcasts.append(msg)
        config = FakeConfig()
        return LiveCaptionManager(config=config, broadcast=fake_broadcast, session_id="s1")

    @pytest.mark.asyncio
    async def test_first_text_always_broadcast(self, manager, broadcasts):
        """First interim for a chunk_id is always sent."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1
        assert broadcasts[0]["text"] == "hello"
        assert broadcasts[0]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_grow_appends_broadcast(self, manager, broadcasts):
        """Text that grows (startswith previous) is broadcast."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_shrink_suppressed(self, manager, broadcasts):
        """Text that shrinks is NOT broadcast."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1  # Only the first one

    @pytest.mark.asyncio
    async def test_correction_longer_broadcast(self, manager, broadcasts):
        """Text that is rewritten but longer is broadcast as correction."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="Hello World! And more"), is_final=False)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "correction"

    @pytest.mark.asyncio
    async def test_duplicate_suppressed(self, manager, broadcasts):
        """Exact same text is suppressed."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1

    @pytest.mark.asyncio
    async def test_final_always_broadcast(self, manager, broadcasts):
        """is_final=True always broadcasts regardless of text length."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=True)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "final"
        assert broadcasts[1]["is_final"] is True

    @pytest.mark.asyncio
    async def test_final_cleans_displayed_text(self, manager, broadcasts):
        """After final, a new text for same chunk_id starts fresh."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="done"), is_final=True)
        # New text for same chunk_id should be treated as first
        await manager.handle_interim_update(FakeChunk(text="new"), is_final=False)
        assert len(broadcasts) == 3
        assert broadcasts[2]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_stats_track_suppressed(self, manager, broadcasts):
        """Stats counters track suppressed shrinks and duplicates."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)  # shrink
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)  # dup (against last displayed "hello world", not suppressed "hello")
        stats = manager.stats
        assert stats["interim_updates_sent"] == 1
        assert stats["interim_shrinks_suppressed"] >= 1

    @pytest.mark.asyncio
    async def test_different_chunk_ids_independent(self, manager, broadcasts):
        """Different chunk_ids maintain separate grow state."""
        await manager.handle_interim_update(FakeChunk(chunk_id="c1", text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(chunk_id="c2", text="hi"), is_final=False)
        # Shrink on c1 should be suppressed, but c2 is independent
        await manager.handle_interim_update(FakeChunk(chunk_id="c1", text="hello"), is_final=False)
        assert len(broadcasts) == 2  # c1 first + c2 first; c1 shrink suppressed

    @pytest.mark.asyncio
    async def test_cleanup_stale_displayed_text(self, manager, broadcasts):
        """cleanup_stale_displayed_text clears internal state."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert manager.stats["displayed_text_entries"] == 1
        removed = manager.cleanup_stale_displayed_text()
        assert removed == 1
        assert manager.stats["displayed_text_entries"] == 0

    @pytest.mark.asyncio
    async def test_translated_mode_filters_interim(self, manager, broadcasts):
        """Display mode 'translated' filters non-final interim updates."""
        manager._config.display_mode = "translated"
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 0
        assert manager.stats["interim_updates_filtered"] == 1

    @pytest.mark.asyncio
    async def test_interim_disabled_filters_non_final(self, manager, broadcasts):
        """When interim captions disabled, only finals pass."""
        manager._config.enable_interim_captions = False
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 0
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=True)
        assert len(broadcasts) == 1
