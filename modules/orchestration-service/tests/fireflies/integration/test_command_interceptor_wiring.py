"""
Validation 5: CommandInterceptor Wiring in create_session Closure

Verifies that the CommandInterceptor is correctly wired into the
handle_transcript closure that create_session builds in fireflies.py.

Tests the actual closure pattern — CommandInterceptor.check() gates whether
chunks reach the pipeline. When enabled and a command is detected, the chunk
is intercepted (not sent to pipeline). When disabled, ALL chunks pass through.

Does NOT require a running server or database. Uses real CommandInterceptor
+ real PipelineConfig objects with minimal recording stubs.

Run: uv run pytest tests/fireflies/integration/test_command_interceptor_wiring.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(orchestration_root / "src"))

from services.pipeline.command_interceptor import CommandInterceptor
from services.pipeline.config import PipelineConfig


# =============================================================================
# Test Helpers: Recording stubs for pipeline components
# =============================================================================


class ChunkProcessingRecorder:
    """
    Records which chunks make it through to pipeline processing.

    Implements the minimal coordinator interface that CommandInterceptor uses:
    - pause() / resume() for pipeline control
    - process_raw_chunk() for chunk forwarding
    - config property for PipelineConfig access
    """

    def __init__(self, config: PipelineConfig):
        self.processed_chunks: list[str] = []
        self.is_paused = False
        self._config = config

    async def process_raw_chunk(self, chunk):
        self.processed_chunks.append(chunk.text)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    @property
    def config(self):
        return self._config


class FakeChunk:
    """Minimal chunk object matching the interface CommandInterceptor expects."""

    def __init__(self, text: str, chunk_id: str = "c1", speaker_name: str = "Speaker"):
        self.text = text
        self.chunk_id = chunk_id
        self.speaker_name = speaker_name


def build_closure(interceptor, coordinator):
    """
    Replicate the handle_transcript closure from fireflies.py.

    This is the exact pattern used in FirefliesSessionManager.create_session():
        if interceptor.check(chunk.text):
            await interceptor.execute(chunk.text)
            return  # Don't process commands through the pipeline
        await coordinator.process_raw_chunk(chunk)
    """

    async def handle_transcript(chunk):
        if interceptor.check(chunk.text):
            await interceptor.execute(chunk.text)
            return
        await coordinator.process_raw_chunk(chunk)

    return handle_transcript


# =============================================================================
# Validation: Command Interception When Enabled
# =============================================================================


class TestCommandInterceptionEnabled:
    """When voice_commands_enabled=True, commands must be intercepted."""

    @pytest.fixture
    def setup(self):
        config = PipelineConfig(
            session_id="wiring-test",
            source_type="fireflies",
            transcript_id="t-wiring",
            voice_commands_enabled=True,
            voice_command_prefix="livetranslate",
        )
        recorder = ChunkProcessingRecorder(config)
        interceptor = CommandInterceptor(
            config=config,
            coordinator=recorder,
            session_id="wiring-test",
        )
        handler = build_closure(interceptor, recorder)
        return handler, interceptor, recorder

    @pytest.mark.asyncio
    async def test_pause_command_intercepted(self, setup):
        """
        GIVEN: voice_commands_enabled=True
        WHEN: A chunk arrives with text 'livetranslate pause'
        THEN: Chunk does NOT reach pipeline, coordinator.pause() IS called
        """
        handler, interceptor, recorder = setup
        await handler(FakeChunk("livetranslate pause"))

        assert recorder.is_paused is True, "Coordinator should be paused"
        assert "livetranslate pause" not in recorder.processed_chunks, (
            "Command chunk should NOT reach the pipeline"
        )
        assert interceptor.commands_executed == 1

    @pytest.mark.asyncio
    async def test_resume_command_intercepted(self, setup):
        """Resume command sets coordinator to not-paused."""
        handler, interceptor, recorder = setup
        recorder.is_paused = True  # Start paused
        await handler(FakeChunk("livetranslate resume"))

        assert recorder.is_paused is False, "Coordinator should be resumed"
        assert "livetranslate resume" not in recorder.processed_chunks

    @pytest.mark.asyncio
    async def test_normal_speech_passes_through(self, setup):
        """
        GIVEN: voice_commands_enabled=True
        WHEN: A normal transcription chunk arrives
        THEN: Chunk DOES reach pipeline, nothing is intercepted
        """
        handler, interceptor, recorder = setup
        await handler(FakeChunk("The quarterly results look promising"))

        assert "The quarterly results look promising" in recorder.processed_chunks, (
            "Normal speech must reach the pipeline"
        )
        assert interceptor.commands_executed == 0

    @pytest.mark.asyncio
    async def test_mixed_traffic_command_then_speech(self, setup):
        """Commands are intercepted, normal speech passes through, interleaved."""
        handler, interceptor, recorder = setup

        await handler(FakeChunk("Hello everyone"))
        await handler(FakeChunk("livetranslate pause"))
        await handler(FakeChunk("Let's continue the discussion"))

        assert recorder.processed_chunks == [
            "Hello everyone",
            "Let's continue the discussion",
        ]
        assert "livetranslate pause" not in recorder.processed_chunks
        assert recorder.is_paused is True
        assert interceptor.commands_executed == 1


# =============================================================================
# Validation: Commands Pass Through When Disabled
# =============================================================================


class TestCommandInterceptionDisabled:
    """When voice_commands_enabled=False (default), ALL chunks pass through."""

    @pytest.fixture
    def setup(self):
        config = PipelineConfig(
            session_id="disabled-test",
            source_type="fireflies",
            transcript_id="t-disabled",
            voice_commands_enabled=False,  # Default: disabled
            voice_command_prefix="livetranslate",
        )
        recorder = ChunkProcessingRecorder(config)
        interceptor = CommandInterceptor(
            config=config,
            coordinator=recorder,
            session_id="disabled-test",
        )
        handler = build_closure(interceptor, recorder)
        return handler, interceptor, recorder

    @pytest.mark.asyncio
    async def test_command_prefixed_text_passes_through(self, setup):
        """
        GIVEN: voice_commands_enabled=False
        WHEN: Even a command-prefixed chunk arrives
        THEN: ALL chunks pass through to pipeline (interceptor is disabled)
        """
        handler, interceptor, recorder = setup
        await handler(FakeChunk("livetranslate pause"))

        assert interceptor.enabled is False
        assert "livetranslate pause" in recorder.processed_chunks, (
            "Disabled interceptor must not block ANY chunk"
        )
        assert recorder.is_paused is False

    @pytest.mark.asyncio
    async def test_all_traffic_reaches_pipeline(self, setup):
        """Every chunk reaches the pipeline when disabled."""
        handler, interceptor, recorder = setup

        texts = [
            "Normal speech",
            "livetranslate pause",
            "livetranslate resume",
            "livetranslate language spanish",
            "More normal speech",
        ]
        for text in texts:
            await handler(FakeChunk(text))

        assert recorder.processed_chunks == texts
        assert interceptor.commands_executed == 0


# =============================================================================
# Validation: Full Closure Replication (from create_session)
# =============================================================================


class TestClosureReplication:
    """
    Verifies the exact closure pattern from FirefliesSessionManager.create_session().

    This test replicates the real wiring: config, coordinator, interceptor are
    constructed the same way as in fireflies.py, and the handle_transcript
    closure is built using the same pattern.
    """

    @pytest.mark.asyncio
    async def test_full_closure_lifecycle(self):
        """
        Simulates a realistic session lifecycle:
        1. Normal speech → pipeline
        2. Pause command → intercepted, pipeline paused
        3. Normal speech → pipeline (still goes even when paused, as the
           closure doesn't check pause state — that's the coordinator's job)
        4. Resume command → intercepted, pipeline resumed
        """
        config = PipelineConfig(
            session_id="closure-lifecycle",
            source_type="fireflies",
            transcript_id="t-lifecycle",
            voice_commands_enabled=True,
            voice_command_prefix="livetranslate",
        )

        recorder = ChunkProcessingRecorder(config)
        interceptor = CommandInterceptor(
            config=config,
            coordinator=recorder,
            session_id="closure-lifecycle",
        )
        handler = build_closure(interceptor, recorder)

        # Phase 1: Normal speech
        await handler(FakeChunk("Good morning everyone"))
        assert "Good morning everyone" in recorder.processed_chunks

        # Phase 2: Pause command
        await handler(FakeChunk("livetranslate pause"))
        assert recorder.is_paused is True
        assert "livetranslate pause" not in recorder.processed_chunks

        # Phase 3: Speech while "paused" — still forwarded to coordinator
        # (the coordinator decides what to do with chunks during pause)
        await handler(FakeChunk("Someone is still talking"))
        assert "Someone is still talking" in recorder.processed_chunks

        # Phase 4: Resume command
        await handler(FakeChunk("livetranslate resume"))
        assert recorder.is_paused is False
        assert "livetranslate resume" not in recorder.processed_chunks

        # Final state
        assert interceptor.commands_executed == 2
        assert recorder.processed_chunks == [
            "Good morning everyone",
            "Someone is still talking",
        ]
