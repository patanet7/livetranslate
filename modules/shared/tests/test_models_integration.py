"""Integration test: verify all shared models are importable from the top-level package."""


def test_top_level_imports():
    """All shared types must be importable from livetranslate_common directly."""
    from livetranslate_common import (
        AudioChunk,
        BackendConfig,
        ModelInfo,
        Segment,
        TranscriptionResult,
        TranslationContext,
        TranslationRequest,
        TranslationResponse,
    )
    assert TranscriptionResult.__name__ == "TranscriptionResult"
    assert BackendConfig.__name__ == "BackendConfig"


def test_ws_message_imports():
    """WebSocket messages must be importable from livetranslate_common.models.ws_messages."""
    from livetranslate_common.models.ws_messages import (
        PROTOCOL_VERSION,
        BackendSwitchedMessage,
        ConfigMessage,
        ConnectedMessage,
        LanguageDetectedMessage,
        parse_ws_message,
    )
    assert PROTOCOL_VERSION >= 1
    assert parse_ws_message('{"type": "connected", "session_id": "x"}') is not None


def test_models_subpackage_reexports():
    """The models subpackage __init__ must re-export all types."""
    from livetranslate_common.models import (
        AudioChunk,
        BackendConfig,
        ModelInfo,
        Segment,
        TranscriptionResult,
        TranslationContext,
        TranslationRequest,
        TranslationResponse,
    )
    seg = Segment(text="hi", start_ms=0, end_ms=100, confidence=0.9)
    chunk = AudioChunk(data=b"\x00", timestamp_ms=0, sequence_number=0, source_id="test")
    ctx = TranslationContext(text="a", translation="b")
    assert seg.text == "hi"
    assert chunk.source_id == "test"
    assert ctx.translation == "b"
