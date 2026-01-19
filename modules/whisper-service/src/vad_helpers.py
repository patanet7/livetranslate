#!/usr/bin/env python3
"""
VAD Processing Helpers

Simplifies complex VAD result handling logic with clear, testable functions.

Usage:
    from vad_helpers import VADEventType, parse_vad_event, should_buffer_audio

    event_type = parse_vad_event(vad_result)
    should_buffer = should_buffer_audio(event_type, current_status)
"""

from enum import Enum

from type_definitions import VADResult


class VADEventType(Enum):
    """VAD event types for clearer state management"""

    NO_CHANGE = "no_change"  # No VAD event detected
    SPEECH_START = "speech_start"  # Speech started (pure start, no end)
    SPEECH_END = "speech_end"  # Speech ended (pure end, no start)
    SPEECH_RESTART = "speech_restart"  # Speech ended AND immediately restarted


class VADStatus(Enum):
    """Current VAD status"""

    VOICE = "voice"  # Currently in speech
    NONVOICE = "nonvoice"  # Currently in silence


def parse_vad_event(vad_result: VADResult | None) -> VADEventType:
    """
    Parse VAD result into a clear event type.

    Simplifies the complex conditional logic in session_manager.py lines 341-373.

    Args:
        vad_result: VAD detection result (can be None, have 'start', 'end', or both)

    Returns:
        VADEventType indicating what happened

    Examples:
        >>> parse_vad_event(None)
        VADEventType.NO_CHANGE

        >>> parse_vad_event({'start': 1.0})
        VADEventType.SPEECH_START

        >>> parse_vad_event({'end': 2.0})
        VADEventType.SPEECH_END

        >>> parse_vad_event({'start': 1.0, 'end': 2.0})
        VADEventType.SPEECH_RESTART
    """
    if vad_result is None:
        return VADEventType.NO_CHANGE

    has_start = "start" in vad_result
    has_end = "end" in vad_result

    if has_start and has_end:
        # Both events - speech ended and immediately restarted
        return VADEventType.SPEECH_RESTART
    elif has_start:
        # Pure start event
        return VADEventType.SPEECH_START
    elif has_end:
        # Pure end event
        return VADEventType.SPEECH_END
    else:
        # Empty dict (shouldn't happen, but handle gracefully)
        return VADEventType.NO_CHANGE


def should_buffer_audio(event_type: VADEventType, current_status: VADStatus) -> bool:
    """
    Determine if audio chunk should be buffered based on VAD state.

    Key principle: ONLY buffer speech audio, NEVER buffer silence.
    This prevents Whisper hallucinations (Milestone 1 baseline pattern).

    Args:
        event_type: Parsed VAD event type
        current_status: Current VAD status

    Returns:
        True if audio should be buffered, False if it should be skipped

    Examples:
        >>> should_buffer_audio(VADEventType.SPEECH_START, VADStatus.NONVOICE)
        True  # Speech just started, buffer it

        >>> should_buffer_audio(VADEventType.SPEECH_END, VADStatus.VOICE)
        False  # Speech ended, entering silence

        >>> should_buffer_audio(VADEventType.NO_CHANGE, VADStatus.VOICE)
        True  # Ongoing speech, keep buffering

        >>> should_buffer_audio(VADEventType.NO_CHANGE, VADStatus.NONVOICE)
        False  # Ongoing silence, don't buffer
    """
    if event_type == VADEventType.SPEECH_START:
        # Speech started - buffer this chunk
        return True

    elif event_type == VADEventType.SPEECH_END:
        # Speech ended - don't buffer (entering silence)
        return False

    elif event_type == VADEventType.SPEECH_RESTART:
        # Speech restarted - keep buffering
        return True

    elif event_type == VADEventType.NO_CHANGE:
        # No event - buffer if we're in speech, skip if in silence
        return current_status == VADStatus.VOICE

    return False


def should_process_buffer(event_type: VADEventType) -> bool:
    """
    Determine if buffered audio should be processed.

    Processing happens when speech ends (VAD boundary).

    Args:
        event_type: Parsed VAD event type

    Returns:
        True if buffer should be processed now

    Examples:
        >>> should_process_buffer(VADEventType.SPEECH_END)
        True  # Speech ended, process accumulated buffer

        >>> should_process_buffer(VADEventType.SPEECH_RESTART)
        True  # Speech ended then restarted, process buffer from ended segment

        >>> should_process_buffer(VADEventType.SPEECH_START)
        False  # Speech just started, wait for more audio

        >>> should_process_buffer(VADEventType.NO_CHANGE)
        False  # No change, keep buffering
    """
    return event_type in [VADEventType.SPEECH_END, VADEventType.SPEECH_RESTART]


def update_vad_status(event_type: VADEventType, current_status: VADStatus) -> VADStatus:
    """
    Update VAD status based on event.

    Args:
        event_type: Parsed VAD event type
        current_status: Current VAD status

    Returns:
        New VAD status

    Examples:
        >>> update_vad_status(VADEventType.SPEECH_START, VADStatus.NONVOICE)
        VADStatus.VOICE

        >>> update_vad_status(VADEventType.SPEECH_END, VADStatus.VOICE)
        VADStatus.NONVOICE

        >>> update_vad_status(VADEventType.SPEECH_RESTART, VADStatus.VOICE)
        VADStatus.VOICE  # Stay in voice status

        >>> update_vad_status(VADEventType.NO_CHANGE, VADStatus.VOICE)
        VADStatus.VOICE  # No change
    """
    if event_type == VADEventType.SPEECH_START:
        return VADStatus.VOICE

    elif event_type == VADEventType.SPEECH_END:
        return VADStatus.NONVOICE

    elif event_type == VADEventType.SPEECH_RESTART:
        # Speech restarted - stay in VOICE
        return VADStatus.VOICE

    # No change
    return current_status


def get_vad_action_plan(
    vad_result: VADResult | None, current_status: VADStatus
) -> tuple[bool, bool, VADStatus]:
    """
    Get complete action plan from VAD result.

    Convenience function that combines all VAD decision logic.

    Args:
        vad_result: VAD detection result
        current_status: Current VAD status

    Returns:
        Tuple of (should_buffer, should_process, new_status)

    Example:
        should_buffer, should_process, new_status = get_vad_action_plan(
            vad_result, current_vad_status
        )

        if should_buffer:
            buffer.append(audio_chunk)

        if should_process:
            process_buffer(buffer)

        current_vad_status = new_status
    """
    event_type = parse_vad_event(vad_result)

    should_buffer = should_buffer_audio(event_type, current_status)
    should_process = should_process_buffer(event_type)
    new_status = update_vad_status(event_type, current_status)

    return should_buffer, should_process, new_status


# Assertion helpers for invariant checking
def assert_valid_vad_state(status: VADStatus, buffer_size: int):
    """
    Assert VAD state invariants.

    Args:
        status: Current VAD status
        buffer_size: Current buffer size

    Raises:
        AssertionError: If invariants are violated
    """
    # Invariant: If in NONVOICE, buffer should be empty or we're about to process it
    # (This is checked before adding to buffer, so slight buffer during transition is OK)

    # Invariant: Buffer size should never be negative
    assert buffer_size >= 0, f"Invalid buffer size: {buffer_size}"


def assert_valid_audio_chunk(audio_chunk) -> None:
    """
    Assert audio chunk is valid.

    Args:
        audio_chunk: Audio numpy array

    Raises:
        AssertionError: If audio chunk is invalid
    """
    import numpy as np

    assert audio_chunk is not None, "Audio chunk is None"
    assert isinstance(
        audio_chunk, np.ndarray
    ), f"Audio chunk must be ndarray, got {type(audio_chunk)}"
    assert len(audio_chunk) > 0, "Audio chunk is empty"
    assert audio_chunk.dtype in [
        np.float32,
        np.float64,
    ], f"Audio must be float32/float64, got {audio_chunk.dtype}"
    assert not np.isnan(audio_chunk).any(), "Audio contains NaN values"
    assert not np.isinf(audio_chunk).any(), "Audio contains infinite values"
