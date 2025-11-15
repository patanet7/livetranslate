#!/usr/bin/env python3
"""
Time Correlation Engine - Orchestration Service Integration

Correlates Google Meet speaker timeline with our high-quality transcription
and translation results based on timestamps. Handles timing differences,
audio delays, and provides confidence scoring for correlations.
Now integrated directly into the orchestration service.

Features:
- Timestamp-based correlation between external and internal data
- Audio delay compensation and timing tolerance
- Speaker transition handling and overlap detection
- Confidence scoring for correlations
- Real-time correlation processing
- Database integration for persistent storage
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExternalSpeakerEvent:
    """Speaker event from external source (Google Meet)."""

    speaker_id: str
    speaker_name: str
    event_type: str  # 'speaking_start', 'speaking_end', 'join', 'leave'
    timestamp: float
    duration: Optional[float] = None
    confidence: float = 1.0
    source: str = "google_meet"
    metadata: Dict[str, Any] = None


@dataclass
class InternalTranscriptionResult:
    """Transcription result from our internal processing."""

    segment_id: str
    text: str
    start_timestamp: float
    end_timestamp: float
    language: str
    confidence: float
    session_id: str
    audio_file_id: Optional[str] = None
    processing_metadata: Dict[str, Any] = None


@dataclass
class CorrelationResult:
    """Result of correlating external speaker data with internal transcription."""

    correlation_id: str
    external_event: ExternalSpeakerEvent
    internal_result: InternalTranscriptionResult
    correlation_confidence: float
    timing_offset: float  # Difference between external and internal timestamps
    correlation_type: str  # 'exact', 'interpolated', 'inferred'
    timestamp: float
    metadata: Dict[str, Any] = None


@dataclass
class CorrelationConfig:
    """Configuration for time correlation."""

    timing_tolerance: float = 2.0  # seconds
    audio_delay_compensation: float = 0.5  # seconds
    min_correlation_confidence: float = 0.7
    overlap_handling: str = "weighted"  # 'weighted', 'latest', 'strongest'
    speaker_transition_buffer: float = 0.3  # seconds
    interpolation_max_gap: float = 5.0  # seconds


class TimingAnalyzer:
    """Analyzes timing patterns and calculates offsets."""

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.timing_history: deque = deque(maxlen=100)
        self.average_offset: float = 0.0
        self.offset_variance: float = 0.0

    def analyze_timing_offset(
        self, external_time: float, internal_time: float
    ) -> Tuple[float, float]:
        """
        Analyze timing offset between external and internal timestamps.

        Args:
            external_time: Timestamp from external source
            internal_time: Timestamp from internal processing

        Returns:
            Tuple of (offset, confidence)
        """
        offset = external_time - internal_time

        # Store in history for trend analysis
        self.timing_history.append(offset)

        # Update average offset
        if len(self.timing_history) > 5:
            self.average_offset = np.mean(list(self.timing_history))
            self.offset_variance = np.var(list(self.timing_history))

        # Calculate confidence based on consistency
        if len(self.timing_history) < 3:
            confidence = 0.5  # Low confidence with limited data
        else:
            # Higher confidence for consistent offsets
            deviation = abs(offset - self.average_offset)
            max_deviation = 3.0  # seconds
            confidence = max(0.1, 1.0 - (deviation / max_deviation))

        return offset, confidence

    def compensate_audio_delay(self, timestamp: float) -> float:
        """Apply audio delay compensation to timestamp."""
        return timestamp + self.config.audio_delay_compensation + self.average_offset

    def is_within_tolerance(self, time1: float, time2: float) -> bool:
        """Check if two timestamps are within tolerance."""
        return abs(time1 - time2) <= self.config.timing_tolerance


class SpeakerStateTracker:
    """Tracks current speaker states and transitions."""

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.active_speakers: Dict[str, float] = {}  # speaker_id -> start_time
        self.recent_transitions: deque = deque(maxlen=50)

    def update_speaker_state(self, speaker_id: str, event_type: str, timestamp: float):
        """Update speaker state based on event."""
        if event_type == "speaking_start":
            self.active_speakers[speaker_id] = timestamp
        elif event_type == "speaking_end":
            if speaker_id in self.active_speakers:
                duration = timestamp - self.active_speakers[speaker_id]
                self.recent_transitions.append(
                    {
                        "speaker_id": speaker_id,
                        "start_time": self.active_speakers[speaker_id],
                        "end_time": timestamp,
                        "duration": duration,
                    }
                )
                del self.active_speakers[speaker_id]

    def get_active_speakers_at_time(self, timestamp: float) -> List[str]:
        """Get speakers who were active at a given time."""
        active = []

        # Check currently active speakers
        for speaker_id, start_time in self.active_speakers.items():
            if start_time <= timestamp:
                active.append(speaker_id)

        # Check recent transitions
        for transition in self.recent_transitions:
            if transition["start_time"] <= timestamp <= transition["end_time"]:
                active.append(transition["speaker_id"])

        return list(set(active))

    def get_most_likely_speaker(self, timestamp: float) -> Optional[str]:
        """Get the most likely speaker at a given time."""
        active_speakers = self.get_active_speakers_at_time(timestamp)

        if not active_speakers:
            return None
        elif len(active_speakers) == 1:
            return active_speakers[0]
        else:
            # Handle overlapping speakers - return most recent
            most_recent = None
            latest_start = 0

            for speaker_id in active_speakers:
                if speaker_id in self.active_speakers:
                    start_time = self.active_speakers[speaker_id]
                    if start_time > latest_start:
                        latest_start = start_time
                        most_recent = speaker_id

            return most_recent


class TimeCorrelationEngine:
    """
    Main engine for correlating external speaker timeline with internal transcription.
    Integrated with orchestration service bot management and database.
    """

    def __init__(
        self,
        session_id: str,
        config: CorrelationConfig = None,
        bot_manager=None,
        database_manager=None,
    ):
        self.session_id = session_id
        self.config = config or CorrelationConfig()
        self.bot_manager = bot_manager
        self.database_manager = database_manager

        # Components
        self.timing_analyzer = TimingAnalyzer(self.config)
        self.speaker_tracker = SpeakerStateTracker(self.config)

        # Data storage
        self.external_events: List[ExternalSpeakerEvent] = []
        self.internal_results: List[InternalTranscriptionResult] = []
        self.correlations: List[CorrelationResult] = []

        # Pending correlation queue
        self.pending_internal: deque = deque(maxlen=100)
        self.pending_external: deque = deque(maxlen=100)

        # Performance tracking
        self.total_correlations = 0
        self.successful_correlations = 0
        self.average_confidence = 0.0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"TimeCorrelationEngine initialized for session: {session_id}")
        logger.info(f"  Timing tolerance: {self.config.timing_tolerance}s")
        logger.info(
            f"  Audio delay compensation: {self.config.audio_delay_compensation}s"
        )

    def add_external_event(self, event: ExternalSpeakerEvent) -> bool:
        """Add external speaker event for correlation."""
        with self.lock:
            try:
                self.external_events.append(event)
                self.pending_external.append(event)

                # Update speaker state
                self.speaker_tracker.update_speaker_state(
                    event.speaker_id, event.event_type, event.timestamp
                )

                # Try immediate correlation
                self._attempt_correlations()

                logger.debug(
                    f"Added external event: {event.speaker_id} {event.event_type} at {event.timestamp}"
                )
                return True

            except Exception as e:
                logger.error(f"Error adding external event: {e}")
                return False

    def add_internal_result(self, result: InternalTranscriptionResult) -> bool:
        """Add internal transcription result for correlation."""
        with self.lock:
            try:
                self.internal_results.append(result)
                self.pending_internal.append(result)

                # Try immediate correlation
                correlations_made = self._attempt_correlations()

                # Store correlations in database if available
                if correlations_made and self.bot_manager and self.database_manager:
                    for correlation in correlations_made:
                        asyncio.create_task(self._store_correlation(correlation))

                logger.debug(
                    f"Added internal result: {result.text[:50]}... at {result.start_timestamp}"
                )
                return True

            except Exception as e:
                logger.error(f"Error adding internal result: {e}")
                return False

    async def _store_correlation(self, correlation: CorrelationResult):
        """Store correlation result in database."""
        try:
            correlation_data = {
                "google_transcript_id": None,  # Will be set if we have the transcript ID
                "inhouse_transcript_id": correlation.internal_result.segment_id,
                "correlation_confidence": correlation.correlation_confidence,
                "timing_offset": correlation.timing_offset,
                "correlation_type": correlation.correlation_type,
                "correlation_method": correlation.metadata.get("method", "unknown"),
                "speaker_id": correlation.external_event.speaker_id,
                "start_timestamp": correlation.internal_result.start_timestamp,
                "end_timestamp": correlation.internal_result.end_timestamp,
                "correlation_metadata": {
                    "external_event": asdict(correlation.external_event),
                    "internal_result": asdict(correlation.internal_result),
                    "metadata": correlation.metadata or {},
                },
            }

            await self.bot_manager.store_correlation(self.session_id, correlation_data)

        except Exception as e:
            logger.error(f"Error storing correlation: {e}")

    def _attempt_correlations(self):
        """Attempt to correlate pending external events with internal results."""
        correlations_made = []

        # Try to correlate pending internal results
        for internal_result in list(self.pending_internal):
            correlation = self._find_best_correlation(internal_result)
            if correlation:
                correlations_made.append(correlation)
                self.correlations.append(correlation)
                self.pending_internal.remove(internal_result)

                # Update statistics
                self.total_correlations += 1
                if (
                    correlation.correlation_confidence
                    >= self.config.min_correlation_confidence
                ):
                    self.successful_correlations += 1

                self._update_average_confidence(correlation.correlation_confidence)

        return correlations_made

    def _find_best_correlation(
        self, internal_result: InternalTranscriptionResult
    ) -> Optional[CorrelationResult]:
        """Find the best correlation for an internal transcription result."""
        best_correlation = None
        best_confidence = 0.0

        # Compensate for audio delay
        compensated_start = self.timing_analyzer.compensate_audio_delay(
            internal_result.start_timestamp
        )
        compensated_end = self.timing_analyzer.compensate_audio_delay(
            internal_result.end_timestamp
        )

        # Method 1: Direct timeline correlation
        direct_correlation = self._correlate_with_timeline(
            internal_result, compensated_start, compensated_end
        )
        if (
            direct_correlation
            and direct_correlation.correlation_confidence > best_confidence
        ):
            best_correlation = direct_correlation
            best_confidence = direct_correlation.correlation_confidence

        # Method 2: Speaker state inference
        inferred_correlation = self._infer_from_speaker_state(
            internal_result, compensated_start, compensated_end
        )
        if (
            inferred_correlation
            and inferred_correlation.correlation_confidence > best_confidence
        ):
            best_correlation = inferred_correlation
            best_confidence = inferred_correlation.correlation_confidence

        # Method 3: Interpolation from nearby events
        interpolated_correlation = self._interpolate_from_nearby_events(
            internal_result, compensated_start, compensated_end
        )
        if (
            interpolated_correlation
            and interpolated_correlation.correlation_confidence > best_confidence
        ):
            best_correlation = interpolated_correlation
            best_confidence = interpolated_correlation.correlation_confidence

        return (
            best_correlation
            if best_confidence >= self.config.min_correlation_confidence
            else None
        )

    def _correlate_with_timeline(
        self,
        internal_result: InternalTranscriptionResult,
        start_time: float,
        end_time: float,
    ) -> Optional[CorrelationResult]:
        """Correlate with explicit timeline events."""
        for event in self.external_events:
            if event.event_type not in ["speaking_start", "speaking_end"]:
                continue

            # Check if timing overlaps
            if self.timing_analyzer.is_within_tolerance(event.timestamp, start_time):
                offset, timing_confidence = self.timing_analyzer.analyze_timing_offset(
                    event.timestamp, internal_result.start_timestamp
                )

                correlation_confidence = timing_confidence * event.confidence

                correlation = CorrelationResult(
                    correlation_id=f"timeline_{internal_result.segment_id}_{event.speaker_id}",
                    external_event=event,
                    internal_result=internal_result,
                    correlation_confidence=correlation_confidence,
                    timing_offset=offset,
                    correlation_type="exact",
                    timestamp=time.time(),
                    metadata={
                        "method": "timeline_correlation",
                        "timing_confidence": timing_confidence,
                    },
                )

                return correlation

        return None

    def _infer_from_speaker_state(
        self,
        internal_result: InternalTranscriptionResult,
        start_time: float,
        end_time: float,
    ) -> Optional[CorrelationResult]:
        """Infer speaker from current speaker state."""
        # Get most likely speaker at the time
        likely_speaker = self.speaker_tracker.get_most_likely_speaker(start_time)

        if likely_speaker:
            # Find corresponding external event
            matching_event = None
            for event in self.external_events:
                if (
                    event.speaker_id == likely_speaker
                    and event.event_type == "speaking_start"
                ):
                    if (
                        abs(event.timestamp - start_time)
                        <= self.config.timing_tolerance * 2
                    ):
                        matching_event = event
                        break

            if matching_event:
                offset, timing_confidence = self.timing_analyzer.analyze_timing_offset(
                    matching_event.timestamp, internal_result.start_timestamp
                )

                # Lower confidence for inferred correlations
                correlation_confidence = timing_confidence * 0.8

                correlation = CorrelationResult(
                    correlation_id=f"inferred_{internal_result.segment_id}_{likely_speaker}",
                    external_event=matching_event,
                    internal_result=internal_result,
                    correlation_confidence=correlation_confidence,
                    timing_offset=offset,
                    correlation_type="inferred",
                    timestamp=time.time(),
                    metadata={
                        "method": "speaker_state_inference",
                        "timing_confidence": timing_confidence,
                    },
                )

                return correlation

        return None

    def _interpolate_from_nearby_events(
        self,
        internal_result: InternalTranscriptionResult,
        start_time: float,
        end_time: float,
    ) -> Optional[CorrelationResult]:
        """Interpolate speaker from nearby timeline events."""
        # Find events before and after
        before_events = [e for e in self.external_events if e.timestamp < start_time]
        after_events = [e for e in self.external_events if e.timestamp > end_time]

        if not before_events:
            return None

        # Get most recent before event
        recent_event = max(before_events, key=lambda e: e.timestamp)

        # Check if gap is within interpolation range
        gap = start_time - recent_event.timestamp
        if gap > self.config.interpolation_max_gap:
            return None

        offset, timing_confidence = self.timing_analyzer.analyze_timing_offset(
            recent_event.timestamp, internal_result.start_timestamp
        )

        # Reduce confidence based on gap size
        gap_factor = 1.0 - (gap / self.config.interpolation_max_gap)
        correlation_confidence = timing_confidence * gap_factor * 0.6

        correlation = CorrelationResult(
            correlation_id=f"interpolated_{internal_result.segment_id}_{recent_event.speaker_id}",
            external_event=recent_event,
            internal_result=internal_result,
            correlation_confidence=correlation_confidence,
            timing_offset=offset,
            correlation_type="interpolated",
            timestamp=time.time(),
            metadata={
                "method": "timeline_interpolation",
                "gap_seconds": gap,
                "gap_factor": gap_factor,
                "timing_confidence": timing_confidence,
            },
        )

        return correlation

    def _update_average_confidence(self, new_confidence: float):
        """Update running average confidence."""
        if self.total_correlations == 1:
            self.average_confidence = new_confidence
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_confidence = (
                1 - alpha
            ) * self.average_confidence + alpha * new_confidence

    def get_correlations(
        self, start_time: float = None, end_time: float = None
    ) -> List[Dict]:
        """Get correlations for a time range."""
        with self.lock:
            filtered = []

            for correlation in self.correlations:
                internal_time = correlation.internal_result.start_timestamp

                if start_time and internal_time < start_time:
                    continue
                if end_time and internal_time > end_time:
                    continue

                filtered.append(
                    {
                        "correlation_id": correlation.correlation_id,
                        "speaker_id": correlation.external_event.speaker_id,
                        "speaker_name": correlation.external_event.speaker_name,
                        "text": correlation.internal_result.text,
                        "start_timestamp": correlation.internal_result.start_timestamp,
                        "end_timestamp": correlation.internal_result.end_timestamp,
                        "correlation_confidence": correlation.correlation_confidence,
                        "timing_offset": correlation.timing_offset,
                        "correlation_type": correlation.correlation_type,
                        "language": correlation.internal_result.language,
                        "transcription_confidence": correlation.internal_result.confidence,
                    }
                )

            return sorted(filtered, key=lambda x: x["start_timestamp"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        with self.lock:
            success_rate = self.successful_correlations / max(
                1, self.total_correlations
            )

            return {
                "session_id": self.session_id,
                "total_external_events": len(self.external_events),
                "total_internal_results": len(self.internal_results),
                "total_correlations": self.total_correlations,
                "successful_correlations": self.successful_correlations,
                "success_rate": success_rate,
                "average_confidence": self.average_confidence,
                "pending_internal": len(self.pending_internal),
                "pending_external": len(self.pending_external),
                "average_timing_offset": self.timing_analyzer.average_offset,
                "timing_variance": self.timing_analyzer.offset_variance,
                "config": asdict(self.config),
            }

    def export_correlated_transcript(self, format: str = "json") -> str:
        """Export correlated transcript with speaker attribution."""
        correlations = self.get_correlations()

        if format == "json":
            return json.dumps(
                {
                    "session_id": self.session_id,
                    "correlations": correlations,
                    "statistics": self.get_statistics(),
                },
                indent=2,
                default=str,
            )

        elif format == "text":
            lines = []
            for corr in correlations:
                timestamp = datetime.fromtimestamp(corr["start_timestamp"]).strftime(
                    "%H:%M:%S"
                )
                confidence = f"({corr['correlation_confidence']:.2f})"
                lines.append(
                    f"[{timestamp}] {corr['speaker_name']}: {corr['text']} {confidence}"
                )
            return "\n".join(lines)

        elif format == "srt":
            lines = []
            for i, corr in enumerate(correlations, 1):
                start_time = self._format_srt_time(corr["start_timestamp"])
                end_time = self._format_srt_time(corr["end_timestamp"])
                speaker_text = f"{corr['speaker_name']}: {corr['text']}"

                lines.extend([str(i), f"{start_time} --> {end_time}", speaker_text, ""])
            return "\n".join(lines)

        return json.dumps(correlations, indent=2, default=str)

    def _format_srt_time(self, timestamp: float) -> str:
        """Format timestamp for SRT format."""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S,000")


# Factory functions
def create_correlation_engine(
    session_id: str, bot_manager=None, database_manager=None, **config_kwargs
) -> TimeCorrelationEngine:
    """Create a time correlation engine with optional configuration."""
    config = CorrelationConfig(**config_kwargs)
    return TimeCorrelationEngine(session_id, config, bot_manager, database_manager)


# Example usage
async def main():
    """Example usage of the time correlation engine."""
    engine = create_correlation_engine("test-session-123")

    # Simulate external events (from Google Meet)
    external_events = [
        ExternalSpeakerEvent(
            "speaker_john_doe",
            "John Doe",
            "speaking_start",
            time.time(),
            source="google_meet",
        ),
        ExternalSpeakerEvent(
            "speaker_jane_smith",
            "Jane Smith",
            "join",
            time.time() + 5,
            source="google_meet",
        ),
        ExternalSpeakerEvent(
            "speaker_jane_smith",
            "Jane Smith",
            "speaking_start",
            time.time() + 10,
            source="google_meet",
        ),
    ]

    # Simulate internal transcription results
    internal_results = [
        InternalTranscriptionResult(
            "seg1",
            "Hello everyone welcome to the meeting",
            time.time() + 0.5,
            time.time() + 3.5,
            "en",
            0.95,
            "test-session-123",
        ),
        InternalTranscriptionResult(
            "seg2",
            "Thanks for having me glad to be here",
            time.time() + 10.3,
            time.time() + 12.8,
            "en",
            0.92,
            "test-session-123",
        ),
    ]

    # Add events and results
    for event in external_events:
        engine.add_external_event(event)

    for result in internal_results:
        engine.add_internal_result(result)

    # Get correlations
    correlations = engine.get_correlations()
    print(f"Correlations: {json.dumps(correlations, indent=2, default=str)}")

    # Get statistics
    stats = engine.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")

    # Export transcript
    transcript = engine.export_correlated_transcript("text")
    print(f"Transcript:\n{transcript}")


if __name__ == "__main__":
    asyncio.run(main())
