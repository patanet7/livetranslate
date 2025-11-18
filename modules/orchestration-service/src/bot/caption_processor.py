#!/usr/bin/env python3
"""
Google Meet Caption Processor - Orchestration Service Integration

Extracts speaker timeline information from Google Meet captions and manages
participant data for time-based correlation with our audio processing.
Now integrated directly into the orchestration service.

Features:
- Real-time caption parsing from Google Meet
- Speaker timeline extraction with precise timestamps
- Participant management (join/leave events)
- Speaker name and role tracking
- Integration with bot session management and database
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a meeting participant."""

    speaker_id: str
    display_name: str
    email: Optional[str] = None
    role: Optional[str] = None  # 'organizer', 'participant', 'guest'
    join_time: Optional[float] = None
    leave_time: Optional[float] = None
    is_active: bool = True
    total_speaking_time: float = 0.0
    utterance_count: int = 0


@dataclass
class CaptionSegment:
    """A single caption segment with speaker and timing information."""

    speaker_id: str
    speaker_name: str
    text: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    confidence: float = 1.0
    caption_source: str = "google_meet"
    segment_id: Optional[str] = None


@dataclass
class SpeakerTimelineEvent:
    """Event in the speaker timeline (speaking, joining, leaving, etc.)."""

    event_type: str  # 'speaking_start', 'speaking_end', 'join', 'leave', 'role_change'
    speaker_id: str
    timestamp: float
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None


class GoogleMeetCaptionParser:
    """Parser for Google Meet caption formats."""

    def __init__(self):
        # Common patterns for Google Meet captions
        self.caption_patterns = [
            # Pattern: "John Doe: Hello everyone"
            re.compile(r"^([^:]+):\s*(.+)$"),
            # Pattern: "[John Doe] Hello everyone"
            re.compile(r"^\[([^\]]+)\]\s*(.+)$"),
            # Pattern: "John Doe - Hello everyone"
            re.compile(r"^([^-]+)-\s*(.+)$"),
            # Pattern: "<John Doe> Hello everyone"
            re.compile(r"^<([^>]+)>\s*(.+)$"),
        ]

        # Patterns for system messages
        self.system_patterns = [
            re.compile(r"(.+) joined the meeting"),
            re.compile(r"(.+) left the meeting"),
            re.compile(r"(.+) is presenting"),
            re.compile(r"(.+) stopped presenting"),
            re.compile(r"Meeting recording started"),
            re.compile(r"Meeting recording stopped"),
        ]

    def parse_caption(
        self, caption_text: str, timestamp: float
    ) -> Tuple[Optional[CaptionSegment], Optional[SpeakerTimelineEvent]]:
        """
        Parse a caption line and extract speaker and text information.

        Args:
            caption_text: Raw caption text from Google Meet
            timestamp: Timestamp of the caption

        Returns:
            Tuple of (CaptionSegment, SpeakerTimelineEvent) - either may be None
        """
        caption_text = caption_text.strip()
        if not caption_text:
            return None, None

        # Check for system messages first
        timeline_event = self._parse_system_message(caption_text, timestamp)
        if timeline_event:
            return None, timeline_event

        # Try to parse speaker caption
        for pattern in self.caption_patterns:
            match = pattern.match(caption_text)
            if match:
                speaker_name = match.group(1).strip()
                text = match.group(2).strip()

                # Generate speaker ID from name
                speaker_id = self._generate_speaker_id(speaker_name)

                caption_segment = CaptionSegment(
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    text=text,
                    start_timestamp=timestamp,
                    segment_id=f"caption_{int(timestamp)}_{hash(text) % 10000}",
                )

                return caption_segment, None

        # If no pattern matches, treat as unknown speaker
        logger.debug(f"Unmatched caption format: {caption_text}")
        return None, None

    def _parse_system_message(
        self, text: str, timestamp: float
    ) -> Optional[SpeakerTimelineEvent]:
        """Parse system messages for timeline events."""
        for pattern in self.system_patterns:
            match = pattern.match(text)
            if match:
                if "joined" in text:
                    speaker_name = match.group(1).strip()
                    speaker_id = self._generate_speaker_id(speaker_name)
                    return SpeakerTimelineEvent(
                        event_type="join",
                        speaker_id=speaker_id,
                        timestamp=timestamp,
                        metadata={"speaker_name": speaker_name},
                    )
                elif "left" in text:
                    speaker_name = match.group(1).strip()
                    speaker_id = self._generate_speaker_id(speaker_name)
                    return SpeakerTimelineEvent(
                        event_type="leave",
                        speaker_id=speaker_id,
                        timestamp=timestamp,
                        metadata={"speaker_name": speaker_name},
                    )
                elif "presenting" in text:
                    speaker_name = match.group(1).strip()
                    speaker_id = self._generate_speaker_id(speaker_name)
                    event_type = (
                        "start_presenting"
                        if "is presenting" in text
                        else "stop_presenting"
                    )
                    return SpeakerTimelineEvent(
                        event_type=event_type,
                        speaker_id=speaker_id,
                        timestamp=timestamp,
                        metadata={"speaker_name": speaker_name},
                    )

        return None

    def _generate_speaker_id(self, speaker_name: str) -> str:
        """Generate consistent speaker ID from speaker name."""
        # Simple approach: normalize name and use as ID
        # In practice, you might want more sophisticated matching
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", speaker_name.lower())
        return f"speaker_{normalized}"


class SpeakerTimelineManager:
    """Manages speaker timeline and participant information."""

    def __init__(self, session_id: str, bot_manager=None, database_manager=None):
        self.session_id = session_id
        self.bot_manager = bot_manager
        self.database_manager = database_manager

        # Speaker and timeline management
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.timeline_events: List[SpeakerTimelineEvent] = []
        self.caption_segments: List[CaptionSegment] = []

        # Current speaking state
        self.current_speakers: Dict[str, float] = {}  # speaker_id -> start_time

        # Performance tracking
        self.total_captions = 0
        self.total_speakers = 0
        self.session_start_time = time.time()

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"SpeakerTimelineManager initialized for session: {session_id}")

    def add_speaker(self, speaker_info: SpeakerInfo) -> bool:
        """Add or update speaker information."""
        with self.lock:
            if speaker_info.speaker_id not in self.speakers:
                self.total_speakers += 1
                logger.info(
                    f"New speaker added: {speaker_info.display_name} ({speaker_info.speaker_id})"
                )

            self.speakers[speaker_info.speaker_id] = speaker_info

            # Store in database if available
            if self.bot_manager and self.database_manager:
                try:
                    asyncio.create_task(self._store_participant_data(speaker_info))
                except Exception as e:
                    logger.warning(f"Failed to store participant data: {e}")

            return True

    async def _store_participant_data(self, speaker_info: SpeakerInfo):
        """Store participant data in database."""
        try:
            participant_data = {
                "google_participant_id": speaker_info.speaker_id,
                "display_name": speaker_info.display_name,
                "email": speaker_info.email,
                "join_time": datetime.fromtimestamp(speaker_info.join_time)
                if speaker_info.join_time
                else None,
                "leave_time": datetime.fromtimestamp(speaker_info.leave_time)
                if speaker_info.leave_time
                else None,
                "total_speaking_time": speaker_info.total_speaking_time,
                "participant_metadata": {
                    "role": speaker_info.role,
                    "utterance_count": speaker_info.utterance_count,
                    "is_active": speaker_info.is_active,
                },
            }

            await self.bot_manager.store_participant(self.session_id, participant_data)

        except Exception as e:
            logger.error(f"Error storing participant data: {e}")

    def process_caption_segment(self, caption: CaptionSegment) -> bool:
        """Process a caption segment and update speaker timeline."""
        with self.lock:
            try:
                # Add speaker if not exists
                if caption.speaker_id not in self.speakers:
                    speaker_info = SpeakerInfo(
                        speaker_id=caption.speaker_id,
                        display_name=caption.speaker_name,
                        join_time=caption.start_timestamp,
                    )
                    self.add_speaker(speaker_info)

                # Update speaker speaking state
                self._update_speaking_state(caption)

                # Store caption segment
                self.caption_segments.append(caption)
                self.total_captions += 1

                # Store caption as transcript in database
                if self.bot_manager and self.database_manager:
                    try:
                        asyncio.create_task(self._store_caption_transcript(caption))
                    except Exception as e:
                        logger.warning(f"Failed to store caption transcript: {e}")

                # Update speaker statistics
                speaker = self.speakers[caption.speaker_id]
                speaker.utterance_count += 1

                logger.debug(
                    f"Processed caption: {caption.speaker_name} - {caption.text[:50]}..."
                )
                return True

            except Exception as e:
                logger.error(f"Error processing caption segment: {e}")
                return False

    async def _store_caption_transcript(self, caption: CaptionSegment):
        """Store caption as transcript in database."""
        try:
            transcript_data = {
                "text": caption.text,
                "source_type": "google_meet",
                "language": "auto",  # Google Meet doesn't always provide language
                "start_timestamp": caption.start_timestamp,
                "end_timestamp": caption.end_timestamp or caption.start_timestamp,
                "speaker_id": caption.speaker_id,
                "speaker_name": caption.speaker_name,
                "confidence_score": caption.confidence,
                "google_transcript_entry_id": caption.segment_id,
                "processing_metadata": {
                    "caption_source": caption.caption_source,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            await self.bot_manager.store_transcript(self.session_id, transcript_data)

        except Exception as e:
            logger.error(f"Error storing caption transcript: {e}")

    def process_timeline_event(self, event: SpeakerTimelineEvent) -> bool:
        """Process a timeline event (join, leave, etc.)."""
        with self.lock:
            try:
                # Add speaker if not exists (for join events)
                if event.event_type == "join" and event.speaker_id not in self.speakers:
                    speaker_name = event.metadata.get("speaker_name", event.speaker_id)
                    speaker_info = SpeakerInfo(
                        speaker_id=event.speaker_id,
                        display_name=speaker_name,
                        join_time=event.timestamp,
                        is_active=True,
                    )
                    self.add_speaker(speaker_info)

                # Update speaker state based on event
                if event.speaker_id in self.speakers:
                    speaker = self.speakers[event.speaker_id]

                    if event.event_type == "leave":
                        speaker.leave_time = event.timestamp
                        speaker.is_active = False
                    elif event.event_type == "join":
                        speaker.join_time = event.timestamp
                        speaker.is_active = True

                # Store timeline event
                self.timeline_events.append(event)

                # Store event in database
                if self.bot_manager and self.database_manager:
                    try:
                        asyncio.create_task(self._store_timeline_event(event))
                    except Exception as e:
                        logger.warning(f"Failed to store timeline event: {e}")

                logger.info(f"Timeline event: {event.event_type} - {event.speaker_id}")
                return True

            except Exception as e:
                logger.error(f"Error processing timeline event: {e}")
                return False

    async def _store_timeline_event(self, event: SpeakerTimelineEvent):
        """Store timeline event in database."""
        try:
            event_data = {
                "event_type": f"speaker_{event.event_type}",
                "event_subtype": event.event_type,
                "event_data": {
                    "speaker_id": event.speaker_id,
                    "duration": event.duration,
                    "metadata": event.metadata or {},
                },
                "timestamp": datetime.fromtimestamp(event.timestamp),
                "source_component": "caption_processor",
                "severity": "info",
            }

            await self.bot_manager.store_session_event(self.session_id, event_data)

        except Exception as e:
            logger.error(f"Error storing timeline event: {e}")

    def _update_speaking_state(self, caption: CaptionSegment):
        """Update current speaking state based on caption."""
        speaker_id = caption.speaker_id
        current_time = caption.start_timestamp

        # If speaker was already speaking, calculate duration
        if speaker_id in self.current_speakers:
            speaking_duration = current_time - self.current_speakers[speaker_id]
            self.speakers[speaker_id].total_speaking_time += speaking_duration

        # Update current speaking time
        self.current_speakers[speaker_id] = current_time

        # Create speaking timeline events
        if speaker_id not in self.current_speakers:
            speaking_event = SpeakerTimelineEvent(
                event_type="speaking_start",
                speaker_id=speaker_id,
                timestamp=current_time,
            )
            self.timeline_events.append(speaking_event)

    def get_speaker_timeline(
        self, start_time: float = None, end_time: float = None
    ) -> List[Dict]:
        """Get speaker timeline for a time range."""
        with self.lock:
            timeline = []

            for event in self.timeline_events:
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue

                timeline.append(
                    {
                        "event_type": event.event_type,
                        "speaker_id": event.speaker_id,
                        "timestamp": event.timestamp,
                        "duration": event.duration,
                        "metadata": event.metadata or {},
                    }
                )

            return sorted(timeline, key=lambda x: x["timestamp"])

    def get_speaker_segments(
        self, speaker_id: str = None, start_time: float = None, end_time: float = None
    ) -> List[Dict]:
        """Get caption segments for a speaker or time range."""
        with self.lock:
            segments = []

            for caption in self.caption_segments:
                if speaker_id and caption.speaker_id != speaker_id:
                    continue
                if start_time and caption.start_timestamp < start_time:
                    continue
                if end_time and caption.start_timestamp > end_time:
                    continue

                segments.append(
                    {
                        "speaker_id": caption.speaker_id,
                        "speaker_name": caption.speaker_name,
                        "text": caption.text,
                        "start_timestamp": caption.start_timestamp,
                        "end_timestamp": caption.end_timestamp,
                        "confidence": caption.confidence,
                        "segment_id": caption.segment_id,
                    }
                )

            return sorted(segments, key=lambda x: x["start_timestamp"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive timeline statistics."""
        with self.lock:
            runtime = time.time() - self.session_start_time

            speaker_stats = []
            for speaker in self.speakers.values():
                speaker_stats.append(
                    {
                        "speaker_id": speaker.speaker_id,
                        "display_name": speaker.display_name,
                        "is_active": speaker.is_active,
                        "total_speaking_time": speaker.total_speaking_time,
                        "utterance_count": speaker.utterance_count,
                        "join_time": speaker.join_time,
                        "leave_time": speaker.leave_time,
                    }
                )

            return {
                "session_id": self.session_id,
                "runtime_seconds": runtime,
                "total_speakers": self.total_speakers,
                "active_speakers": sum(
                    1 for s in self.speakers.values() if s.is_active
                ),
                "total_captions": self.total_captions,
                "total_timeline_events": len(self.timeline_events),
                "current_speakers": list(self.current_speakers.keys()),
                "speaker_stats": speaker_stats,
            }


class GoogleMeetCaptionProcessor:
    """
    Main class for processing Google Meet captions and managing speaker timeline.
    Integrated with orchestration service bot management.
    """

    def __init__(self, session_id: str, bot_manager=None, database_manager=None):
        self.session_id = session_id
        self.bot_manager = bot_manager
        self.database_manager = database_manager

        # Components
        self.parser = GoogleMeetCaptionParser()
        self.timeline_manager = SpeakerTimelineManager(
            session_id, bot_manager, database_manager
        )

        # State management
        self.is_processing = False
        self.processing_thread = None

        # Callbacks
        self.on_speaker_event = None
        self.on_caption_processed = None
        self.on_error = None

        logger.info(f"GoogleMeetCaptionProcessor initialized for session: {session_id}")

    def set_speaker_event_callback(
        self, callback: Callable[[SpeakerTimelineEvent], None]
    ):
        """Set callback for speaker timeline events."""
        self.on_speaker_event = callback

    def set_caption_callback(self, callback: Callable[[CaptionSegment], None]):
        """Set callback for processed captions."""
        self.on_caption_processed = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback

    async def start_processing(self) -> bool:
        """Start caption processing."""
        if self.is_processing:
            logger.warning("Caption processing already active")
            return False

        try:
            self.is_processing = True

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.processing_thread.start()

            logger.info("Started Google Meet caption processing")
            return True

        except Exception as e:
            logger.error(f"Failed to start caption processing: {e}")
            if self.on_error:
                self.on_error(f"Processing start failed: {e}")
            return False

    async def stop_processing(self) -> bool:
        """Stop caption processing."""
        if not self.is_processing:
            return True

        try:
            self.is_processing = False

            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)

            # Send final timeline data to database if available
            if self.bot_manager and self.database_manager:
                await self._finalize_timeline_data()

            logger.info("Stopped Google Meet caption processing")
            return True

        except Exception as e:
            logger.error(f"Error stopping caption processing: {e}")
            return False

    async def _finalize_timeline_data(self):
        """Finalize timeline data in database."""
        try:
            # Update final speaker statistics
            for speaker_info in self.timeline_manager.speakers.values():
                await self.timeline_manager._store_participant_data(speaker_info)

            # Store final timeline summary event
            final_event_data = {
                "event_type": "caption_processing_completed",
                "event_data": self.timeline_manager.get_statistics(),
                "source_component": "caption_processor",
                "severity": "info",
            }

            await self.bot_manager.store_session_event(
                self.session_id, final_event_data
            )

        except Exception as e:
            logger.error(f"Error finalizing timeline data: {e}")

    def process_caption_line(self, caption_text: str, timestamp: float = None) -> bool:
        """Process a single caption line."""
        try:
            timestamp = timestamp or time.time()

            # Parse caption
            caption_segment, timeline_event = self.parser.parse_caption(
                caption_text, timestamp
            )

            # Process caption segment
            if caption_segment:
                success = self.timeline_manager.process_caption_segment(caption_segment)
                if success and self.on_caption_processed:
                    self.on_caption_processed(caption_segment)

            # Process timeline event
            if timeline_event:
                success = self.timeline_manager.process_timeline_event(timeline_event)
                if success and self.on_speaker_event:
                    self.on_speaker_event(timeline_event)

            return True

        except Exception as e:
            logger.error(f"Error processing caption line: {e}")
            if self.on_error:
                self.on_error(f"Caption processing error: {e}")
            return False

    def _processing_loop(self):
        """Main processing loop for real-time captions."""
        logger.info("Caption processing loop started")

        while self.is_processing:
            try:
                # This is where we would interface with Google Meet's caption API
                # For now, simulate caption processing
                caption_data = self._capture_google_meet_captions()

                if caption_data:
                    for caption_line, timestamp in caption_data:
                        self.process_caption_line(caption_line, timestamp)

                time.sleep(0.1)  # 100ms intervals

            except Exception as e:
                logger.error(f"Error in caption processing loop: {e}")
                if self.on_error:
                    self.on_error(f"Processing loop error: {e}")
                time.sleep(1.0)  # Pause on error

        logger.info("Caption processing loop ended")

    def _capture_google_meet_captions(self) -> Optional[List[Tuple[str, float]]]:
        """
        Capture captions from Google Meet.

        This is a placeholder for actual Google Meet caption API integration.
        """
        # TODO: Implement actual Google Meet caption capture
        # For now, return empty to test the pipeline
        return []

    def get_current_timeline(self) -> Dict[str, Any]:
        """Get current speaker timeline data."""
        return {
            "session_id": self.session_id,
            "timeline": self.timeline_manager.get_speaker_timeline(),
            "speakers": {
                speaker_id: asdict(speaker_info)
                for speaker_id, speaker_info in self.timeline_manager.speakers.items()
            },
            "statistics": self.timeline_manager.get_statistics(),
        }


# Factory functions
def create_caption_processor(
    session_id: str, bot_manager=None, database_manager=None
) -> GoogleMeetCaptionProcessor:
    """Create a caption processor instance."""
    return GoogleMeetCaptionProcessor(session_id, bot_manager, database_manager)


# Example usage
async def main():
    """Example usage of the caption processor."""
    processor = create_caption_processor("test-session-123")

    # Set up callbacks
    def on_caption(caption):
        print(f"Caption: {caption.speaker_name} - {caption.text}")

    def on_speaker_event(event):
        print(f"Speaker event: {event.event_type} - {event.speaker_id}")

    def on_error(error):
        print(f"Error: {error}")

    processor.set_caption_callback(on_caption)
    processor.set_speaker_event_callback(on_speaker_event)
    processor.set_error_callback(on_error)

    # Test caption processing
    test_captions = [
        ("John Doe: Hello everyone, welcome to our meeting", time.time()),
        ("Jane Smith joined the meeting", time.time() + 1),
        ("Jane Smith: Thanks John, glad to be here", time.time() + 2),
        ("Bob Johnson: Can everyone hear me okay?", time.time() + 3),
        ("John Doe left the meeting", time.time() + 30),
    ]

    for caption_text, timestamp in test_captions:
        processor.process_caption_line(caption_text, timestamp)
        await asyncio.sleep(0.5)

    # Print final timeline
    timeline = processor.get_current_timeline()
    print(f"Final timeline: {json.dumps(timeline, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
