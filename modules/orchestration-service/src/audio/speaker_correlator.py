#!/usr/bin/env python3
"""
Speaker Correlation Manager - Orchestration Service

Optional speaker correlation system that links Whisper-detected speakers with external sources
(Google Meet, manual input, etc.). Designed to be graceful and non-blocking for testing scenarios.

Key Features:
- Optional correlation (system works without it)
- Manual speaker mapping support for testing
- Loopback audio handling
- Graceful fallback when Google Meet data unavailable
- Database persistence for correlations
- Confidence scoring and validation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import (
    CorrelationType,
    SpeakerCorrelation,
    create_speaker_correlation,
)

logger = logging.getLogger(__name__)


class CorrelationMethod(str, Enum):
    """Methods for speaker correlation."""

    MANUAL = "manual"  # Manually specified by user
    TEMPORAL = "temporal"  # Time-based correlation
    ACOUSTIC = "acoustic"  # Audio feature matching
    GOOGLE_MEET_API = "google_meet_api"  # Official Google Meet API
    FALLBACK = "fallback"  # Default assignment


@dataclass
class ManualSpeakerMapping:
    """Manual speaker mapping for testing."""

    whisper_speaker_id: str
    display_name: str
    real_name: str | None = None
    notes: str | None = None
    confidence: float = 1.0  # Manual mappings have full confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "whisper_speaker_id": self.whisper_speaker_id,
            "display_name": self.display_name,
            "real_name": self.real_name,
            "notes": self.notes,
            "confidence": self.confidence,
        }


@dataclass
class LoopbackSpeakerInfo:
    """Speaker information for loopback audio testing."""

    estimated_speaker_count: int = 1
    primary_speaker_name: str = "Primary Speaker"
    secondary_speakers: list[str] = field(default_factory=list)
    audio_source_description: str = "Loopback Audio"
    mixing_detected: bool = False  # Multiple speakers detected in single stream

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_speaker_count": self.estimated_speaker_count,
            "primary_speaker_name": self.primary_speaker_name,
            "secondary_speakers": self.secondary_speakers,
            "audio_source_description": self.audio_source_description,
            "mixing_detected": self.mixing_detected,
        }


@dataclass
class CorrelationResult:
    """Result of speaker correlation attempt."""

    success: bool
    method_used: CorrelationMethod
    correlations: list[SpeakerCorrelation]
    confidence_score: float
    processing_time_ms: float
    notes: str | None = None
    fallback_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "method_used": self.method_used.value,
            "correlation_count": len(self.correlations),
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "notes": self.notes,
            "fallback_applied": self.fallback_applied,
        }


class SpeakerCorrelationManager:
    """
    Optional speaker correlation manager designed for graceful testing scenarios.

    This manager handles speaker correlation between Whisper-detected speakers and
    external sources, with emphasis on manual testing and loopback audio scenarios.
    """

    def __init__(
        self,
        database_adapter=None,
        enable_google_meet_correlation: bool = False,
        enable_manual_correlation: bool = True,
        enable_fallback_correlation: bool = True,
        correlation_timeout: float = 5.0,
        min_confidence_threshold: float = 0.3,
    ):
        self.database_adapter = database_adapter
        self.enable_google_meet_correlation = enable_google_meet_correlation
        self.enable_manual_correlation = enable_manual_correlation
        self.enable_fallback_correlation = enable_fallback_correlation
        self.correlation_timeout = correlation_timeout
        self.min_confidence_threshold = min_confidence_threshold

        # Manual mappings storage
        self.manual_mappings: dict[str, list[ManualSpeakerMapping]] = {}  # session_id -> mappings
        self.loopback_configs: dict[str, LoopbackSpeakerInfo] = {}  # session_id -> config

        # Correlation cache
        self.correlation_cache: dict[str, list[SpeakerCorrelation]] = {}

        # Statistics
        self.correlation_stats = {
            "total_attempts": 0,
            "successful_correlations": 0,
            "manual_correlations": 0,
            "fallback_correlations": 0,
            "failed_correlations": 0,
        }

        logger.info(
            f"SpeakerCorrelationManager initialized with Google Meet: {enable_google_meet_correlation}"
        )

    async def set_manual_speaker_mapping(
        self, session_id: str, mappings: list[ManualSpeakerMapping]
    ) -> bool:
        """Set manual speaker mappings for a session (for testing)."""
        try:
            self.manual_mappings[session_id] = mappings

            logger.info(
                f"Set manual speaker mappings for session {session_id}: {len(mappings)} mappings"
            )
            for mapping in mappings:
                logger.debug(f"  {mapping.whisper_speaker_id} -> {mapping.display_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to set manual mappings for session {session_id}: {e}")
            return False

    async def set_loopback_config(self, session_id: str, config: LoopbackSpeakerInfo) -> bool:
        """Set loopback audio configuration for a session."""
        try:
            self.loopback_configs[session_id] = config

            logger.info(
                f"Set loopback config for session {session_id}: {config.audio_source_description}"
            )
            logger.debug(f"  Estimated speakers: {config.estimated_speaker_count}")
            logger.debug(f"  Primary speaker: {config.primary_speaker_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to set loopback config for session {session_id}: {e}")
            return False

    async def correlate_speakers(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        google_meet_speakers: list[dict[str, Any]] | None = None,
        start_timestamp: float = 0.0,
        end_timestamp: float = 0.0,
        force_method: CorrelationMethod | None = None,
    ) -> CorrelationResult:
        """
        Correlate Whisper speakers with external sources.

        This is the main correlation function that tries different methods
        and gracefully falls back when data is unavailable.
        """
        start_time = time.time()
        self.correlation_stats["total_attempts"] += 1

        try:
            logger.info(f"Starting speaker correlation for session {session_id}")
            logger.debug(f"  Whisper speakers: {len(whisper_speakers)}")
            logger.debug(
                f"  Google Meet speakers: {len(google_meet_speakers) if google_meet_speakers else 0}"
            )

            # Determine correlation method
            correlation_method = await self._determine_correlation_method(
                session_id, whisper_speakers, google_meet_speakers, force_method
            )

            logger.info(f"Using correlation method: {correlation_method.value}")

            # Perform correlation based on method
            correlations = []
            confidence_score = 0.0
            notes = None
            fallback_applied = False

            if correlation_method == CorrelationMethod.MANUAL:
                correlations, confidence_score, notes = await self._correlate_manual(
                    session_id, whisper_speakers, start_timestamp, end_timestamp
                )

            elif correlation_method == CorrelationMethod.GOOGLE_MEET_API:
                (
                    correlations,
                    confidence_score,
                    notes,
                ) = await self._correlate_google_meet(
                    session_id,
                    whisper_speakers,
                    google_meet_speakers,
                    start_timestamp,
                    end_timestamp,
                )

            elif correlation_method == CorrelationMethod.TEMPORAL:
                correlations, confidence_score, notes = await self._correlate_temporal(
                    session_id,
                    whisper_speakers,
                    google_meet_speakers,
                    start_timestamp,
                    end_timestamp,
                )

            elif correlation_method == CorrelationMethod.FALLBACK:
                correlations, confidence_score, notes = await self._correlate_fallback(
                    session_id, whisper_speakers, start_timestamp, end_timestamp
                )
                fallback_applied = True

            # Validate correlations
            valid_correlations = [
                c for c in correlations if c.correlation_confidence >= self.min_confidence_threshold
            ]

            if len(valid_correlations) < len(correlations):
                logger.warning(
                    f"Filtered {len(correlations) - len(valid_correlations)} low-confidence correlations"
                )
                correlations = valid_correlations

            # Store correlations in database if available
            if self.database_adapter and correlations:
                await self._store_correlations(correlations)

            # Cache correlations
            self.correlation_cache[session_id] = correlations

            # Update statistics
            if correlations:
                self.correlation_stats["successful_correlations"] += 1
                if correlation_method == CorrelationMethod.MANUAL:
                    self.correlation_stats["manual_correlations"] += 1
                elif correlation_method == CorrelationMethod.FALLBACK:
                    self.correlation_stats["fallback_correlations"] += 1
            else:
                self.correlation_stats["failed_correlations"] += 1

            processing_time = (time.time() - start_time) * 1000

            result = CorrelationResult(
                success=len(correlations) > 0,
                method_used=correlation_method,
                correlations=correlations,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                notes=notes,
                fallback_applied=fallback_applied,
            )

            logger.info(
                f"Speaker correlation completed: {len(correlations)} correlations, confidence: {confidence_score:.2f}"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Speaker correlation failed for session {session_id}: {e}")
            self.correlation_stats["failed_correlations"] += 1

            return CorrelationResult(
                success=False,
                method_used=CorrelationMethod.FALLBACK,
                correlations=[],
                confidence_score=0.0,
                processing_time_ms=processing_time,
                notes=f"Correlation failed: {e!s}",
                fallback_applied=True,
            )

    async def _determine_correlation_method(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        google_meet_speakers: list[dict[str, Any]] | None,
        force_method: CorrelationMethod | None,
    ) -> CorrelationMethod:
        """Determine the best correlation method to use."""

        if force_method:
            return force_method

        # Check for manual mappings first (highest priority for testing)
        if self.enable_manual_correlation and session_id in self.manual_mappings:
            return CorrelationMethod.MANUAL

        # Check for Google Meet data
        if (
            self.enable_google_meet_correlation
            and google_meet_speakers
            and len(google_meet_speakers) > 0
        ):
            return CorrelationMethod.GOOGLE_MEET_API

        # Fallback method
        if self.enable_fallback_correlation:
            return CorrelationMethod.FALLBACK

        # No correlation possible - this should cause a failure
        raise ValueError("No correlation methods available or enabled")

    async def _correlate_manual(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        start_timestamp: float,
        end_timestamp: float,
    ) -> tuple[list[SpeakerCorrelation], float, str | None]:
        """Correlate using manual mappings."""

        correlations = []
        manual_mappings = self.manual_mappings.get(session_id, [])

        if not manual_mappings:
            return [], 0.0, "No manual mappings found"

        mapping_dict = {m.whisper_speaker_id: m for m in manual_mappings}

        for speaker in whisper_speakers:
            speaker_id = speaker.get("speaker_id", speaker.get("id", "unknown"))

            if speaker_id in mapping_dict:
                mapping = mapping_dict[speaker_id]

                correlation = create_speaker_correlation(
                    session_id=session_id,
                    whisper_speaker_id=speaker_id,
                    correlation_confidence=mapping.confidence,
                    correlation_type=CorrelationType.MANUAL,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    correlation_method="manual_mapping",
                    external_speaker_name=mapping.display_name,
                    external_speaker_id=f"manual_{speaker_id}",
                    correlation_metadata={
                        "display_name": mapping.display_name,
                        "real_name": mapping.real_name,
                        "notes": mapping.notes,
                        "manual_mapping": True,
                    },
                )
                correlations.append(correlation)

        confidence_score = (
            sum(c.correlation_confidence for c in correlations) / len(correlations)
            if correlations
            else 0.0
        )
        notes = f"Manual correlation: {len(correlations)} speakers mapped"

        return correlations, confidence_score, notes

    async def _correlate_google_meet(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        google_meet_speakers: list[dict[str, Any]],
        start_timestamp: float,
        end_timestamp: float,
    ) -> tuple[list[SpeakerCorrelation], float, str | None]:
        """Correlate using Google Meet speaker data."""

        # This would implement temporal alignment between Whisper and Google Meet speakers
        # For now, return basic correlation based on order

        correlations = []

        for i, whisper_speaker in enumerate(whisper_speakers):
            if i < len(google_meet_speakers):
                gmeet_speaker = google_meet_speakers[i]
                speaker_id = whisper_speaker.get(
                    "speaker_id", whisper_speaker.get("id", f"speaker_{i}")
                )

                correlation = create_speaker_correlation(
                    session_id=session_id,
                    whisper_speaker_id=speaker_id,
                    correlation_confidence=0.7,  # Moderate confidence for basic correlation
                    correlation_type=CorrelationType.TEMPORAL,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    correlation_method="google_meet_temporal",
                    external_speaker_name=gmeet_speaker.get("name", f"Google Meet Speaker {i + 1}"),
                    external_speaker_id=gmeet_speaker.get("id", f"gmeet_{i}"),
                    google_meet_speaker_name=gmeet_speaker.get("name"),
                    correlation_metadata={
                        "google_meet_speaker": gmeet_speaker,
                        "correlation_method": "basic_temporal",
                    },
                )
                correlations.append(correlation)

        confidence_score = 0.7 if correlations else 0.0
        notes = f"Google Meet correlation: {len(correlations)} speakers correlated"

        return correlations, confidence_score, notes

    async def _correlate_temporal(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        google_meet_speakers: list[dict[str, Any]],
        start_timestamp: float,
        end_timestamp: float,
    ) -> tuple[list[SpeakerCorrelation], float, str | None]:
        """Correlate using temporal alignment."""

        # Advanced temporal correlation would go here
        # For now, delegate to Google Meet correlation
        return await self._correlate_google_meet(
            session_id,
            whisper_speakers,
            google_meet_speakers,
            start_timestamp,
            end_timestamp,
        )

    async def _correlate_fallback(
        self,
        session_id: str,
        whisper_speakers: list[dict[str, Any]],
        start_timestamp: float,
        end_timestamp: float,
    ) -> tuple[list[SpeakerCorrelation], float, str | None]:
        """Fallback correlation using loopback config or generic names."""

        correlations = []
        loopback_config = self.loopback_configs.get(session_id)

        for i, speaker in enumerate(whisper_speakers):
            speaker_id = speaker.get("speaker_id", speaker.get("id", f"speaker_{i}"))

            # Determine speaker name
            if loopback_config:
                if i == 0:
                    display_name = loopback_config.primary_speaker_name
                elif i - 1 < len(loopback_config.secondary_speakers):
                    display_name = loopback_config.secondary_speakers[i - 1]
                else:
                    display_name = f"Speaker {i + 1}"

                external_id = f"loopback_{speaker_id}"
                notes = f"Loopback audio: {loopback_config.audio_source_description}"
            else:
                display_name = f"Speaker {i + 1}"
                external_id = f"fallback_{speaker_id}"
                notes = "Generic fallback naming"

            correlation = create_speaker_correlation(
                session_id=session_id,
                whisper_speaker_id=speaker_id,
                correlation_confidence=0.5,  # Moderate confidence for fallback
                correlation_type=CorrelationType.FALLBACK,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                correlation_method="fallback_naming",
                external_speaker_name=display_name,
                external_speaker_id=external_id,
                correlation_metadata={
                    "fallback_method": True,
                    "loopback_config": loopback_config.to_dict() if loopback_config else None,
                    "notes": notes,
                },
            )
            correlations.append(correlation)

        confidence_score = 0.5 if correlations else 0.0
        notes = f"Fallback correlation: {len(correlations)} speakers assigned generic names"

        return correlations, confidence_score, notes

    async def _store_correlations(self, correlations: list[SpeakerCorrelation]) -> bool:
        """Store correlations in database."""
        try:
            if not self.database_adapter:
                return True  # No database, no problem

            for correlation in correlations:
                await self.database_adapter.store_speaker_correlation(correlation)

            logger.debug(f"Stored {len(correlations)} speaker correlations in database")
            return True

        except Exception as e:
            logger.error(f"Failed to store speaker correlations: {e}")
            return False

    async def get_session_correlations(self, session_id: str) -> list[SpeakerCorrelation]:
        """Get cached correlations for a session."""
        return self.correlation_cache.get(session_id, [])

    async def clear_session_correlations(self, session_id: str) -> bool:
        """Clear correlations for a session."""
        try:
            if session_id in self.correlation_cache:
                del self.correlation_cache[session_id]

            if session_id in self.manual_mappings:
                del self.manual_mappings[session_id]

            if session_id in self.loopback_configs:
                del self.loopback_configs[session_id]

            logger.info(f"Cleared correlations for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear correlations for session {session_id}: {e}")
            return False

    def get_correlation_statistics(self) -> dict[str, Any]:
        """Get correlation statistics."""
        total = self.correlation_stats["total_attempts"]

        return {
            **self.correlation_stats,
            "success_rate": self.correlation_stats["successful_correlations"] / total
            if total > 0
            else 0.0,
            "manual_rate": self.correlation_stats["manual_correlations"] / total
            if total > 0
            else 0.0,
            "fallback_rate": self.correlation_stats["fallback_correlations"] / total
            if total > 0
            else 0.0,
        }

    async def test_correlation_flow(self, session_id: str) -> dict[str, Any]:
        """Test the correlation flow with sample data (for development)."""

        # Sample Whisper speakers
        whisper_speakers = [
            {"speaker_id": "speaker_0", "confidence": 0.9},
            {"speaker_id": "speaker_1", "confidence": 0.8},
        ]

        # Test manual mapping
        manual_mappings = [
            ManualSpeakerMapping("speaker_0", "Test User 1", "John Doe", "Primary speaker"),
            ManualSpeakerMapping("speaker_1", "Test User 2", "Jane Smith", "Secondary speaker"),
        ]

        await self.set_manual_speaker_mapping(session_id, manual_mappings)

        # Test loopback config
        loopback_config = LoopbackSpeakerInfo(
            estimated_speaker_count=2,
            primary_speaker_name="Loopback Primary",
            secondary_speakers=["Loopback Secondary"],
            audio_source_description="Test Loopback Audio",
        )

        await self.set_loopback_config(session_id, loopback_config)

        # Test correlation
        result = await self.correlate_speakers(
            session_id=session_id,
            whisper_speakers=whisper_speakers,
            start_timestamp=0.0,
            end_timestamp=10.0,
        )

        return {
            "test_session_id": session_id,
            "correlation_result": result.to_dict(),
            "correlations": [
                {
                    "whisper_speaker": c.whisper_speaker_id,
                    "external_speaker": c.external_speaker_name,
                    "confidence": c.correlation_confidence,
                    "method": c.correlation_method,
                }
                for c in result.correlations
            ],
            "statistics": self.get_correlation_statistics(),
        }


def create_speaker_correlation_manager(
    database_adapter=None,
    enable_google_meet_correlation: bool = False,
    enable_manual_correlation: bool = True,
    enable_fallback_correlation: bool = True,
    **kwargs,
) -> SpeakerCorrelationManager:
    """Factory function for creating SpeakerCorrelationManager."""

    return SpeakerCorrelationManager(
        database_adapter=database_adapter,
        enable_google_meet_correlation=enable_google_meet_correlation,
        enable_manual_correlation=enable_manual_correlation,
        enable_fallback_correlation=enable_fallback_correlation,
        **kwargs,
    )


# Example usage
async def main():
    """Example usage of SpeakerCorrelationManager."""

    # Create manager for testing (no Google Meet, manual correlation enabled)
    manager = create_speaker_correlation_manager(
        enable_google_meet_correlation=False,
        enable_manual_correlation=True,
        enable_fallback_correlation=True,
    )

    # Test the correlation flow
    test_result = await manager.test_correlation_flow("test_session_123")
    print(f"Test result: {json.dumps(test_result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
