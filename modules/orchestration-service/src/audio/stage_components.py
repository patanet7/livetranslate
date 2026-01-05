#!/usr/bin/env python3
"""
Modular Audio Processing Stage Components

Each stage is a self-contained component that can be used independently
or as part of the complete pipeline. This enables frontend testing of
individual stages and flexible pipeline configuration.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    """Processing stage status."""

    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    BYPASSED = "bypassed"


@dataclass
class StageResult:
    """Result from a processing stage."""

    stage_name: str
    status: StageStatus
    processed_audio: np.ndarray
    processing_time_ms: float
    input_level_db: float
    output_level_db: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    stage_config: Optional[Dict[str, Any]] = None


@dataclass
class StagePerformanceTarget:
    """Optional performance targets for a stage."""

    target_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality_score: Optional[float] = None
    max_cpu_usage_percent: Optional[float] = None
    max_memory_usage_mb: Optional[float] = None


class BaseAudioStage(ABC):
    """
    Base class for all audio processing stages.

    Each stage is responsible for:
    1. Processing audio data
    2. Monitoring performance metrics
    3. Providing metadata about processing
    4. Handling errors gracefully
    """

    def __init__(self, stage_name: str, config: Any, sample_rate: int = 16000):
        self.stage_name = stage_name
        self.config = config
        self.sample_rate = sample_rate
        self.performance_target = StagePerformanceTarget()

        # Performance tracking
        self.total_processing_time = 0.0
        self.total_chunks_processed = 0
        self.processing_history: List[float] = []
        self.error_count = 0

        # Stage state
        self.is_enabled = True
        self.is_initialized = False

        logger.debug(f"Initialized stage: {stage_name}")

    def set_performance_target(self, target: StagePerformanceTarget):
        """Set performance targets for this stage."""
        self.performance_target = target

    def enable(self, enabled: bool = True):
        """Enable or disable this stage."""
        self.is_enabled = enabled
        logger.debug(f"Stage {self.stage_name} {'enabled' if enabled else 'disabled'}")

    def reset_statistics(self):
        """Reset performance statistics."""
        self.total_processing_time = 0.0
        self.total_chunks_processed = 0
        self.processing_history.clear()
        self.error_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for this stage."""
        if self.total_chunks_processed == 0:
            return {
                "stage_name": self.stage_name,
                "chunks_processed": 0,
                "average_processing_time_ms": 0.0,
                "total_processing_time_ms": 0.0,
                "error_count": 0,
                "error_rate": 0.0,
                "is_enabled": self.is_enabled,
            }

        avg_time = self.total_processing_time / self.total_chunks_processed
        recent_history = self.processing_history[-100:]  # Last 100 chunks

        return {
            "stage_name": self.stage_name,
            "chunks_processed": self.total_chunks_processed,
            "average_processing_time_ms": avg_time,
            "recent_average_ms": np.mean(recent_history) if recent_history else 0.0,
            "min_processing_time_ms": np.min(recent_history) if recent_history else 0.0,
            "max_processing_time_ms": np.max(recent_history) if recent_history else 0.0,
            "total_processing_time_ms": self.total_processing_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_chunks_processed,
            "is_enabled": self.is_enabled,
            "performance_target": {
                "target_latency_ms": self.performance_target.target_latency_ms,
                "max_latency_ms": self.performance_target.max_latency_ms,
                "target_met": self._is_performance_target_met(),
            },
        }

    def _is_performance_target_met(self) -> bool:
        """Check if performance targets are being met."""
        if not self.processing_history:
            return True

        recent_avg = np.mean(self.processing_history[-10:])  # Last 10 chunks

        if self.performance_target.target_latency_ms:
            if recent_avg > self.performance_target.target_latency_ms:
                return False

        if self.performance_target.max_latency_ms:
            if recent_avg > self.performance_target.max_latency_ms:
                return False

        return True

    def _calculate_audio_level(self, audio_data: np.ndarray) -> float:
        """Calculate RMS level in dB."""
        if len(audio_data) == 0:
            return -80.0

        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 1e-10:
            return -80.0

        return 20 * np.log10(rms)

    def process(self, audio_data: np.ndarray) -> StageResult:
        """
        Process audio through this stage with full monitoring.

        Args:
            audio_data: Input audio data

        Returns:
            StageResult containing processed audio and metadata
        """
        if not self.is_enabled:
            return StageResult(
                stage_name=self.stage_name,
                status=StageStatus.BYPASSED,
                processed_audio=audio_data,
                processing_time_ms=0.0,
                input_level_db=self._calculate_audio_level(audio_data),
                output_level_db=self._calculate_audio_level(audio_data),
                metadata={"reason": "stage_disabled"},
            )

        start_time = time.time()
        input_level = self._calculate_audio_level(audio_data)

        try:
            # Call the stage-specific processing
            processed_audio, stage_metadata = self._process_audio(audio_data)

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            output_level = self._calculate_audio_level(processed_audio)

            # Update statistics
            self.total_processing_time += processing_time_ms
            self.total_chunks_processed += 1
            self.processing_history.append(processing_time_ms)

            # Keep history manageable
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-500:]

            # Check performance targets
            performance_warnings = []
            if self.performance_target.target_latency_ms:
                if processing_time_ms > self.performance_target.target_latency_ms:
                    performance_warnings.append(
                        {
                            "type": "latency_target_exceeded",
                            "actual_ms": processing_time_ms,
                            "target_ms": self.performance_target.target_latency_ms,
                            "exceeded_by_ms": processing_time_ms
                            - self.performance_target.target_latency_ms,
                        }
                    )

            if self.performance_target.max_latency_ms:
                if processing_time_ms > self.performance_target.max_latency_ms:
                    performance_warnings.append(
                        {
                            "type": "max_latency_exceeded",
                            "actual_ms": processing_time_ms,
                            "max_ms": self.performance_target.max_latency_ms,
                            "exceeded_by_ms": processing_time_ms
                            - self.performance_target.max_latency_ms,
                        }
                    )

            # Create result
            result = StageResult(
                stage_name=self.stage_name,
                status=StageStatus.COMPLETED,
                processed_audio=processed_audio,
                processing_time_ms=processing_time_ms,
                input_level_db=input_level,
                output_level_db=output_level,
                metadata={
                    **stage_metadata,
                    "performance_warnings": performance_warnings,
                    "level_change_db": output_level - input_level,
                },
                stage_config=self._get_stage_config(),
            )

            return result

        except Exception as e:
            self.error_count += 1
            processing_time_ms = (time.time() - start_time) * 1000

            logger.error(f"Stage {self.stage_name} processing failed: {e}")

            return StageResult(
                stage_name=self.stage_name,
                status=StageStatus.ERROR,
                processed_audio=audio_data,  # Return original audio on error
                processing_time_ms=processing_time_ms,
                input_level_db=input_level,
                output_level_db=input_level,
                error_message=str(e),
                metadata={"error_type": type(e).__name__},
            )

    @abstractmethod
    def _process_audio(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stage-specific audio processing implementation.

        Args:
            audio_data: Input audio data

        Returns:
            Tuple of (processed_audio, stage_metadata)
        """
        pass

    @abstractmethod
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration for metadata."""
        pass

    def update_config(self, new_config: Any):
        """Update stage configuration."""
        self.config = new_config
        logger.debug(f"Updated config for stage {self.stage_name}")


class ModularAudioPipeline:
    """
    Modular audio processing pipeline that processes audio through
    a sequence of individual stage components.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.stages: List[BaseAudioStage] = []
        self.stage_map: Dict[str, BaseAudioStage] = {}

        # Pipeline statistics
        self.total_chunks_processed = 0
        self.total_pipeline_time = 0.0
        self.pipeline_history: List[float] = []

        logger.info("Initialized modular audio pipeline")

    def add_stage(self, stage: BaseAudioStage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self.stage_map[stage.stage_name] = stage
        logger.info(f"Added stage: {stage.stage_name}")

    def remove_stage(self, stage_name: str):
        """Remove a stage from the pipeline."""
        if stage_name in self.stage_map:
            stage = self.stage_map[stage_name]
            self.stages.remove(stage)
            del self.stage_map[stage_name]
            logger.info(f"Removed stage: {stage_name}")

    def get_stage(self, stage_name: str) -> Optional[BaseAudioStage]:
        """Get a stage by name."""
        return self.stage_map.get(stage_name)

    def enable_stage(self, stage_name: str, enabled: bool = True):
        """Enable or disable a specific stage."""
        if stage_name in self.stage_map:
            self.stage_map[stage_name].enable(enabled)

    def set_stage_performance_target(
        self, stage_name: str, target: StagePerformanceTarget
    ):
        """Set performance target for a specific stage."""
        if stage_name in self.stage_map:
            self.stage_map[stage_name].set_performance_target(target)

    def process_chunk(
        self, audio_data: np.ndarray, stop_after_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio chunk through the pipeline.

        Args:
            audio_data: Input audio data
            stop_after_stage: Optional stage name to stop processing after

        Returns:
            Dictionary containing processed audio and comprehensive metadata
        """
        pipeline_start_time = time.time()

        # Initialize result structure
        result = {
            "final_audio": audio_data,
            "stage_results": {},
            "pipeline_metadata": {
                "total_processing_time_ms": 0.0,
                "stages_processed": [],
                "stages_bypassed": [],
                "stages_with_errors": [],
                "performance_warnings": [],
                "input_level_db": 0.0,
                "output_level_db": 0.0,
                "level_change_db": 0.0,
                "stopped_after_stage": stop_after_stage,
            },
        }

        current_audio = audio_data.copy()
        result["pipeline_metadata"]["input_level_db"] = self._calculate_audio_level(
            audio_data
        )

        # Process through each stage
        for stage in self.stages:
            stage_result = stage.process(current_audio)
            result["stage_results"][stage.stage_name] = stage_result

            # Update pipeline metadata
            if stage_result.status == StageStatus.COMPLETED:
                result["pipeline_metadata"]["stages_processed"].append(stage.stage_name)
                current_audio = stage_result.processed_audio
            elif stage_result.status == StageStatus.BYPASSED:
                result["pipeline_metadata"]["stages_bypassed"].append(stage.stage_name)
            elif stage_result.status == StageStatus.ERROR:
                result["pipeline_metadata"]["stages_with_errors"].append(
                    stage.stage_name
                )

            # Collect performance warnings
            if "performance_warnings" in stage_result.metadata:
                result["pipeline_metadata"]["performance_warnings"].extend(
                    stage_result.metadata["performance_warnings"]
                )

            # Stop processing if requested
            if stop_after_stage and stage.stage_name == stop_after_stage:
                break

        # Calculate final metrics
        pipeline_time_ms = (time.time() - pipeline_start_time) * 1000
        result["final_audio"] = current_audio
        result["pipeline_metadata"]["total_processing_time_ms"] = pipeline_time_ms
        result["pipeline_metadata"]["output_level_db"] = self._calculate_audio_level(
            current_audio
        )
        result["pipeline_metadata"]["level_change_db"] = (
            result["pipeline_metadata"]["output_level_db"]
            - result["pipeline_metadata"]["input_level_db"]
        )

        # Update pipeline statistics
        self.total_chunks_processed += 1
        self.total_pipeline_time += pipeline_time_ms
        self.pipeline_history.append(pipeline_time_ms)

        # Keep history manageable
        if len(self.pipeline_history) > 1000:
            self.pipeline_history = self.pipeline_history[-500:]

        return result

    def _calculate_audio_level(self, audio_data: np.ndarray) -> float:
        """Calculate RMS level in dB."""
        if len(audio_data) == 0:
            return -80.0

        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 1e-10:
            return -80.0

        return 20 * np.log10(rms)

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stage_stats = {
            stage_name: stage.get_statistics()
            for stage_name, stage in self.stage_map.items()
        }

        recent_history = self.pipeline_history[-100:]  # Last 100 chunks

        return {
            "pipeline_stats": {
                "total_chunks_processed": self.total_chunks_processed,
                "total_pipeline_time_ms": self.total_pipeline_time,
                "average_pipeline_time_ms": (
                    self.total_pipeline_time / self.total_chunks_processed
                    if self.total_chunks_processed > 0
                    else 0.0
                ),
                "recent_average_ms": np.mean(recent_history) if recent_history else 0.0,
                "min_pipeline_time_ms": np.min(recent_history)
                if recent_history
                else 0.0,
                "max_pipeline_time_ms": np.max(recent_history)
                if recent_history
                else 0.0,
                "stages_count": len(self.stages),
                "enabled_stages": [s.stage_name for s in self.stages if s.is_enabled],
                "disabled_stages": [
                    s.stage_name for s in self.stages if not s.is_enabled
                ],
            },
            "stage_stats": stage_stats,
        }

    def reset_all_statistics(self):
        """Reset all pipeline and stage statistics."""
        self.total_chunks_processed = 0
        self.total_pipeline_time = 0.0
        self.pipeline_history.clear()

        for stage in self.stages:
            stage.reset_statistics()

    def process_single_stage(
        self, stage_name: str, audio_data: np.ndarray
    ) -> Optional[StageResult]:
        """Process audio through a single stage only."""
        if stage_name not in self.stage_map:
            return None

        return self.stage_map[stage_name].process(audio_data)
