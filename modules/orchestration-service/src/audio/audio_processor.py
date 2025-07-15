#!/usr/bin/env python3
"""
Audio Processing Pipeline - Orchestration Service

Modular audio processing pipeline that uses individual stage components
for flexible configuration and comprehensive monitoring.

Features:
- Modular stage architecture for independent testing
- Real-time performance monitoring and database storage
- Configurable performance targets per stage
- Comprehensive error handling and recovery
- WebSocket integration for real-time updates
- Database aggregation of performance metrics
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from dataclasses import asdict

from .config import (
    AudioProcessingConfig,
    AudioConfigurationManager,
    VADConfig,
    VoiceFilterConfig,
    NoiseReductionConfig,
    VoiceEnhancementConfig,
    EqualizerConfig,
    SpectralDenoisingConfig,
    ConventionalDenoisingConfig,
    LUFSNormalizationConfig,
    AGCConfig,
    CompressionConfig,
    LimiterConfig,
    QualityConfig,
    VADMode,
    NoiseReductionMode,
    CompressionMode,
    AGCMode,
)
from .models import QualityMetrics
from .stage_components import ModularAudioPipeline, StagePerformanceTarget
from .stages import (
    VADStage,
    VoiceFilterStage,
    NoiseReductionStage,
    VoiceEnhancementStage,
    EqualizerStage,
    SpectralDenoisingStage,
    ConventionalDenoisingStage,
    LUFSNormalizationStage,
    AGCStage,
    CompressionStage,
    LimiterStage
)
# Import with try/catch to handle missing database module gracefully
try:
    from database.processing_metrics import get_metrics_manager
except ImportError:
    # Fallback if database module not available
    def get_metrics_manager(database_url=None):
        return None

logger = logging.getLogger(__name__)


class AudioPipelineProcessor:
    """
    Modular audio processing pipeline that uses individual stage components
    for flexible configuration and comprehensive monitoring.
    """
    
    def __init__(self, config: AudioProcessingConfig, sample_rate: int = 16000, database_url: str = None):
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize modular pipeline
        self.pipeline = ModularAudioPipeline(sample_rate)
        
        # Initialize database metrics manager with lazy import
        self.metrics_manager = None
        if database_url:
            try:
                metrics_manager_func = get_metrics_manager
                self.metrics_manager = metrics_manager_func(database_url)
            except Exception as e:
                logger.warning(f"Could not initialize metrics manager: {e}")
                self.metrics_manager = None
        
        # Initialize stage components
        self._initialize_stages()
        
        # Set optional performance targets
        self._set_performance_targets()
        
        logger.info(f"Modular AudioPipelineProcessor initialized with preset: {config.preset_name}")
    
    def _initialize_stages(self):
        """Initialize all processing stages."""
        # Create stage instances
        vad_stage = VADStage(self.config.vad, self.sample_rate)
        voice_filter_stage = VoiceFilterStage(self.config.voice_filter, self.sample_rate)
        noise_reduction_stage = NoiseReductionStage(self.config.noise_reduction, self.sample_rate)
        voice_enhancement_stage = VoiceEnhancementStage(self.config.voice_enhancement, self.sample_rate)
        equalizer_stage = EqualizerStage(self.config.equalizer, self.sample_rate)
        spectral_denoising_stage = SpectralDenoisingStage(self.config.spectral_denoising, self.sample_rate)
        conventional_denoising_stage = ConventionalDenoisingStage(self.config.conventional_denoising, self.sample_rate)
        lufs_normalization_stage = LUFSNormalizationStage(self.config.lufs_normalization, self.sample_rate)
        agc_stage = AGCStage(self.config.agc, self.sample_rate)
        compression_stage = CompressionStage(self.config.compression, self.sample_rate)
        limiter_stage = LimiterStage(self.config.limiter, self.sample_rate)
        
        # Add stages to pipeline in order
        self.pipeline.add_stage(vad_stage)
        self.pipeline.add_stage(voice_filter_stage)
        self.pipeline.add_stage(noise_reduction_stage)
        self.pipeline.add_stage(voice_enhancement_stage)
        self.pipeline.add_stage(equalizer_stage)
        self.pipeline.add_stage(spectral_denoising_stage)
        self.pipeline.add_stage(conventional_denoising_stage)
        self.pipeline.add_stage(lufs_normalization_stage)
        self.pipeline.add_stage(agc_stage)
        self.pipeline.add_stage(compression_stage)
        self.pipeline.add_stage(limiter_stage)
        
        # Enable/disable stages based on config
        for stage_name in self.config.enabled_stages:
            self.pipeline.enable_stage(stage_name, True)
        
        # Disable stages not in enabled list
        all_stages = ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "equalizer", "spectral_denoising", "conventional_denoising", "lufs_normalization", "agc", "compression", "limiter"]
        for stage_name in all_stages:
            if stage_name not in self.config.enabled_stages:
                self.pipeline.enable_stage(stage_name, False)
    
    def _set_performance_targets(self):
        """Set optional performance targets for each stage."""
        # These are optional targets that can be configured
        # Default targets based on real-time processing requirements
        
        performance_targets = {
            "vad": StagePerformanceTarget(target_latency_ms=5.0, max_latency_ms=10.0),
            "voice_filter": StagePerformanceTarget(target_latency_ms=8.0, max_latency_ms=15.0),
            "noise_reduction": StagePerformanceTarget(target_latency_ms=15.0, max_latency_ms=25.0),
            "voice_enhancement": StagePerformanceTarget(target_latency_ms=10.0, max_latency_ms=20.0),
            "equalizer": StagePerformanceTarget(target_latency_ms=12.0, max_latency_ms=22.0),
            "spectral_denoising": StagePerformanceTarget(target_latency_ms=20.0, max_latency_ms=35.0),
            "conventional_denoising": StagePerformanceTarget(target_latency_ms=8.0, max_latency_ms=15.0),
            "lufs_normalization": StagePerformanceTarget(target_latency_ms=18.0, max_latency_ms=30.0),
            "agc": StagePerformanceTarget(target_latency_ms=12.0, max_latency_ms=20.0),
            "compression": StagePerformanceTarget(target_latency_ms=8.0, max_latency_ms=15.0),
            "limiter": StagePerformanceTarget(target_latency_ms=6.0, max_latency_ms=12.0)
        }
        
        # Apply targets to stages
        for stage_name, target in performance_targets.items():
            self.pipeline.set_stage_performance_target(stage_name, target)
    
    def update_config(self, config: AudioProcessingConfig):
        """Update processing configuration."""
        self.config = config
        
        # Update individual stage configs
        if self.pipeline.get_stage("vad"):
            self.pipeline.get_stage("vad").update_config(config.vad)
        if self.pipeline.get_stage("voice_filter"):
            self.pipeline.get_stage("voice_filter").update_config(config.voice_filter)
        if self.pipeline.get_stage("noise_reduction"):
            self.pipeline.get_stage("noise_reduction").update_config(config.noise_reduction)
        if self.pipeline.get_stage("voice_enhancement"):
            self.pipeline.get_stage("voice_enhancement").update_config(config.voice_enhancement)
        if self.pipeline.get_stage("equalizer"):
            self.pipeline.get_stage("equalizer").update_config(config.equalizer)
        if self.pipeline.get_stage("spectral_denoising"):
            self.pipeline.get_stage("spectral_denoising").update_config(config.spectral_denoising)
        if self.pipeline.get_stage("conventional_denoising"):
            self.pipeline.get_stage("conventional_denoising").update_config(config.conventional_denoising)
        if self.pipeline.get_stage("lufs_normalization"):
            self.pipeline.get_stage("lufs_normalization").update_config(config.lufs_normalization)
        if self.pipeline.get_stage("agc"):
            self.pipeline.get_stage("agc").update_config(config.agc)
        if self.pipeline.get_stage("compression"):
            self.pipeline.get_stage("compression").update_config(config.compression)
        if self.pipeline.get_stage("limiter"):
            self.pipeline.get_stage("limiter").update_config(config.limiter)
        
        # Update stage enable/disable status
        for stage_name in self.config.enabled_stages:
            self.pipeline.enable_stage(stage_name, True)
        
        all_stages = ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "equalizer", "spectral_denoising", "conventional_denoising", "lufs_normalization", "agc", "compression", "limiter"]
        for stage_name in all_stages:
            if stage_name not in self.config.enabled_stages:
                self.pipeline.enable_stage(stage_name, False)
        
        logger.info(f"Audio pipeline config updated to preset: {config.preset_name}")
    
    def process_audio_chunk(self, audio_data: np.ndarray, session_id: str = None, chunk_id: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio chunk through the modular pipeline with comprehensive monitoring.
        
        Args:
            audio_data: Input audio data
            session_id: Optional session identifier for tracking
            chunk_id: Optional chunk identifier for tracking
            
        Returns:
            Tuple of (processed_audio, processing_metadata)
        """
        try:
            # Process through modular pipeline
            pipeline_result = self.pipeline.process_chunk(audio_data)
            
            # Store metrics in database if manager is available
            if self.metrics_manager:
                try:
                    self.metrics_manager.store_pipeline_metrics(pipeline_result, session_id, chunk_id)
                except Exception as e:
                    logger.warning(f"Failed to store pipeline metrics: {e}")
            
            # Extract processed audio and metadata
            processed_audio = pipeline_result["final_audio"]
            processing_metadata = pipeline_result["pipeline_metadata"]
            
            # Add stage results for compatibility
            processing_metadata["stage_results"] = pipeline_result["stage_results"]
            
            # Add legacy fields for backward compatibility
            processing_metadata["stages_applied"] = processing_metadata.get("stages_processed", [])
            processing_metadata["vad_result"] = self._extract_vad_result(pipeline_result)
            processing_metadata["quality_metrics"] = self._calculate_quality_metrics(audio_data, processed_audio)
            processing_metadata["bypassed"] = len(processing_metadata.get("stages_with_errors", [])) > 0
            
            return processed_audio, processing_metadata
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            
            # Return original audio with error metadata
            error_metadata = {
                "stages_applied": [],
                "stage_results": {},
                "vad_result": None,
                "quality_metrics": None,
                "total_processing_time_ms": 0,
                "bypassed": True,
                "error": str(e),
                "pipeline_metadata": {
                    "stages_processed": [],
                    "stages_bypassed": [],
                    "stages_with_errors": ["pipeline"],
                    "performance_warnings": [{"type": "pipeline_error", "message": str(e)}]
                }
            }
            
            return audio_data, error_metadata
    
    def _extract_vad_result(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract VAD result from pipeline results."""
        stage_results = pipeline_result.get("stage_results", {})
        vad_result = stage_results.get("vad")
        
        if vad_result and vad_result.metadata:
            return {
                "voice_detected": vad_result.metadata.get("voice_detected", False),
                "confidence": vad_result.metadata.get("confidence", 0.0)
            }
        
        return None
    
    def _calculate_quality_metrics(self, input_audio: np.ndarray, output_audio: np.ndarray) -> Dict[str, Any]:
        """Calculate quality metrics for processed audio."""
        try:
            # Calculate basic quality metrics
            input_rms = np.sqrt(np.mean(input_audio ** 2))
            output_rms = np.sqrt(np.mean(output_audio ** 2))
            
            # SNR estimation (simplified)
            signal_power = np.mean(output_audio ** 2)
            noise_power = np.mean((output_audio - input_audio) ** 2)
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            
            return {
                "input_rms": input_rms,
                "output_rms": output_rms,
                "level_change_db": 20 * np.log10(output_rms / max(input_rms, 1e-10)),
                "estimated_snr_db": snr,
                "dynamic_range": np.max(output_audio) - np.min(output_audio)
            }
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return {}
    
    def process_single_stage(self, stage_name: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio through a single stage only."""
        result = self.pipeline.process_single_stage(stage_name, audio_data)
        
        if result:
            return {
                "processed_audio": result.processed_audio,
                "stage_result": result,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            }
        
        return None
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return self.pipeline.get_pipeline_statistics()
    
    def get_stage_statistics(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific stage."""
        stage = self.pipeline.get_stage(stage_name)
        return stage.get_statistics() if stage else None
    
    def reset_all_statistics(self):
        """Reset all pipeline and stage statistics."""
        self.pipeline.reset_all_statistics()
    
    def get_database_statistics(self, hours: int = 24, session_id: str = None) -> Dict[str, Any]:
        """Get processing statistics from database."""
        if self.metrics_manager:
            return self.metrics_manager.get_processing_statistics(hours, session_id)
        return {"error": "Database metrics not available"}
    
    def get_real_time_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics from database."""
        if self.metrics_manager:
            return self.metrics_manager.get_real_time_metrics(minutes)
        return {"error": "Database metrics not available"}
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old database metrics."""
        if self.metrics_manager:
            self.metrics_manager.cleanup_old_metrics(days_to_keep)
    
    def set_stage_performance_target(self, stage_name: str, target: StagePerformanceTarget):
        """Set performance target for a specific stage."""
        self.pipeline.set_stage_performance_target(stage_name, target)
    
    def enable_stage(self, stage_name: str, enabled: bool = True):
        """Enable or disable a specific stage."""
        self.pipeline.enable_stage(stage_name, enabled)
    
    def get_stage_config(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get current configuration for a specific stage."""
        stage = self.pipeline.get_stage(stage_name)
        if stage:
            return stage._get_stage_config()
        return None


def create_audio_pipeline_processor(config: AudioProcessingConfig, sample_rate: int = 16000, database_url: str = None) -> AudioPipelineProcessor:
    """Factory function to create an AudioPipelineProcessor instance."""
    return AudioPipelineProcessor(config, sample_rate, database_url)