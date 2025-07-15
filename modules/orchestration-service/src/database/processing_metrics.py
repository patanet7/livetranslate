#!/usr/bin/env python3
"""
Processing Metrics Database Models

Database schema and operations for storing audio processing
performance metrics and statistics for long-term analysis.
"""

import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class ProcessingMetrics(Base):
    """Store processing metrics for individual audio chunks."""
    
    __tablename__ = 'processing_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String(255), nullable=True)
    chunk_id = Column(String(255), nullable=True)
    
    # Pipeline metrics
    total_processing_time_ms = Column(Float, nullable=False)
    input_level_db = Column(Float, nullable=True)
    output_level_db = Column(Float, nullable=True)
    level_change_db = Column(Float, nullable=True)
    
    # Quality metrics
    input_quality_score = Column(Float, nullable=True)
    output_quality_score = Column(Float, nullable=True)
    quality_improvement = Column(Float, nullable=True)
    
    # Processing status
    stages_processed = Column(Text, nullable=True)  # JSON array
    stages_bypassed = Column(Text, nullable=True)   # JSON array
    stages_with_errors = Column(Text, nullable=True)  # JSON array
    performance_warnings = Column(Text, nullable=True)  # JSON array
    
    # System metrics
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    real_time_factor = Column(Float, nullable=True)
    
    # Metadata
    audio_duration_ms = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channel_count = Column(Integer, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_session_id', 'session_id'),
        Index('idx_processing_time', 'total_processing_time_ms'),
        Index('idx_quality_score', 'output_quality_score'),
    )


class StageMetrics(Base):
    """Store individual stage processing metrics."""
    
    __tablename__ = 'stage_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String(255), nullable=True)
    chunk_id = Column(String(255), nullable=True)
    
    # Stage identification
    stage_name = Column(String(100), nullable=False)
    stage_order = Column(Integer, nullable=False)
    
    # Performance metrics
    processing_time_ms = Column(Float, nullable=False)
    input_level_db = Column(Float, nullable=True)
    output_level_db = Column(Float, nullable=True)
    level_change_db = Column(Float, nullable=True)
    
    # Stage status
    status = Column(String(50), nullable=False)  # COMPLETED, BYPASSED, ERROR
    error_message = Column(Text, nullable=True)
    
    # Performance warnings
    target_latency_ms = Column(Float, nullable=True)
    max_latency_ms = Column(Float, nullable=True)
    latency_target_met = Column(Boolean, nullable=True)
    
    # Stage-specific metadata
    stage_metadata = Column(Text, nullable=True)  # JSON
    stage_config = Column(Text, nullable=True)    # JSON
    
    # Indexes
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_session_id', 'session_id'),
        Index('idx_stage_name', 'stage_name'),
        Index('idx_processing_time', 'processing_time_ms'),
        Index('idx_status', 'status'),
    )


class ProcessingStatistics(Base):
    """Store aggregated processing statistics."""
    
    __tablename__ = 'processing_statistics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Time window for aggregation
    time_window_minutes = Column(Integer, nullable=False)  # 1, 5, 15, 60, etc.
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    
    # Aggregate metrics
    total_chunks_processed = Column(Integer, nullable=False)
    average_processing_time_ms = Column(Float, nullable=False)
    min_processing_time_ms = Column(Float, nullable=False)
    max_processing_time_ms = Column(Float, nullable=False)
    p95_processing_time_ms = Column(Float, nullable=False)
    p99_processing_time_ms = Column(Float, nullable=False)
    
    # Quality metrics
    average_quality_score = Column(Float, nullable=True)
    average_quality_improvement = Column(Float, nullable=True)
    
    # Error metrics
    error_count = Column(Integer, nullable=False, default=0)
    error_rate = Column(Float, nullable=False, default=0.0)
    
    # Performance warnings
    latency_warnings_count = Column(Integer, nullable=False, default=0)
    latency_warnings_rate = Column(Float, nullable=False, default=0.0)
    
    # System metrics
    average_cpu_usage = Column(Float, nullable=True)
    average_memory_usage_mb = Column(Float, nullable=True)
    average_real_time_factor = Column(Float, nullable=True)
    
    # Stage-specific statistics (JSON)
    stage_statistics = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_window_start', 'window_start'),
        Index('idx_time_window', 'time_window_minutes'),
        Index('idx_processing_time', 'average_processing_time_ms'),
    )


class ProcessingMetricsManager:
    """Manages processing metrics database operations."""
    
    def __init__(self, database_url: str = "sqlite:///processing_metrics.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        logger.info(f"Initialized processing metrics database: {database_url}")
    
    def store_pipeline_metrics(self, pipeline_result: Dict[str, Any], session_id: str = None, chunk_id: str = None):
        """Store pipeline processing metrics."""
        try:
            with self.SessionLocal() as session:
                # Extract pipeline metadata
                pipeline_meta = pipeline_result.get("pipeline_metadata", {})
                
                # Create pipeline metrics record
                pipeline_metrics = ProcessingMetrics(
                    session_id=session_id,
                    chunk_id=chunk_id,
                    total_processing_time_ms=pipeline_meta.get("total_processing_time_ms", 0.0),
                    input_level_db=pipeline_meta.get("input_level_db", 0.0),
                    output_level_db=pipeline_meta.get("output_level_db", 0.0),
                    level_change_db=pipeline_meta.get("level_change_db", 0.0),
                    stages_processed=json.dumps(pipeline_meta.get("stages_processed", [])),
                    stages_bypassed=json.dumps(pipeline_meta.get("stages_bypassed", [])),
                    stages_with_errors=json.dumps(pipeline_meta.get("stages_with_errors", [])),
                    performance_warnings=json.dumps(pipeline_meta.get("performance_warnings", [])),
                    sample_rate=16000  # Default sample rate
                )
                
                session.add(pipeline_metrics)
                session.commit()
                
                # Store individual stage metrics
                self._store_stage_metrics(session, pipeline_result, session_id, chunk_id)
                
        except Exception as e:
            logger.error(f"Failed to store pipeline metrics: {e}")
    
    def _store_stage_metrics(self, db_session, pipeline_result: Dict[str, Any], session_id: str, chunk_id: str):
        """Store individual stage metrics."""
        stage_results = pipeline_result.get("stage_results", {})
        
        for stage_order, (stage_name, stage_result) in enumerate(stage_results.items()):
            try:
                stage_metrics = StageMetrics(
                    session_id=session_id,
                    chunk_id=chunk_id,
                    stage_name=stage_name,
                    stage_order=stage_order,
                    processing_time_ms=stage_result.processing_time_ms,
                    input_level_db=stage_result.input_level_db,
                    output_level_db=stage_result.output_level_db,
                    level_change_db=stage_result.output_level_db - stage_result.input_level_db,
                    status=stage_result.status.value,
                    error_message=stage_result.error_message,
                    stage_metadata=json.dumps(stage_result.metadata) if stage_result.metadata else None,
                    stage_config=json.dumps(stage_result.stage_config) if stage_result.stage_config else None
                )
                
                db_session.add(stage_metrics)
                
            except Exception as e:
                logger.error(f"Failed to store stage metrics for {stage_name}: {e}")
    
    def get_processing_statistics(self, hours: int = 24, session_id: str = None) -> Dict[str, Any]:
        """Get processing statistics for the specified time period."""
        try:
            with self.SessionLocal() as session:
                # Calculate time window
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                
                # Query pipeline metrics
                query = session.query(ProcessingMetrics).filter(
                    ProcessingMetrics.timestamp >= start_time,
                    ProcessingMetrics.timestamp <= end_time
                )
                
                if session_id:
                    query = query.filter(ProcessingMetrics.session_id == session_id)
                
                pipeline_metrics = query.all()
                
                if not pipeline_metrics:
                    return {"total_chunks": 0, "time_period_hours": hours}
                
                # Calculate statistics
                processing_times = [m.total_processing_time_ms for m in pipeline_metrics]
                quality_scores = [m.output_quality_score for m in pipeline_metrics if m.output_quality_score is not None]
                
                stats = {
                    "time_period_hours": hours,
                    "total_chunks": len(pipeline_metrics),
                    "processing_time_stats": {
                        "average_ms": np.mean(processing_times),
                        "min_ms": np.min(processing_times),
                        "max_ms": np.max(processing_times),
                        "p95_ms": np.percentile(processing_times, 95),
                        "p99_ms": np.percentile(processing_times, 99)
                    },
                    "quality_stats": {
                        "average_score": np.mean(quality_scores) if quality_scores else 0.0,
                        "min_score": np.min(quality_scores) if quality_scores else 0.0,
                        "max_score": np.max(quality_scores) if quality_scores else 0.0
                    },
                    "error_stats": {
                        "chunks_with_errors": len([m for m in pipeline_metrics if m.stages_with_errors and json.loads(m.stages_with_errors)]),
                        "error_rate": 0.0  # Calculate error rate
                    }
                }
                
                # Calculate error rate
                if stats["total_chunks"] > 0:
                    stats["error_stats"]["error_rate"] = stats["error_stats"]["chunks_with_errors"] / stats["total_chunks"]
                
                # Get stage-specific statistics
                stats["stage_stats"] = self._get_stage_statistics(session, start_time, end_time, session_id)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {"error": str(e)}
    
    def _get_stage_statistics(self, db_session, start_time: datetime, end_time: datetime, session_id: str = None) -> Dict[str, Any]:
        """Get stage-specific statistics."""
        try:
            query = db_session.query(StageMetrics).filter(
                StageMetrics.timestamp >= start_time,
                StageMetrics.timestamp <= end_time
            )
            
            if session_id:
                query = query.filter(StageMetrics.session_id == session_id)
            
            stage_metrics = query.all()
            
            # Group by stage name
            stage_stats = {}
            for metric in stage_metrics:
                stage_name = metric.stage_name
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {
                        "total_processed": 0,
                        "processing_times": [],
                        "error_count": 0,
                        "bypass_count": 0
                    }
                
                stage_stats[stage_name]["total_processed"] += 1
                stage_stats[stage_name]["processing_times"].append(metric.processing_time_ms)
                
                if metric.status == "ERROR":
                    stage_stats[stage_name]["error_count"] += 1
                elif metric.status == "BYPASSED":
                    stage_stats[stage_name]["bypass_count"] += 1
            
            # Calculate statistics for each stage
            for stage_name, stats in stage_stats.items():
                processing_times = stats["processing_times"]
                if processing_times:
                    stats["average_processing_time_ms"] = np.mean(processing_times)
                    stats["min_processing_time_ms"] = np.min(processing_times)
                    stats["max_processing_time_ms"] = np.max(processing_times)
                    stats["p95_processing_time_ms"] = np.percentile(processing_times, 95)
                else:
                    stats["average_processing_time_ms"] = 0.0
                    stats["min_processing_time_ms"] = 0.0
                    stats["max_processing_time_ms"] = 0.0
                    stats["p95_processing_time_ms"] = 0.0
                
                # Calculate rates
                if stats["total_processed"] > 0:
                    stats["error_rate"] = stats["error_count"] / stats["total_processed"]
                    stats["bypass_rate"] = stats["bypass_count"] / stats["total_processed"]
                else:
                    stats["error_rate"] = 0.0
                    stats["bypass_rate"] = 0.0
                
                # Remove raw processing times from output
                del stats["processing_times"]
            
            return stage_stats
            
        except Exception as e:
            logger.error(f"Failed to get stage statistics: {e}")
            return {}
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Remove old metrics to keep database size manageable."""
        try:
            with self.SessionLocal() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                # Delete old processing metrics
                deleted_pipeline = session.query(ProcessingMetrics).filter(
                    ProcessingMetrics.timestamp < cutoff_date
                ).delete()
                
                # Delete old stage metrics
                deleted_stage = session.query(StageMetrics).filter(
                    StageMetrics.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_pipeline} pipeline metrics and {deleted_stage} stage metrics older than {days_to_keep} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def get_real_time_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for the last few minutes."""
        try:
            with self.SessionLocal() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=minutes)
                
                recent_metrics = session.query(ProcessingMetrics).filter(
                    ProcessingMetrics.timestamp >= start_time,
                    ProcessingMetrics.timestamp <= end_time
                ).order_by(ProcessingMetrics.timestamp.desc()).limit(100).all()
                
                if not recent_metrics:
                    return {"chunks_processed": 0, "time_window_minutes": minutes}
                
                # Calculate real-time statistics
                processing_times = [m.total_processing_time_ms for m in recent_metrics]
                
                return {
                    "time_window_minutes": minutes,
                    "chunks_processed": len(recent_metrics),
                    "current_average_ms": np.mean(processing_times),
                    "current_max_ms": np.max(processing_times),
                    "recent_chunks": [
                        {
                            "timestamp": m.timestamp.isoformat(),
                            "processing_time_ms": m.total_processing_time_ms,
                            "input_level_db": m.input_level_db,
                            "output_level_db": m.output_level_db,
                            "stages_processed": json.loads(m.stages_processed) if m.stages_processed else []
                        }
                        for m in recent_metrics[:10]  # Last 10 chunks
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {"error": str(e)}


# Global metrics manager instance
metrics_manager = None

def get_metrics_manager(database_url: str = "sqlite:///processing_metrics.db") -> ProcessingMetricsManager:
    """Get or create the global metrics manager."""
    global metrics_manager
    if metrics_manager is None:
        metrics_manager = ProcessingMetricsManager(database_url)
    return metrics_manager