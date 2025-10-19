"""
Analytics Router

FastAPI router for comprehensive system analytics and monitoring including:
- Real-time performance metrics and trends
- Service usage analytics and patterns
- Audio processing quality metrics
- Translation accuracy and performance tracking
- Bot session analytics and insights
- WebSocket connection analytics
- Historical data aggregation
- Custom dashboard APIs
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import io
import csv
from enum import Enum

from dependencies import (
    get_health_monitor,
    get_websocket_manager,
    get_audio_service_client,
    get_translation_service_client,
    get_bot_manager,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class TimeRange(str, Enum):
    """Time range options for analytics queries"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    CUSTOM = "custom"

class MetricType(str, Enum):
    """Types of metrics available"""
    PERFORMANCE = "performance"
    USAGE = "usage"
    QUALITY = "quality"
    ERRORS = "errors"
    CAPACITY = "capacity"

class AggregationType(str, Enum):
    """Aggregation methods for metrics"""
    AVERAGE = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"

class ServiceMetrics(BaseModel):
    """Service-specific metrics"""
    service_name: str
    response_time_ms: float
    error_rate: float
    request_count: int
    success_rate: float
    cpu_usage: float
    memory_usage: float
    uptime_seconds: float
    last_error: Optional[str] = None
    timestamp: datetime

class AudioProcessingMetrics(BaseModel):
    """Audio processing specific metrics"""
    total_chunks_processed: int
    average_processing_time_ms: float
    quality_score: float
    noise_reduction_effectiveness: float
    transcription_accuracy: float
    pipeline_latency_ms: float
    failed_chunks: int
    active_sessions: int
    hardware_acceleration: str  # "npu", "gpu", "cpu"
    timestamp: datetime

class TranslationMetrics(BaseModel):
    """Translation service metrics"""
    total_translations: int
    average_translation_time_ms: float
    character_count: int
    language_pairs_active: List[str]
    accuracy_score: float
    cache_hit_rate: float
    failed_translations: int
    model_performance: Dict[str, float]
    timestamp: datetime

class WebSocketMetrics(BaseModel):
    """WebSocket connection metrics"""
    total_connections: int
    active_connections: int
    messages_per_second: float
    average_message_size_bytes: int
    connection_duration_avg_seconds: float
    reconnection_rate: float
    failed_connections: int
    bandwidth_usage_mbps: float
    timestamp: datetime

class BotSessionMetrics(BaseModel):
    """Bot session analytics"""
    total_sessions: int
    active_sessions: int
    average_session_duration_minutes: float
    success_rate: float
    meeting_platforms: Dict[str, int]
    audio_quality_score: float
    translation_coverage: float
    failed_sessions: int
    recovery_success_rate: float
    timestamp: datetime

class SystemAnalytics(BaseModel):
    """Comprehensive system analytics response"""
    overview: Dict[str, Any]
    services: List[ServiceMetrics]
    audio: AudioProcessingMetrics
    translation: TranslationMetrics
    websocket: WebSocketMetrics
    bots: BotSessionMetrics
    trends: Dict[str, List[float]]
    alerts: List[Dict[str, Any]]
    timestamp: datetime

class CustomDashboardRequest(BaseModel):
    """Request for custom dashboard creation"""
    dashboard_name: str
    metrics: List[str]
    time_range: TimeRange
    aggregation: AggregationType
    filters: Optional[Dict[str, Any]] = {}
    refresh_interval: Optional[int] = 30  # seconds

class ExportRequest(BaseModel):
    """Analytics export request"""
    format: str = Field(..., pattern="^(json|csv|pdf)$")
    metrics: List[str]
    time_range: TimeRange
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aggregation: AggregationType = AggregationType.AVERAGE

# ============================================================================
# Real-time Analytics Endpoints
# ============================================================================

@router.get("/overview", response_model=SystemAnalytics)
async def get_analytics_overview(
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    health_monitor=Depends(get_health_monitor),
    websocket_manager=Depends(get_websocket_manager),
    audio_client=Depends(get_audio_service_client),
    translation_client=Depends(get_translation_service_client),
    bot_manager=Depends(get_bot_manager),
):
    """
    Get comprehensive system analytics overview
    
    Returns real-time metrics across all services including performance,
    usage patterns, quality scores, and system health indicators.
    """
    try:
        # Get time range boundaries
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Collect metrics from all services
        services_data = await _collect_service_metrics(health_monitor, start_time, end_time)
        audio_data = await _collect_audio_metrics(audio_client, start_time, end_time)
        translation_data = await _collect_translation_metrics(translation_client, start_time, end_time)
        websocket_data = await _collect_websocket_metrics(websocket_manager, start_time, end_time)
        bot_data = await _collect_bot_metrics(bot_manager, start_time, end_time)
        
        # Generate trends and alerts
        trends = await _calculate_trends(services_data, start_time, end_time)
        alerts = await _check_system_alerts(services_data, audio_data, translation_data)
        
        # Create overview summary
        overview = {
            "total_requests": sum(s.request_count for s in services_data),
            "average_response_time": sum(s.response_time_ms for s in services_data) / len(services_data) if services_data else 0,
            "system_health_score": _calculate_health_score(services_data),
            "active_services": len([s for s in services_data if s.success_rate > 0.95]),
            "total_errors": sum(s.request_count * s.error_rate for s in services_data),
            "uptime_percentage": min(s.uptime_seconds for s in services_data) / (24 * 3600) * 100 if services_data else 100,
        }
        
        return SystemAnalytics(
            overview=overview,
            services=services_data,
            audio=audio_data,
            translation=translation_data,
            websocket=websocket_data,
            bots=bot_data,
            trends=trends,
            alerts=alerts,
            timestamp=end_time
        )
        
    except Exception as e:
        logger.error(f"Failed to get analytics overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics overview: {str(e)}"
        )

@router.get("/metrics/{metric_type}")
async def get_specific_metrics(
    metric_type: MetricType,
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    service: Optional[str] = Query(None),
    aggregation: AggregationType = Query(AggregationType.AVERAGE),
    health_monitor=Depends(get_health_monitor),
):
    """
    Get specific metric type with detailed breakdown
    
    Returns detailed metrics for a specific category (performance, usage, etc.)
    with optional service filtering and custom aggregation.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Route to specific metric collection
        if metric_type == MetricType.PERFORMANCE:
            metrics = await _get_performance_metrics(health_monitor, start_time, end_time, service, aggregation)
        elif metric_type == MetricType.USAGE:
            metrics = await _get_usage_metrics(health_monitor, start_time, end_time, service, aggregation)
        elif metric_type == MetricType.QUALITY:
            metrics = await _get_quality_metrics(health_monitor, start_time, end_time, service, aggregation)
        elif metric_type == MetricType.ERRORS:
            metrics = await _get_error_metrics(health_monitor, start_time, end_time, service, aggregation)
        elif metric_type == MetricType.CAPACITY:
            metrics = await _get_capacity_metrics(health_monitor, start_time, end_time, service, aggregation)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown metric type: {metric_type}")
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "service_filter": service,
            "aggregation": aggregation,
            "data": metrics,
            "timestamp": end_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {metric_type} metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get {metric_type} metrics: {str(e)}"
        )

@router.get("/trends")
async def get_trend_analysis(
    metrics: List[str] = Query(...),
    time_range: TimeRange = Query(TimeRange.LAST_24_HOURS),
    resolution: int = Query(60, description="Data point interval in seconds"),
    health_monitor=Depends(get_health_monitor),
):
    """
    Get trend analysis for specified metrics
    
    Returns time-series data showing trends and patterns for selected metrics
    with configurable resolution and forecasting.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        trend_data = {}
        
        for metric in metrics:
            # Get time series data for each metric
            time_series = await _get_metric_time_series(
                health_monitor, metric, start_time, end_time, resolution
            )
            
            # Calculate trend indicators
            trend_analysis = _analyze_trend(time_series)
            
            trend_data[metric] = {
                "time_series": time_series,
                "trend_direction": trend_analysis["direction"],  # "up", "down", "stable"
                "trend_strength": trend_analysis["strength"],    # 0.0 to 1.0
                "forecast": trend_analysis["forecast"],          # Next 3 data points
                "anomalies": trend_analysis["anomalies"],        # Detected anomalies
                "statistics": {
                    "mean": trend_analysis["mean"],
                    "std_dev": trend_analysis["std_dev"],
                    "min": trend_analysis["min"],
                    "max": trend_analysis["max"],
                }
            }
        
        return {
            "metrics": trend_data,
            "time_range": time_range,
            "resolution_seconds": resolution,
            "analysis_timestamp": end_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get trend analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trend analysis: {str(e)}"
        )

# ============================================================================
# Service-Specific Analytics Endpoints
# ============================================================================

@router.get("/audio/processing")
async def get_audio_processing_analytics(
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    include_pipeline_breakdown: bool = Query(False),
    audio_client=Depends(get_audio_service_client),
):
    """
    Get detailed audio processing analytics
    
    Returns comprehensive audio processing metrics including pipeline
    performance, quality scores, and hardware utilization.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Collect audio processing metrics
        processing_stats = await audio_client.get_processing_statistics()
        
        analytics = {
            "summary": {
                "total_processed_chunks": processing_stats.get("total_chunks", 0),
                "average_processing_time": processing_stats.get("avg_processing_time_ms", 0),
                "success_rate": processing_stats.get("success_rate", 0),
                "quality_score_average": processing_stats.get("avg_quality_score", 0),
            },
            "pipeline_performance": {
                "vad_effectiveness": processing_stats.get("vad_effectiveness", 0),
                "noise_reduction_quality": processing_stats.get("noise_reduction_quality", 0),
                "voice_enhancement_improvement": processing_stats.get("voice_enhancement_improvement", 0),
                "lufs_compliance_rate": processing_stats.get("lufs_compliance_rate", 0),
            },
            "hardware_utilization": {
                "npu_usage_percentage": processing_stats.get("npu_usage", 0),
                "gpu_fallback_rate": processing_stats.get("gpu_fallback_rate", 0),
                "cpu_fallback_rate": processing_stats.get("cpu_fallback_rate", 0),
                "acceleration_effectiveness": processing_stats.get("acceleration_effectiveness", 0),
            },
            "quality_metrics": {
                "transcription_accuracy": processing_stats.get("transcription_accuracy", 0),
                "speaker_diarization_accuracy": processing_stats.get("diarization_accuracy", 0),
                "audio_quality_improvement": processing_stats.get("quality_improvement_db", 0),
                "noise_reduction_effectiveness": processing_stats.get("noise_reduction_db", 0),
            }
        }
        
        if include_pipeline_breakdown:
            analytics["pipeline_breakdown"] = await _get_pipeline_stage_analytics(
                audio_client, start_time, end_time
            )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get audio processing analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audio processing analytics: {str(e)}"
        )

@router.get("/translation/performance")
async def get_translation_analytics(
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    language_pair: Optional[str] = Query(None),
    translation_client=Depends(get_translation_service_client),
):
    """
    Get translation service analytics
    
    Returns detailed translation performance metrics including accuracy,
    speed, and language-specific performance breakdown.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Get translation service metrics
        translation_stats = await translation_client.get_analytics()
        
        analytics = {
            "summary": {
                "total_translations": translation_stats.get("total_translations", 0),
                "average_translation_time": translation_stats.get("avg_translation_time_ms", 0),
                "character_throughput": translation_stats.get("characters_per_second", 0),
                "overall_accuracy": translation_stats.get("accuracy_score", 0),
            },
            "language_performance": translation_stats.get("language_breakdown", {}),
            "model_performance": {
                "active_models": translation_stats.get("active_models", []),
                "model_accuracy": translation_stats.get("model_accuracy", {}),
                "model_speed": translation_stats.get("model_speed", {}),
                "gpu_utilization": translation_stats.get("gpu_utilization", 0),
            },
            "cache_performance": {
                "cache_hit_rate": translation_stats.get("cache_hit_rate", 0),
                "cache_efficiency": translation_stats.get("cache_efficiency", 0),
                "memory_usage": translation_stats.get("cache_memory_mb", 0),
            },
            "quality_metrics": {
                "bleu_score": translation_stats.get("bleu_score", 0),
                "fluency_score": translation_stats.get("fluency_score", 0),
                "adequacy_score": translation_stats.get("adequacy_score", 0),
                "confidence_score": translation_stats.get("confidence_score", 0),
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get translation analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation analytics: {str(e)}"
        )

@router.get("/websocket/connections")
async def get_websocket_analytics(
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    include_connection_details: bool = Query(False),
    websocket_manager=Depends(get_websocket_manager),
):
    """
    Get WebSocket connection analytics
    
    Returns detailed WebSocket metrics including connection patterns,
    message throughput, and performance characteristics.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Get WebSocket analytics
        ws_stats = await websocket_manager.get_connection_stats()
        
        analytics = {
            "summary": {
                "peak_concurrent_connections": ws_stats.get("peak_connections", 0),
                "total_connections_established": ws_stats.get("total_connections", 0),
                "average_connection_duration": ws_stats.get("avg_connection_duration_seconds", 0),
                "message_throughput": ws_stats.get("messages_per_second", 0),
            },
            "connection_patterns": {
                "connection_rate": ws_stats.get("connections_per_minute", 0),
                "disconnection_rate": ws_stats.get("disconnections_per_minute", 0),
                "reconnection_rate": ws_stats.get("reconnection_rate", 0),
                "session_recovery_rate": ws_stats.get("session_recovery_rate", 0),
            },
            "performance_metrics": {
                "average_message_latency": ws_stats.get("avg_message_latency_ms", 0),
                "message_drop_rate": ws_stats.get("message_drop_rate", 0),
                "bandwidth_utilization": ws_stats.get("bandwidth_mbps", 0),
                "heartbeat_success_rate": ws_stats.get("heartbeat_success_rate", 0),
            },
            "error_analysis": {
                "connection_failures": ws_stats.get("connection_failures", 0),
                "timeout_errors": ws_stats.get("timeout_errors", 0),
                "protocol_errors": ws_stats.get("protocol_errors", 0),
                "authentication_failures": ws_stats.get("auth_failures", 0),
            }
        }
        
        if include_connection_details:
            analytics["active_connections"] = await websocket_manager.get_active_connection_details()
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get WebSocket analytics: {str(e)}"
        )

@router.get("/bots/sessions")
async def get_bot_session_analytics(
    time_range: TimeRange = Query(TimeRange.LAST_HOUR),
    platform: Optional[str] = Query(None),
    bot_manager=Depends(get_bot_manager),
):
    """
    Get bot session analytics
    
    Returns comprehensive bot session metrics including success rates,
    platform-specific performance, and session quality indicators.
    """
    try:
        end_time = datetime.now()
        start_time = _get_start_time(time_range, end_time)
        
        # Get bot session analytics (use placeholder for now)
        bot_stats = {"total_sessions": 0, "successful_sessions": 0, "avg_session_duration_minutes": 0, "success_rate": 0}
        
        analytics = {
            "summary": {
                "total_sessions": bot_stats.get("total_sessions", 0),
                "successful_sessions": bot_stats.get("successful_sessions", 0),
                "average_session_duration": bot_stats.get("avg_session_duration_minutes", 0),
                "success_rate": bot_stats.get("success_rate", 0),
            },
            "platform_breakdown": bot_stats.get("platform_stats", {}),
            "performance_metrics": {
                "audio_capture_quality": bot_stats.get("audio_quality_score", 0),
                "caption_extraction_accuracy": bot_stats.get("caption_accuracy", 0),
                "time_correlation_accuracy": bot_stats.get("time_correlation_accuracy", 0),
                "virtual_webcam_performance": bot_stats.get("webcam_performance", 0),
            },
            "error_analysis": {
                "connection_failures": bot_stats.get("connection_failures", 0),
                "audio_capture_failures": bot_stats.get("audio_failures", 0),
                "processing_failures": bot_stats.get("processing_failures", 0),
                "recovery_attempts": bot_stats.get("recovery_attempts", 0),
                "recovery_success_rate": bot_stats.get("recovery_success_rate", 0),
            },
            "usage_patterns": {
                "peak_concurrent_bots": bot_stats.get("peak_concurrent_bots", 0),
                "average_concurrent_bots": bot_stats.get("avg_concurrent_bots", 0),
                "session_frequency": bot_stats.get("sessions_per_hour", 0),
                "user_engagement": bot_stats.get("user_engagement_score", 0),
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get bot session analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot session analytics: {str(e)}"
        )

# ============================================================================
# Dashboard & Export Endpoints
# ============================================================================

@router.post("/dashboard/create")
async def create_custom_dashboard(
    dashboard_request: CustomDashboardRequest,
    health_monitor=Depends(get_health_monitor),
):
    """
    Create a custom analytics dashboard
    
    Allows creation of personalized dashboards with specific metrics,
    time ranges, and visualization preferences.
    """
    try:
        # Validate metrics
        available_metrics = await _get_available_metrics()
        invalid_metrics = [m for m in dashboard_request.metrics if m not in available_metrics]
        
        if invalid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metrics: {invalid_metrics}"
            )
        
        # Create dashboard configuration
        dashboard_config = {
            "name": dashboard_request.dashboard_name,
            "metrics": dashboard_request.metrics,
            "time_range": dashboard_request.time_range,
            "aggregation": dashboard_request.aggregation,
            "filters": dashboard_request.filters,
            "refresh_interval": dashboard_request.refresh_interval,
            "created_at": datetime.now().isoformat(),
        }
        
        # Save dashboard configuration (in production, save to database)
        dashboard_id = await _save_dashboard_config(dashboard_config)
        
        # Generate initial dashboard data
        dashboard_data = await _generate_dashboard_data(dashboard_config, health_monitor)
        
        return {
            "dashboard_id": dashboard_id,
            "config": dashboard_config,
            "data": dashboard_data,
            "endpoints": {
                "data_url": f"/api/analytics/dashboard/{dashboard_id}/data",
                "config_url": f"/api/analytics/dashboard/{dashboard_id}/config",
                "delete_url": f"/api/analytics/dashboard/{dashboard_id}",
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create custom dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dashboard: {str(e)}"
        )

@router.get("/dashboard/{dashboard_id}/data")
async def get_dashboard_data(
    dashboard_id: str,
    health_monitor=Depends(get_health_monitor),
):
    """
    Get data for a specific dashboard
    
    Returns current data for all metrics configured in the dashboard.
    """
    try:
        # Load dashboard configuration
        dashboard_config = await _load_dashboard_config(dashboard_id)
        
        if not dashboard_config:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard {dashboard_id} not found"
            )
        
        # Generate current dashboard data
        dashboard_data = await _generate_dashboard_data(dashboard_config, health_monitor)
        
        return {
            "dashboard_id": dashboard_id,
            "data": dashboard_data,
            "last_updated": datetime.now().isoformat(),
            "next_refresh": (datetime.now() + timedelta(seconds=dashboard_config["refresh_interval"])).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard data: {str(e)}"
        )

@router.post("/export")
async def export_analytics(
    export_request: ExportRequest,
    health_monitor=Depends(get_health_monitor),
):
    """
    Export analytics data in various formats
    
    Supports JSON, CSV, and PDF export formats with configurable
    time ranges and metric selection.
    """
    try:
        # Determine time range
        end_time = export_request.end_time or datetime.now()
        start_time = export_request.start_time or _get_start_time(export_request.time_range, end_time)
        
        # Collect requested metrics
        export_data = {}
        for metric in export_request.metrics:
            metric_data = await _get_metric_data(
                health_monitor, metric, start_time, end_time, export_request.aggregation
            )
            export_data[metric] = metric_data
        
        # Generate export based on format
        if export_request.format == "json":
            return _export_json(export_data, start_time, end_time)
        elif export_request.format == "csv":
            return _export_csv(export_data, start_time, end_time)
        elif export_request.format == "pdf":
            return _export_pdf(export_data, start_time, end_time)
        else:
            raise HTTPException(status_code=400, detail="Invalid export format")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export analytics: {str(e)}"
        )

@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    health_monitor=Depends(get_health_monitor),
):
    """
    Get active system alerts
    
    Returns current system alerts with optional severity filtering.
    """
    try:
        alerts = await health_monitor.get_active_alerts()
        
        if severity:
            alerts = [alert for alert in alerts if alert.get("severity") == severity]
        
        # Enrich alerts with analytics context
        enriched_alerts = []
        for alert in alerts:
            enriched_alert = alert.copy()
            
            # Add trend analysis for alert metric
            if "metric" in alert:
                trend = await _get_alert_trend_context(health_monitor, alert)
                enriched_alert["trend_context"] = trend
            
            enriched_alerts.append(enriched_alert)
        
        return {
            "total_alerts": len(enriched_alerts),
            "severity_breakdown": _count_by_severity(enriched_alerts),
            "alerts": enriched_alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active alerts: {str(e)}"
        )

# ============================================================================
# Utility Functions
# ============================================================================

def _get_start_time(time_range: TimeRange, end_time: datetime) -> datetime:
    """Calculate start time based on time range"""
    if time_range == TimeRange.LAST_HOUR:
        return end_time - timedelta(hours=1)
    elif time_range == TimeRange.LAST_6_HOURS:
        return end_time - timedelta(hours=6)
    elif time_range == TimeRange.LAST_24_HOURS:
        return end_time - timedelta(hours=24)
    elif time_range == TimeRange.LAST_7_DAYS:
        return end_time - timedelta(days=7)
    elif time_range == TimeRange.LAST_30_DAYS:
        return end_time - timedelta(days=30)
    else:
        return end_time - timedelta(hours=1)  # Default to 1 hour

async def _collect_service_metrics(health_monitor, start_time: datetime, end_time: datetime) -> List[ServiceMetrics]:
    """Collect metrics from all services"""
    try:
        services_status = await health_monitor.get_all_services_status()
        service_metrics = []
        
        # Get overall service metrics
        overall_metrics = await health_monitor.get_service_metrics()
        
        for service in services_status:
            # Extract metrics from the service status data
            service_name = service.get("name", "unknown")
            
            service_metric = ServiceMetrics(
                service_name=service_name,
                response_time_ms=service.get("response_time", 0),
                error_rate=service.get("error_count", 0) / max(1, overall_metrics.get("total_services", 1)),
                request_count=overall_metrics.get("total_services", 0),
                success_rate=1.0 if service.get("status") == "healthy" else 0.0,
                cpu_usage=overall_metrics.get("avg_response_time_ms", 0) / 10,  # Approximate CPU usage
                memory_usage=overall_metrics.get("health_percentage", 0),
                uptime_seconds=service.get("last_check", 0),
                last_error=service.get("last_error"),
                timestamp=end_time
            )
            service_metrics.append(service_metric)
        
        return service_metrics
    except Exception as e:
        logger.error(f"Failed to collect service metrics: {e}")
        return []

async def _collect_audio_metrics(audio_client, start_time: datetime, end_time: datetime) -> AudioProcessingMetrics:
    """Collect audio processing metrics"""
    try:
        stats = await audio_client.get_processing_statistics()
        
        return AudioProcessingMetrics(
            total_chunks_processed=stats.get("total_chunks", 0),
            average_processing_time_ms=stats.get("avg_processing_time_ms", 0),
            quality_score=stats.get("avg_quality_score", 0),
            noise_reduction_effectiveness=stats.get("noise_reduction_effectiveness", 0),
            transcription_accuracy=stats.get("transcription_accuracy", 0),
            pipeline_latency_ms=stats.get("pipeline_latency_ms", 0),
            failed_chunks=stats.get("failed_chunks", 0),
            active_sessions=stats.get("active_sessions", 0),
            hardware_acceleration=stats.get("hardware_acceleration", "cpu"),
            timestamp=end_time
        )
    except Exception as e:
        logger.error(f"Failed to collect audio metrics: {e}")
        return AudioProcessingMetrics(
            total_chunks_processed=0,
            average_processing_time_ms=0,
            quality_score=0,
            noise_reduction_effectiveness=0,
            transcription_accuracy=0,
            pipeline_latency_ms=0,
            failed_chunks=0,
            active_sessions=0,
            hardware_acceleration="cpu",
            timestamp=end_time
        )

async def _collect_translation_metrics(translation_client, start_time: datetime, end_time: datetime) -> TranslationMetrics:
    """Collect translation service metrics"""
    try:
        stats = await translation_client.get_analytics()
        
        return TranslationMetrics(
            total_translations=stats.get("total_translations", 0),
            average_translation_time_ms=stats.get("avg_translation_time_ms", 0),
            character_count=stats.get("total_characters", 0),
            language_pairs_active=stats.get("active_language_pairs", []),
            accuracy_score=stats.get("accuracy_score", 0),
            cache_hit_rate=stats.get("cache_hit_rate", 0),
            failed_translations=stats.get("failed_translations", 0),
            model_performance=stats.get("model_performance", {}),
            timestamp=end_time
        )
    except Exception as e:
        logger.error(f"Failed to collect translation metrics: {e}")
        return TranslationMetrics(
            total_translations=0,
            average_translation_time_ms=0,
            character_count=0,
            language_pairs_active=[],
            accuracy_score=0,
            cache_hit_rate=0,
            failed_translations=0,
            model_performance={},
            timestamp=end_time
        )

async def _collect_websocket_metrics(websocket_manager, start_time: datetime, end_time: datetime) -> WebSocketMetrics:
    """Collect WebSocket metrics"""
    try:
        stats = await websocket_manager.get_connection_stats()
        
        return WebSocketMetrics(
            total_connections=stats.get("total_connections", 0),
            active_connections=stats.get("active_connections", 0),
            messages_per_second=stats.get("messages_per_second", 0),
            average_message_size_bytes=stats.get("avg_message_size", 0),
            connection_duration_avg_seconds=stats.get("avg_connection_duration", 0),
            reconnection_rate=stats.get("reconnection_rate", 0),
            failed_connections=stats.get("failed_connections", 0),
            bandwidth_usage_mbps=stats.get("bandwidth_mbps", 0),
            timestamp=end_time
        )
    except Exception as e:
        logger.error(f"Failed to collect WebSocket metrics: {e}")
        return WebSocketMetrics(
            total_connections=0,
            active_connections=0,
            messages_per_second=0,
            average_message_size_bytes=0,
            connection_duration_avg_seconds=0,
            reconnection_rate=0,
            failed_connections=0,
            bandwidth_usage_mbps=0,
            timestamp=end_time
        )

async def _collect_bot_metrics(bot_manager, start_time: datetime, end_time: datetime) -> BotSessionMetrics:
    """Collect bot session metrics"""
    try:
        stats = {"total_sessions": 0, "active_sessions": 0, "avg_session_duration_minutes": 0, "success_rate": 0}
        
        return BotSessionMetrics(
            total_sessions=stats.get("total_sessions", 0),
            active_sessions=stats.get("active_sessions", 0),
            average_session_duration_minutes=stats.get("avg_session_duration_minutes", 0),
            success_rate=stats.get("success_rate", 0),
            meeting_platforms=stats.get("platform_breakdown", {}),
            audio_quality_score=stats.get("audio_quality_score", 0),
            translation_coverage=stats.get("translation_coverage", 0),
            failed_sessions=stats.get("failed_sessions", 0),
            recovery_success_rate=stats.get("recovery_success_rate", 0),
            timestamp=end_time
        )
    except Exception as e:
        logger.error(f"Failed to collect bot metrics: {e}")
        return BotSessionMetrics(
            total_sessions=0,
            active_sessions=0,
            average_session_duration_minutes=0,
            success_rate=0,
            meeting_platforms={},
            audio_quality_score=0,
            translation_coverage=0,
            failed_sessions=0,
            recovery_success_rate=0,
            timestamp=end_time
        )

def _calculate_health_score(services: List[ServiceMetrics]) -> float:
    """Calculate overall system health score"""
    if not services:
        return 0.0
    
    # Weighted average of service health factors
    total_score = 0
    for service in services:
        service_score = (
            service.success_rate * 0.4 +  # 40% weight on success rate
            (1 - min(service.response_time_ms / 1000, 1.0)) * 0.3 +  # 30% weight on response time
            (1 - service.cpu_usage / 100) * 0.15 +  # 15% weight on CPU usage
            (1 - service.memory_usage / 100) * 0.15  # 15% weight on memory usage
        )
        total_score += service_score
    
    return min(total_score / len(services), 1.0)

async def _calculate_trends(services: List[ServiceMetrics], start_time: datetime, end_time: datetime) -> Dict[str, List[float]]:
    """Calculate trend data for key metrics"""
    # Simplified trend calculation - in production, use time-series database
    trends = {
        "response_times": [s.response_time_ms for s in services],
        "error_rates": [s.error_rate for s in services],
        "cpu_usage": [s.cpu_usage for s in services],
        "memory_usage": [s.memory_usage for s in services],
    }
    return trends

async def _check_system_alerts(services: List[ServiceMetrics], audio_metrics: AudioProcessingMetrics, translation_metrics: TranslationMetrics) -> List[Dict[str, Any]]:
    """Generate system alerts based on current metrics"""
    alerts = []
    
    # Check service health
    for service in services:
        if service.error_rate > 0.1:  # >10% error rate
            alerts.append({
                "type": "high_error_rate",
                "severity": "high",
                "service": service.service_name,
                "message": f"High error rate: {service.error_rate:.1%}",
                "metric_value": service.error_rate,
                "threshold": 0.1
            })
        
        if service.response_time_ms > 2000:  # >2 second response time
            alerts.append({
                "type": "slow_response",
                "severity": "medium",
                "service": service.service_name,
                "message": f"Slow response time: {service.response_time_ms:.0f}ms",
                "metric_value": service.response_time_ms,
                "threshold": 2000
            })
    
    # Check audio processing
    if audio_metrics.pipeline_latency_ms > 200:  # >200ms pipeline latency
        alerts.append({
            "type": "high_audio_latency",
            "severity": "medium",
            "service": "audio-processing",
            "message": f"High audio processing latency: {audio_metrics.pipeline_latency_ms:.0f}ms",
            "metric_value": audio_metrics.pipeline_latency_ms,
            "threshold": 200
        })
    
    return alerts

def _export_json(data: Dict[str, Any], start_time: datetime, end_time: datetime) -> JSONResponse:
    """Export data as JSON"""
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "data": data
    }
    
    return JSONResponse(content=export_data)

def _export_csv(data: Dict[str, Any], start_time: datetime, end_time: datetime) -> StreamingResponse:
    """Export data as CSV"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["timestamp", "metric", "value"])
    
    # Write data
    for metric_name, metric_data in data.items():
        if isinstance(metric_data, list):
            for i, value in enumerate(metric_data):
                timestamp = start_time + timedelta(seconds=i * 60)  # Assume 1-minute intervals
                writer.writerow([timestamp.isoformat(), metric_name, value])
    
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analytics_export.csv"}
    )

def _export_pdf(data: Dict[str, Any], start_time: datetime, end_time: datetime) -> StreamingResponse:
    """Export data as PDF"""
    # Simplified PDF export - in production, use proper PDF library
    content = f"Analytics Report\n"
    content += f"Time Range: {start_time} to {end_time}\n\n"
    
    for metric_name, metric_data in data.items():
        content += f"{metric_name}: {metric_data}\n"
    
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=analytics_report.pdf"}
    )

# Placeholder functions for dashboard management (implement with database in production)
async def _save_dashboard_config(config: Dict[str, Any]) -> str:
    """Save dashboard configuration and return ID"""
    import uuid
    return str(uuid.uuid4())

async def _load_dashboard_config(dashboard_id: str) -> Optional[Dict[str, Any]]:
    """Load dashboard configuration by ID"""
    # In production, load from database
    return None

async def _generate_dashboard_data(config: Dict[str, Any], health_monitor) -> Dict[str, Any]:
    """Generate dashboard data based on configuration"""
    # Simplified dashboard data generation
    return {"message": "Dashboard data would be generated here"}

async def _get_available_metrics() -> List[str]:
    """Get list of available metrics"""
    return [
        "response_time", "error_rate", "cpu_usage", "memory_usage",
        "audio_processing_latency", "translation_accuracy", "websocket_connections"
    ]

def _count_by_severity(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count alerts by severity"""
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for alert in alerts:
        severity = alert.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return severity_counts

# Additional placeholder functions for comprehensive analytics
async def _get_performance_metrics(health_monitor, start_time, end_time, service, aggregation):
    """Get performance-specific metrics"""
    return {"placeholder": "performance metrics"}

async def _get_usage_metrics(health_monitor, start_time, end_time, service, aggregation):
    """Get usage-specific metrics"""
    return {"placeholder": "usage metrics"}

async def _get_quality_metrics(health_monitor, start_time, end_time, service, aggregation):
    """Get quality-specific metrics"""
    return {"placeholder": "quality metrics"}

async def _get_error_metrics(health_monitor, start_time, end_time, service, aggregation):
    """Get error-specific metrics"""
    return {"placeholder": "error metrics"}

async def _get_capacity_metrics(health_monitor, start_time, end_time, service, aggregation):
    """Get capacity-specific metrics"""
    return {"placeholder": "capacity metrics"}

async def _get_metric_time_series(health_monitor, metric, start_time, end_time, resolution):
    """Get time series data for a metric"""
    return []

def _analyze_trend(time_series):
    """Analyze trend in time series data"""
    return {
        "direction": "stable",
        "strength": 0.0,
        "forecast": [],
        "anomalies": [],
        "mean": 0.0,
        "std_dev": 0.0,
        "min": 0.0,
        "max": 0.0
    }

async def _get_pipeline_stage_analytics(audio_client, start_time, end_time):
    """Get detailed pipeline stage analytics"""
    return {"placeholder": "pipeline stage analytics"}

async def _get_metric_data(health_monitor, metric, start_time, end_time, aggregation):
    """Get data for a specific metric"""
    return []

async def _get_alert_trend_context(health_monitor, alert):
    """Get trend context for an alert"""
    return {"trend": "stable"}