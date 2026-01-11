"""
Bot Analytics Router

Bot analytics and performance monitoring endpoints including:
- Bot analytics (/analytics)
- Session data (/session, /sessions)
- Performance metrics (/performance)
- Quality reports (/quality-report)
- Database analytics (/analytics/database)
- Session analytics (/analytics/sessions)
- Quality analytics (/analytics/quality)
"""

from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status

from ._shared import (
    create_bot_router,
    BotAnalyticsResponse,
    logger,
    get_error_response,
    validate_bot_exists,
)
from dependencies import get_bot_manager

# Create router for bot analytics
router = create_bot_router()


@router.get("/{bot_id}/analytics")
async def get_bot_analytics(
    bot_id: str, bot_manager=Depends(get_bot_manager)
) -> BotAnalyticsResponse:
    """
    Get bot analytics and performance metrics

    Returns comprehensive analytics for a bot including transcription
    quality, translation accuracy, and system performance metrics.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Get bot analytics from database
        analytics = await bot_manager.get_bot_analytics(bot_id)
        if not analytics:
            # Return basic analytics if none found
            analytics = {
                "bot_id": bot_id,
                "total_sessions": 0,
                "active_sessions": 0,
                "total_audio_files": 0,
                "total_transcripts": 0,
                "total_translations": 0,
                "total_correlations": 0,
                "average_confidence": 0.0,
                "quality_metrics": {
                    "transcription_accuracy": 0.0,
                    "translation_quality": 0.0,
                    "correlation_success_rate": 0.0,
                    "average_processing_time": 0.0,
                },
                "performance_stats": {
                    "uptime_percentage": 0.0,
                    "error_rate": 0.0,
                    "success_rate": 0.0,
                },
                "usage_patterns": {
                    "most_active_hours": [],
                    "average_session_duration": 0.0,
                    "popular_languages": [],
                },
            }

        return BotAnalyticsResponse(
            bot_id=bot_id, analytics=analytics, timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot analytics: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot analytics: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/{bot_id}/session")
async def get_bot_session(
    bot_id: str, session_id: Optional[str] = None, bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Get bot session data

    Returns session information including audio files, transcripts,
    translations, and time correlation data.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Get session data
        if session_id:
            # Get specific session data
            session_data = await bot_manager.get_comprehensive_session_data(session_id)
            if not session_data:
                raise get_error_response(
                    status.HTTP_404_NOT_FOUND,
                    f"Session {session_id} not found for bot {bot_id}",
                    {"bot_id": bot_id, "session_id": session_id},
                )
        else:
            # Get current active session for the bot
            session_data = await bot_manager.get_current_bot_session(bot_id)
            if not session_data:
                return {
                    "bot_id": bot_id,
                    "current_session": None,
                    "message": "No active session found for this bot",
                }

        return {"bot_id": bot_id, "session_data": session_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot session: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot session: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/{bot_id}/sessions")
async def list_bot_sessions(
    bot_id: str,
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """
    List all sessions for a specific bot

    Returns paginated list of sessions with optional status filtering.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Get sessions list
        sessions = await bot_manager.list_bot_sessions(
            bot_id, limit=limit, offset=offset, status_filter=status_filter
        )

        return {
            "bot_id": bot_id,
            "sessions": sessions,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(sessions) if len(sessions) < limit else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list bot sessions: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to list bot sessions: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/{bot_id}/performance")
async def get_bot_performance_metrics(
    bot_id: str,
    timeframe: str = "24h",  # 1h, 24h, 7d, 30d
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """
    Get detailed performance metrics for a bot

    Returns performance data including latency, error rates,
    and quality metrics over specified timeframe.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Get performance metrics
        metrics = await bot_manager.get_bot_performance_metrics(bot_id, timeframe)
        if not metrics:
            metrics = {
                "bot_id": bot_id,
                "timeframe": timeframe,
                "processing_metrics": {
                    "average_audio_processing_time": 0.0,
                    "average_transcription_time": 0.0,
                    "average_translation_time": 0.0,
                    "average_correlation_time": 0.0,
                    "total_processing_time": 0.0,
                },
                "quality_metrics": {
                    "average_transcription_confidence": 0.0,
                    "average_translation_confidence": 0.0,
                    "correlation_success_rate": 0.0,
                    "audio_quality_score": 0.0,
                },
                "error_metrics": {
                    "total_errors": 0,
                    "audio_errors": 0,
                    "transcription_errors": 0,
                    "translation_errors": 0,
                    "correlation_errors": 0,
                    "error_rate": 0.0,
                },
                "throughput_metrics": {
                    "audio_files_processed": 0,
                    "transcripts_generated": 0,
                    "translations_generated": 0,
                    "correlations_created": 0,
                    "average_throughput": 0.0,
                },
            }

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot performance metrics: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot performance metrics: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/{bot_id}/quality-report")
async def get_bot_quality_report(
    bot_id: str, session_id: Optional[str] = None, bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Get detailed quality report for bot operations

    Returns comprehensive quality analysis including confidence distributions,
    accuracy metrics, and quality trends.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Get quality report
        report = await bot_manager.get_bot_quality_report(bot_id, session_id)
        if not report:
            report = {
                "bot_id": bot_id,
                "session_id": session_id,
                "confidence_distribution": {
                    "transcription": {"high": 0, "medium": 0, "low": 0},
                    "translation": {"high": 0, "medium": 0, "low": 0},
                    "correlation": {"high": 0, "medium": 0, "low": 0},
                },
                "language_analysis": {
                    "detected_languages": [],
                    "translation_pairs": [],
                    "language_accuracy": {},
                },
                "speaker_analysis": {
                    "total_speakers": 0,
                    "speaker_attribution_accuracy": 0.0,
                    "speaker_consistency": 0.0,
                },
                "temporal_analysis": {
                    "average_segment_length": 0.0,
                    "timing_accuracy": 0.0,
                    "correlation_delay": 0.0,
                },
                "overall_quality_score": 0.0,
                "recommendations": [],
            }

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot quality report: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot quality report: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/analytics/database")
async def get_database_analytics(
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """
    Get comprehensive database analytics

    Returns database statistics including storage usage,
    session counts, and data distribution.
    """
    try:
        analytics = await bot_manager.get_database_analytics()
        if not analytics:
            analytics = {
                "total_sessions": 0,
                "active_sessions": 0,
                "recent_sessions_24h": 0,
                "total_audio_files": 0,
                "total_transcripts": 0,
                "total_translations": 0,
                "total_correlations": 0,
                "storage_usage_bytes": 0,
                "storage_usage_mb": 0.0,
                "data_distribution": {
                    "sessions_by_status": {},
                    "files_by_format": {},
                    "transcripts_by_language": {},
                    "translations_by_language": {},
                },
                "performance_summary": {
                    "average_session_duration": 0.0,
                    "average_file_size": 0.0,
                    "average_confidence_scores": {
                        "transcription": 0.0,
                        "translation": 0.0,
                        "correlation": 0.0,
                    },
                },
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get database analytics: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get database analytics: {str(e)}",
        )


@router.get("/analytics/sessions")
async def get_session_analytics(
    timeframe: str = "24h",
    group_by: str = "hour",  # hour, day, week
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """
    Get session analytics with time-based grouping

    Returns session activity patterns over specified timeframe.
    """
    try:
        analytics = await bot_manager.get_session_analytics(timeframe, group_by)
        if not analytics:
            analytics = {
                "timeframe": timeframe,
                "group_by": group_by,
                "session_activity": [],
                "peak_usage": {
                    "peak_hour": None,
                    "peak_sessions": 0,
                    "peak_date": None,
                },
                "trends": {
                    "sessions_trend": "stable",  # increasing, decreasing, stable
                    "duration_trend": "stable",
                    "quality_trend": "stable",
                },
                "summary": {
                    "total_sessions": 0,
                    "average_session_duration": 0.0,
                    "success_rate": 0.0,
                    "most_active_bots": [],
                },
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get session analytics: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get session analytics: {str(e)}",
        )


@router.get("/analytics/quality")
async def get_quality_analytics(
    timeframe: str = "24h", bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Get quality analytics across all bots and sessions

    Returns comprehensive quality metrics and trends.
    """
    try:
        analytics = await bot_manager.get_quality_analytics(timeframe)
        if not analytics:
            analytics = {
                "timeframe": timeframe,
                "confidence_trends": {
                    "transcription": {
                        "average": 0.0,
                        "trend": "stable",
                        "distribution": {"high": 0, "medium": 0, "low": 0},
                    },
                    "translation": {
                        "average": 0.0,
                        "trend": "stable",
                        "distribution": {"high": 0, "medium": 0, "low": 0},
                    },
                    "correlation": {
                        "average": 0.0,
                        "trend": "stable",
                        "distribution": {"high": 0, "medium": 0, "low": 0},
                    },
                },
                "language_performance": {},
                "error_analysis": {
                    "total_errors": 0,
                    "error_by_type": {},
                    "error_trend": "stable",
                    "most_common_errors": [],
                },
                "processing_performance": {
                    "average_latency": 0.0,
                    "latency_trend": "stable",
                    "throughput": 0.0,
                    "throughput_trend": "stable",
                },
                "recommendations": [],
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get quality analytics: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get quality analytics: {str(e)}",
        )
