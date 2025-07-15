#!/usr/bin/env python3
"""
Real-time Dashboard Component

Provides real-time performance metrics, analytics, and visualization
for the orchestration service dashboard.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and maintains real-time metrics"""

    def __init__(self, max_data_points: int = 100):
        """Initialize metrics collector"""
        self.max_data_points = max_data_points
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_data_points))
        self.current_metrics = {}
        self._lock = threading.RLock()

    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value"""
        timestamp = timestamp or time.time()

        with self._lock:
            self.metrics_history[name].append({"timestamp": timestamp, "value": value})
            self.current_metrics[name] = value

    def get_metric_history(
        self, name: str, duration_seconds: Optional[int] = None
    ) -> List[Dict]:
        """Get metric history for a given duration"""
        with self._lock:
            history = list(self.metrics_history[name])

            if duration_seconds:
                cutoff_time = time.time() - duration_seconds
                history = [h for h in history if h["timestamp"] >= cutoff_time]

            return history

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        with self._lock:
            return self.current_metrics.copy()

    def calculate_average(self, name: str, duration_seconds: int = 300) -> float:
        """Calculate average for a metric over duration"""
        history = self.get_metric_history(name, duration_seconds)
        if not history:
            return 0.0

        return sum(h["value"] for h in history) / len(history)

    def calculate_rate(self, name: str, duration_seconds: int = 60) -> float:
        """Calculate rate of change for a metric"""
        history = self.get_metric_history(name, duration_seconds)
        if len(history) < 2:
            return 0.0

        recent = history[-1]
        older = history[0]
        time_diff = recent["timestamp"] - older["timestamp"]

        if time_diff <= 0:
            return 0.0

        value_diff = recent["value"] - older["value"]
        return value_diff / time_diff


class PerformanceAnalyzer:
    """Analyzes performance trends and generates insights"""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance analyzer"""
        self.metrics_collector = metrics_collector
        self.thresholds = {
            "response_time": {"warning": 1000, "critical": 5000},  # ms
            "error_rate": {"warning": 0.05, "critical": 0.1},  # 5%, 10%
            "cpu_usage": {"warning": 0.7, "critical": 0.9},  # 70%, 90%
            "memory_usage": {"warning": 0.8, "critical": 0.95},  # 80%, 95%
        }

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and generate insights"""
        current_metrics = self.metrics_collector.get_current_metrics()
        insights = {
            "status": "healthy",
            "warnings": [],
            "critical_issues": [],
            "recommendations": [],
            "performance_score": 100,
        }

        # Analyze each metric
        for metric_name, value in current_metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]

                if value >= threshold["critical"]:
                    insights["critical_issues"].append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold["critical"],
                            "message": f"{metric_name} is critically high: {value}",
                        }
                    )
                    insights["status"] = "critical"
                    insights["performance_score"] -= 30

                elif value >= threshold["warning"]:
                    insights["warnings"].append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold["warning"],
                            "message": f"{metric_name} is elevated: {value}",
                        }
                    )
                    if insights["status"] == "healthy":
                        insights["status"] = "warning"
                    insights["performance_score"] -= 10

        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(current_metrics)

        # Ensure performance score doesn't go below 0
        insights["performance_score"] = max(0, insights["performance_score"])

        return insights

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Response time recommendations
        response_time = metrics.get("avg_response_time", 0)
        if response_time > 2000:
            recommendations.append(
                "Consider optimizing API endpoints or adding caching"
            )

        # Error rate recommendations
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.05:
            recommendations.append(
                "Investigate error causes and improve error handling"
            )

        # Connection recommendations
        active_connections = metrics.get("active_connections", 0)
        max_connections = metrics.get("max_connections", 10000)
        if active_connections / max_connections > 0.8:
            recommendations.append("Consider increasing connection pool size")

        # Memory recommendations
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 0.8:
            recommendations.append(
                "Monitor memory usage and consider garbage collection tuning"
            )

        return recommendations

    def get_performance_trends(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get performance trends over time"""
        duration_seconds = duration_minutes * 60
        trends = {}

        for metric_name in ["response_time", "error_rate", "active_connections"]:
            history = self.metrics_collector.get_metric_history(
                metric_name, duration_seconds
            )
            if len(history) >= 2:
                trend_direction = "stable"
                recent_avg = sum(h["value"] for h in history[-10:]) / min(
                    10, len(history)
                )
                older_avg = sum(h["value"] for h in history[:10]) / min(
                    10, len(history)
                )

                if recent_avg > older_avg * 1.1:
                    trend_direction = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trend_direction = "decreasing"

                trends[metric_name] = {
                    "direction": trend_direction,
                    "recent_average": recent_avg,
                    "change_percentage": ((recent_avg - older_avg) / older_avg * 100)
                    if older_avg > 0
                    else 0,
                }

        return trends


class RealTimeDashboard:
    """Real-time dashboard with metrics collection and analysis"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize real-time dashboard"""
        self.config = config
        self.refresh_interval = config.get("refresh_interval", 5)
        self.max_data_points = config.get("max_data_points", 100)
        self.enable_real_time = config.get("enable_real_time", True)

        # Components
        self.metrics_collector = MetricsCollector(self.max_data_points)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)

        # State
        self.running = False
        self.start_time = time.time()
        self._collection_thread = None
        self._websocket_manager = None  # Will be set by orchestration service

        # Dashboard data
        self.dashboard_data = {
            "system_overview": {},
            "performance_metrics": {},
            "service_status": {},
            "alerts": [],
            "trends": {},
        }

        logger.info("Real-time dashboard initialized")

    def set_websocket_manager(self, websocket_manager):
        """Set WebSocket manager instance for real-time updates"""
        self._websocket_manager = websocket_manager
        logger.info("WebSocket manager attached to dashboard")

    async def start(self):
        """Start the dashboard"""
        self.running = True

        if self.enable_real_time:
            self._collection_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self._collection_thread.start()

        logger.info("Real-time dashboard started")

    async def stop(self):
        """Stop the dashboard"""
        self.running = False

        if self._collection_thread:
            self._collection_thread.join(timeout=10)

        logger.info("Real-time dashboard stopped")

    def update_metrics(self, orchestration_service):
        """Update metrics from orchestration service components"""
        try:
            current_time = time.time()

            # WebSocket metrics
            if hasattr(orchestration_service, "websocket_manager"):
                ws_stats = orchestration_service.websocket_manager.get_statistics()
                self.metrics_collector.record_metric(
                    "active_connections", ws_stats.get("active_connections", 0)
                )
                self.metrics_collector.record_metric(
                    "total_connections", ws_stats.get("total_connections", 0)
                )
                self.metrics_collector.record_metric(
                    "active_sessions", ws_stats.get("active_sessions", 0)
                )

            # API Gateway metrics
            if hasattr(orchestration_service, "api_gateway"):
                gateway_metrics = orchestration_service.api_gateway.get_metrics()
                self.metrics_collector.record_metric(
                    "total_requests", gateway_metrics.get("total_requests", 0)
                )
                self.metrics_collector.record_metric(
                    "avg_response_time", gateway_metrics.get("average_response_time", 0)
                )

                # Calculate error rate
                total_requests = gateway_metrics.get("total_requests", 0)
                failed_requests = gateway_metrics.get("failed_requests", 0)
                error_rate = (failed_requests / max(total_requests, 1)) * 100
                self.metrics_collector.record_metric("error_rate", error_rate)

            # Health Monitor metrics
            if hasattr(orchestration_service, "health_monitor"):
                health_metrics = orchestration_service.health_monitor.get_metrics()
                service_status = (
                    orchestration_service.health_monitor.get_all_service_status()
                )

                # Overall system health score
                summary = service_status.get("summary", {})
                total_services = summary.get("total_services", 1)
                healthy_services = summary.get("healthy_services", 0)
                health_score = (healthy_services / total_services) * 100
                self.metrics_collector.record_metric(
                    "system_health_score", health_score
                )

            # Update dashboard data
            self._update_dashboard_data(orchestration_service)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    def _update_dashboard_data(self, orchestration_service):
        """Update dashboard data structure"""
        try:
            # System overview
            self.dashboard_data["system_overview"] = {
                "uptime": time.time() - self.start_time,
                "status": "running",
                "performance_score": self.performance_analyzer.analyze_performance().get(
                    "performance_score", 100
                ),
                "timestamp": time.time(),
            }

            # Performance metrics
            current_metrics = self.metrics_collector.get_current_metrics()
            self.dashboard_data["performance_metrics"] = {
                "active_connections": current_metrics.get("active_connections", 0),
                "total_requests": current_metrics.get("total_requests", 0),
                "avg_response_time": current_metrics.get("avg_response_time", 0),
                "error_rate": current_metrics.get("error_rate", 0),
                "system_health_score": current_metrics.get("system_health_score", 100),
            }

            # Service status
            if hasattr(orchestration_service, "health_monitor"):
                self.dashboard_data[
                    "service_status"
                ] = orchestration_service.health_monitor.get_all_service_status()

            # Performance analysis
            analysis = self.performance_analyzer.analyze_performance()
            self.dashboard_data["alerts"] = analysis.get("warnings", []) + analysis.get(
                "critical_issues", []
            )

            # Trends
            self.dashboard_data[
                "trends"
            ] = self.performance_analyzer.get_performance_trends()

        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()

    def get_metric_history(
        self, metric_name: str, duration_minutes: int = 60
    ) -> List[Dict]:
        """Get metric history for charting"""
        duration_seconds = duration_minutes * 60
        return self.metrics_collector.get_metric_history(metric_name, duration_seconds)

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for WebSocket updates"""
        return {
            "current_metrics": self.metrics_collector.get_current_metrics(),
            "performance_analysis": self.performance_analyzer.analyze_performance(),
            "timestamp": time.time(),
        }

    async def broadcast_metrics_update(self):
        """Broadcast metrics update via WebSocket"""
        if self._websocket_manager and self.enable_real_time:
            try:
                metrics_data = self.get_real_time_metrics()
                
                # Broadcast to all connected clients
                for connection_id, connection in self._websocket_manager.connections.items():
                    try:
                        await connection.send_message({
                            "type": "metrics:update",
                            "data": metrics_data,
                            "timestamp": int(time.time() * 1000),
                        })
                    except Exception as e:
                        logger.error(f"Failed to send metrics update to {connection_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to broadcast metrics update: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        return {
            "component": "real_time_dashboard",
            "status": "running" if self.running else "stopped",
            "uptime": time.time() - self.start_time,
            "configuration": {
                "refresh_interval": self.refresh_interval,
                "max_data_points": self.max_data_points,
                "enable_real_time": self.enable_real_time,
            },
            "metrics_count": len(self.metrics_collector.current_metrics),
            "data_points": sum(
                len(history)
                for history in self.metrics_collector.metrics_history.values()
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get dashboard metrics"""
        return {
            "metrics_collected": len(self.metrics_collector.current_metrics),
            "total_data_points": sum(
                len(history)
                for history in self.metrics_collector.metrics_history.values()
            ),
            "performance_score": self.performance_analyzer.analyze_performance().get(
                "performance_score", 100
            ),
        }

    def _collection_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                # This will be called with the orchestration service instance
                # For now, just sleep
                time.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.refresh_interval)
