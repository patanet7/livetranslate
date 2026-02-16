#!/usr/bin/env python3
"""
Service Health Monitor

Provides comprehensive health monitoring, alerting, and auto-recovery
for backend services. Integrates with the monitoring-service configuration.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import requests
from livetranslate_common.logging import get_logger

logger = get_logger()


class HealthStatus(Enum):
    """Health status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthAlert:
    """Health alert representation"""

    def __init__(
        self,
        service_name: str,
        level: AlertLevel,
        message: str,
        timestamp: datetime | None = None,
    ):
        self.service_name = service_name
        self.level = level
        self.message = message
        self.timestamp = timestamp or datetime.now(UTC)
        self.resolved = False
        self.resolution_time = None

    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolution_time = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_name": self.service_name,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
        }


class ServiceHealthTracker:
    """Tracks health of a single service"""

    def __init__(self, name: str, url: str, health_endpoint: str = "/health"):
        self.name = name
        self.url = url
        self.health_endpoint = health_endpoint
        self.status = HealthStatus.UNKNOWN
        self.last_check = 0
        self.last_success = 0
        self.last_failure = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.response_times = deque(maxlen=100)  # Store last 100 response times
        self.error_history = deque(maxlen=50)  # Store last 50 errors
        self.uptime_percentage = 100.0
        self.total_checks = 0
        self.successful_checks = 0

        # Health thresholds
        self.failure_threshold = 3
        self.recovery_threshold = 2
        self.response_time_threshold = 5000  # 5 seconds

    def record_health_check(
        self,
        success: bool,
        response_time: float | None = None,
        error: str | None = None,
    ):
        """Record a health check result"""
        self.total_checks += 1
        current_time = time.time()

        if success:
            self.last_success = current_time
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.successful_checks += 1

            if response_time:
                self.response_times.append(response_time)

            # Update status based on consecutive successes
            if (
                self.status == HealthStatus.UNHEALTHY
                and self.consecutive_successes >= self.recovery_threshold
            ):
                self.status = HealthStatus.RECOVERING
            elif (
                self.status == HealthStatus.RECOVERING
                and self.consecutive_successes >= self.recovery_threshold * 2
            ) or self.status == HealthStatus.UNKNOWN:
                self.status = HealthStatus.HEALTHY

            # Check for degraded performance
            if response_time and response_time > self.response_time_threshold:
                if self.status == HealthStatus.HEALTHY:
                    self.status = HealthStatus.DEGRADED

        else:
            self.last_failure = current_time
            self.consecutive_failures += 1
            self.consecutive_successes = 0

            if error:
                self.error_history.append({"timestamp": current_time, "error": error})

            # Update status based on consecutive failures
            if self.consecutive_failures >= self.failure_threshold:
                self.status = HealthStatus.UNHEALTHY

        # Update last check time
        self.last_check = current_time

        # Calculate uptime percentage
        if self.total_checks > 0:
            self.uptime_percentage = (self.successful_checks / self.total_checks) * 100

    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary"""
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status.value,
            "last_check": self.last_check,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "uptime_percentage": round(self.uptime_percentage, 2),
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "average_response_time": round(self.get_average_response_time(), 2),
            "recent_errors": list(self.error_history)[-5:],  # Last 5 errors
            "response_time_trend": list(self.response_times)[-10:],  # Last 10 response times
        }


class ServiceHealthMonitor:
    """Comprehensive service health monitoring system"""

    def __init__(
        self,
        health_check_interval: int = 10,
        alert_threshold: int = 3,
        recovery_threshold: int = 2,
        auto_recovery: bool = True,
    ):
        """
        Initialize health monitor

        Args:
            health_check_interval: Interval between health checks (seconds)
            alert_threshold: Number of failures before alerting
            recovery_threshold: Number of successes needed for recovery
            auto_recovery: Enable automatic recovery attempts
        """
        self.health_check_interval = health_check_interval
        self.alert_threshold = alert_threshold
        self.recovery_threshold = recovery_threshold
        self.auto_recovery = auto_recovery

        # Service tracking
        self.services: dict[str, ServiceHealthTracker] = {}
        self.alerts: list[HealthAlert] = []
        self.alert_callbacks: list[Callable] = []

        # Monitoring state
        self.running = False
        self.start_time = time.time()
        self._monitor_thread = None
        self._lock = threading.RLock()

        # Metrics
        self.metrics = {
            "total_health_checks": 0,
            "successful_health_checks": 0,
            "failed_health_checks": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
            "auto_recovery_attempts": 0,
            "auto_recovery_successes": 0,
        }

        # Initialize default services from environment
        self._initialize_default_services()

        logger.info("Service health monitor initialized")

    def _initialize_default_services(self):
        """Initialize default service configurations"""
        import os

        default_services = {
            "whisper": {
                "url": os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001"),
                "health_endpoint": "/health",
            },
            "speaker": {
                "url": os.getenv("SPEAKER_SERVICE_URL", "http://localhost:5002"),
                "health_endpoint": "/health",
            },
            "translation": {
                "url": os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003"),
                "health_endpoint": "/api/health",
            },
        }

        for service_name, config in default_services.items():
            self.register_service(service_name, config["url"], config["health_endpoint"])

    async def start(self):
        """Start the health monitor"""
        with self._lock:
            if self.running:
                return

            self.running = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()

            logger.info("Service health monitor started")

    async def stop(self):
        """Stop the health monitor"""
        with self._lock:
            self.running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=10)

            logger.info("Service health monitor stopped")

    def register_service(self, name: str, url: str, health_endpoint: str = "/health"):
        """Register a service for monitoring"""
        with self._lock:
            tracker = ServiceHealthTracker(name, url, health_endpoint)
            tracker.failure_threshold = self.alert_threshold
            tracker.recovery_threshold = self.recovery_threshold

            self.services[name] = tracker

            logger.info(f"Registered service for monitoring: {name} at {url}")

    def unregister_service(self, name: str):
        """Unregister a service"""
        with self._lock:
            if name in self.services:
                del self.services[name]
                logger.info(f"Unregistered service: {name}")

    def check_service_health(self, service_name: str) -> bool:
        """Perform health check on a specific service"""
        with self._lock:
            tracker = self.services.get(service_name)
            if not tracker:
                logger.warning(f"Service not registered: {service_name}")
                return False

        try:
            start_time = time.time()
            health_url = f"{tracker.url.rstrip('/')}{tracker.health_endpoint}"

            response = requests.get(health_url, timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            success = response.status_code == 200

            # Record the result
            tracker.record_health_check(success, response_time)

            # Update metrics
            with self._lock:
                self.metrics["total_health_checks"] += 1
                if success:
                    self.metrics["successful_health_checks"] += 1
                else:
                    self.metrics["failed_health_checks"] += 1

            # Handle status changes
            self._handle_status_change(tracker, success)

            return success

        except Exception as e:
            error_msg = str(e)
            tracker.record_health_check(False, error=error_msg)

            with self._lock:
                self.metrics["total_health_checks"] += 1
                self.metrics["failed_health_checks"] += 1

            self._handle_status_change(tracker, False, error_msg)

            logger.warning(f"Health check failed for {service_name}: {error_msg}")
            return False

    def _handle_status_change(
        self, tracker: ServiceHealthTracker, success: bool, error: str | None = None
    ):
        """Handle service status changes and generate alerts"""
        previous_status = tracker.status

        # Check for status change after recording
        if previous_status != tracker.status:
            logger.info(
                f"Service {tracker.name} status changed: {previous_status.value} -> {tracker.status.value}"
            )

            # Generate alerts
            if tracker.status == HealthStatus.UNHEALTHY:
                self._generate_alert(
                    tracker.name,
                    AlertLevel.ERROR,
                    f"Service {tracker.name} is unhealthy ({tracker.consecutive_failures} consecutive failures)",
                )

                # Attempt auto-recovery
                if self.auto_recovery:
                    self._attempt_auto_recovery(tracker)

            elif tracker.status == HealthStatus.DEGRADED:
                self._generate_alert(
                    tracker.name,
                    AlertLevel.WARNING,
                    f"Service {tracker.name} performance degraded (avg response time: {tracker.get_average_response_time():.2f}ms)",
                )

            elif tracker.status == HealthStatus.HEALTHY and previous_status in [
                HealthStatus.UNHEALTHY,
                HealthStatus.RECOVERING,
            ]:
                self._generate_alert(
                    tracker.name,
                    AlertLevel.INFO,
                    f"Service {tracker.name} has recovered",
                )

                # Resolve previous alerts
                self._resolve_alerts(tracker.name)

    def _generate_alert(self, service_name: str, level: AlertLevel, message: str):
        """Generate and store an alert"""
        alert = HealthAlert(service_name, level, message)

        with self._lock:
            self.alerts.append(alert)
            self.metrics["alerts_generated"] += 1

            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts.pop(0)

        logger.log(
            logging.ERROR if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
            f"Health Alert [{level.value.upper()}] {service_name}: {message}",
        )

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _resolve_alerts(self, service_name: str):
        """Resolve all unresolved alerts for a service"""
        with self._lock:
            for alert in self.alerts:
                if alert.service_name == service_name and not alert.resolved:
                    alert.resolve()
                    self.metrics["alerts_resolved"] += 1

    def _attempt_auto_recovery(self, tracker: ServiceHealthTracker):
        """Attempt automatic service recovery"""
        if not self.auto_recovery:
            return

        logger.info(f"Attempting auto-recovery for service: {tracker.name}")

        with self._lock:
            self.metrics["auto_recovery_attempts"] += 1

        # This is a placeholder for actual recovery logic
        # In a real implementation, this might:
        # - Restart the service
        # - Clear caches
        # - Restart containers
        # - Notify orchestration systems

        # For now, just log the attempt
        self._generate_alert(
            tracker.name,
            AlertLevel.INFO,
            f"Auto-recovery attempted for service {tracker.name}",
        )

    def get_all_service_status(self) -> dict[str, Any]:
        """Get status of all monitored services"""
        with self._lock:
            service_statuses = {}

            for name, tracker in self.services.items():
                service_statuses[name] = tracker.get_health_summary()

            # Calculate overall system health
            total_services = len(self.services)
            healthy_services = sum(
                1 for t in self.services.values() if t.status == HealthStatus.HEALTHY
            )
            degraded_services = sum(
                1 for t in self.services.values() if t.status == HealthStatus.DEGRADED
            )
            unhealthy_services = sum(
                1 for t in self.services.values() if t.status == HealthStatus.UNHEALTHY
            )

            overall_status = "healthy"
            if unhealthy_services > 0:
                overall_status = "critical"
            elif degraded_services > 0:
                overall_status = "degraded"

            return {
                "services": service_statuses,
                "summary": {
                    "total_services": total_services,
                    "healthy_services": healthy_services,
                    "degraded_services": degraded_services,
                    "unhealthy_services": unhealthy_services,
                    "overall_status": overall_status,
                    "overall_uptime": (
                        sum(t.uptime_percentage for t in self.services.values())
                        / max(total_services, 1)
                    ),
                },
                "timestamp": time.time(),
            }

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent alerts"""
        with self._lock:
            recent_alerts = self.alerts[-limit:] if limit else self.alerts
            return [alert.to_dict() for alert in reversed(recent_alerts)]

    def get_metrics(self) -> dict[str, Any]:
        """Get health monitor metrics"""
        with self._lock:
            uptime = time.time() - self.start_time

            return {
                **self.metrics,
                "uptime": uptime,
                "health_checks_per_minute": (
                    self.metrics["total_health_checks"] / max(uptime / 60, 1)
                ),
                "alert_rate": (
                    self.metrics["alerts_generated"] / max(uptime / 3600, 1)  # alerts per hour
                ),
                "monitored_services": len(self.services),
                "active_alerts": len([a for a in self.alerts if not a.resolved]),
            }

    def get_status(self) -> dict[str, Any]:
        """Get health monitor status"""
        return {
            "component": "health_monitor",
            "status": "running" if self.running else "stopped",
            "uptime": time.time() - self.start_time,
            "configuration": {
                "health_check_interval": self.health_check_interval,
                "alert_threshold": self.alert_threshold,
                "recovery_threshold": self.recovery_threshold,
                "auto_recovery": self.auto_recovery,
                "monitored_services": list(self.services.keys()),
            },
            "metrics": self.get_metrics(),
        }

    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check all services
                for service_name in list(self.services.keys()):
                    if not self.running:
                        break

                    self.check_service_health(service_name)

                # Sleep until next check
                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.health_check_interval)
