#!/usr/bin/env python3
"""
Enhanced Bot Lifecycle Manager

Advanced lifecycle management for Google Meet bots with sophisticated monitoring,
recovery strategies, resource management, and performance optimization.

Features:
- Advanced health monitoring with predictive analytics
- Multi-tier recovery strategies (restart, relocate, replace)
- Resource optimization and automatic scaling
- Performance profiling and optimization
- Comprehensive lifecycle events and metrics
- SLA monitoring and compliance tracking
"""

import asyncio
import contextlib
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifecycleStage(Enum):
    """Bot lifecycle stages."""

    REQUESTED = "requested"
    INITIALIZING = "initializing"
    SPAWNING = "spawning"
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


class RecoveryStrategy(Enum):
    """Recovery strategies for failed bots."""

    RESTART = "restart"  # Restart the same bot instance
    RELOCATE = "relocate"  # Move to different resources
    REPLACE = "replace"  # Create new bot instance
    GRACEFUL_FAIL = "graceful_fail"  # Allow graceful failure
    ESCALATE = "escalate"  # Escalate to manual intervention


class HealthStatus(Enum):
    """Bot health status levels."""

    EXCELLENT = "excellent"  # >95% performance
    GOOD = "good"  # 80-95% performance
    DEGRADED = "degraded"  # 60-80% performance
    POOR = "poor"  # 40-60% performance
    CRITICAL = "critical"  # <40% performance
    UNKNOWN = "unknown"  # No data available


@dataclass
class LifecycleEvent:
    """Lifecycle event record."""

    event_id: str
    bot_id: str
    stage_from: LifecycleStage
    stage_to: LifecycleStage
    timestamp: datetime
    duration_ms: float | None = None
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = None


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for a bot."""

    bot_id: str
    timestamp: datetime
    health_status: HealthStatus
    performance_score: float  # 0.0 - 1.0
    resource_usage: dict[str, float]  # CPU, memory, etc.
    response_times: list[float]  # Recent response times
    error_rate: float  # 0.0 - 1.0
    uptime_percentage: float  # 0.0 - 1.0
    quality_metrics: dict[str, float]  # Translation quality, etc.
    sla_compliance: bool
    predicted_failure_probability: float  # 0.0 - 1.0


@dataclass
class RecoveryAttempt:
    """Recovery attempt record."""

    attempt_id: str
    bot_id: str
    strategy: RecoveryStrategy
    timestamp: datetime
    reason: str
    success: bool
    duration_seconds: float
    resource_cost: float
    metadata: dict[str, Any] = None


class PredictiveAnalyzer:
    """Predictive analytics for bot health and failure prediction."""

    def __init__(self):
        self.health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.failure_patterns: dict[str, list[dict]] = defaultdict(list)
        self.performance_baselines: dict[str, dict] = {}

    def analyze_health_trend(
        self, bot_id: str, recent_metrics: list[HealthMetrics]
    ) -> dict[str, Any]:
        """Analyze health trends and predict future issues."""
        if not recent_metrics:
            return {"trend": "unknown", "prediction": "insufficient_data"}

        # Store in history
        for metric in recent_metrics:
            self.health_history[bot_id].append(
                {
                    "timestamp": metric.timestamp.timestamp(),
                    "performance_score": metric.performance_score,
                    "error_rate": metric.error_rate,
                    "response_time": statistics.mean(metric.response_times)
                    if metric.response_times
                    else 0,
                }
            )

        history = list(self.health_history[bot_id])
        if len(history) < 3:
            return {"trend": "insufficient_data", "prediction": "unknown"}

        # Analyze performance trend
        scores = [h["performance_score"] for h in history[-10:]]
        error_rates = [h["error_rate"] for h in history[-10:]]
        response_times = [h["response_time"] for h in history[-10:]]

        # Calculate trends
        performance_trend = self._calculate_trend(scores)
        error_trend = self._calculate_trend(error_rates)
        latency_trend = self._calculate_trend(response_times)

        # Predict failure probability
        failure_probability = self._predict_failure_probability(
            performance_trend, error_trend, latency_trend, recent_metrics[-1]
        )

        # Determine overall trend
        if performance_trend < -0.1 or error_trend > 0.1 or latency_trend > 0.2:
            trend = "declining"
        elif performance_trend > 0.1 and error_trend < -0.05 and latency_trend < -0.1:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "performance_trend": performance_trend,
            "error_trend": error_trend,
            "latency_trend": latency_trend,
            "failure_probability": failure_probability,
            "recommendation": self._generate_recommendation(
                trend, failure_probability, recent_metrics[-1]
            ),
        }

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate trend using simple linear regression."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # Linear regression: y = mx + b
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    def _predict_failure_probability(
        self,
        performance_trend: float,
        error_trend: float,
        latency_trend: float,
        current_metrics: HealthMetrics,
    ) -> float:
        """Predict probability of failure in the next 5 minutes."""
        probability = 0.0

        # Factor in current health status
        if current_metrics.health_status == HealthStatus.CRITICAL:
            probability += 0.7
        elif current_metrics.health_status == HealthStatus.POOR:
            probability += 0.4
        elif current_metrics.health_status == HealthStatus.DEGRADED:
            probability += 0.2

        # Factor in trends
        if performance_trend < -0.2:
            probability += 0.3
        if error_trend > 0.15:
            probability += 0.3
        if latency_trend > 0.3:
            probability += 0.2

        # Factor in specific metrics
        if current_metrics.error_rate > 0.1:
            probability += 0.2
        if current_metrics.performance_score < 0.5:
            probability += 0.3
        if not current_metrics.sla_compliance:
            probability += 0.2

        return min(1.0, probability)

    def _generate_recommendation(
        self, trend: str, failure_probability: float, current_metrics: HealthMetrics
    ) -> str:
        """Generate recommendations based on analysis."""
        if failure_probability > 0.8:
            return "immediate_intervention_required"
        elif failure_probability > 0.6:
            return "schedule_maintenance"
        elif failure_probability > 0.4:
            return "monitor_closely"
        elif trend == "declining":
            return "investigate_performance_degradation"
        elif current_metrics.health_status == HealthStatus.DEGRADED:
            return "optimize_resources"
        else:
            return "continue_monitoring"


class ResourceManager:
    """Manages resource allocation and optimization for bots."""

    def __init__(self):
        self.resource_limits = {
            "max_cpu_percent": 80.0,
            "max_memory_percent": 75.0,
            "max_concurrent_bots": 10,
            "min_available_memory_gb": 2.0,
        }
        self.resource_usage_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def check_resource_availability(
        self, required_resources: dict[str, float] | None = None
    ) -> tuple[bool, dict[str, Any]]:
        """Check if resources are available for new bot."""
        try:
            # Get current system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            available = True
            constraints = {}

            # Check CPU
            if cpu_percent > self.resource_limits["max_cpu_percent"]:
                available = False
                constraints["cpu"] = (
                    f"Current: {cpu_percent}%, Limit: {self.resource_limits['max_cpu_percent']}%"
                )

            # Check Memory
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            if memory_percent > self.resource_limits["max_memory_percent"]:
                available = False
                constraints["memory_percent"] = (
                    f"Current: {memory_percent}%, Limit: {self.resource_limits['max_memory_percent']}%"
                )

            if memory_available_gb < self.resource_limits["min_available_memory_gb"]:
                available = False
                constraints["memory_available"] = (
                    f"Available: {memory_available_gb:.1f}GB, Required: {self.resource_limits['min_available_memory_gb']}GB"
                )

            # Check disk space (warn if < 10GB)
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb < 10.0:
                constraints["disk_warning"] = f"Low disk space: {disk_free_gb:.1f}GB free"

            resource_status = {
                "available": available,
                "constraints": constraints,
                "current_usage": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory_available_gb,
                    "disk_free_gb": disk_free_gb,
                },
            }

            return available, resource_status

        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return False, {"error": str(e)}

    def optimize_bot_resources(
        self, bot_id: str, current_usage: dict[str, float]
    ) -> dict[str, Any]:
        """Optimize resource allocation for a bot."""
        # Store usage history
        self.resource_usage_history[bot_id].append(
            {"timestamp": time.time(), "usage": current_usage.copy()}
        )

        recommendations = {}

        # Analyze usage patterns
        history = list(self.resource_usage_history[bot_id])
        if len(history) >= 5:
            cpu_usage = [h["usage"].get("cpu_percent", 0) for h in history]
            memory_usage = [h["usage"].get("memory_percent", 0) for h in history]

            avg_cpu = statistics.mean(cpu_usage)
            avg_memory = statistics.mean(memory_usage)

            # Generate recommendations
            if avg_cpu > 70:
                recommendations["cpu"] = "consider_cpu_optimization"
            elif avg_cpu < 20:
                recommendations["cpu"] = "cpu_underutilized"

            if avg_memory > 80:
                recommendations["memory"] = "consider_memory_optimization"
            elif avg_memory < 30:
                recommendations["memory"] = "memory_underutilized"

        return {
            "recommendations": recommendations,
            "current_usage": current_usage,
            "optimization_score": self._calculate_optimization_score(current_usage),
        }

    def _calculate_optimization_score(self, usage: dict[str, float]) -> float:
        """Calculate optimization score (0.0 - 1.0, higher is better)."""
        cpu = usage.get("cpu_percent", 0)
        memory = usage.get("memory_percent", 0)

        # Ideal ranges: CPU 30-70%, Memory 40-75%
        cpu_score = 1.0 - abs(50 - cpu) / 50.0
        memory_score = 1.0 - abs(57.5 - memory) / 57.5

        return max(0.0, min(1.0, (cpu_score + memory_score) / 2.0))


class AdvancedRecoveryManager:
    """Advanced recovery management with multiple strategies."""

    def __init__(self, bot_manager, resource_manager: ResourceManager):
        self.bot_manager = bot_manager
        self.resource_manager = resource_manager
        self.recovery_history: dict[str, list[RecoveryAttempt]] = defaultdict(list)
        self.strategy_success_rates: dict[RecoveryStrategy, float] = {}

    async def attempt_recovery(
        self,
        bot_id: str,
        health_metrics: HealthMetrics,
        prediction_analysis: dict[str, Any],
    ) -> RecoveryAttempt:
        """Attempt recovery with the best strategy for the situation."""
        # Determine optimal recovery strategy
        strategy = self._select_recovery_strategy(bot_id, health_metrics, prediction_analysis)

        attempt_id = f"recovery_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            success = await self._execute_recovery_strategy(bot_id, strategy, health_metrics)
            duration = time.time() - start_time

            # Calculate resource cost (simplified)
            resource_cost = self._calculate_recovery_cost(strategy, duration, success)

            attempt = RecoveryAttempt(
                attempt_id=attempt_id,
                bot_id=bot_id,
                strategy=strategy,
                timestamp=datetime.now(),
                reason=f"Health: {health_metrics.health_status.value}, Prediction: {prediction_analysis.get('recommendation', 'unknown')}",
                success=success,
                duration_seconds=duration,
                resource_cost=resource_cost,
                metadata={
                    "initial_health": asdict(health_metrics),
                    "prediction_analysis": prediction_analysis,
                    "strategy_selected": strategy.value,
                },
            )

            # Store attempt
            self.recovery_history[bot_id].append(attempt)

            # Update strategy success rates
            self._update_strategy_success_rate(strategy, success)

            logger.info(
                f"Recovery attempt {attempt_id} for bot {bot_id}: {strategy.value} - {'SUCCESS' if success else 'FAILED'}"
            )
            return attempt

        except Exception as e:
            duration = time.time() - start_time

            attempt = RecoveryAttempt(
                attempt_id=attempt_id,
                bot_id=bot_id,
                strategy=strategy,
                timestamp=datetime.now(),
                reason=f"Recovery failed: {e!s}",
                success=False,
                duration_seconds=duration,
                resource_cost=1.0,  # High cost for failed attempts
                metadata={"error": str(e)},
            )

            self.recovery_history[bot_id].append(attempt)
            self._update_strategy_success_rate(strategy, False)

            logger.error(f"Recovery attempt {attempt_id} failed: {e}")
            return attempt

    def _select_recovery_strategy(
        self,
        bot_id: str,
        health_metrics: HealthMetrics,
        prediction_analysis: dict[str, Any],
    ) -> RecoveryStrategy:
        """Select the optimal recovery strategy."""
        # Get bot's recovery history
        history = self.recovery_history.get(bot_id, [])
        recent_attempts = [
            a for a in history if (datetime.now() - a.timestamp).seconds < 300
        ]  # Last 5 minutes

        # Check if we've tried too many times recently
        if len(recent_attempts) >= 3:
            return RecoveryStrategy.ESCALATE

        # Check resource availability
        (
            resources_available,
            resource_status,
        ) = self.resource_manager.check_resource_availability()

        # Decision matrix based on health status and conditions
        if health_metrics.health_status == HealthStatus.CRITICAL:
            if health_metrics.error_rate > 0.5:
                return RecoveryStrategy.REPLACE  # Too many errors, replace
            elif not resources_available:
                return RecoveryStrategy.GRACEFUL_FAIL  # No resources for replacement
            else:
                return RecoveryStrategy.RESTART  # Try restart first

        elif health_metrics.health_status == HealthStatus.POOR:
            if prediction_analysis.get("failure_probability", 0) > 0.7:
                return RecoveryStrategy.REPLACE  # High failure probability
            elif resource_status["current_usage"]["cpu_percent"] > 85:
                return RecoveryStrategy.RELOCATE  # Resource contention
            else:
                return RecoveryStrategy.RESTART

        elif health_metrics.health_status == HealthStatus.DEGRADED:
            if prediction_analysis.get("performance_trend", 0) < -0.3:
                return RecoveryStrategy.RESTART  # Declining performance
            else:
                return RecoveryStrategy.RESTART  # Standard restart

        else:
            return RecoveryStrategy.RESTART  # Default to restart

    async def _execute_recovery_strategy(
        self, bot_id: str, strategy: RecoveryStrategy, health_metrics: HealthMetrics
    ) -> bool:
        """Execute the selected recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RESTART:
                return await self._restart_bot(bot_id)
            elif strategy == RecoveryStrategy.RELOCATE:
                return await self._relocate_bot(bot_id)
            elif strategy == RecoveryStrategy.REPLACE:
                return await self._replace_bot(bot_id)
            elif strategy == RecoveryStrategy.GRACEFUL_FAIL:
                return await self._graceful_fail_bot(bot_id)
            elif strategy == RecoveryStrategy.ESCALATE:
                return await self._escalate_bot_issue(bot_id, health_metrics)
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Error executing recovery strategy {strategy} for bot {bot_id}: {e}")
            return False

    async def _restart_bot(self, bot_id: str) -> bool:
        """Restart a bot instance."""
        try:
            # Get bot instance
            bot = self.bot_manager.active_bots.get(bot_id)
            if not bot:
                return False

            # Stop current processes
            await self.bot_manager._stop_processing(bot_id)

            # Wait briefly
            await asyncio.sleep(2)

            # Restart processes
            success = await self.bot_manager._start_bot_processes(bot)

            if success:
                with self.bot_manager.lock:
                    bot.status = self.bot_manager.BotStatus.ACTIVE
                    bot.last_activity = datetime.now()
                    bot.error_count = 0  # Reset error count on successful restart

                logger.info(f"Successfully restarted bot: {bot_id}")

            return success

        except Exception as e:
            logger.error(f"Error restarting bot {bot_id}: {e}")
            return False

    async def _relocate_bot(self, bot_id: str) -> bool:
        """Relocate bot to different resources (simplified implementation)."""
        # For now, this is similar to restart but with resource optimization
        logger.info(f"Relocating bot {bot_id} (implementing resource optimization)")
        return await self._restart_bot(bot_id)

    async def _replace_bot(self, bot_id: str) -> bool:
        """Replace bot with a new instance."""
        try:
            # Get original bot instance
            bot = self.bot_manager.active_bots.get(bot_id)
            if not bot:
                return False

            # Create new bot request with same parameters
            new_bot_id = await self.bot_manager.request_bot(bot.meeting_request)

            if new_bot_id:
                # Terminate old bot
                await self.bot_manager.terminate_bot(bot_id)
                logger.info(f"Replaced bot {bot_id} with new bot {new_bot_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error replacing bot {bot_id}: {e}")
            return False

    async def _graceful_fail_bot(self, bot_id: str) -> bool:
        """Gracefully fail a bot when recovery is not possible."""
        try:
            logger.warning(f"Gracefully failing bot {bot_id} - recovery not possible")
            await self.bot_manager.terminate_bot(bot_id)
            return True
        except Exception as e:
            logger.error(f"Error in graceful failure for bot {bot_id}: {e}")
            return False

    async def _escalate_bot_issue(self, bot_id: str, health_metrics: HealthMetrics) -> bool:
        """Escalate bot issue for manual intervention."""
        logger.critical(
            f"Escalating bot {bot_id} for manual intervention - automatic recovery failed"
        )

        # TODO: Implement actual escalation (notifications, tickets, etc.)
        # For now, just log and mark for manual intervention

        return True

    def _calculate_recovery_cost(
        self, strategy: RecoveryStrategy, duration: float, success: bool
    ) -> float:
        """Calculate the resource cost of a recovery attempt."""
        base_costs = {
            RecoveryStrategy.RESTART: 0.1,
            RecoveryStrategy.RELOCATE: 0.3,
            RecoveryStrategy.REPLACE: 0.8,
            RecoveryStrategy.GRACEFUL_FAIL: 0.05,
            RecoveryStrategy.ESCALATE: 1.0,
        }

        cost = base_costs.get(strategy, 0.5)

        # Increase cost for failed attempts
        if not success:
            cost *= 2.0

        # Factor in duration
        cost += duration / 300.0  # Add cost based on time (5 minutes = 1.0 cost unit)

        return min(2.0, cost)  # Cap at 2.0

    def _update_strategy_success_rate(self, strategy: RecoveryStrategy, success: bool):
        """Update success rate statistics for recovery strategies."""
        if strategy not in self.strategy_success_rates:
            self.strategy_success_rates[strategy] = 0.5  # Start with neutral rate

        # Exponential moving average
        alpha = 0.1
        current_rate = self.strategy_success_rates[strategy]
        new_rate = current_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha

        self.strategy_success_rates[strategy] = new_rate


class BotLifecycleManager:
    """
    Enhanced bot lifecycle manager with advanced monitoring, recovery, and optimization.
    """

    def __init__(self, bot_manager):
        self.bot_manager = bot_manager

        # Component managers
        self.predictive_analyzer = PredictiveAnalyzer()
        self.resource_manager = ResourceManager()
        self.recovery_manager = AdvancedRecoveryManager(bot_manager, self.resource_manager)

        # Lifecycle tracking
        self.lifecycle_events: dict[str, list[LifecycleEvent]] = defaultdict(list)
        self.health_metrics_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # SLA tracking
        self.sla_targets = {
            "uptime_percentage": 99.5,
            "response_time_ms": 2000,
            "error_rate_max": 0.05,
            "recovery_time_minutes": 5,
        }

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task = None

        logger.info("Enhanced Bot Lifecycle Manager initialized")

    async def start_monitoring(self):
        """Start background monitoring and management."""
        if self.monitoring_active:
            logger.warning("Lifecycle monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started enhanced bot lifecycle monitoring")

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
        logger.info("Stopped enhanced bot lifecycle monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    async def _perform_health_checks(self):
        """Perform comprehensive health checks on all bots."""
        try:
            with self.bot_manager.lock:
                active_bots = list(self.bot_manager.active_bots.values())

            for bot in active_bots:
                try:
                    # Collect health metrics
                    health_metrics = await self._collect_health_metrics(bot)

                    # Store metrics
                    self.health_metrics_history[bot.bot_id].append(health_metrics)

                    # Analyze trends and predict issues
                    recent_metrics = list(self.health_metrics_history[bot.bot_id])[-10:]
                    analysis = self.predictive_analyzer.analyze_health_trend(
                        bot.bot_id, recent_metrics
                    )

                    # Check if intervention is needed
                    if self._should_intervene(health_metrics, analysis):
                        await self._initiate_intervention(bot.bot_id, health_metrics, analysis)

                    # Optimize resources
                    if health_metrics.health_status in [
                        HealthStatus.GOOD,
                        HealthStatus.EXCELLENT,
                    ]:
                        self.resource_manager.optimize_bot_resources(
                            bot.bot_id, health_metrics.resource_usage
                        )

                except Exception as e:
                    logger.error(f"Error in health check for bot {bot.bot_id}: {e}")

        except Exception as e:
            logger.error(f"Error in health checks: {e}")

    async def _collect_health_metrics(self, bot) -> HealthMetrics:
        """Collect comprehensive health metrics for a bot."""
        try:
            # Calculate performance score
            performance_score = self._calculate_performance_score(bot)

            # Get resource usage
            resource_usage = self._get_resource_usage(bot)

            # Calculate health status
            health_status = self._determine_health_status(performance_score, bot)

            # Get response times (simulated for now)
            response_times = [bot.performance_stats.get("average_latency", 0.5)] * 5

            # Calculate error rate
            total_messages = bot.performance_stats.get("messages_processed", 1)
            error_rate = bot.error_count / max(1, total_messages)

            # Calculate uptime
            uptime_percentage = self._calculate_uptime_percentage(bot)

            # Check SLA compliance
            sla_compliance = self._check_sla_compliance(
                performance_score, error_rate, response_times
            )

            return HealthMetrics(
                bot_id=bot.bot_id,
                timestamp=datetime.now(),
                health_status=health_status,
                performance_score=performance_score,
                resource_usage=resource_usage,
                response_times=response_times,
                error_rate=error_rate,
                uptime_percentage=uptime_percentage,
                quality_metrics=self._get_quality_metrics(bot),
                sla_compliance=sla_compliance,
                predicted_failure_probability=0.0,  # Will be set by predictive analyzer
            )

        except Exception as e:
            logger.error(f"Error collecting health metrics for bot {bot.bot_id}: {e}")
            return HealthMetrics(
                bot_id=bot.bot_id,
                timestamp=datetime.now(),
                health_status=HealthStatus.UNKNOWN,
                performance_score=0.0,
                resource_usage={},
                response_times=[],
                error_rate=1.0,
                uptime_percentage=0.0,
                quality_metrics={},
                sla_compliance=False,
                predicted_failure_probability=1.0,
            )

    def _calculate_performance_score(self, bot) -> float:
        """Calculate overall performance score for a bot."""
        score = 1.0

        # Factor in health score
        score *= bot.health_score

        # Factor in error rate
        total_messages = bot.performance_stats.get("messages_processed", 1)
        error_rate = bot.error_count / max(1, total_messages)
        score *= max(0.0, 1.0 - error_rate * 5)  # Penalize errors heavily

        # Factor in activity
        if bot.last_activity:
            seconds_since_activity = (datetime.now() - bot.last_activity).total_seconds()
            if seconds_since_activity > 300:  # 5 minutes
                score *= 0.5  # Penalize inactivity

        # Factor in status
        if bot.status.value == "error":
            score *= 0.1
        elif bot.status.value == "active":
            score *= 1.0
        else:
            score *= 0.7

        return max(0.0, min(1.0, score))

    def _get_resource_usage(self, bot) -> dict[str, float]:
        """Get resource usage for a bot (simplified)."""
        return {
            "cpu_percent": 45.0,  # Simulated
            "memory_percent": 60.0,  # Simulated
            "network_io": 10.5,  # Simulated
        }

    def _determine_health_status(self, performance_score: float, bot) -> HealthStatus:
        """Determine health status based on performance score."""
        if performance_score >= 0.95:
            return HealthStatus.EXCELLENT
        elif performance_score >= 0.80:
            return HealthStatus.GOOD
        elif performance_score >= 0.60:
            return HealthStatus.DEGRADED
        elif performance_score >= 0.40:
            return HealthStatus.POOR
        elif performance_score > 0.0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNKNOWN

    def _calculate_uptime_percentage(self, bot) -> float:
        """Calculate uptime percentage for a bot."""
        if not bot.created_at:
            return 0.0

        total_time = (datetime.now() - bot.created_at).total_seconds()
        if total_time <= 0:
            return 100.0

        # Estimate downtime based on error count and status
        estimated_downtime = bot.error_count * 30  # 30 seconds per error
        if bot.status.value == "error":
            estimated_downtime += 300  # Add 5 minutes for current error state

        uptime = max(0, total_time - estimated_downtime)
        return min(100.0, (uptime / total_time) * 100.0)

    def _get_quality_metrics(self, bot) -> dict[str, float]:
        """Get quality metrics for a bot."""
        return {
            "transcription_quality": 0.92,  # Simulated
            "translation_quality": 0.88,  # Simulated
            "audio_quality": 0.95,  # Simulated
            "correlation_accuracy": 0.85,  # Simulated
        }

    def _check_sla_compliance(
        self, performance_score: float, error_rate: float, response_times: list[float]
    ) -> bool:
        """Check if bot is meeting SLA targets."""
        # Check error rate
        if error_rate > self.sla_targets["error_rate_max"]:
            return False

        # Check response times
        if response_times and max(response_times) * 1000 > self.sla_targets["response_time_ms"]:
            return False

        # Check performance
        if performance_score < 0.8:  # 80% minimum performance for SLA
            return False

        return True

    def _should_intervene(self, health_metrics: HealthMetrics, analysis: dict[str, Any]) -> bool:
        """Determine if intervention is needed."""
        # Critical health status
        if health_metrics.health_status == HealthStatus.CRITICAL:
            return True

        # High failure probability
        if analysis.get("failure_probability", 0) > 0.6:
            return True

        # SLA violation
        if not health_metrics.sla_compliance:
            return True

        # Poor performance with declining trend
        return bool(health_metrics.health_status == HealthStatus.POOR and analysis.get("trend") == "declining")

    async def _initiate_intervention(
        self, bot_id: str, health_metrics: HealthMetrics, analysis: dict[str, Any]
    ):
        """Initiate intervention for a problematic bot."""
        logger.warning(
            f"Initiating intervention for bot {bot_id}: {health_metrics.health_status.value}"
        )

        # Record lifecycle event
        event = LifecycleEvent(
            event_id=f"intervention_{uuid.uuid4().hex[:8]}",
            bot_id=bot_id,
            stage_from=LifecycleStage.ACTIVE,
            stage_to=LifecycleStage.RECOVERING,
            timestamp=datetime.now(),
            metadata={
                "health_status": health_metrics.health_status.value,
                "analysis": analysis,
                "intervention_reason": analysis.get("recommendation", "unknown"),
            },
        )
        self.lifecycle_events[bot_id].append(event)

        # Attempt recovery
        recovery_attempt = await self.recovery_manager.attempt_recovery(
            bot_id, health_metrics, analysis
        )

        # Update lifecycle event with result
        event.success = recovery_attempt.success
        event.duration_ms = recovery_attempt.duration_seconds * 1000

        if recovery_attempt.success:
            logger.info(f"Intervention successful for bot {bot_id}")
        else:
            logger.error(f"Intervention failed for bot {bot_id}")

    def get_lifecycle_statistics(self) -> dict[str, Any]:
        """Get comprehensive lifecycle statistics."""
        total_events = sum(len(events) for events in self.lifecycle_events.values())
        total_bots = len(self.lifecycle_events)

        # Calculate success rates
        intervention_events = []
        for events in self.lifecycle_events.values():
            intervention_events.extend([e for e in events if "intervention" in e.event_id])

        intervention_success_rate = 0.0
        if intervention_events:
            successful_interventions = sum(1 for e in intervention_events if e.success)
            intervention_success_rate = successful_interventions / len(intervention_events)

        # Recovery statistics
        all_recoveries = []
        for recoveries in self.recovery_manager.recovery_history.values():
            all_recoveries.extend(recoveries)

        recovery_success_rate = 0.0
        if all_recoveries:
            successful_recoveries = sum(1 for r in all_recoveries if r.success)
            recovery_success_rate = successful_recoveries / len(all_recoveries)

        return {
            "total_bots_managed": total_bots,
            "total_lifecycle_events": total_events,
            "intervention_success_rate": intervention_success_rate,
            "recovery_success_rate": recovery_success_rate,
            "total_recovery_attempts": len(all_recoveries),
            "strategy_success_rates": dict(self.recovery_manager.strategy_success_rates),
            "sla_targets": self.sla_targets,
            "average_recovery_time": statistics.mean([r.duration_seconds for r in all_recoveries])
            if all_recoveries
            else 0,
        }


# Factory function
def create_lifecycle_manager(bot_manager) -> BotLifecycleManager:
    """Create a bot lifecycle manager."""
    return BotLifecycleManager(bot_manager)


# Example usage
async def main():
    """Example usage of the enhanced bot lifecycle manager."""
    # This would be integrated with the actual bot manager
    print("Enhanced Bot Lifecycle Manager - Example Usage")

    # Create lifecycle manager (would use actual bot manager)
    lifecycle_manager = BotLifecycleManager(None)

    # Start monitoring
    await lifecycle_manager.start_monitoring()

    # Run for a while
    await asyncio.sleep(10)

    # Get statistics
    stats = lifecycle_manager.get_lifecycle_statistics()
    print(f"Lifecycle statistics: {json.dumps(stats, indent=2, default=str)}")

    # Stop monitoring
    await lifecycle_manager.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
