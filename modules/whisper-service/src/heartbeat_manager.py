#!/usr/bin/env python3
"""
WebSocket Heartbeat Manager

Implements connection heartbeat mechanism for real-time audio streaming WebSocket server.
Provides automatic detection of disconnected clients, connection health monitoring,
and graceful cleanup of stale connections.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HeartbeatState(Enum):
    """Heartbeat connection states"""

    HEALTHY = "healthy"  # Receiving regular heartbeats
    WARNING = "warning"  # Missed some heartbeats
    CRITICAL = "critical"  # Missed many heartbeats
    DISCONNECTED = "disconnected"  # Connection considered dead


@dataclass
class HeartbeatInfo:
    """Information about connection heartbeat status"""

    connection_id: str
    last_heartbeat: datetime
    last_pong: datetime
    ping_count: int = 0
    pong_count: int = 0
    missed_heartbeats: int = 0
    state: HeartbeatState = HeartbeatState.HEALTHY
    round_trip_times: list = field(default_factory=list)  # RTT measurements in ms

    def get_average_rtt(self) -> float:
        """Get average round-trip time in milliseconds"""
        if not self.round_trip_times:
            return 0.0
        return sum(self.round_trip_times) / len(self.round_trip_times)

    def get_last_rtt(self) -> float:
        """Get last round-trip time in milliseconds"""
        return self.round_trip_times[-1] if self.round_trip_times else 0.0

    def add_rtt_measurement(self, rtt_ms: float):
        """Add RTT measurement and maintain rolling window"""
        self.round_trip_times.append(rtt_ms)
        # Keep only last 10 measurements
        if len(self.round_trip_times) > 10:
            self.round_trip_times.pop(0)

    def update_heartbeat_state(self, max_missed: int = 3):
        """Update heartbeat state based on missed heartbeats"""
        if self.missed_heartbeats == 0:
            self.state = HeartbeatState.HEALTHY
        elif self.missed_heartbeats <= max_missed // 2:
            self.state = HeartbeatState.WARNING
        elif self.missed_heartbeats <= max_missed:
            self.state = HeartbeatState.CRITICAL
        else:
            self.state = HeartbeatState.DISCONNECTED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "connection_id": self.connection_id,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "last_pong": self.last_pong.isoformat(),
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
            "missed_heartbeats": self.missed_heartbeats,
            "state": self.state.value,
            "average_rtt_ms": self.get_average_rtt(),
            "last_rtt_ms": self.get_last_rtt(),
            "rtt_measurements": self.round_trip_times.copy(),
        }


class HeartbeatManager:
    """Manages WebSocket connection heartbeats"""

    def __init__(
        self,
        ping_interval: int = 30,
        pong_timeout: int = 10,
        max_missed_heartbeats: int = 3,
        cleanup_interval: int = 60,
    ):
        """
        Initialize heartbeat manager

        Args:
            ping_interval: Interval between ping messages (seconds)
            pong_timeout: Timeout waiting for pong response (seconds)
            max_missed_heartbeats: Maximum missed heartbeats before disconnect
            cleanup_interval: Interval for cleanup tasks (seconds)
        """
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout
        self.max_missed_heartbeats = max_missed_heartbeats
        self.cleanup_interval = cleanup_interval

        # Connection tracking
        self.heartbeats: dict[str, HeartbeatInfo] = {}
        self.pending_pings: dict[str, datetime] = {}  # connection_id -> ping_time

        # Callbacks
        self.on_connection_lost: Callable[[str], None] | None = None
        self.on_connection_warning: Callable[[str, HeartbeatInfo], None] | None = None
        self.socketio_instance = None

        # Threading
        self._lock = threading.RLock()
        self._heartbeat_task = None
        self._cleanup_task = None
        self._running = False

        # Statistics
        self.total_pings_sent = 0
        self.total_pongs_received = 0
        self.total_timeouts = 0
        self.total_disconnections = 0

    def set_socketio_instance(self, socketio_instance):
        """Set the SocketIO instance for sending ping messages"""
        self.socketio_instance = socketio_instance

    def set_connection_lost_callback(self, callback: Callable[[str], None]):
        """Set callback for when connection is lost"""
        self.on_connection_lost = callback

    def set_connection_warning_callback(self, callback: Callable[[str, HeartbeatInfo], None]):
        """Set callback for connection warnings"""
        self.on_connection_warning = callback

    def start(self):
        """Start the heartbeat manager"""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Start heartbeat task
            self._heartbeat_task = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_task.start()

            # Start cleanup task
            self._cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_task.start()

            logger.info(
                f"Heartbeat manager started (ping_interval={self.ping_interval}s, "
                f"pong_timeout={self.pong_timeout}s, max_missed={self.max_missed_heartbeats})"
            )

    def stop(self):
        """Stop the heartbeat manager"""
        with self._lock:
            self._running = False

            if self._heartbeat_task:
                self._heartbeat_task.join(timeout=5)
            if self._cleanup_task:
                self._cleanup_task.join(timeout=5)

            logger.info("Heartbeat manager stopped")

    def add_connection(self, connection_id: str):
        """Add a connection to heartbeat monitoring"""
        with self._lock:
            now = datetime.now()
            self.heartbeats[connection_id] = HeartbeatInfo(
                connection_id=connection_id, last_heartbeat=now, last_pong=now
            )
            logger.debug(f"Added connection {connection_id} to heartbeat monitoring")

    def remove_connection(self, connection_id: str):
        """Remove a connection from heartbeat monitoring"""
        with self._lock:
            self.heartbeats.pop(connection_id, None)
            self.pending_pings.pop(connection_id, None)
            logger.debug(f"Removed connection {connection_id} from heartbeat monitoring")

    def handle_pong(self, connection_id: str, ping_timestamp: str | None = None):
        """Handle pong response from client"""
        with self._lock:
            heartbeat = self.heartbeats.get(connection_id)
            if not heartbeat:
                logger.warning(f"Received pong from unknown connection: {connection_id}")
                return

            now = datetime.now()
            heartbeat.last_pong = now
            heartbeat.pong_count += 1
            heartbeat.missed_heartbeats = 0  # Reset missed count

            # Calculate RTT if ping timestamp provided
            if ping_timestamp and connection_id in self.pending_pings:
                try:
                    ping_time = self.pending_pings.pop(connection_id)
                    rtt_ms = (now - ping_time).total_seconds() * 1000
                    heartbeat.add_rtt_measurement(rtt_ms)
                    logger.debug(f"Connection {connection_id} RTT: {rtt_ms:.2f}ms")
                except Exception as e:
                    logger.warning(f"Failed to calculate RTT for {connection_id}: {e}")

            # Update state
            heartbeat.update_heartbeat_state(self.max_missed_heartbeats)

            self.total_pongs_received += 1
            logger.debug(
                f"Received pong from {connection_id} (missed: {heartbeat.missed_heartbeats})"
            )

    def handle_client_heartbeat(self, connection_id: str):
        """Handle heartbeat message from client (alternative to ping/pong)"""
        with self._lock:
            heartbeat = self.heartbeats.get(connection_id)
            if not heartbeat:
                logger.warning(f"Received heartbeat from unknown connection: {connection_id}")
                return

            heartbeat.last_heartbeat = datetime.now()
            heartbeat.missed_heartbeats = 0
            heartbeat.update_heartbeat_state(self.max_missed_heartbeats)

            logger.debug(f"Received heartbeat from {connection_id}")

    def send_ping(self, connection_id: str) -> bool:
        """Send ping to specific connection"""
        if not self.socketio_instance:
            logger.error("SocketIO instance not set - cannot send ping")
            return False

        try:
            now = datetime.now()
            ping_data = {
                "type": "ping",
                "timestamp": now.isoformat(),
                "connection_id": connection_id,
            }

            # Send ping via SocketIO
            self.socketio_instance.emit("ping", ping_data, room=connection_id)

            # Track pending ping
            with self._lock:
                self.pending_pings[connection_id] = now
                heartbeat = self.heartbeats.get(connection_id)
                if heartbeat:
                    heartbeat.ping_count += 1

            self.total_pings_sent += 1
            logger.debug(f"Sent ping to {connection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send ping to {connection_id}: {e}")
            return False

    def get_connection_health(self, connection_id: str) -> HeartbeatInfo | None:
        """Get health information for a specific connection"""
        with self._lock:
            return self.heartbeats.get(connection_id)

    def get_all_connections_health(self) -> dict[str, HeartbeatInfo]:
        """Get health information for all connections"""
        with self._lock:
            return self.heartbeats.copy()

    def get_unhealthy_connections(self) -> dict[str, HeartbeatInfo]:
        """Get connections that are not healthy"""
        with self._lock:
            return {
                conn_id: heartbeat
                for conn_id, heartbeat in self.heartbeats.items()
                if heartbeat.state != HeartbeatState.HEALTHY
            }

    def get_statistics(self) -> dict[str, Any]:
        """Get heartbeat manager statistics"""
        with self._lock:
            total_connections = len(self.heartbeats)
            state_counts = {}

            for heartbeat in self.heartbeats.values():
                state = heartbeat.state.value
                state_counts[state] = state_counts.get(state, 0) + 1

            avg_rtt = 0.0
            if self.heartbeats:
                rtts = [
                    h.get_average_rtt() for h in self.heartbeats.values() if h.get_average_rtt() > 0
                ]
                avg_rtt = sum(rtts) / len(rtts) if rtts else 0.0

            return {
                "total_connections": total_connections,
                "state_distribution": state_counts,
                "total_pings_sent": self.total_pings_sent,
                "total_pongs_received": self.total_pongs_received,
                "total_timeouts": self.total_timeouts,
                "total_disconnections": self.total_disconnections,
                "average_rtt_ms": avg_rtt,
                "pending_pings": len(self.pending_pings),
                "configuration": {
                    "ping_interval": self.ping_interval,
                    "pong_timeout": self.pong_timeout,
                    "max_missed_heartbeats": self.max_missed_heartbeats,
                    "cleanup_interval": self.cleanup_interval,
                },
            }

    def _heartbeat_loop(self):
        """Main heartbeat loop"""
        while self._running:
            try:
                self._send_pings()
                self._check_timeouts()
                time.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.ping_interval)

    def _send_pings(self):
        """Send ping messages to all connections"""
        with self._lock:
            connections = list(self.heartbeats.keys())

        for connection_id in connections:
            if self._running:
                self.send_ping(connection_id)

    def _check_timeouts(self):
        """Check for ping timeouts and update connection states"""
        now = datetime.now()
        timeout_threshold = timedelta(seconds=self.pong_timeout)

        with self._lock:
            # Check pending pings for timeouts
            expired_pings = []
            for connection_id, ping_time in self.pending_pings.items():
                if now - ping_time > timeout_threshold:
                    expired_pings.append(connection_id)

            # Handle timeouts
            for connection_id in expired_pings:
                self.pending_pings.pop(connection_id, None)
                heartbeat = self.heartbeats.get(connection_id)

                if heartbeat:
                    heartbeat.missed_heartbeats += 1
                    heartbeat.update_heartbeat_state(self.max_missed_heartbeats)

                    self.total_timeouts += 1

                    logger.warning(
                        f"Ping timeout for {connection_id} "
                        f"(missed: {heartbeat.missed_heartbeats}/{self.max_missed_heartbeats})"
                    )

                    # Check if connection should be considered lost
                    if heartbeat.state == HeartbeatState.DISCONNECTED:
                        logger.error(f"Connection {connection_id} considered disconnected")
                        self.total_disconnections += 1

                        if self.on_connection_lost:
                            try:
                                self.on_connection_lost(connection_id)
                            except Exception as e:
                                logger.error(f"Error in connection lost callback: {e}")

                    elif heartbeat.state in [HeartbeatState.WARNING, HeartbeatState.CRITICAL]:
                        if self.on_connection_warning:
                            try:
                                self.on_connection_warning(connection_id, heartbeat)
                            except Exception as e:
                                logger.error(f"Error in connection warning callback: {e}")

    def _cleanup_loop(self):
        """Cleanup loop for removing disconnected connections"""
        while self._running:
            try:
                self._cleanup_disconnected_connections()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)

    def _cleanup_disconnected_connections(self):
        """Remove connections that have been disconnected for too long"""
        cleanup_threshold = timedelta(minutes=5)  # Remove after 5 minutes
        now = datetime.now()

        with self._lock:
            disconnected_connections = []

            for connection_id, heartbeat in self.heartbeats.items():
                if (
                    heartbeat.state == HeartbeatState.DISCONNECTED
                    and now - heartbeat.last_pong > cleanup_threshold
                ):
                    disconnected_connections.append(connection_id)

            for connection_id in disconnected_connections:
                self.remove_connection(connection_id)
                logger.info(f"Cleaned up disconnected connection: {connection_id}")


# Global heartbeat manager instance
heartbeat_manager = HeartbeatManager()
