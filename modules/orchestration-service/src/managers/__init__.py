"""
Manager Classes

Central management components for the orchestration service.
"""

from .config_manager import ConfigManager, OrchestrationConfig
from .unified_config_manager import UnifiedConfigurationManager
from .unified_bot_manager import UnifiedBotManager
from .websocket_manager import WebSocketManager, ConnectionInfo, SessionInfo
from .health_monitor import HealthMonitor, HealthStatus, ServiceHealth
from .bot_manager import BotManager, BotStatus, BotInstance, MeetingRequest

__all__ = [
    "ConfigManager",
    "OrchestrationConfig",
    "UnifiedConfigurationManager",
    "UnifiedBotManager",
    "WebSocketManager",
    "ConnectionInfo",
    "SessionInfo",
    "HealthMonitor",
    "HealthStatus",
    "ServiceHealth",
    "BotManager",
    "BotStatus",
    "BotInstance",
    "MeetingRequest",
]
