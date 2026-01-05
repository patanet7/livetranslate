#!/usr/bin/env python3
"""
Unified Configuration Manager - Orchestration Service

Provides a single facade interface for all configuration management across the system.
Coordinates between multiple specialized configuration managers while maintaining
their specific functionality through a clean unified API.

Consolidates:
- ConfigManager (managers/config_manager.py) - Core orchestration config
- AudioConfigurationManager (audio/config.py) - Audio processing config
- ConfigurationSyncManager (audio/config_sync.py) - Cross-service sync
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path

# Import specialized configuration managers
from .config_manager import ConfigManager
from audio.config import AudioConfigurationManager
from audio.config_sync import ConfigurationSyncManager

logger = logging.getLogger(__name__)


class UnifiedConfigurationManager:
    """
    Unified facade for all configuration management in the orchestration service.

    Provides a single entry point for configuration operations while delegating
    to specialized managers based on configuration type and scope.

    Architecture:
    - Facade Pattern: Single interface for multiple config systems
    - Delegation: Routes requests to appropriate specialized managers
    - Event Coordination: Synchronizes changes across all managers
    - Callback System: Notifies components of configuration changes
    """

    def __init__(self):
        """Initialize unified configuration manager with all specialized managers."""
        logger.info("Initializing UnifiedConfigurationManager")

        # Initialize specialized configuration managers
        self.core_config = ConfigManager()
        self.audio_config = AudioConfigurationManager()
        self.sync_manager = ConfigurationSyncManager()

        # Configuration change callbacks
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []

        # Cross-manager coordination state
        self._coordination_enabled = True
        self._sync_lock = asyncio.Lock()

        # Register for change notifications from specialized managers
        self._setup_manager_coordination()

        logger.info("UnifiedConfigurationManager initialized successfully")

    def _setup_manager_coordination(self):
        """Set up coordination between specialized configuration managers."""
        try:
            # Register callbacks with core config manager
            if hasattr(self.core_config, "add_change_callback"):
                self.core_config.add_change_callback(self._handle_core_config_change)

            # Register callbacks with audio config manager
            if hasattr(self.audio_config, "add_change_callback"):
                self.audio_config.add_change_callback(self._handle_audio_config_change)

            # Register callbacks with sync manager
            if hasattr(self.sync_manager, "add_change_callback"):
                self.sync_manager.add_change_callback(self._handle_sync_config_change)

            logger.info("Manager coordination established")

        except Exception as e:
            logger.warning(f"Could not establish full manager coordination: {e}")

    # ========================================================================
    # Core Configuration Operations
    # ========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Routes to appropriate manager based on key prefix.
        """
        try:
            # Route to appropriate manager based on key prefix
            if key.startswith("audio."):
                return self.audio_config.get(key, default)
            elif key.startswith("sync."):
                return self.sync_manager.get(key, default)
            else:
                return self.core_config.get(key, default)

        except Exception as e:
            logger.error(f"Error getting config key '{key}': {e}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value using dot notation.
        Routes to appropriate manager and coordinates changes.
        """
        try:
            old_value = self.get(key)

            # Route to appropriate manager
            success = False
            if key.startswith("audio."):
                success = self.audio_config.set(key, value)
            elif key.startswith("sync."):
                success = self.sync_manager.set(key, value)
            else:
                success = self.core_config.set(key, value)

            # Notify callbacks if change was successful
            if success and self._coordination_enabled:
                self._notify_change_callbacks(key, old_value, value)

            return success

        except Exception as e:
            logger.error(f"Error setting config key '{key}': {e}")
            return False

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section from appropriate manager."""
        try:
            if section in ["audio", "audio_processing"]:
                return self.audio_config.get_section(section)
            elif section in ["sync", "synchronization"]:
                return self.sync_manager.get_section(section)
            else:
                return self.core_config.get_section(section)

        except Exception as e:
            logger.error(f"Error getting config section '{section}': {e}")
            return {}

    def update_section(self, section: str, values: Dict[str, Any]) -> bool:
        """Update entire configuration section in appropriate manager."""
        try:
            # Route to appropriate manager
            if section in ["audio", "audio_processing"]:
                return self.audio_config.update_section(section, values)
            elif section in ["sync", "synchronization"]:
                return self.sync_manager.update_section(section, values)
            else:
                return self.core_config.update_section(section, values)

        except Exception as e:
            logger.error(f"Error updating config section '{section}': {e}")
            return False

    # ========================================================================
    # Service Configuration (Core Manager Delegation)
    # ========================================================================

    def get_service_url(self, service_name: str) -> str:
        """Get service URL from core configuration."""
        return self.core_config.get_service_url(service_name)

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration from core manager."""
        return self.core_config.get_database_config()

    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration from core manager."""
        return self.core_config.get_websocket_config()

    def get_bot_config(self) -> Dict[str, Any]:
        """Get bot configuration from core manager."""
        return self.core_config.get_bot_config()

    # ========================================================================
    # Audio Configuration (Audio Manager Delegation)
    # ========================================================================

    def get_audio_processing_config(self) -> Dict[str, Any]:
        """Get audio processing configuration from audio manager."""
        return self.audio_config.get_processing_config()

    def get_audio_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get audio configuration preset."""
        return self.audio_config.get_preset(preset_name)

    def save_audio_preset(self, preset_name: str, config: Dict[str, Any]) -> bool:
        """Save audio configuration preset."""
        return self.audio_config.save_preset(preset_name, config)

    def get_audio_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session-specific audio configuration."""
        return self.audio_config.get_session_config(session_id)

    # ========================================================================
    # Cross-Service Synchronization (Sync Manager Delegation)
    # ========================================================================

    async def sync_configuration(
        self, target_services: Optional[List[str]] = None
    ) -> bool:
        """Synchronize configuration across services."""
        return await self.sync_manager.sync_configuration(target_services)

    async def get_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations from all services."""
        return await self.sync_manager.get_service_configs()

    async def detect_configuration_drift(self) -> Dict[str, Any]:
        """Detect configuration drift between services."""
        return await self.sync_manager.detect_configuration_drift()

    # ========================================================================
    # Configuration File Operations
    # ========================================================================

    def load_config_file(self, file_path: str) -> bool:
        """Load configuration from file, routing to appropriate manager."""
        try:
            path = Path(file_path)

            # Route based on file name patterns
            if "audio" in path.name.lower():
                return self.audio_config.load_config_file(file_path)
            elif "sync" in path.name.lower():
                return self.sync_manager.load_config_file(file_path)
            else:
                return self.core_config.load_config_file(file_path)

        except Exception as e:
            logger.error(f"Error loading config file '{file_path}': {e}")
            return False

    def save_config_file(self, file_path: str, section: Optional[str] = None) -> bool:
        """Save configuration to file from appropriate manager."""
        try:
            path = Path(file_path)

            # Route based on section or file name
            if section and section.startswith("audio"):
                return self.audio_config.save_config_file(file_path, section)
            elif section and section.startswith("sync"):
                return self.sync_manager.save_config_file(file_path, section)
            elif "audio" in path.name.lower():
                return self.audio_config.save_config_file(file_path)
            elif "sync" in path.name.lower():
                return self.sync_manager.save_config_file(file_path)
            else:
                return self.core_config.save_config_file(file_path, section)

        except Exception as e:
            logger.error(f"Error saving config file '{file_path}': {e}")
            return False

    # ========================================================================
    # Configuration Change Management
    # ========================================================================

    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add callback for configuration changes."""
        self._change_callbacks.append(callback)
        logger.debug(f"Added config change callback: {callback.__name__}")

    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Remove configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            logger.debug(f"Removed config change callback: {callback.__name__}")

    def _notify_change_callbacks(self, key: str, old_value: Any, new_value: Any):
        """Notify all registered callbacks of configuration changes."""
        for callback in self._change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(
                    f"Error in config change callback {callback.__name__}: {e}"
                )

    def _handle_core_config_change(self, key: str, old_value: Any, new_value: Any):
        """Handle configuration changes from core manager."""
        logger.debug(f"Core config change: {key} = {new_value}")
        if self._coordination_enabled:
            self._notify_change_callbacks(f"core.{key}", old_value, new_value)

    def _handle_audio_config_change(self, key: str, old_value: Any, new_value: Any):
        """Handle configuration changes from audio manager."""
        logger.debug(f"Audio config change: {key} = {new_value}")
        if self._coordination_enabled:
            self._notify_change_callbacks(f"audio.{key}", old_value, new_value)

    def _handle_sync_config_change(self, key: str, old_value: Any, new_value: Any):
        """Handle configuration changes from sync manager."""
        logger.debug(f"Sync config change: {key} = {new_value}")
        if self._coordination_enabled:
            self._notify_change_callbacks(f"sync.{key}", old_value, new_value)

    # ========================================================================
    # Validation and Health Checks
    # ========================================================================

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configurations across managers."""
        validation_results = {
            "overall_status": "valid",
            "managers": {},
            "errors": [],
            "warnings": [],
        }

        try:
            # Validate core configuration
            if hasattr(self.core_config, "validate"):
                core_result = self.core_config.validate()
                validation_results["managers"]["core"] = core_result

            # Validate audio configuration
            if hasattr(self.audio_config, "validate"):
                audio_result = self.audio_config.validate()
                validation_results["managers"]["audio"] = audio_result

            # Validate sync configuration
            if hasattr(self.sync_manager, "validate"):
                sync_result = self.sync_manager.validate()
                validation_results["managers"]["sync"] = sync_result

            # Aggregate results
            for manager, result in validation_results["managers"].items():
                if isinstance(result, dict):
                    if result.get("status") != "valid":
                        validation_results["overall_status"] = "invalid"
                    if "errors" in result:
                        validation_results["errors"].extend(result["errors"])
                    if "warnings" in result:
                        validation_results["warnings"].extend(result["warnings"])

        except Exception as e:
            validation_results["overall_status"] = "error"
            validation_results["errors"].append(f"Validation error: {e}")

        return validation_results

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of unified configuration system."""
        health_status = {
            "status": "healthy",
            "managers": {},
            "coordination": "enabled" if self._coordination_enabled else "disabled",
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # Check each manager health
            managers = [
                ("core", self.core_config),
                ("audio", self.audio_config),
                ("sync", self.sync_manager),
            ]

            for name, manager in managers:
                try:
                    if hasattr(manager, "health_check"):
                        manager_health = await manager.health_check()
                    else:
                        manager_health = {
                            "status": "healthy",
                            "note": "no health check method",
                        }

                    health_status["managers"][name] = manager_health

                    if manager_health.get("status") != "healthy":
                        health_status["status"] = "degraded"

                except Exception as e:
                    health_status["managers"][name] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health_status["status"] = "unhealthy"

        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)

        return health_status

    # ========================================================================
    # Coordination Control
    # ========================================================================

    def enable_coordination(self):
        """Enable cross-manager coordination."""
        self._coordination_enabled = True
        logger.info("Configuration coordination enabled")

    def disable_coordination(self):
        """Disable cross-manager coordination."""
        self._coordination_enabled = False
        logger.info("Configuration coordination disabled")

    async def sync_all_configurations(self) -> bool:
        """Synchronize all configurations across all managers and services."""
        async with self._sync_lock:
            try:
                logger.info("Starting unified configuration synchronization")

                # Sync within local managers first
                core_sync = (
                    self.core_config.reload()
                    if hasattr(self.core_config, "reload")
                    else True
                )
                audio_sync = (
                    self.audio_config.reload()
                    if hasattr(self.audio_config, "reload")
                    else True
                )

                # Sync across services
                service_sync = await self.sync_manager.sync_configuration()

                success = core_sync and audio_sync and service_sync

                if success:
                    logger.info(
                        "Unified configuration synchronization completed successfully"
                    )
                else:
                    logger.warning(
                        "Unified configuration synchronization completed with issues"
                    )

                return success

            except Exception as e:
                logger.error(f"Error during unified configuration synchronization: {e}")
                return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations from all managers."""
        try:
            all_configs = {}

            # Get core configuration
            if hasattr(self.core_config, "get_all"):
                all_configs["core"] = self.core_config.get_all()

            # Get audio configuration
            if hasattr(self.audio_config, "get_all"):
                all_configs["audio"] = self.audio_config.get_all()

            # Get sync configuration
            if hasattr(self.sync_manager, "get_all"):
                all_configs["sync"] = self.sync_manager.get_all()

            return all_configs

        except Exception as e:
            logger.error(f"Error getting all configurations: {e}")
            return {}

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configuration managers and their status."""
        return {
            "unified_manager": {
                "status": "active",
                "coordination_enabled": self._coordination_enabled,
                "callback_count": len(self._change_callbacks),
            },
            "core_manager": {
                "status": "active" if self.core_config else "inactive",
                "type": type(self.core_config).__name__,
            },
            "audio_manager": {
                "status": "active" if self.audio_config else "inactive",
                "type": type(self.audio_config).__name__,
            },
            "sync_manager": {
                "status": "active" if self.sync_manager else "inactive",
                "type": type(self.sync_manager).__name__,
            },
        }
