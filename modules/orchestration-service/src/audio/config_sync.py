#!/usr/bin/env python3
"""
Configuration Synchronization System - Orchestration Service

Manages configuration synchronization between:
- Whisper Service
- Orchestration Coordinator  
- Frontend Settings Pages

Ensures all components have consistent configuration and provides:
- Real-time configuration updates
- Bidirectional configuration sync
- Configuration validation
- Preset management
- Hot-reloading capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import asdict
import aiohttp
from pathlib import Path

from .models import AudioChunkingConfig
from .whisper_compatibility import (
    WhisperCompatibilityManager, 
    WhisperServiceConfig,
    get_frontend_compatible_config,
    CONFIGURATION_PRESETS
)

logger = logging.getLogger(__name__)


class ConfigurationSyncManager:
    """
    Manages configuration synchronization between all audio processing components.
    Ensures whisper service, orchestration coordinator, and frontend stay in sync.
    """
    
    def __init__(self, 
                 whisper_service_url: str = "http://localhost:5001",
                 orchestration_service_url: str = "http://localhost:3000",
                 translation_service_url: str = "http://localhost:5003"):
        self.whisper_service_url = whisper_service_url
        self.orchestration_service_url = orchestration_service_url
        self.translation_service_url = translation_service_url
        self.compatibility_manager = WhisperCompatibilityManager()
        
        # Configuration cache
        self._whisper_config: Optional[Dict[str, Any]] = None
        self._orchestration_config: Optional[AudioChunkingConfig] = None
        self._translation_config: Optional[Dict[str, Any]] = None
        self._last_sync_time: Optional[datetime] = None
        
        # Sync callbacks
        self._config_change_callbacks: List[Callable] = []
        
        # Configuration file paths
        self.config_dir = Path(__file__).parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.sync_config_file = self.config_dir / "sync_config.json"
        
    async def initialize(self):
        """Initialize the configuration sync manager."""
        try:
            # Load saved configuration if exists
            await self._load_saved_configuration()
            
            # Fetch current configurations from services
            await self._fetch_all_configurations()
            
            # Validate compatibility
            compatibility_check = await self._validate_configuration_compatibility()
            
            if not compatibility_check["compatible"]:
                logger.warning(f"Configuration compatibility issues found: {compatibility_check['issues']}")
                # Attempt to reconcile configurations
                await self._reconcile_configurations()
            
            logger.info("Configuration sync manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration sync manager: {e}")
            # Use default configurations
            await self._initialize_default_configurations()
    
    async def _fetch_all_configurations(self):
        """Fetch current configurations from all services."""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch whisper service configuration
                try:
                    whisper_url = f"{self.whisper_service_url}/api/config"
                    async with session.get(whisper_url, timeout=5) as response:
                        if response.status == 200:
                            self._whisper_config = await response.json()
                            logger.info("✅ Retrieved whisper service configuration")
                        else:
                            logger.warning(f"Whisper service config request failed: {response.status}")
                except Exception as e:
                    logger.warning(f"Failed to fetch whisper config: {e}")
                    self._whisper_config = None
                
                # Fetch translation service configuration
                try:
                    translation_url = f"{self.translation_service_url}/api/config"
                    async with session.get(translation_url, timeout=5) as response:
                        if response.status == 200:
                            self._translation_config = await response.json()
                            logger.info("✅ Retrieved translation service configuration")
                        else:
                            logger.warning(f"Translation service config request failed: {response.status}")
                except Exception as e:
                    logger.warning(f"Failed to fetch translation config: {e}")
                    self._translation_config = None
            
            # Get orchestration configuration from compatibility manager
            self._orchestration_config = self.compatibility_manager.orchestration_config
            self._last_sync_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to fetch configurations: {e}")
    
    async def _validate_configuration_compatibility(self) -> Dict[str, Any]:
        """Validate that all configurations are compatible."""
        compatibility_result = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "sync_required": False
        }
        
        if not self._whisper_config or not self._orchestration_config:
            compatibility_result["compatible"] = False
            compatibility_result["issues"].append("Missing configuration from one or more services")
            return compatibility_result
        
        # Compare key parameters
        whisper_cfg = self._whisper_config.get("configuration", {})
        orch_cfg = self._orchestration_config.dict()
        
        # Check timing parameters
        if abs(whisper_cfg.get("inference_interval", 3.0) - orch_cfg.get("chunk_duration", 3.0)) > 0.1:
            compatibility_result["warnings"].append(
                f"Timing mismatch: whisper inference_interval ({whisper_cfg.get('inference_interval')}) "
                f"vs orchestration chunk_duration ({orch_cfg.get('chunk_duration')})"
            )
            compatibility_result["sync_required"] = True
        
        if abs(whisper_cfg.get("overlap_duration", 0.2) - orch_cfg.get("overlap_duration", 0.2)) > 0.05:
            compatibility_result["warnings"].append(
                f"Overlap mismatch: whisper ({whisper_cfg.get('overlap_duration')}) "
                f"vs orchestration ({orch_cfg.get('overlap_duration')})"
            )
            compatibility_result["sync_required"] = True
        
        # Check buffer compatibility - allow reasonable differences
        whisper_buffer = whisper_cfg.get("buffer_duration", 4.0)
        orch_buffer = orch_cfg.get("buffer_duration", 5.0)
        buffer_diff = abs(whisper_buffer - orch_buffer)
        
        # Only flag as incompatible if difference is significant (>2 seconds)
        if buffer_diff > 2.0:
            compatibility_result["issues"].append(
                f"Significant buffer size difference: whisper buffer ({whisper_buffer}s) "
                f"vs orchestration buffer ({orch_buffer}s) - difference: {buffer_diff:.1f}s"
            )
            compatibility_result["compatible"] = False
        elif buffer_diff > 0.5:
            # Minor difference - just note it
            compatibility_result["warnings"].append(
                f"Minor buffer size difference: whisper ({whisper_buffer}s) "
                f"vs orchestration ({orch_buffer}s) - acceptable difference: {buffer_diff:.1f}s"
            )
        
        return compatibility_result
    
    async def _reconcile_configurations(self):
        """Reconcile configuration differences between services."""
        logger.info("🔄 Reconciling configuration differences")
        
        try:
            # Use whisper service settings as the source of truth for timing
            if self._whisper_config:
                whisper_cfg = self._whisper_config.get("configuration", {})
                
                # Update orchestration config to match whisper settings
                updated_config = self._orchestration_config.dict()
                updated_config.update({
                    "chunk_duration": whisper_cfg.get("inference_interval", 3.0),
                    "overlap_duration": whisper_cfg.get("overlap_duration", 0.2),
                    "processing_interval": whisper_cfg.get("inference_interval", 3.0) * 0.8,
                    "buffer_duration": max(
                        whisper_cfg.get("buffer_duration", 4.0),
                        updated_config.get("buffer_duration", 4.0)
                    )
                })
                
                self._orchestration_config = AudioChunkingConfig(**updated_config)
                
                # Save reconciled configuration
                await self._save_configuration()
                
                logger.info("✅ Configuration reconciliation completed")
                
                # Notify callbacks of configuration change
                await self._notify_config_change("reconciliation")
        
        except Exception as e:
            logger.error(f"Failed to reconcile configurations: {e}")
    
    async def _initialize_default_configurations(self):
        """Initialize with default configurations if services are unavailable."""
        logger.info("🔧 Initializing default configurations")
        
        self._whisper_config = {
            "service_mode": "legacy",
            "orchestration_mode": False,
            "configuration": {
                "sample_rate": 16000,
                "buffer_duration": 4.0,
                "inference_interval": 3.0,
                "overlap_duration": 0.2,
                "enable_vad": True,
                "default_model": "whisper-base",
                "max_concurrent_requests": 10
            }
        }
        
        self._orchestration_config = AudioChunkingConfig()
        self._last_sync_time = datetime.utcnow()
        
        await self._save_configuration()
    
    async def get_unified_configuration(self) -> Dict[str, Any]:
        """Get unified configuration for all components."""
        if not self._whisper_config or not self._orchestration_config or not self._translation_config:
            await self._fetch_all_configurations()
        
        return {
            "whisper_service": self._whisper_config,
            "orchestration_service": {
                "chunking_config": self._orchestration_config.dict() if self._orchestration_config else {},
                "service_mode": "orchestration" if self._whisper_config and self._whisper_config.get("orchestration_mode") else "legacy"
            },
            "translation_service": self._translation_config,
            "frontend_compatible": get_frontend_compatible_config(),
            "sync_info": {
                "last_sync": self._last_sync_time.isoformat() if self._last_sync_time else None,
                "sync_source": "configuration_sync_manager",
                "configuration_version": "1.1",
                "services_synced": ["whisper", "orchestration", "translation"]
            },
            "presets": CONFIGURATION_PRESETS
        }
    
    async def update_configuration(self, 
                                 component: str, 
                                 config_updates: Dict[str, Any],
                                 propagate: bool = True) -> Dict[str, Any]:
        """
        Update configuration for a specific component and optionally propagate to others.
        
        Args:
            component: 'whisper', 'orchestration', or 'unified'
            config_updates: Configuration updates to apply
            propagate: Whether to propagate changes to other components
        """
        update_result = {
            "success": False,
            "component": component,
            "changes_applied": {},
            "propagation_results": {},
            "errors": []
        }
        
        try:
            if component == "whisper":
                update_result = await self._update_whisper_configuration(config_updates, propagate)
            elif component == "orchestration":
                update_result = await self._update_orchestration_configuration(config_updates, propagate)
            elif component == "translation":
                update_result = await self._update_translation_configuration(config_updates, propagate)
            elif component == "unified":
                update_result = await self._update_unified_configuration(config_updates)
            else:
                update_result["errors"].append(f"Unknown component: {component}")
                return update_result
            
            # Save updated configuration
            if update_result["success"]:
                await self._save_configuration()
                await self._notify_config_change(component)
            
        except Exception as e:
            logger.error(f"Failed to update {component} configuration: {e}")
            update_result["errors"].append(str(e))
        
        return update_result
    
    async def _update_whisper_configuration(self, 
                                          config_updates: Dict[str, Any], 
                                          propagate: bool) -> Dict[str, Any]:
        """Update whisper service configuration."""
        result = {
            "success": False,
            "component": "whisper",
            "changes_applied": {},
            "propagation_results": {},
            "errors": []
        }
        
        try:
            # Send configuration updates to whisper service
            async with aiohttp.ClientSession() as session:
                # Note: Whisper service may need restart for some config changes
                # For now, we'll update our local cache and notify
                
                if self._whisper_config:
                    original_config = self._whisper_config.get("configuration", {}).copy()
                    
                    # Apply updates to local cache
                    for key, value in config_updates.items():
                        if key in original_config:
                            original_config[key] = value
                            result["changes_applied"][key] = value
                    
                    self._whisper_config["configuration"] = original_config
                    result["success"] = True
                    
                    # Propagate to orchestration if requested
                    if propagate:
                        orchestration_updates = self._map_whisper_to_orchestration_config(config_updates)
                        if orchestration_updates:
                            prop_result = await self._update_orchestration_configuration(orchestration_updates, False)
                            result["propagation_results"]["orchestration"] = prop_result
                
                else:
                    result["errors"].append("Whisper configuration not available")
        
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    async def _update_orchestration_configuration(self, 
                                                config_updates: Dict[str, Any], 
                                                propagate: bool) -> Dict[str, Any]:
        """Update orchestration service configuration."""
        result = {
            "success": False,
            "component": "orchestration",
            "changes_applied": {},
            "propagation_results": {},
            "errors": []
        }
        
        try:
            if self._orchestration_config:
                # Create updated config
                current_config = self._orchestration_config.dict()
                
                for key, value in config_updates.items():
                    if key in current_config:
                        current_config[key] = value
                        result["changes_applied"][key] = value
                
                # Validate updated configuration
                try:
                    self._orchestration_config = AudioChunkingConfig(**current_config)
                    result["success"] = True
                    
                    # Propagate to whisper if requested
                    if propagate:
                        whisper_updates = self._map_orchestration_to_whisper_config(config_updates)
                        if whisper_updates:
                            prop_result = await self._update_whisper_configuration(whisper_updates, False)
                            result["propagation_results"]["whisper"] = prop_result
                
                except Exception as validation_error:
                    result["errors"].append(f"Configuration validation failed: {validation_error}")
            else:
                result["errors"].append("Orchestration configuration not available")
        
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    async def _update_translation_configuration(self, 
                                              config_updates: Dict[str, Any], 
                                              propagate: bool) -> Dict[str, Any]:
        """Update translation service configuration."""
        result = {
            "success": False,
            "component": "translation",
            "changes_applied": {},
            "propagation_results": {},
            "errors": []
        }
        
        try:
            # Send configuration updates to translation service
            async with aiohttp.ClientSession() as session:
                try:
                    translation_url = f"{self.translation_service_url}/api/config"
                    
                    # Apply updates to local cache
                    if self._translation_config:
                        original_config = self._translation_config.copy()
                        
                        # Apply updates to local cache
                        for key, value in config_updates.items():
                            # Common translation parameters
                            if key in ["backend", "model_name", "temperature", "max_tokens", "gpu_enable"]:
                                original_config[key] = value
                                result["changes_applied"][key] = value
                        
                        # Send updates to translation service
                        async with session.post(
                            translation_url, 
                            json=config_updates,
                            timeout=10
                        ) as response:
                            if response.status == 200:
                                self._translation_config = original_config
                                result["success"] = True
                                logger.info("✅ Translation configuration updated successfully")
                            else:
                                result["errors"].append(f"Translation service update failed: {response.status}")
                    
                    else:
                        result["errors"].append("Translation configuration not available")
                
                except Exception as e:
                    logger.warning(f"Failed to update translation service: {e}")
                    result["errors"].append(f"Translation service communication error: {str(e)}")
                    
                    # For now, we'll just update our local cache
                    if self._translation_config:
                        for key, value in config_updates.items():
                            if key in ["backend", "model_name", "temperature", "max_tokens", "gpu_enable"]:
                                self._translation_config[key] = value
                                result["changes_applied"][key] = value
                        result["success"] = True
                        logger.info("✅ Translation configuration updated in cache (service unavailable)")
        
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    async def _update_unified_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration across all components."""
        result = {
            "success": True,
            "component": "unified",
            "changes_applied": {},
            "propagation_results": {},
            "errors": []
        }
        
        try:
            # Separate updates by component
            whisper_updates = {}
            orchestration_updates = {}
            translation_updates = {}
            
            for key, value in config_updates.items():
                # Whisper service parameters
                if key in ["sample_rate", "buffer_duration", "inference_interval", "overlap_duration", "enable_vad"]:
                    whisper_updates[key] = value
                # Orchestration service parameters
                if key in ["chunk_duration", "overlap_duration", "processing_interval", "buffer_duration"]:
                    # Map to orchestration naming
                    if key == "inference_interval":
                        orchestration_updates["chunk_duration"] = value
                    else:
                        orchestration_updates[key] = value
                # Translation service parameters
                if key in ["backend", "model_name", "temperature", "max_tokens", "gpu_enable"]:
                    translation_updates[key] = value
            
            # Apply updates to all components
            if whisper_updates:
                whisper_result = await self._update_whisper_configuration(whisper_updates, False)
                result["propagation_results"]["whisper"] = whisper_result
                if not whisper_result["success"]:
                    result["success"] = False
                    result["errors"].extend(whisper_result["errors"])
            
            if orchestration_updates:
                orch_result = await self._update_orchestration_configuration(orchestration_updates, False)
                result["propagation_results"]["orchestration"] = orch_result
                if not orch_result["success"]:
                    result["success"] = False
                    result["errors"].extend(orch_result["errors"])
            
            if translation_updates:
                trans_result = await self._update_translation_configuration(translation_updates, False)
                result["propagation_results"]["translation"] = trans_result
                if not trans_result["success"]:
                    result["success"] = False
                    result["errors"].extend(trans_result["errors"])
            
            # Combine changes applied
            result["changes_applied"].update(config_updates)
        
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
        
        return result
    
    def _map_whisper_to_orchestration_config(self, whisper_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Map whisper configuration updates to orchestration configuration."""
        mapping = {
            "inference_interval": "chunk_duration",
            "overlap_duration": "overlap_duration",
            "buffer_duration": "buffer_duration"
        }
        
        return {mapping[key]: value for key, value in whisper_updates.items() if key in mapping}
    
    def _map_orchestration_to_whisper_config(self, orchestration_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Map orchestration configuration updates to whisper configuration."""
        mapping = {
            "chunk_duration": "inference_interval",
            "overlap_duration": "overlap_duration",
            "buffer_duration": "buffer_duration"
        }
        
        return {mapping[key]: value for key, value in orchestration_updates.items() if key in mapping}
    
    async def apply_preset(self, preset_name: str) -> Dict[str, Any]:
        """Apply a configuration preset to all components."""
        if preset_name not in CONFIGURATION_PRESETS:
            return {
                "success": False,
                "error": f"Unknown preset: {preset_name}",
                "available_presets": list(CONFIGURATION_PRESETS.keys())
            }
        
        preset = CONFIGURATION_PRESETS[preset_name]
        
        # Apply preset as unified configuration
        result = await self.update_configuration("unified", preset, propagate=False)
        result["preset_applied"] = preset_name
        result["preset_description"] = preset.get("description", "")
        
        return result
    
    def add_config_change_callback(self, callback: Callable):
        """Add a callback to be notified when configuration changes."""
        self._config_change_callbacks.append(callback)
    
    async def _notify_config_change(self, source: str):
        """Notify all registered callbacks of configuration changes."""
        for callback in self._config_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(source, await self.get_unified_configuration())
                else:
                    callback(source, await self.get_unified_configuration())
            except Exception as e:
                logger.error(f"Configuration change callback failed: {e}")
    
    async def _save_configuration(self):
        """Save current configuration to file."""
        try:
            config_data = {
                "whisper_config": self._whisper_config,
                "orchestration_config": self._orchestration_config.dict() if self._orchestration_config else None,
                "translation_config": self._translation_config,
                "last_sync": self._last_sync_time.isoformat() if self._last_sync_time else None,
                "saved_at": datetime.utcnow().isoformat(),
                "version": "1.1"
            }
            
            with open(self.sync_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug("Configuration saved to file")
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def _load_saved_configuration(self):
        """Load saved configuration from file."""
        try:
            if self.sync_config_file.exists():
                with open(self.sync_config_file, 'r') as f:
                    config_data = json.load(f)
                
                self._whisper_config = config_data.get("whisper_config")
                
                orch_config_data = config_data.get("orchestration_config")
                if orch_config_data:
                    self._orchestration_config = AudioChunkingConfig(**orch_config_data)
                
                self._translation_config = config_data.get("translation_config")
                
                last_sync_str = config_data.get("last_sync")
                if last_sync_str:
                    self._last_sync_time = datetime.fromisoformat(last_sync_str)
                
                version = config_data.get("version", "1.0")
                logger.info(f"✅ Loaded saved configuration (version {version})")
        
        except Exception as e:
            logger.warning(f"Failed to load saved configuration: {e}")
    
    async def sync_with_services(self) -> Dict[str, Any]:
        """Force synchronization with all services."""
        sync_result = {
            "success": True,
            "sync_time": datetime.utcnow().isoformat(),
            "services_synced": [],
            "errors": []
        }
        
        try:
            # Fetch fresh configurations
            await self._fetch_all_configurations()
            sync_result["services_synced"].append("configuration_fetch")
            
            # Validate compatibility
            compatibility = await self._validate_configuration_compatibility()
            
            if not compatibility["compatible"]:
                await self._reconcile_configurations()
                sync_result["services_synced"].append("configuration_reconciliation")
            
            # Update sync time
            self._last_sync_time = datetime.utcnow()
            await self._save_configuration()
            
            sync_result["compatibility_status"] = compatibility
            
        except Exception as e:
            sync_result["success"] = False
            sync_result["errors"].append(str(e))
        
        return sync_result


# Global configuration sync manager instance
_config_sync_manager: Optional[ConfigurationSyncManager] = None


async def get_config_sync_manager() -> ConfigurationSyncManager:
    """Get the global configuration sync manager instance."""
    global _config_sync_manager
    
    if _config_sync_manager is None:
        _config_sync_manager = ConfigurationSyncManager()
        await _config_sync_manager.initialize()
    
    return _config_sync_manager


async def sync_all_configurations() -> Dict[str, Any]:
    """Convenience function to sync all configurations."""
    manager = await get_config_sync_manager()
    return await manager.sync_with_services()


async def get_unified_configuration() -> Dict[str, Any]:
    """Get unified configuration for all components."""
    manager = await get_config_sync_manager()
    return await manager.get_unified_configuration()


async def update_configuration(component: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration for a specific component."""
    manager = await get_config_sync_manager()
    return await manager.update_configuration(component, updates)


async def apply_configuration_preset(preset_name: str) -> Dict[str, Any]:
    """Apply a configuration preset."""
    manager = await get_config_sync_manager()
    return await manager.apply_preset(preset_name)