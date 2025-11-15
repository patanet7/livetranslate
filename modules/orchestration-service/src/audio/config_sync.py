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
import hashlib
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import asdict
from enum import Enum
import aiohttp
from pathlib import Path
import threading
from collections import deque

from .models import AudioChunkingConfig
from .whisper_compatibility import (
    WhisperCompatibilityManager, 
    get_frontend_compatible_config,
    CONFIGURATION_PRESETS
)

logger = logging.getLogger(__name__)


class ConfigurationValidationError(Exception):
    """Raised when configuration validation fails"""
    def __init__(self, message: str, validation_errors: List[str]):
        super().__init__(message)
        self.validation_errors = validation_errors


class ConfigurationDriftError(Exception):
    """Raised when configuration drift is detected"""
    def __init__(self, message: str, drift_details: Dict[str, Any]):
        super().__init__(message)
        self.drift_details = drift_details


class ConfigurationConflictError(Exception):
    """Raised when configuration conflicts are detected"""
    def __init__(self, message: str, conflict_details: Dict[str, Any]):
        super().__init__(message)
        self.conflict_details = conflict_details


class ConfigurationVersion:
    """Represents a configuration version with metadata"""
    def __init__(self, config: Dict[str, Any], version: str = None):
        self.config = config
        self.version = version or self._generate_version()
        self.timestamp = datetime.utcnow()
        self.checksum = self._calculate_checksum()
        
    def _generate_version(self) -> str:
        """Generate version string based on timestamp"""
        return f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _calculate_checksum(self) -> str:
        """Calculate configuration checksum for drift detection"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ConfigurationStatus(Enum):
    """Configuration synchronization status"""
    SYNCHRONIZED = "synchronized"
    DRIFT_DETECTED = "drift_detected"
    CONFLICT = "conflict"
    VALIDATION_ERROR = "validation_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    ROLLBACK_REQUIRED = "rollback_required"


class ConfigurationValidator:
    """Validates configuration changes and compatibility"""
    
    def __init__(self):
        self.validation_rules = {
            'whisper': self._validate_whisper_config,
            'orchestration': self._validate_orchestration_config,
            'translation': self._validate_translation_config,
            'unified': self._validate_unified_config
        }
        
    def validate_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for a specific component"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "compatibility_issues": []
        }
        
        try:
            if component in self.validation_rules:
                component_result = self.validation_rules[component](config)
                validation_result.update(component_result)
            else:
                validation_result["errors"].append(f"Unknown component: {component}")
                validation_result["valid"] = False
                
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
            
        return validation_result
    
    def _validate_whisper_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate whisper service configuration"""
        result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}
        
        # Check required fields
        required_fields = ["sample_rate", "buffer_duration", "inference_interval"]
        for field in required_fields:
            if field not in config:
                result["errors"].append(f"Missing required field: {field}")
        
        # Validate sample rate
        if "sample_rate" in config:
            sample_rate = config["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                result["errors"].append(f"Invalid sample_rate: {sample_rate}. Must be one of: 8000, 16000, 22050, 44100, 48000")
        
        # Validate timing parameters
        if "inference_interval" in config:
            interval = config["inference_interval"]
            if not isinstance(interval, (int, float)) or interval < 0.5 or interval > 30.0:
                result["errors"].append(f"Invalid inference_interval: {interval}. Must be between 0.5 and 30.0 seconds")
        
        if "buffer_duration" in config:
            buffer = config["buffer_duration"]
            if not isinstance(buffer, (int, float)) or buffer < 1.0 or buffer > 60.0:
                result["errors"].append(f"Invalid buffer_duration: {buffer}. Must be between 1.0 and 60.0 seconds")
        
        # Check for optimization suggestions
        if "inference_interval" in config and config["inference_interval"] > 10.0:
            result["suggestions"].append("Consider reducing inference_interval for better real-time performance")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def _validate_orchestration_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate orchestration service configuration"""
        result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}
        
        # Check chunk duration
        if "chunk_duration" in config:
            duration = config["chunk_duration"]
            if not isinstance(duration, (int, float)) or duration < 1.0 or duration > 30.0:
                result["errors"].append(f"Invalid chunk_duration: {duration}. Must be between 1.0 and 30.0 seconds")
        
        # Check overlap duration
        if "overlap_duration" in config:
            overlap = config["overlap_duration"]
            if not isinstance(overlap, (int, float)) or overlap < 0.0 or overlap > 5.0:
                result["errors"].append(f"Invalid overlap_duration: {overlap}. Must be between 0.0 and 5.0 seconds")
        
        # Check consistency between chunk and overlap
        if "chunk_duration" in config and "overlap_duration" in config:
            if config["overlap_duration"] >= config["chunk_duration"]:
                result["errors"].append("overlap_duration must be less than chunk_duration")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def _validate_translation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate translation service configuration"""
        result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}
        
        # Check backend
        if "backend" in config:
            backend = config["backend"]
            valid_backends = ["vllm", "ollama", "triton", "local"]
            if backend not in valid_backends:
                result["errors"].append(f"Invalid backend: {backend}. Must be one of: {valid_backends}")
        
        # Check model parameters
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                result["errors"].append(f"Invalid temperature: {temp}. Must be between 0.0 and 2.0")
        
        if "max_tokens" in config:
            tokens = config["max_tokens"]
            if not isinstance(tokens, int) or tokens < 1 or tokens > 4096:
                result["errors"].append(f"Invalid max_tokens: {tokens}. Must be between 1 and 4096")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def _validate_unified_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate unified configuration across all components"""
        result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}
        
        # Validate each component's config
        for component in ["whisper", "orchestration", "translation"]:
            if component in config:
                component_result = self.validate_config(component, config[component])
                if not component_result["valid"]:
                    result["errors"].extend([f"{component}: {error}" for error in component_result["errors"]])
                result["warnings"].extend([f"{component}: {warning}" for warning in component_result["warnings"]])
                result["suggestions"].extend([f"{component}: {suggestion}" for suggestion in component_result["suggestions"]])
        
        result["valid"] = len(result["errors"]) == 0
        return result


class ConfigurationVersionManager:
    """Manages configuration versions and rollback functionality"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.versions_dir = config_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        self.current_version = 1
        self.version_history = deque(maxlen=50)  # Keep last 50 versions
        self.lock = threading.Lock()
        
    def create_version(self, config: Dict[str, Any], description: str = "") -> int:
        """Create a new configuration version"""
        with self.lock:
            self.current_version += 1
            version_data = {
                "version": self.current_version,
                "timestamp": datetime.utcnow().isoformat(),
                "description": description,
                "configuration": config,
                "checksum": self._calculate_checksum(config)
            }
            
            # Save version to file
            version_file = self.versions_dir / f"config_v{self.current_version}.json"
            with open(version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            # Add to history
            self.version_history.append(version_data)
            
            logger.info(f"Created configuration version {self.current_version}: {description}")
            return self.current_version
    
    def get_version(self, version: int) -> Optional[Dict[str, Any]]:
        """Get a specific configuration version"""
        try:
            version_file = self.versions_dir / f"config_v{version}.json"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version {version}: {e}")
        return None
    
    def rollback_to_version(self, target_version: int) -> Dict[str, Any]:
        """Rollback to a specific configuration version"""
        version_data = self.get_version(target_version)
        if not version_data:
            raise ValueError(f"Version {target_version} not found")
        
        # Verify checksum
        config = version_data["configuration"]
        expected_checksum = version_data["checksum"]
        actual_checksum = self._calculate_checksum(config)
        
        if expected_checksum != actual_checksum:
            raise ValueError(f"Version {target_version} integrity check failed")
        
        logger.info(f"Rolling back to configuration version {target_version}")
        return {
            "success": True,
            "version": target_version,
            "configuration": config,
            "rollback_description": version_data.get("description", "")
        }
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List available configuration versions"""
        versions = []
        for version_data in reversed(list(self.version_history)[-limit:]):
            versions.append({
                "version": version_data["version"],
                "timestamp": version_data["timestamp"],
                "description": version_data["description"],
                "checksum": version_data["checksum"]
            })
        return versions
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate configuration checksum for integrity verification"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ConfigurationDriftDetector:
    """Detects configuration drift between services"""
    
    def __init__(self):
        self.baseline_configs = {}
        self.drift_thresholds = {
            "timing_drift_ms": 100,
            "parameter_variance": 0.1,
            "checksum_mismatch": True
        }
        
    def set_baseline(self, service: str, config: Dict[str, Any]) -> None:
        """Set baseline configuration for drift detection"""
        self.baseline_configs[service] = {
            "config": copy.deepcopy(config),
            "checksum": self._calculate_config_hash(config),
            "timestamp": datetime.utcnow()
        }
        
    def detect_drift(self, service: str, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect configuration drift for a service"""
        if service not in self.baseline_configs:
            return {
                "drift_detected": False,
                "reason": "No baseline configuration set"
            }
        
        baseline = self.baseline_configs[service]
        current_hash = self._calculate_config_hash(current_config)
        
        drift_result = {
            "drift_detected": False,
            "drift_details": [],
            "severity": "none",
            "recommendations": []
        }
        
        # Check for checksum mismatch
        if baseline["checksum"] != current_hash:
            drift_result["drift_detected"] = True
            drift_result["severity"] = "medium"
            
            # Detailed analysis
            baseline_config = baseline["config"]
            differences = self._find_config_differences(baseline_config, current_config)
            
            for diff in differences:
                drift_result["drift_details"].append(diff)
                
                # Assess severity
                if "timing" in diff["key"].lower() or "interval" in diff["key"].lower():
                    drift_result["severity"] = "high"
                    drift_result["recommendations"].append(
                        f"Timing parameter drift detected in {diff['key']}. Consider synchronization."
                    )
        
        return drift_result
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash for configuration comparison"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _find_config_differences(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find specific differences between configurations"""
        differences = []
        
        def compare_nested(base, curr, path=""):
            if isinstance(base, dict) and isinstance(curr, dict):
                all_keys = set(base.keys()) | set(curr.keys())
                for key in all_keys:
                    new_path = f"{path}.{key}" if path else key
                    if key not in base:
                        differences.append({
                            "type": "added",
                            "key": new_path,
                            "current_value": curr[key]
                        })
                    elif key not in curr:
                        differences.append({
                            "type": "removed",
                            "key": new_path,
                            "baseline_value": base[key]
                        })
                    else:
                        compare_nested(base[key], curr[key], new_path)
            elif base != curr:
                differences.append({
                    "type": "modified",
                    "key": path,
                    "baseline_value": base,
                    "current_value": curr
                })
        
        compare_nested(baseline, current)
        return differences


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
        
        # Enhanced validation components
        self.validator = ConfigurationValidator()
        self.version_manager = ConfigurationVersionManager(self.config_dir)
        self.drift_detector = ConfigurationDriftDetector()
        
        # Configuration state tracking
        self._configuration_checksums = {}
        self._sync_lock = asyncio.Lock()
        self._rollback_stack = deque(maxlen=10)
        
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
            
        except ConfigurationValidationError as e:
            update_result["errors"].extend(e.validation_errors)
            update_result["success"] = False
        except ConfigurationConflictError as e:
            update_result["errors"].append(f"Configuration conflict: {str(e)}")
            update_result["success"] = False
        except Exception as e:
            logger.error(f"Failed to update {component} configuration: {e}")
            update_result["errors"].append(str(e))
            update_result["success"] = False
        
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
            async with aiohttp.ClientSession():
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
    
    async def validate_configuration_before_apply(self, config: Dict[str, Any], component: str = "unified") -> Dict[str, Any]:
        """Enhanced validation before applying configuration changes"""
        async with self._sync_lock:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "drift_detected": False,
                "conflict_resolution_required": False,
                "rollback_recommended": False
            }
            
            try:
                # Basic validation
                component_validation = self.validator.validate_config(component, config)
                validation_result.update(component_validation)
                
                # Drift detection
                if component in self._configuration_checksums:
                    drift_result = self.drift_detector.detect_drift(component, config)
                    if drift_result["drift_detected"]:
                        validation_result["drift_detected"] = True
                        validation_result["drift_details"] = drift_result["drift_details"]
                        validation_result["warnings"].append(f"Configuration drift detected for {component}")
                
                # Conflict detection across services
                conflict_result = await self._detect_configuration_conflicts(config, component)
                if conflict_result["conflicts_found"]:
                    validation_result["conflict_resolution_required"] = True
                    validation_result["conflicts"] = conflict_result["conflicts"]
                
                # Version impact assessment
                if not validation_result["valid"]:
                    validation_result["rollback_recommended"] = True
                    
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Validation failed: {str(e)}")
                logger.error(f"Configuration validation error: {e}")
            
            return validation_result
    
    async def _detect_configuration_conflicts(self, new_config: Dict[str, Any], component: str) -> Dict[str, Any]:
        """Detect conflicts between new configuration and existing service configurations"""
        conflict_result = {
            "conflicts_found": False,
            "conflicts": [],
            "severity": "none"
        }
        
        try:
            # Fetch current configurations from all services
            current_configs = {
                "whisper": self._whisper_config,
                "orchestration": asdict(self._orchestration_config) if self._orchestration_config else {},
                "translation": self._translation_config
            }
            
            # Check for parameter conflicts
            for service, current_config in current_configs.items():
                if not current_config or service == component:
                    continue
                    
                conflicts = self._find_parameter_conflicts(new_config, current_config, service)
                if conflicts:
                    conflict_result["conflicts_found"] = True
                    conflict_result["conflicts"].extend(conflicts)
                    
                    # Assess severity
                    for conflict in conflicts:
                        if conflict["type"] in ["sample_rate", "timing", "hardware"]:
                            conflict_result["severity"] = "high"
                        elif conflict_result["severity"] != "high":
                            conflict_result["severity"] = "medium"
                            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            
        return conflict_result
    
    def _find_parameter_conflicts(self, new_config: Dict[str, Any], existing_config: Dict[str, Any], service: str) -> List[Dict[str, Any]]:
        """Find specific parameter conflicts between configurations"""
        conflicts = []
        
        # Critical parameters that must be synchronized
        critical_params = {
            "sample_rate": "Sample rate mismatch can cause audio processing issues",
            "chunk_size": "Chunk size differences may cause buffering problems", 
            "overlap": "Overlap parameter mismatch can cause audio gaps",
            "model": "Model differences may cause inconsistent results"
        }
        
        for param, description in critical_params.items():
            new_value = self._get_nested_value(new_config, param)
            existing_value = self._get_nested_value(existing_config, param)
            
            if new_value is not None and existing_value is not None and new_value != existing_value:
                conflicts.append({
                    "type": param,
                    "service": service,
                    "new_value": new_value,
                    "existing_value": existing_value,
                    "description": description,
                    "resolution_required": True
                })
                
        return conflicts
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get value from nested configuration dict"""
        keys = key.split('.')
        value = config
        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            return value
        except (KeyError, TypeError):
            return None
    
    async def resolve_configuration_conflicts(self, conflicts: List[Dict[str, Any]], resolution_strategy: str = "auto") -> Dict[str, Any]:
        """Resolve configuration conflicts using specified strategy"""
        resolution_result = {
            "success": True,
            "resolved_conflicts": [],
            "unresolved_conflicts": [],
            "actions_taken": []
        }
        
        try:
            for conflict in conflicts:
                if resolution_strategy == "auto":
                    resolution = await self._auto_resolve_conflict(conflict)
                elif resolution_strategy == "prefer_whisper":
                    resolution = self._prefer_service_resolution(conflict, "whisper")
                elif resolution_strategy == "prefer_orchestration":
                    resolution = self._prefer_service_resolution(conflict, "orchestration")
                else:
                    resolution = {"success": False, "reason": "Unknown resolution strategy"}
                
                if resolution["success"]:
                    resolution_result["resolved_conflicts"].append({
                        "conflict": conflict,
                        "resolution": resolution
                    })
                    resolution_result["actions_taken"].append(resolution["action"])
                else:
                    resolution_result["unresolved_conflicts"].append(conflict)
                    
        except Exception as e:
            resolution_result["success"] = False
            resolution_result["error"] = str(e)
            logger.error(f"Conflict resolution failed: {e}")
        
        return resolution_result
    
    async def _auto_resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically resolve a configuration conflict using smart defaults"""
        conflict_type = conflict["type"]
        
        if conflict_type == "sample_rate":
            # Prefer 16kHz for whisper compatibility
            return {
                "success": True,
                "action": f"Set sample rate to 16000 Hz for {conflict['service']}",
                "resolved_value": 16000
            }
        elif conflict_type == "chunk_size":
            # Use conservative chunk size
            return {
                "success": True,
                "action": f"Set chunk size to 1024 for {conflict['service']}",
                "resolved_value": 1024
            }
        elif conflict_type == "model":
            # Prefer whisper-base for compatibility
            return {
                "success": True,
                "action": f"Set model to whisper-base for {conflict['service']}",
                "resolved_value": "whisper-base"
            }
        else:
            return {
                "success": False,
                "reason": f"No auto-resolution available for {conflict_type}"
            }
    
    def _prefer_service_resolution(self, conflict: Dict[str, Any], preferred_service: str) -> Dict[str, Any]:
        """Resolve conflict by preferring configuration from specific service"""
        if conflict["service"] == preferred_service:
            return {
                "success": True,
                "action": f"Keep {preferred_service} configuration for {conflict['type']}",
                "resolved_value": conflict["existing_value"]
            }
        else:
            return {
                "success": True,
                "action": f"Update {conflict['service']} to match {preferred_service} configuration",
                "resolved_value": conflict["new_value"]
            }
    
    async def create_configuration_checkpoint(self, description: str = "Manual checkpoint") -> Dict[str, Any]:
        """Create a configuration checkpoint for rollback purposes"""
        try:
            current_config = {
                "whisper": self._whisper_config,
                "orchestration": asdict(self._orchestration_config) if self._orchestration_config else {},
                "translation": self._translation_config,
                "sync_metadata": {
                    "last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
                    "checksums": self._configuration_checksums.copy()
                }
            }
            
            version = self.version_manager.create_version(current_config, description)
            
            # Add to rollback stack
            self._rollback_stack.append({
                "version": version,
                "timestamp": datetime.utcnow(),
                "description": description
            })
            
            return {
                "success": True,
                "version": version,
                "description": description,
                "checkpoint_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create configuration checkpoint: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def rollback_configuration(self, target_version: Optional[int] = None) -> Dict[str, Any]:
        """Rollback configuration to a previous version"""
        try:
            if target_version is None and self._rollback_stack:
                # Rollback to latest checkpoint
                target_version = self._rollback_stack[-1]["version"]
            
            if target_version is None:
                return {
                    "success": False,
                    "error": "No rollback version specified and no checkpoints available"
                }
            
            rollback_result = self.version_manager.rollback_to_version(target_version)
            
            if rollback_result["success"]:
                # Apply rolled back configuration
                config = rollback_result["configuration"]
                
                # Set configurations
                self._whisper_config = config.get("whisper")
                if config.get("orchestration"):
                    self._orchestration_config = AudioChunkingConfig(**config["orchestration"])
                self._translation_config = config.get("translation")
                
                # Restore metadata
                if "sync_metadata" in config:
                    metadata = config["sync_metadata"]
                    if metadata.get("checksums"):
                        self._configuration_checksums = metadata["checksums"]
                
                # Save configuration
                await self._save_configuration()
                
                # Sync with services
                await self.sync_with_services()
                
                logger.info(f"Successfully rolled back to configuration version {target_version}")
                
                return {
                    "success": True,
                    "version": target_version,
                    "description": rollback_result.get("rollback_description", ""),
                    "rollback_time": datetime.utcnow().isoformat()
                }
            else:
                return rollback_result
                
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


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