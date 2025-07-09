#!/usr/bin/env python3
"""
Configuration Manager

Handles configuration loading, validation, and hot-reloading for the orchestration service.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management with validation and hot-reloading"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path
        self.config = {}
        self.default_config = self._get_default_config()
        
        # Load configuration
        self.load_config()
        
        logger.info("Configuration manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "orchestration": {
                "service_name": "orchestration-service",
                "version": "2.0.0",
                "log_level": "INFO"
            },
            "frontend": {
                "host": "0.0.0.0",
                "port": 3000,
                "workers": 4,
                "secret_key": os.getenv('SECRET_KEY', 'orchestration-secret-key')
            },
            "websocket": {
                "max_connections": 10000,
                "connection_timeout": 1800,  # 30 minutes
                "max_connections_per_ip": 50,
                "cleanup_interval": 300,  # 5 minutes
                "heartbeat_interval": 30
            },
            "gateway": {
                "timeout": 30,
                "retries": 3,
                "circuit_breaker_threshold": 5,
                "request_timeout": 30
            },
            "monitoring": {
                "health_check_interval": 10,
                "alert_threshold": 3,
                "recovery_threshold": 2,
                "auto_recovery": True,
                "metrics_retention": 3600  # 1 hour
            },
            "dashboard": {
                "refresh_interval": 5,
                "max_data_points": 100,
                "enable_real_time": True
            },
            "services": {
                "whisper": {
                    "url": os.getenv('WHISPER_SERVICE_URL', 'http://localhost:5001'),
                    "health_endpoint": "/health",
                    "timeout": 30
                },
                "speaker": {
                    "url": os.getenv('SPEAKER_SERVICE_URL', 'http://localhost:5002'),
                    "health_endpoint": "/health",
                    "timeout": 30
                },
                "translation": {
                    "url": os.getenv('TRANSLATION_SERVICE_URL', 'http://localhost:5003'),
                    "health_endpoint": "/api/health",
                    "timeout": 30
                }
            },
            "security": {
                "cors_origins": ["*"],
                "rate_limiting": False,
                "max_request_size": "100MB"
            },
            "logging": {
                "level": os.getenv('LOG_LEVEL', 'INFO'),
                "format": "json",
                "file": None
            }
        }
    
    def load_config(self):
        """Load configuration from file or environment"""
        # Start with defaults
        self.config = self.default_config.copy()
        
        # Load from file if specified
        if self.config_path and os.path.exists(self.config_path):
            try:
                file_config = self._load_config_file(self.config_path)
                self.config = self._merge_configs(self.config, file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        path_obj = Path(path)
        
        with open(path, 'r') as f:
            if path_obj.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path_obj.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path_obj.suffix}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'HOST': ['frontend', 'host'],
            'PORT': ['frontend', 'port'],
            'SECRET_KEY': ['frontend', 'secret_key'],
            'LOG_LEVEL': ['logging', 'level'],
            'WEBSOCKET_MAX_CONNECTIONS': ['websocket', 'max_connections'],
            'WEBSOCKET_TIMEOUT': ['websocket', 'connection_timeout'],
            'HEALTH_CHECK_INTERVAL': ['monitoring', 'health_check_interval'],
            'GATEWAY_TIMEOUT': ['gateway', 'timeout'],
            'WHISPER_SERVICE_URL': ['services', 'whisper', 'url'],
            'SPEAKER_SERVICE_URL': ['services', 'speaker', 'url'],
            'TRANSLATION_SERVICE_URL': ['services', 'translation', 'url']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set in config
                self._set_nested_config(self.config, config_path, converted_value)
                logger.debug(f"Environment override: {env_var} -> {'.'.join(config_path)}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_config(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate port ranges
        port = self.config.get('frontend', {}).get('port', 3000)
        if not (1 <= port <= 65535):
            errors.append(f"Invalid port: {port}")
        
        # Validate timeouts
        timeouts = [
            ('websocket.connection_timeout', self.config.get('websocket', {}).get('connection_timeout', 1800)),
            ('gateway.timeout', self.config.get('gateway', {}).get('timeout', 30)),
            ('monitoring.health_check_interval', self.config.get('monitoring', {}).get('health_check_interval', 10))
        ]
        
        for name, value in timeouts:
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"Invalid timeout {name}: {value}")
        
        # Validate service URLs
        services = self.config.get('services', {})
        for service_name, service_config in services.items():
            url = service_config.get('url', '')
            if not url.startswith(('http://', 'https://')):
                errors.append(f"Invalid service URL for {service_name}: {url}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
        logger.info("Configuration updated")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.load_config()
        logger.info("Configuration reloaded")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        path_obj = Path(save_path)
        
        with open(save_path, 'w') as f:
            if path_obj.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif path_obj.suffix.lower() == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported save format: {path_obj.suffix}")
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        return self.config.get('services', {}).get(service_name, {})
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.config.get(component_name, {})