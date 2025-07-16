#!/usr/bin/env python3
"""
NPU Power Management Utilities

Power optimization and thermal management for Intel NPU devices
with adaptive performance scaling and battery optimization.
"""

import os
import time
import logging
import threading
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class PowerProfile(Enum):
    """Power management profiles"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    REALTIME = "realtime"
    BATTERY = "battery"


@dataclass
class PowerMetrics:
    """Power consumption and performance metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    temperature: Optional[float] = None
    power_watts: Optional[float] = None
    battery_percent: Optional[float] = None
    inference_time: float = 0.0
    thermal_throttling: bool = False


class PowerManager:
    """
    NPU Power Management and Performance Optimization
    
    Manages power consumption, thermal throttling, and performance scaling
    for Intel NPU devices with adaptive behavior based on system conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize power manager with configuration"""
        self.config = config or {}
        self.current_profile = PowerProfile.BALANCED
        self.monitoring_enabled = True
        self.monitoring_interval = 5.0  # seconds
        
        # Power state
        self.metrics = PowerMetrics()
        self.thresholds = self._load_thresholds()
        self.callbacks = {}
        
        # Threading
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.metrics_lock = threading.Lock()
        
        # Performance tracking
        self.inference_times = []
        self.max_history = 100
        
        # Battery optimization
        self.battery_profiles = self._load_battery_profiles()
        
        logger.info(f"NPU Power Manager initialized with profile: {self.current_profile.value}")
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load power and thermal thresholds from configuration"""
        default_thresholds = {
            "temp_warning": 75.0,
            "temp_critical": 85.0,
            "temp_safe": 65.0,
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "battery_low": 20.0,
            "battery_critical": 10.0
        }
        
        thermal_config = self.config.get("thermal_management", {})
        return {**default_thresholds, **thermal_config}
    
    def _load_battery_profiles(self) -> Dict[str, PowerProfile]:
        """Load battery level to power profile mappings"""
        battery_config = self.config.get("battery_optimization", {})
        return {
            "high": PowerProfile(battery_config.get("high_battery_profile", "performance")),
            "medium": PowerProfile(battery_config.get("medium_battery_profile", "balanced")),
            "low": PowerProfile(battery_config.get("low_battery_profile", "power_saver")),
            "critical": PowerProfile(battery_config.get("critical_battery_profile", "power_saver"))
        }
    
    def set_power_profile(self, profile: PowerProfile) -> bool:
        """Set active power profile"""
        try:
            old_profile = self.current_profile
            self.current_profile = profile
            
            # Apply profile settings
            self._apply_profile_settings(profile)
            
            logger.info(f"Power profile changed: {old_profile.value} → {profile.value}")
            
            # Notify callbacks
            self._notify_callbacks("profile_changed", {
                "old_profile": old_profile.value,
                "new_profile": profile.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set power profile {profile.value}: {e}")
            return False
    
    def _apply_profile_settings(self, profile: PowerProfile):
        """Apply profile-specific settings"""
        profile_configs = self.config.get("power_profiles", {})
        profile_config = profile_configs.get(profile.value, {})
        
        # Set environment variables for OpenVINO optimization
        openvino_config = profile_config.get("openvino", {})
        for key, value in openvino_config.items():
            env_key = f"OPENVINO_{key}"
            os.environ[env_key] = str(value)
            logger.debug(f"Set {env_key}={value}")
        
        # Update monitoring interval based on profile
        if profile == PowerProfile.PERFORMANCE:
            self.monitoring_interval = 2.0  # More frequent monitoring
        elif profile == PowerProfile.POWER_SAVER:
            self.monitoring_interval = 10.0  # Less frequent monitoring
        else:
            self.monitoring_interval = 5.0  # Default
    
    def start_monitoring(self):
        """Start power and thermal monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Power monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Power monitoring started")
    
    def stop_power_monitoring(self):
        """Stop power monitoring"""
        if self.monitor_thread:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=5.0)
            logger.info("Power monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                self._update_metrics()
                self._check_thresholds()
                self._adaptive_profile_adjustment()
            except Exception as e:
                logger.error(f"Error in power monitoring: {e}")
    
    def _update_metrics(self):
        """Update power and performance metrics"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            with self.metrics_lock:
                # CPU and memory usage
                self.metrics.cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.memory_percent = psutil.virtual_memory().percent
                
                # Temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Try to get CPU temperature
                        cpu_temps = temps.get('cpu_thermal', temps.get('coretemp', []))
                        if cpu_temps:
                            self.metrics.temperature = cpu_temps[0].current
                except:
                    pass
                
                # Battery information
                try:
                    battery = psutil.sensors_battery()
                    if battery:
                        self.metrics.battery_percent = battery.percent
                except:
                    pass
                
        except Exception as e:
            logger.debug(f"Error updating metrics: {e}")
    
    def _check_thresholds(self):
        """Check if metrics exceed thresholds and take action"""
        with self.metrics_lock:
            metrics = self.metrics
        
        # Temperature monitoring
        if metrics.temperature:
            if metrics.temperature > self.thresholds["temp_critical"]:
                logger.critical(f"Critical temperature: {metrics.temperature:.1f}°C")
                self._emergency_throttle()
            elif metrics.temperature > self.thresholds["temp_warning"]:
                logger.warning(f"High temperature: {metrics.temperature:.1f}°C")
                self._thermal_throttle()
        
        # CPU usage monitoring
        if metrics.cpu_percent > self.thresholds["cpu_high"]:
            logger.debug(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory usage monitoring  
        if metrics.memory_percent > self.thresholds["memory_high"]:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Battery monitoring
        if metrics.battery_percent is not None:
            if metrics.battery_percent < self.thresholds["battery_critical"]:
                logger.critical(f"Critical battery: {metrics.battery_percent:.1f}%")
                self.set_power_profile(PowerProfile.POWER_SAVER)
            elif metrics.battery_percent < self.thresholds["battery_low"]:
                logger.warning(f"Low battery: {metrics.battery_percent:.1f}%")
                if self.current_profile == PowerProfile.PERFORMANCE:
                    self.set_power_profile(PowerProfile.BALANCED)
    
    def _thermal_throttle(self):
        """Apply thermal throttling"""
        if self.current_profile != PowerProfile.POWER_SAVER:
            logger.info("Applying thermal throttling")
            self.metrics.thermal_throttling = True
            # Consider switching to power saver mode
            if self.current_profile == PowerProfile.PERFORMANCE:
                self.set_power_profile(PowerProfile.BALANCED)
    
    def _emergency_throttle(self):
        """Apply emergency thermal throttling"""
        logger.critical("Applying emergency thermal throttling")
        self.metrics.thermal_throttling = True
        self.set_power_profile(PowerProfile.POWER_SAVER)
        
        # Notify callbacks about emergency state
        self._notify_callbacks("emergency_throttle", {
            "temperature": self.metrics.temperature,
            "action": "emergency_power_saver"
        })
    
    def _adaptive_profile_adjustment(self):
        """Adaptive power profile adjustment based on system state"""
        if not self.config.get("battery_optimization", {}).get("adaptive_profiles", False):
            return
        
        with self.metrics_lock:
            battery_percent = self.metrics.battery_percent
        
        if battery_percent is None:
            return  # No battery info available
        
        # Determine optimal profile based on battery level
        if battery_percent > 80:
            optimal_profile = self.battery_profiles["high"]
        elif battery_percent > 50:
            optimal_profile = self.battery_profiles["medium"]
        elif battery_percent > 20:
            optimal_profile = self.battery_profiles["low"]
        else:
            optimal_profile = self.battery_profiles["critical"]
        
        # Switch if different from current profile
        if optimal_profile != self.current_profile:
            logger.info(f"Adaptive profile switch based on battery ({battery_percent:.1f}%)")
            self.set_power_profile(optimal_profile)
    
    def record_inference_time(self, inference_time: float):
        """Record inference time for performance tracking"""
        with self.metrics_lock:
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_history:
                self.inference_times.pop(0)
            
            self.metrics.inference_time = inference_time
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self.metrics_lock:
            if not self.inference_times:
                return {}
            
            return {
                "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
                "min_inference_time": min(self.inference_times),
                "max_inference_time": max(self.inference_times),
                "recent_inference_time": self.inference_times[-1] if self.inference_times else 0.0,
                "total_inferences": len(self.inference_times)
            }
    
    def get_current_metrics(self) -> PowerMetrics:
        """Get current power metrics"""
        with self.metrics_lock:
            return PowerMetrics(
                cpu_percent=self.metrics.cpu_percent,
                memory_percent=self.metrics.memory_percent,
                temperature=self.metrics.temperature,
                power_watts=self.metrics.power_watts,
                battery_percent=self.metrics.battery_percent,
                inference_time=self.metrics.inference_time,
                thermal_throttling=self.metrics.thermal_throttling
            )
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for power events"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
    
    def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """Notify registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
    
    def get_profile_config(self, profile: PowerProfile = None) -> Dict[str, Any]:
        """Get configuration for a power profile"""
        if profile is None:
            profile = self.current_profile
        
        profile_configs = self.config.get("power_profiles", {})
        return profile_configs.get(profile.value, {})
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize power profile for specific workload"""
        workload_profiles = {
            "real_time": PowerProfile.REALTIME,
            "batch": PowerProfile.PERFORMANCE,
            "interactive": PowerProfile.BALANCED,
            "background": PowerProfile.POWER_SAVER
        }
        
        optimal_profile = workload_profiles.get(workload_type, PowerProfile.BALANCED)
        if optimal_profile != self.current_profile:
            logger.info(f"Optimizing for {workload_type} workload")
            self.set_power_profile(optimal_profile)
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_power_monitoring()


# Global power manager instance
_power_manager = None

def get_power_manager(config: Dict[str, Any] = None) -> PowerManager:
    """Get global power manager instance"""
    global _power_manager
    if _power_manager is None:
        _power_manager = PowerManager(config)
    return _power_manager