#!/usr/bin/env python3
"""
NPU Device Detection Utilities

Hardware detection and capability assessment for Intel NPU devices
with fallback support for GPU and CPU.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Device detection and capability assessment for NPU optimization"""
    
    def __init__(self):
        self.core = None
        self.available_devices = []
        self.device_properties = {}
        
        if OPENVINO_AVAILABLE:
            try:
                self.core = ov.Core()
                self.available_devices = self.core.available_devices
                self._analyze_devices()
            except Exception as e:
                logger.warning(f"OpenVINO core initialization failed: {e}")
    
    def _analyze_devices(self):
        """Analyze available devices and their capabilities"""
        for device in self.available_devices:
            try:
                # Get device properties
                properties = {}
                
                if device.startswith("NPU"):
                    properties.update(self._get_npu_properties(device))
                elif device.startswith("GPU"):
                    properties.update(self._get_gpu_properties(device))
                elif device == "CPU":
                    properties.update(self._get_cpu_properties(device))
                
                self.device_properties[device] = properties
                logger.info(f"Device {device} capabilities: {properties}")
                
            except Exception as e:
                logger.warning(f"Could not analyze device {device}: {e}")
    
    def _get_npu_properties(self, device: str) -> Dict:
        """Get NPU-specific properties"""
        properties = {
            "type": "NPU",
            "power_efficient": True,
            "recommended_precision": "FP16",
            "supports_int8": True,
            "max_batch_size": 1,
            "optimal_for": "real_time_inference"
        }
        
        try:
            if self.core:
                # Get NPU-specific properties from OpenVINO
                supported_properties = self.core.get_property(device, "SUPPORTED_PROPERTIES")
                properties["supported_properties"] = str(supported_properties)
                
                # Check for specific NPU capabilities
                try:
                    perf_hint = self.core.get_property(device, "PERFORMANCE_HINT")
                    properties["performance_hint"] = str(perf_hint)
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Could not get detailed NPU properties: {e}")
        
        return properties
    
    def _get_gpu_properties(self, device: str) -> Dict:
        """Get GPU-specific properties"""
        properties = {
            "type": "GPU",
            "power_efficient": False,
            "recommended_precision": "FP16",
            "supports_int8": True,
            "max_batch_size": 4,
            "optimal_for": "throughput"
        }
        
        try:
            if self.core:
                # Get GPU-specific properties
                device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                properties["device_name"] = str(device_name)
                
        except Exception as e:
            logger.debug(f"Could not get detailed GPU properties: {e}")
        
        return properties
    
    def _get_cpu_properties(self, device: str) -> Dict:
        """Get CPU-specific properties"""
        properties = {
            "type": "CPU",
            "power_efficient": True,
            "recommended_precision": "FP32",
            "supports_int8": True,
            "max_batch_size": 8,
            "optimal_for": "compatibility"
        }
        
        try:
            if self.core:
                # Get CPU-specific properties
                device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                properties["device_name"] = str(device_name)
                
                # Get CPU threads
                try:
                    threads = self.core.get_property(device, "INFERENCE_NUM_THREADS")
                    properties["threads"] = int(threads)
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Could not get detailed CPU properties: {e}")
        
        return properties
    
    def detect_best_device(self, preference: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Detect the best available device for inference
        
        Args:
            preference: Preferred device type (NPU, GPU, CPU) or None for auto
            
        Returns:
            Tuple of (device_name, device_properties)
        """
        if not OPENVINO_AVAILABLE:
            return "CPU", {"type": "CPU", "fallback": True}
        
        # Check environment override
        env_device = os.getenv("OPENVINO_DEVICE")
        if env_device and env_device.upper() in self.available_devices:
            device = env_device.upper()
            properties = self.device_properties.get(device, {})
            logger.info(f"Using environment device override: {device}")
            return device, properties
        
        # Honor preference if specified
        if preference:
            pref_upper = preference.upper()
            matching_devices = [d for d in self.available_devices if d.startswith(pref_upper)]
            if matching_devices:
                device = matching_devices[0]
                properties = self.device_properties.get(device, {})
                logger.info(f"Using preferred device: {device}")
                return device, properties
        
        # Auto-detect with priority: NPU > GPU > CPU
        device_priority = ["NPU", "GPU", "CPU"]
        
        for device_type in device_priority:
            matching_devices = [d for d in self.available_devices if d.startswith(device_type)]
            if matching_devices:
                device = matching_devices[0]  # Take first matching device
                properties = self.device_properties.get(device, {})
                
                if device_type == "NPU":
                    logger.info("🚀 Intel NPU detected! Using NPU acceleration")
                elif device_type == "GPU":
                    logger.info("⚡ GPU detected! Using GPU acceleration")
                else:
                    logger.info("💻 Using CPU fallback")
                
                return device, properties
        
        # Fallback to CPU if nothing detected
        logger.warning("No suitable devices detected, falling back to CPU")
        return "CPU", {"type": "CPU", "fallback": True}
    
    def get_device_info(self) -> Dict:
        """Get comprehensive device information"""
        if not OPENVINO_AVAILABLE:
            return {
                "openvino_available": False,
                "available_devices": [],
                "selected_device": "CPU",
                "device_properties": {}
            }
        
        selected_device, selected_properties = self.detect_best_device()
        
        return {
            "openvino_available": True,
            "available_devices": self.available_devices,
            "selected_device": selected_device,
            "device_properties": self.device_properties,
            "selected_properties": selected_properties,
            "npu_available": any(d.startswith("NPU") for d in self.available_devices),
            "gpu_available": any(d.startswith("GPU") for d in self.available_devices),
            "cpu_available": "CPU" in self.available_devices
        }
    
    def is_device_available(self, device_type: str) -> bool:
        """Check if a specific device type is available"""
        device_upper = device_type.upper()
        return any(d.startswith(device_upper) for d in self.available_devices)
    
    def get_fallback_chain(self, primary_device: str) -> List[str]:
        """Get fallback device chain for a primary device"""
        primary_upper = primary_device.upper()
        
        if primary_upper.startswith("NPU"):
            return ["GPU", "CPU"]
        elif primary_upper.startswith("GPU"):
            return ["CPU"]
        else:
            return []  # CPU has no fallback
    
    def validate_device_compatibility(self, device: str, model_path: str) -> bool:
        """Validate if a device can run a specific model"""
        if not OPENVINO_AVAILABLE or not self.core:
            return device.upper() == "CPU"
        
        try:
            # Try to compile the model on the device
            model = self.core.read_model(model_path)
            compiled_model = self.core.compile_model(model, device)
            return True
        except Exception as e:
            logger.debug(f"Device {device} cannot run model {model_path}: {e}")
            return False


# Global device detector instance
_device_detector = None

def get_device_detector() -> DeviceDetector:
    """Get global device detector instance"""
    global _device_detector
    if _device_detector is None:
        _device_detector = DeviceDetector()
    return _device_detector

def detect_hardware() -> Dict:
    """Convenience function to detect hardware"""
    detector = get_device_detector()
    return detector.get_device_info()

def get_optimal_device(preference: Optional[str] = None) -> Tuple[str, Dict]:
    """Convenience function to get optimal device"""
    detector = get_device_detector()
    return detector.detect_best_device(preference)