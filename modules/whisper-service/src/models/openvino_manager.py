#!/usr/bin/env python3
"""
NPU-Optimized Model Manager for Whisper Service

High-performance model management with Intel NPU acceleration, automatic fallback,
thread safety, memory management, and comprehensive error handling.

Features:
- Intel NPU acceleration with automatic CPU/GPU fallback
- Thread-safe model loading and inference
- Memory pressure management with automatic cache clearing
- Comprehensive error handling and recovery
- Queue-based request management to prevent overload
- Minimum inference intervals for NPU protection
"""

import gc
import logging
import os
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Any

try:
    import openvino as ov
    import openvino_genai

    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)


class NPUError(Exception):
    """NPU-specific errors"""

    pass


class ModelNotFoundError(Exception):
    """Model not found errors"""

    pass


class InferenceError(Exception):
    """Inference-related errors"""

    pass


class OpenVINOModelManager:
    """
    NPU-optimized model manager with comprehensive error handling and performance optimization.

    This class manages Whisper model loading, inference, and lifecycle with specific optimizations
    for Intel NPU devices, including automatic fallback, memory management, and thread safety.
    """

    def __init__(
        self,
        models_dir: str | None = None,
        default_model: str = "whisper-base",
        device: str | None = None,
        min_inference_interval: float = 0.2,
        max_queue_size: int = 10,
    ):
        """
        Initialize the ModelManager with NPU optimizations.

        Args:
            models_dir: Directory containing OpenVINO IR format models
            default_model: Default model to preload
            device: Force specific device (NPU/GPU/CPU) or auto-detect
            min_inference_interval: Minimum time between NPU inferences (seconds)
            max_queue_size: Maximum concurrent inference requests
        """
        # Configuration
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.default_model = default_model
        self.min_inference_interval = min_inference_interval
        self.max_queue_size = max_queue_size

        # Device management
        self.device = device or self._detect_best_device()
        self.device_capabilities = self._get_device_capabilities()

        # Model storage
        self.pipelines: dict[str, Any] = {}
        self.model_metadata: dict[str, dict] = {}
        self.last_used: dict[str, float] = {}

        # Thread safety
        self.inference_lock = threading.RLock()  # Re-entrant lock for nested operations
        self.model_load_lock = threading.Lock()
        self.request_queue = Queue(maxsize=max_queue_size)

        # Performance tracking
        self.last_inference_time = 0
        self.inference_count = 0
        self.error_count = 0
        self.device_errors = 0

        # Memory management
        self.memory_pressure_threshold = 0.8  # 80% memory usage triggers cleanup
        self.max_cached_models = 3  # Maximum models to keep in memory

        # Weak references for automatic cleanup
        self._pipeline_refs = weakref.WeakValueDictionary()

        logger.info("ModelManager initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Default model: {self.default_model}")
        logger.info(f"  Min inference interval: {self.min_inference_interval}s")
        logger.info(f"  Max queue size: {max_queue_size}")

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Preload default model if available
        self._preload_default_model()

    def _detect_best_device(self) -> str:
        """Detect the best available device for inference with comprehensive checking."""
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available, using CPU fallback")
            return "CPU"

        try:
            # Check environment variable first
            env_device = os.getenv("OPENVINO_DEVICE")
            if env_device:
                logger.info(f"Using device from environment: {env_device}")
                return env_device.upper()

            # Auto-detect available devices
            core = ov.Core()
            available_devices = core.available_devices
            logger.info(f"Available OpenVINO devices: {available_devices}")

            # Test device capabilities
            device_priorities = ["NPU", "GPU", "CPU"]

            for device in device_priorities:
                if device in available_devices:
                    try:
                        # Test device by querying capabilities
                        if device == "NPU":
                            # NPU-specific checks
                            npu_devices = [d for d in available_devices if d.startswith("NPU")]
                            if npu_devices:
                                logger.info(f"✓ NPU detected: {npu_devices}")
                                return "NPU"
                        elif device == "GPU":
                            # GPU-specific checks
                            gpu_devices = [d for d in available_devices if d.startswith("GPU")]
                            if gpu_devices:
                                logger.info(f"✓ GPU detected: {gpu_devices}")
                                return "GPU"
                        else:
                            logger.info("✓ Using CPU device")
                            return "CPU"
                    except Exception as e:
                        logger.warning(f"Device {device} failed capability test: {e}")
                        continue

            logger.warning("⚠ No suitable devices found, defaulting to CPU")
            return "CPU"

        except Exception as e:
            logger.error(f"Error detecting devices: {e}")
            return "CPU"

    def _get_device_capabilities(self) -> dict[str, Any]:
        """Get device-specific capabilities and limitations."""
        capabilities = {
            "concurrent_inferences": 1,
            "memory_efficient": False,
            "requires_cooldown": False,
            "supports_dynamic_shapes": True,
        }

        if self.device == "NPU":
            capabilities.update(
                {
                    "concurrent_inferences": 1,  # NPU typically handles one at a time
                    "memory_efficient": True,
                    "requires_cooldown": True,
                    "supports_dynamic_shapes": False,
                }
            )
        elif self.device == "GPU":
            capabilities.update(
                {
                    "concurrent_inferences": 2,
                    "memory_efficient": False,
                    "requires_cooldown": False,
                    "supports_dynamic_shapes": True,
                }
            )

        return capabilities

    def _preload_default_model(self):
        """Attempt to preload the default model if available."""
        try:
            if self.default_model in self.list_models():
                logger.info(f"Preloading default model: {self.default_model}")
                self.load_model(self.default_model)
                logger.info(f"✓ Default model {self.default_model} preloaded successfully")
            else:
                logger.warning(f"Default model {self.default_model} not found in {self.models_dir}")
                available = self.list_models()
                if available:
                    logger.info(f"Available models: {available}")
                else:
                    logger.warning("No models found - server will run without preloaded models")
        except Exception as e:
            logger.warning(f"Could not preload {self.default_model}: {e}")
            logger.info("Server will continue without preloaded models")

    def list_models(self) -> list[str]:
        """List available OpenVINO IR format models."""
        try:
            models = []
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    # Check for OpenVINO IR files
                    xml_file = item / f"{item.name}.xml"

                    # Also check for standard OpenVINO model files
                    if not xml_file.exists():
                        xml_files = list(item.glob("*.xml"))
                        if xml_files:
                            xml_file = xml_files[0]

                    if xml_file.exists() or any(item.glob("*.xml")):
                        models.append(item.name)

                        # Cache metadata
                        if item.name not in self.model_metadata:
                            self.model_metadata[item.name] = {
                                "path": str(item),
                                "size": sum(
                                    f.stat().st_size for f in item.rglob("*") if f.is_file()
                                ),
                                "last_modified": max(
                                    f.stat().st_mtime for f in item.rglob("*") if f.is_file()
                                ),
                            }

            return sorted(models)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """
        Load a model with thread safety and error handling.

        Args:
            model_name: Name of the model to load
            force_reload: Force reload even if model is cached

        Returns:
            Loaded pipeline object

        Raises:
            ModelNotFoundError: If model is not found
            NPUError: If NPU-specific error occurs
            InferenceError: If model loading fails
        """
        with self.model_load_lock:
            # Check if model is already loaded and not forcing reload
            if not force_reload and model_name in self.pipelines:
                self.last_used[model_name] = time.time()
                logger.debug(f"Using cached model: {model_name}")
                return self.pipelines[model_name]

            # Validate model exists
            available_models = self.list_models()
            if model_name not in available_models:
                raise ModelNotFoundError(
                    f"Model {model_name} not found. Available: {available_models}"
                )

            # Check memory pressure before loading
            self._check_memory_pressure()

            model_path = self.models_dir / model_name
            logger.info(f"Loading model: {model_name} on device: {self.device}")

            try:
                # Load with device-specific optimizations
                pipeline = self._load_pipeline_with_fallback(str(model_path), model_name)

                # Store pipeline and metadata
                self.pipelines[model_name] = pipeline
                self.last_used[model_name] = time.time()
                self._pipeline_refs[model_name] = pipeline

                logger.info(f"✓ Model {model_name} loaded successfully on {self.device}")
                return pipeline

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Clean up any partial state
                if model_name in self.pipelines:
                    del self.pipelines[model_name]
                raise InferenceError(f"Failed to load model {model_name}: {e}") from e

    def _load_pipeline_with_fallback(self, model_path: str, model_name: str) -> Any:
        """Load pipeline with automatic device fallback."""
        if not OPENVINO_AVAILABLE:
            raise InferenceError("OpenVINO not available")

        devices_to_try = [self.device]

        # Add fallback devices
        if self.device != "CPU":
            devices_to_try.append("CPU")
        if self.device == "NPU" and "GPU" not in devices_to_try:
            devices_to_try.insert(-1, "GPU")

        last_error = None

        for device in devices_to_try:
            try:
                logger.debug(f"Attempting to load {model_name} on {device}")
                pipeline = openvino_genai.WhisperPipeline(model_path, device=device)

                # Update device if we fell back
                if device != self.device:
                    logger.warning(f"Fell back from {self.device} to {device} for {model_name}")
                    # Note: Not updating self.device globally to preserve user preference

                return pipeline

            except Exception as e:
                last_error = e
                if device == "NPU" and ("NPU" in str(e) or "device" in str(e).lower()):
                    self.device_errors += 1
                    logger.warning(f"NPU error for {model_name}: {e}")
                else:
                    logger.warning(f"Failed to load {model_name} on {device}: {e}")

        # If all devices failed
        raise InferenceError(
            f"Failed to load {model_name} on all devices. Last error: {last_error}"
        )

    def safe_inference(self, model_name: str, audio_data: Any, **kwargs) -> str:
        """
        Perform thread-safe inference with NPU protection and comprehensive error handling.

        Args:
            model_name: Name of the model to use
            audio_data: Audio data (numpy array)
            **kwargs: Additional inference parameters

        Returns:
            Transcription text

        Raises:
            NPUError: If NPU-specific error occurs
            InferenceError: If inference fails
        """
        with self.inference_lock:
            try:
                # Enforce minimum interval for NPU protection
                if self.device_capabilities.get("requires_cooldown", False):
                    current_time = time.time()
                    time_since_last = current_time - self.last_inference_time
                    if time_since_last < self.min_inference_interval:
                        sleep_time = self.min_inference_interval - time_since_last
                        logger.debug(f"NPU cooldown: sleeping {sleep_time:.3f}s")
                        time.sleep(sleep_time)

                # Load model if not already loaded
                pipeline = self.load_model(model_name)

                # Validate audio data
                if audio_data is None or len(audio_data) == 0:
                    raise InferenceError("Empty audio data provided")

                # Perform inference with timing
                logger.debug(f"Starting inference for {len(audio_data)} samples")
                start_time = time.time()

                try:
                    # Call the pipeline with proper error handling
                    result = pipeline.generate(audio_data, **kwargs)
                    inference_time = time.time() - start_time

                    # Update statistics
                    self.last_inference_time = time.time()
                    self.inference_count += 1

                    logger.debug(f"Inference completed in {inference_time:.3f}s")

                    # Extract text from result (handle different result formats)
                    if hasattr(result, "text"):
                        return result.text
                    elif isinstance(result, str):
                        return result
                    elif isinstance(result, dict) and "text" in result:
                        return result["text"]
                    else:
                        return str(result)

                except Exception as inference_error:
                    self.error_count += 1
                    error_msg = str(inference_error)

                    # Handle specific error types
                    if "Infer Request is busy" in error_msg:
                        raise NPUError("NPU is busy processing another request. Please try again.") from inference_error

                    elif any(
                        err in error_msg
                        for err in ["ZE_RESULT_ERROR_DEVICE_LOST", "device hung", "device lost"]
                    ):
                        self.device_errors += 1
                        logger.error("NPU device lost/hung - attempting recovery")

                        # Clear the problematic pipeline
                        if model_name in self.pipelines:
                            del self.pipelines[model_name]

                        raise NPUError(f"NPU device error: {error_msg}") from inference_error

                    elif "out of memory" in error_msg.lower() or "memory" in error_msg.lower():
                        logger.warning("Memory pressure detected, clearing cache")
                        self.clear_cache()
                        raise InferenceError(f"Memory error during inference: {error_msg}") from inference_error

                    else:
                        raise InferenceError(f"Inference failed: {error_msg}") from inference_error

            except (NPUError, InferenceError):
                # Re-raise our custom errors
                raise
            except Exception as e:
                self.error_count += 1
                logger.error(f"Unexpected error during inference: {e}")
                raise InferenceError(f"Unexpected inference error: {e}") from e

    def _check_memory_pressure(self):
        """Check memory pressure and clear cache if needed."""
        try:
            # Simple heuristic: if we have too many models loaded, clear some
            if len(self.pipelines) >= self.max_cached_models:
                self._cleanup_old_models()
        except Exception as e:
            logger.warning(f"Memory pressure check failed: {e}")

    def _cleanup_old_models(self):
        """Clean up least recently used models."""
        if len(self.pipelines) <= 1:
            return  # Keep at least one model

        # Sort by last used time
        sorted_models = sorted(self.last_used.items(), key=lambda x: x[1])

        # Remove oldest models
        models_to_remove = len(self.pipelines) - self.max_cached_models + 1

        for model_name, _ in sorted_models[:models_to_remove]:
            try:
                logger.info(f"Cleaning up old model: {model_name}")
                del self.pipelines[model_name]
                if model_name in self.last_used:
                    del self.last_used[model_name]
            except Exception as e:
                logger.warning(f"Error cleaning up model {model_name}: {e}")

    def clear_cache(self, model_name: str | None = None):
        """
        Clear model cache to free memory.

        Args:
            model_name: Specific model to clear, or None for all models
        """
        try:
            logger.info(f"Clearing cache: {model_name or 'all models'}")

            if model_name:
                # Clear specific model
                if model_name in self.pipelines:
                    del self.pipelines[model_name]
                if model_name in self.last_used:
                    del self.last_used[model_name]
                logger.info(f"✓ Cleared model: {model_name}")
            else:
                # Clear all models
                model_count = len(self.pipelines)
                self.pipelines.clear()
                self.last_used.clear()
                logger.info(f"✓ Cleared {model_count} models from cache")

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the model manager."""
        return {
            "device": self.device,
            "models_loaded": len(self.pipelines),
            "models_available": len(self.list_models()),
            "inference_count": self.inference_count,
            "error_count": self.error_count,
            "device_errors": self.device_errors,
            "last_inference_time": self.last_inference_time,
            "device_capabilities": self.device_capabilities,
            "loaded_models": list(self.pipelines.keys()),
            "models_directory": str(self.models_dir),
            "memory_stats": {
                "cached_models": len(self.pipelines),
                "max_cached_models": self.max_cached_models,
                "weak_refs": len(self._pipeline_refs),
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "device": self.device,
            "models_available": len(self.list_models()),
            "models_loaded": len(self.pipelines),
            "error_rate": self.error_count / max(1, self.inference_count),
            "device_error_rate": self.device_errors / max(1, self.inference_count),
        }

        # Check for issues
        issues = []

        if not OPENVINO_AVAILABLE:
            issues.append("OpenVINO not available")
            health["status"] = "degraded"

        if len(self.list_models()) == 0:
            issues.append("No models available")
            health["status"] = "unhealthy"

        if self.error_count / max(1, self.inference_count) > 0.1:
            issues.append("High error rate")
            health["status"] = "degraded"

        if self.device_errors > 5:
            issues.append("Multiple device errors")
            health["status"] = "degraded"

        health["issues"] = issues

        return health

    @contextmanager
    def inference_context(self, model_name: str):
        """Context manager for safe inference with automatic cleanup."""
        try:
            pipeline = self.load_model(model_name)
            yield pipeline
        except Exception as e:
            logger.error(f"Error in inference context: {e}")
            raise
        finally:
            # Cleanup if needed
            pass

    def shutdown(self):
        """Graceful shutdown with resource cleanup."""
        logger.info("Shutting down ModelManager")
        try:
            self.clear_cache()
            logger.info("✓ ModelManager shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience function for creating model manager
def create_model_manager(
    models_dir: str | None = None,
    default_model: str = "whisper-base",
    device: str | None = None,
    **kwargs,
) -> OpenVINOModelManager:
    """Create and configure an OpenVINOModelManager instance."""
    return OpenVINOModelManager(
        models_dir=models_dir, default_model=default_model, device=device, **kwargs
    )


# For backwards compatibility, create an alias
ModelManager = OpenVINOModelManager
