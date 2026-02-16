#!/usr/bin/env python3
"""
Model Factory for Whisper Service

Smart device selection and model creation with automatic fallback.

Device Priority: GPU/MPS → NPU → CPU

Phase 2 Refactoring: Created 2025-10-25
"""

import torch
from livetranslate_common.logging import get_logger

from .base_model import WhisperModel
from .pytorch_manager import PyTorchModelManager

# OpenVINO is OPTIONAL - may not be installed on all systems
try:
    import openvino as ov

    from .openvino_manager import OpenVINOModelManager

    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    OpenVINOModelManager = None  # type: ignore

logger = get_logger()


class ModelFactory:
    """
    Factory for creating appropriate Whisper model implementation based on available hardware.

    Device Selection Priority:
    1. **GPU (CUDA)** - NVIDIA GPUs via PyTorch (Linux/Windows)
    2. **MPS (Metal)** - Apple Silicon via PyTorch (Mac M1/M2/M3)
    3. **NPU (Neural Processing Unit)** - Intel NPU via OpenVINO (if installed)
    4. **CPU** - Universal fallback via PyTorch

    Notes:
    - OpenVINO is OPTIONAL and may not be installed
    - OpenVINO may not work correctly on Mac (versioning issues)
    - PyTorch is always available and provides reliable fallback

    Example:
        ```python
        # Auto-detect best device
        model = ModelFactory.create()

        # Force specific device
        model_gpu = ModelFactory.create(device="cuda")
        model_npu = ModelFactory.create(device="npu")  # Falls back if OpenVINO unavailable
        ```
    """

    @staticmethod
    def create(
        device: str = "auto",
        model_name: str = "large-v3-turbo",
        models_dir: str | None = None,
        **kwargs,
    ) -> WhisperModel:
        """
        Create appropriate model implementation based on device and availability.

        Args:
            device: Device preference ("auto", "cuda", "gpu", "mps", "npu", "cpu")
            model_name: Model identifier (e.g., "whisper-base", "large-v3-turbo")
            models_dir: Optional models directory path
            **kwargs: Additional arguments passed to model manager

        Returns:
            WhisperModel instance (PyTorch or OpenVINO implementation)

        Raises:
            RuntimeError: If no suitable backend is available (should never happen - CPU fallback)

        Example:
            ```python
            # Auto-detect
            model = ModelFactory.create()

            # Specific device
            model = ModelFactory.create(device="cuda", model_name="whisper-base")

            # With configuration
            model = ModelFactory.create(
                device="auto",
                model_name="large-v3-turbo",
                models_dir=".models/pytorch",
                auto_warmup=True
            )
            ```
        """
        # Normalize device string
        device = device.lower()

        # Auto-detect best device
        if device == "auto":
            device = ModelFactory._detect_best_device()
            logger.info(f"[FACTORY] Auto-detected device: {device}")

        # NPU requested - check if OpenVINO available
        if device == "npu":
            if not OPENVINO_AVAILABLE:
                logger.warning(
                    "[FACTORY] NPU requested but OpenVINO not installed. "
                    "Install with: pip install openvino openvino-genai"
                )
                logger.warning("[FACTORY] Falling back to GPU/CPU")
                device = ModelFactory._detect_best_device()
            else:
                # Verify NPU is actually available
                if not ModelFactory._verify_npu_available():
                    logger.warning("[FACTORY] NPU requested but not detected in system")
                    logger.warning("[FACTORY] Falling back to GPU/CPU")
                    device = ModelFactory._detect_best_device()
                else:
                    logger.info(f"[FACTORY] Creating OpenVINO model for NPU: {model_name}")
                    return OpenVINOModelManager(
                        models_dir=models_dir or ".models/openvino",
                        default_model=model_name,
                        device="npu",
                        **kwargs,
                    )

        # PyTorch for GPU/MPS/CPU
        # Normalize "gpu" to "cuda" for PyTorch
        if device == "gpu":
            device = "cuda"

        logger.info(f"[FACTORY] Creating PyTorch model for {device.upper()}: {model_name}")

        # Determine models directory for PyTorch
        if models_dir is None:
            models_dir = ".models/pytorch"

        return PyTorchModelManager(
            models_dir=models_dir,
            warmup_file=kwargs.pop("warmup_file", None),
            auto_warmup=kwargs.pop("auto_warmup", False),
            static_prompt=kwargs.pop("static_prompt", None),
            init_prompt=kwargs.pop("init_prompt", None),
            max_context_tokens=kwargs.pop("max_context_tokens", 223),
            **kwargs,
        )

    @staticmethod
    def _detect_best_device() -> str:
        """
        Detect best available device with fallback chain.

        Priority:
        1. CUDA GPU (PyTorch) - Most reliable, widely available
        2. Apple MPS (PyTorch) - Mac M1/M2/M3
        3. Intel NPU (OpenVINO) - If available and installed
        4. CPU (PyTorch) - Universal fallback

        Returns:
            Device string: "cuda", "mps", "npu", or "cpu"

        Note:
            This prioritizes GPU/MPS over NPU because:
            - PyTorch GPU support is more mature and reliable
            - OpenVINO may not be installed
            - NPU is still emerging technology
        """
        # Check CUDA GPU (NVIDIA)
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"[FACTORY] ✓ CUDA GPU detected: {gpu_name}")
                return "cuda"
            except Exception as e:
                logger.debug(f"[FACTORY] CUDA check failed: {e}")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Additional check: verify MPS is built
                if torch.backends.mps.is_built():
                    logger.info("[FACTORY] ✓ Apple MPS (Metal Performance Shaders) detected")
                    return "mps"
            except Exception as e:
                logger.debug(f"[FACTORY] MPS check failed: {e}")

        # Check NPU (Intel Neural Processing Unit) via OpenVINO
        if OPENVINO_AVAILABLE and ModelFactory._verify_npu_available():
            logger.info("[FACTORY] ✓ Intel NPU detected via OpenVINO")
            return "npu"

        # Fallback to CPU (always available)
        logger.info("[FACTORY] ⚠ Using CPU (no accelerator detected)")
        return "cpu"

    @staticmethod
    def _verify_npu_available() -> bool:
        """
        Verify that Intel NPU is actually available via OpenVINO.

        Returns:
            True if NPU is detected and accessible, False otherwise
        """
        if not OPENVINO_AVAILABLE:
            return False

        try:
            core = ov.Core()
            available_devices = core.available_devices

            # Check for NPU in available devices
            npu_devices = [d for d in available_devices if d.startswith("NPU")]

            if npu_devices:
                logger.debug(f"[FACTORY] NPU devices found: {npu_devices}")
                return True
            else:
                logger.debug(f"[FACTORY] Available devices: {available_devices} (no NPU)")
                return False

        except Exception as e:
            logger.debug(f"[FACTORY] NPU verification failed: {e}")
            return False

    @staticmethod
    def get_available_devices() -> dict:
        """
        Get information about all available devices.

        Returns:
            Dictionary with device availability information

        Example:
            ```python
            devices = ModelFactory.get_available_devices()
            # {
            #     "cuda": {"available": True, "name": "NVIDIA RTX 3090"},
            #     "mps": {"available": False},
            #     "npu": {"available": True, "devices": ["NPU.0"]},
            #     "cpu": {"available": True}
            # }
            ```
        """
        info = {}

        # CUDA GPU
        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        # Apple MPS
        info["mps"] = {
            "available": (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
        }

        # Intel NPU
        if OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                npu_devices = [d for d in core.available_devices if d.startswith("NPU")]
                info["npu"] = {
                    "available": len(npu_devices) > 0,
                    "devices": npu_devices,
                    "openvino_version": ov.__version__ if hasattr(ov, "__version__") else "unknown",
                }
            except Exception as e:
                info["npu"] = {"available": False, "error": str(e)}
        else:
            info["npu"] = {"available": False, "reason": "OpenVINO not installed"}

        # CPU (always available)
        info["cpu"] = {"available": True}

        return info


# Convenience function
def create_model(device: str = "auto", **kwargs) -> WhisperModel:
    """
    Convenience function for creating a model.

    Equivalent to ModelFactory.create()

    Args:
        device: Device preference
        **kwargs: Additional arguments

    Returns:
        WhisperModel instance
    """
    return ModelFactory.create(device=device, **kwargs)


__all__ = [
    "OPENVINO_AVAILABLE",
    "ModelFactory",
    "create_model",
]
