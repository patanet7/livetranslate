#!/usr/bin/env python3
"""
Base Model Protocol for Whisper Service

Defines the common interface that all Whisper model implementations must follow.
This allows for interchangeable backends (PyTorch, OpenVINO, etc.) while maintaining
a consistent API.

Phase 2 Refactoring: Created 2025-10-25
"""

from typing import Protocol, List, Dict, Any, Optional
import numpy as np


class WhisperModel(Protocol):
    """
    Protocol defining the common interface for all Whisper model implementations.

    This Protocol allows different backends (PyTorch, OpenVINO, ONNX, etc.) to be
    used interchangeably while maintaining a consistent API.

    Implementations:
    - PyTorchModelManager: GPU/MPS/CPU backend using openai-whisper
    - OpenVINOModelManager: NPU/GPU/CPU backend using OpenVINO

    Example:
        ```python
        # Both implementations satisfy this Protocol
        pytorch_model: WhisperModel = PyTorchModelManager()
        openvino_model: WhisperModel = OpenVINOModelManager()

        # Can be used interchangeably
        def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
            model.load_model("whisper-base")
            return model.safe_inference("whisper-base", audio)
        ```
    """

    # Core attributes that all implementations must have
    device: str
    models_dir: str

    def load_model(self, model_name: str) -> Any:
        """
        Load a Whisper model by name.

        Args:
            model_name: Model identifier (e.g., "whisper-base", "large-v3-turbo")

        Returns:
            Loaded model object (implementation-specific)

        Raises:
            Exception: If model loading fails

        Example:
            ```python
            model_manager.load_model("whisper-base")
            ```
        """
        ...

    def safe_inference(
        self,
        model_name: str,
        audio_data: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Perform thread-safe inference on audio data.

        Args:
            model_name: Model to use for inference
            audio_data: Audio array (16kHz, mono, float32)
            **kwargs: Additional inference parameters (language, temperature, etc.)

        Returns:
            Transcription result (format varies by implementation)

        Raises:
            Exception: If inference fails

        Example:
            ```python
            audio = np.random.randn(16000).astype(np.float32)
            result = model_manager.safe_inference(
                "whisper-base",
                audio,
                language="en",
                temperature=0.0
            )
            ```
        """
        ...

    def list_models(self) -> List[str]:
        """
        List available Whisper models.

        Returns:
            List of model identifiers

        Example:
            ```python
            models = model_manager.list_models()
            # ["tiny", "base", "small", "medium", "large", "large-v3-turbo"]
            ```
        """
        ...

    def clear_cache(self) -> None:
        """
        Clear cached models to free memory.

        This should:
        - Unload all loaded models
        - Clear any GPU/NPU memory
        - Force garbage collection

        Example:
            ```python
            model_manager.clear_cache()
            ```
        """
        ...


class WhisperModelExtended(WhisperModel, Protocol):
    """
    Extended Protocol with optional advanced features.

    Not all implementations need to support these features, but they're
    commonly useful for production deployments.
    """

    def _detect_best_device(self) -> str:
        """
        Detect the best available device for inference.

        Priority varies by implementation:
        - PyTorch: GPU/MPS → CPU
        - OpenVINO: NPU → GPU → CPU

        Returns:
            Device string (e.g., "cuda", "mps", "npu", "cpu")
        """
        ...

    def _preload_default_model(self) -> None:
        """
        Preload the default model to reduce first-request latency.

        This is called during initialization if auto_warmup is enabled.
        """
        ...


# Type aliases for convenience
ModelType = WhisperModel
ExtendedModelType = WhisperModelExtended


__all__ = [
    'WhisperModel',
    'WhisperModelExtended',
    'ModelType',
    'ExtendedModelType',
]
