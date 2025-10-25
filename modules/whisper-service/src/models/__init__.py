"""
Model Management Package for Whisper Service

This package provides model managers for different backends:
- PyTorchModelManager: PyTorch implementation (GPU/MPS/CPU)
- OpenVINOModelManager: OpenVINO implementation (NPU/GPU/CPU)

Extracted from monolithic whisper_service.py as part of Phase 1 refactoring.
"""

from .pytorch_manager import PyTorchModelManager, ModelManager as PyTorchModelManagerAlias
from .openvino_manager import OpenVINOModelManager, ModelManager as OpenVINOModelManagerAlias

__all__ = [
    'PyTorchModelManager',
    'OpenVINOModelManager',
]
