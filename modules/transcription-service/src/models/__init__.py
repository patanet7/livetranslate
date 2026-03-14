"""
Model Management Package for Whisper Service

This package provides model managers for different backends with a unified interface.

## Phase 2 Refactoring (2025-10-25)
- WhisperModel Protocol: Common interface for all implementations
- ModelFactory: Smart device selection with automatic fallback
- PyTorchModelManager: GPU/MPS/CPU backend (always available)
- OpenVINOModelManager: NPU/GPU/CPU backend (optional, may not be installed)

## Usage

### Recommended: Use Factory Pattern
```python
from models import ModelFactory

# Auto-detect best device
model = ModelFactory.create()

# Force specific device
model = ModelFactory.create(device="cuda")
model = ModelFactory.create(device="npu")  # Falls back if unavailable
```

### Direct Instantiation (Advanced)
```python
from models import PyTorchModelManager, OpenVINOModelManager

pytorch_model = PyTorchModelManager(models_dir=".models/pytorch")
openvino_model = OpenVINOModelManager(models_dir=".models/openvino")
```

## Device Priority
GPU/MPS → NPU → CPU
"""

# Phase 2: Base Protocol and Factory
from .base_model import WhisperModel, WhisperModelExtended
from .model_factory import OPENVINO_AVAILABLE, ModelFactory, create_model
from .openvino_manager import ModelManager as OpenVINOModelManagerAlias, OpenVINOModelManager

# Phase 1: Concrete Implementations
from .pytorch_manager import ModelManager as PyTorchModelManagerAlias, PyTorchModelManager

__all__ = [
    "OPENVINO_AVAILABLE",
    "ModelFactory",
    "OpenVINOModelManager",
    # Phase 1: Concrete Implementations
    "PyTorchModelManager",
    # Phase 2: Protocol and Factory (Recommended)
    "WhisperModel",
    "WhisperModelExtended",
    "create_model",
]
