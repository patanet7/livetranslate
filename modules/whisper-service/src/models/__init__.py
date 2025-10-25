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
from .model_factory import ModelFactory, create_model, OPENVINO_AVAILABLE

# Phase 1: Concrete Implementations
from .pytorch_manager import PyTorchModelManager, ModelManager as PyTorchModelManagerAlias
from .openvino_manager import OpenVINOModelManager, ModelManager as OpenVINOModelManagerAlias

__all__ = [
    # Phase 2: Protocol and Factory (Recommended)
    'WhisperModel',
    'WhisperModelExtended',
    'ModelFactory',
    'create_model',
    'OPENVINO_AVAILABLE',

    # Phase 1: Concrete Implementations
    'PyTorchModelManager',
    'OpenVINOModelManager',
]
