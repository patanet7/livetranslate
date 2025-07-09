# Shared Module

Common utilities, types, and configurations shared across all LiveTranslate modules.

## Overview

The shared module provides:
- Common data types and interfaces
- Utility functions for all modules
- Local inference clients (vLLM, Ollama)
- Configuration management
- Logging utilities
- Error handling patterns

## Local Inference Support

### vLLM Integration
- High-performance inference server for local LLM models
- GPU acceleration support
- Batch processing capabilities
- OpenAI-compatible API

### Ollama Integration
- Easy-to-use local model hosting
- Multiple model format support (GGUF, GGML)
- CPU and GPU inference
- Model management and switching

## Structure

```
shared/
├── src/
│   ├── types/              # Common TypeScript/Python types
│   ├── utils/              # Utility functions
│   ├── inference/          # Local inference clients
│   │   ├── vllm_client.py  # vLLM API client
│   │   ├── ollama_client.py # Ollama API client
│   │   └── base_client.py  # Base inference interface
│   ├── config/             # Configuration management
│   ├── logging/            # Logging utilities
│   └── errors/             # Custom exception classes
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── package.json           # Node.js dependencies
└── README.md              # This file
```

## Installation

```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies (if using TypeScript)
npm install
```

## Usage

### Local Inference Client

```python
from shared.inference import get_inference_client

# Auto-detect available inference backend
client = get_inference_client()

# Explicit vLLM client
vllm_client = get_inference_client('vllm')

# Explicit Ollama client
ollama_client = get_inference_client('ollama')

# Generate text
response = client.generate(
    prompt="Translate to Spanish: Hello world",
    model="llama3.1:8b",
    max_tokens=100
)
```

### Configuration Management

```python
from shared.config import Config

config = Config.load()
print(config.inference.backend)  # 'vllm' or 'ollama'
print(config.inference.model)    # Default model name
```

### Logging

```python
from shared.logging import get_logger

logger = get_logger(__name__)
logger.info("Module started")
``` 