# Hardware Optimization Strategy

## NPU (Intel) - Whisper Service

**Primary Hardware**: Intel Core Ultra (Meteor Lake) with NPU
**Framework**: OpenVINO
**Models**: Whisper (base, small, medium) quantized to INT8
**Performance**: 10x speedup vs CPU

## GPU (NVIDIA) - Translation Service

**Primary Hardware**: NVIDIA RTX 3060+ or cloud GPU
**Framework**: PyTorch + Transformers, vLLM
**Models**: Llama 3.1-8B, NLLB-200
**Performance**: 5-10x speedup vs CPU

## CPU Fallback

All services gracefully fallback to CPU when NPU/GPU unavailable.

## Auto-Detection

Services automatically detect and use best available hardware:
```
NPU → GPU → CPU (Whisper)
GPU → CPU (Translation)
```

See [Container Overview](./README.md) for service details.
