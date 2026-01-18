# Triton Translation Service Integration

This integration adds NVIDIA Triton Inference Server support to the LiveTranslate translation service, providing a production-ready inference backend with the vLLM engine.

## Quick Start

### 1. Deploy with Docker Compose

```bash
cd modules/translation-service
docker-compose -f docker-compose-triton.yml up --build -d
```

### 2. Test the Service

```bash
# Simple test
python test_triton_simple.py

# Manual API test
curl -X POST http://localhost:5003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "Spanish"}'
```

## Architecture

- **Triton Server** (Port 8000): NVIDIA Triton with vLLM backend
- **Translation API** (Port 5003): REST API for translation requests
- **Metrics** (Port 8002): Prometheus metrics endpoint

## Configuration

Key environment variables in `docker-compose-triton.yml`:

```yaml
environment:
  - MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  - TENSOR_PARALLEL_SIZE=1
  - MAX_MODEL_LEN=4096
  - GPU_MEMORY_UTILIZATION=0.9
  - INFERENCE_BACKEND=triton
```

## Files Added

- `Dockerfile.triton` - Triton-based container
- `docker-compose-triton.yml` - Deployment configuration
- `requirements-triton.txt` - Triton-specific dependencies
- `triton-config/` - Triton model configuration
- `scripts/start-triton-translation.sh` - Startup script
- `test_triton_simple.py` - Basic test suite

## Benefits

1. **Production Ready**: Enterprise-grade inference server
2. **High Performance**: Optimized batching and GPU utilization
3. **Monitoring**: Built-in metrics and health checks
4. **Scalability**: Horizontal scaling support
5. **Compatibility**: Drop-in replacement for existing vLLM backend

## Troubleshooting

### Common Issues

1. **GPU Memory**: Reduce `GPU_MEMORY_UTILIZATION` if OOM errors occur
2. **Model Loading**: Check logs for model download/loading issues
3. **Port Conflicts**: Ensure ports 8000, 5003, 8002 are available

### Logs

```bash
# View service logs
docker-compose -f docker-compose-triton.yml logs -f triton-translation

# Check Triton health
curl http://localhost:8000/v2/health

# Check translation service health
curl http://localhost:5003/api/health
```

This integration provides a robust, scalable foundation for production translation workloads while maintaining full API compatibility with the existing translation service.
