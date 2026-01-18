# Debugging Guide

This guide covers common debugging scenarios and techniques for LiveTranslate.

## Table of Contents

- [Service Debugging](#service-debugging)
- [Audio Pipeline Issues](#audio-pipeline-issues)
- [WebSocket Issues](#websocket-issues)
- [Performance Profiling](#performance-profiling)
- [Common Errors](#common-errors)

## Service Debugging

### Using VSCode Debugger

Launch configurations are provided in `.vscode/launch.json`:

1. **Debug Orchestration Service**: Start orchestration service with debugger
2. **Debug Whisper Service**: Start Whisper with breakpoints
3. **Debug Translation Service**: Debug translation backend
4. **Debug Current File**: Debug any Python file

### Command-Line Debugging

```bash
# Run with verbose logging
cd modules/orchestration-service
LOG_LEVEL=DEBUG pdm run python src/main_fastapi.py

# Profile with py-spy
pip install py-spy
py-spy top --pid <PID>
```

## Audio Pipeline Issues

### 422 Validation Errors

If you see 422 errors on `/api/audio/upload`:

1. Check request format matches expected schema
2. Verify audio file is valid (WAV, 16kHz recommended)
3. Check `model` parameter uses correct naming ("whisper-base")

### Audio Not Transcribing

1. Check Whisper service is running: `curl http://localhost:5001/health`
2. Verify audio format: 16kHz, mono, WAV
3. Check device availability (NPU/GPU/CPU fallback)

```bash
# Test audio upload manually
curl -X POST http://localhost:3000/api/audio/upload \
  -F "file=@test.wav" \
  -F "model=whisper-base"
```

## WebSocket Issues

### Connection Dropping

1. Check heartbeat interval (should be < 30s)
2. Verify proxy timeout settings
3. Check for memory pressure on services

### Messages Not Received

1. Verify subscription to correct channels
2. Check message routing configuration
3. Enable WebSocket debug logging

```python
# Enable WebSocket debug logging
import logging
logging.getLogger("websockets").setLevel(logging.DEBUG)
```

## Performance Profiling

### Python Profiling

```bash
# Profile with cProfile
python -m cProfile -o output.prof src/main_fastapi.py

# Visualize with snakeviz
pip install snakeviz
snakeviz output.prof
```

### Memory Profiling

```bash
# Profile memory usage
pip install memory_profiler
python -m memory_profiler src/main_fastapi.py
```

## Common Errors

### "Model not found" Error

**Cause**: Model name mismatch between frontend and backend.

**Solution**: Use standardized model names:
- Correct: `whisper-base`
- Incorrect: `base`
- Incorrect: `whisper_base`

### "NPU device not available"

**Cause**: Intel NPU not detected or drivers not installed.

**Solution**:
1. Verify Intel NPU is available: `lspci | grep -i neural`
2. Install OpenVINO runtime
3. Service will fallback to GPU/CPU automatically

### "CUDA out of memory"

**Cause**: GPU memory exhausted during translation.

**Solution**:
1. Reduce batch size in translation service
2. Use smaller model
3. Enable memory cleanup between batches

## Logging

### Log Levels

Set via `LOG_LEVEL` environment variable:
- `DEBUG`: Verbose debugging information
- `INFO`: Normal operation logs
- `WARNING`: Warning messages
- `ERROR`: Error messages only

### Log Locations

- Orchestration: `logs/orchestration.log`
- Whisper: `logs/whisper.log`
- Translation: `logs/translation.log`

### Structured Logging

All services use `structlog` for structured logging:

```python
import structlog
logger = structlog.get_logger()

logger.info("processing_audio",
    session_id=session_id,
    duration_ms=duration,
    model=model_name)
```
