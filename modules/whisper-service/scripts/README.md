# Development Scripts

Developer tools for whisper-service configuration, validation, and performance testing.

## Scripts

### `validate_config.py`

Validates configuration before starting services. Catches configuration errors early.

**Usage:**
```bash
# Validate current configuration
python scripts/validate_config.py

# Validate with custom .env file
python scripts/validate_config.py --env-file .env.production

# Validate with specific model path
python scripts/validate_config.py --model-path /path/to/model.pt
```

**What it checks:**
- ✅ VAD configuration (threshold, sampling rate, parameters)
- ✅ LID configuration (margins, dwell times, smoothing)
- ✅ Whisper configuration (model path, decoder, languages)
- ✅ Session configuration (log levels, performance settings)
- ✅ Environment variable values

**Example output:**
```
=== Validating VAD Configuration ===
✅ VAD Config Valid
   - Threshold: 0.5
   - Sampling Rate: 16000 Hz
   - Min Silence: 500 ms

=== Validating LID Configuration ===
✅ LID Config Valid
   - LID Hop: 100 ms
   - Confidence Margin: 0.2
   - Min Dwell: 250.0 ms (6 frames)

✅ ALL CONFIGURATIONS VALID
```

### `benchmark.py`

Measures performance characteristics of key components.

**Usage:**
```bash
# Benchmark all components
python scripts/benchmark.py --component all

# Benchmark specific component
python scripts/benchmark.py --component vad
python scripts/benchmark.py --component audio
python scripts/benchmark.py --component buffer

# Benchmark with real audio file
python scripts/benchmark.py --audio test_audio.wav

# Custom number of iterations
python scripts/benchmark.py --component vad --iterations 1000
```

**What it measures:**
- ⚡ VAD detection latency (avg, min, max, p95)
- ⚡ Audio processing operations (RMS, normalization, etc.)
- ⚡ Buffer operations (append, clear)
- ⚡ Real-time factor (for audio files)

**Example output:**
```
=== Benchmarking VAD (iterations=100) ===
✅ VAD Performance:
   - Average: 15.2ms
   - Min: 12.1ms
   - Max: 22.3ms
   - P95: 19.8ms
   - Chunk size: 8000 samples (500.0ms)
   ✅ Performance target met (< 50ms)

=== Benchmarking Audio Processing ===
   RMS Calculation: 0.823ms
   Max Amplitude: 0.412ms
   Normalization: 1.234ms
   Concatenation: 0.567ms
```

## Quick Reference

### Before Starting Service

```bash
# 1. Validate configuration
python scripts/validate_config.py

# 2. Check performance (optional)
python scripts/benchmark.py --component all
```

### After Configuration Changes

```bash
# Validate changes
python scripts/validate_config.py

# Compare performance
python scripts/benchmark.py --component vad --iterations 100
```

### Investigating Performance Issues

```bash
# Benchmark with real audio
python scripts/benchmark.py --audio problematic_audio.wav

# More iterations for stable results
python scripts/benchmark.py --component all --iterations 1000
```

## Adding New Scripts

When adding new development scripts:

1. **Use consistent structure:**
   ```python
   #!/usr/bin/env python3
   """
   Script description
   """
   import sys
   from pathlib import Path

   # Add src to path
   sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

   # Your imports
   from service_config import SessionConfig

   def main():
       # Your code
       pass

   if __name__ == '__main__':
       main()
   ```

2. **Add argparse for CLI options**
3. **Provide clear output with ✅/❌ indicators**
4. **Exit with proper status codes (0=success, 1=failure)**
5. **Document in this README**

## Dependencies

Scripts require the same dependencies as the main service:
```bash
pip install -r requirements.txt
```

## Integration with CI/CD

### Pre-commit Hook

Configuration validation runs automatically:
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Validation runs on git commit
git commit -m "Update config"
# → Validates configuration automatically
```

### CI Pipeline

```yaml
# Example GitHub Actions
- name: Validate Configuration
  run: python scripts/validate_config.py --env-file .env.ci

- name: Run Benchmarks
  run: python scripts/benchmark.py --component all --iterations 50
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Ensure you're running from whisper-service directory
cd modules/whisper-service
python scripts/validate_config.py
```

### Configuration Not Found

Scripts use environment variables. If validation fails:
```bash
# Check current environment
python scripts/validate_config.py

# Use specific .env file
python scripts/validate_config.py --env-file .env.local
```

### Performance Variance

Benchmarks can vary based on system load:
```bash
# Run more iterations for stable results
python scripts/benchmark.py --iterations 500

# Close other applications for accurate measurement
# Run multiple times and compare
```

## See Also

- **Development Guide**: `../DEVELOPMENT.md`
- **Configuration Reference**: `../.env.example`
- **Migration Guide**: `../MIGRATION_GUIDE.md`
