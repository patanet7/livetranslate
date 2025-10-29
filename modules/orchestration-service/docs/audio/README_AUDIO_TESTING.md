# Audio Pipeline Testing Guide

## Overview

The `test_audio_pipeline.py` script allows you to test the complete audio processing pipeline with configurable stages. It processes audio files stage-by-stage and saves intermediate outputs for inspection.

## Directory Structure

```
orchestration-service/
├── test_audio_pipeline.py    # Main testing script
├── test_config.json           # Example configuration
├── input/                     # Place audio files here
│   └── your_audio.wav
└── output/                    # Test results saved here
    └── run_20251020_143022/   # Timestamped test runs
        ├── 00_original.wav
        ├── 01_vad.wav
        ├── 02_voice_filter.wav
        ├── 03_noise_reduction.wav
        ├── ...
        ├── 99_final_output.wav
        ├── test_info.json
        └── processing_results.json
```

## Quick Start

### 1. Add Test Audio

Place an audio file in the `input/` directory:

```bash
cp ~/Downloads/test_meeting.wav ./input/
```

Supported formats: WAV, MP3, OGG, FLAC, WebM, M4A

### 2. Run Basic Test

Use default configuration:

```bash
python test_audio_pipeline.py
```

### 3. Use Custom Configuration

```bash
python test_audio_pipeline.py --config test_config.json
```

### 4. Test Specific File

```bash
python test_audio_pipeline.py --input ./input/my_audio.wav
```

## Configuration

### Using Presets

List available presets:

```bash
python test_audio_pipeline.py --list-presets
```

Create config using preset:

```json
{
  "preset_name": "meeting_optimized"
}
```

### Custom Configuration

Create a JSON config file with custom stages:

```json
{
  "enabled_stages": [
    "vad",
    "voice_filter",
    "noise_reduction",
    "voice_enhancement",
    "equalizer",
    "lufs_normalization",
    "agc",
    "compression",
    "limiter"
  ],
  "sample_rate": 16000,
  "quality": "high",
  "vad": {
    "threshold": 0.5,
    "min_speech_duration": 0.25
  },
  "voice_filter": {
    "low_cutoff": 85,
    "high_cutoff": 8000
  },
  "noise_reduction": {
    "strength": 0.7
  },
  "voice_enhancement": {
    "gain_db": 3.0
  }
}
```

### Available Stages

1. **vad** - Voice Activity Detection
2. **voice_filter** - Voice frequency filtering (85-8000Hz)
3. **noise_reduction** - Spectral noise reduction
4. **spectral_denoising** - Advanced spectral denoising
5. **conventional_denoising** - Traditional denoising
6. **voice_enhancement** - Voice clarity enhancement
7. **equalizer** - Frequency equalization
8. **lufs_normalization** - Loudness normalization (EBU R128)
9. **agc** - Automatic Gain Control
10. **compression** - Dynamic range compression
11. **limiter** - Peak limiting

## Output Files

Each test run creates a timestamped directory in `output/`:

### Audio Files

- `00_original.wav` - Original input audio
- `01_<stage>.wav` - Output after stage 1
- `02_<stage>.wav` - Output after stage 2
- ...
- `99_final_output.wav` - Final processed audio

### Metadata Files

**test_info.json**
```json
{
  "input_file": "./input/test.wav",
  "sample_rate": 16000,
  "duration_seconds": 30.5,
  "num_samples": 488000,
  "config": {...},
  "timestamp": "20251020_143022"
}
```

**processing_results.json**
```json
{
  "stage_results": [
    {
      "stage_name": "vad",
      "stage_index": 1,
      "metrics": {
        "input_rms": 0.1234,
        "output_rms": 0.1456,
        "gain_change_db": 0.54,
        "peak_change_db": 0.32
      }
    }
  ],
  "final_metrics": {
    "gain_change_db": 2.34,
    "peak_change_db": 1.23
  },
  "success": true
}
```

## Common Use Cases

### Test Different Noise Levels

Create configs for different noise reduction strengths:

```bash
# Light noise reduction
cat > config_light.json << EOF
{
  "enabled_stages": ["vad", "noise_reduction"],
  "noise_reduction": {"strength": 0.3}
}
EOF

# Heavy noise reduction
cat > config_heavy.json << EOF
{
  "enabled_stages": ["vad", "noise_reduction"],
  "noise_reduction": {"strength": 0.9}
}
EOF

# Test both
python test_audio_pipeline.py --config config_light.json
python test_audio_pipeline.py --config config_heavy.json
```

### Compare Processing Presets

```bash
# Test each preset
for preset in meeting_optimized voice_optimized minimal_processing; do
  echo "{\"preset_name\": \"$preset\"}" > config_$preset.json
  python test_audio_pipeline.py --config config_$preset.json
done
```

### Debug Specific Stage

Test a single stage in isolation:

```bash
cat > config_debug.json << EOF
{
  "enabled_stages": ["voice_filter"],
  "voice_filter": {
    "low_cutoff": 85,
    "high_cutoff": 8000,
    "filter_order": 5
  }
}
EOF

python test_audio_pipeline.py --config config_debug.json
```

### Batch Testing

Test multiple files:

```bash
for file in ./input/*.wav; do
  echo "Testing: $file"
  python test_audio_pipeline.py --input "$file"
done
```

## Analyzing Results

### Listen to Outputs

Compare the audio quality at each stage:

```bash
cd output/run_20251020_143022

# Original
ffplay 00_original.wav

# After noise reduction
ffplay 03_noise_reduction.wav

# Final output
ffplay 99_final_output.wav
```

### Visualize Metrics

Use the JSON results to create graphs:

```python
import json
import matplotlib.pyplot as plt

with open('output/run_20251020_143022/processing_results.json') as f:
    results = json.load(f)

stages = [s['stage_name'] for s in results['stage_results']]
gains = [s['metrics']['gain_change_db'] for s in results['stage_results']]

plt.bar(stages, gains)
plt.xlabel('Processing Stage')
plt.ylabel('Gain Change (dB)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gain_analysis.png')
```

### Compare Waveforms

```python
import soundfile as sf
import matplotlib.pyplot as plt

original, sr = sf.read('output/run_20251020_143022/00_original.wav')
final, sr = sf.read('output/run_20251020_143022/99_final_output.wav')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

ax1.plot(original)
ax1.set_title('Original Audio')
ax1.set_ylabel('Amplitude')

ax2.plot(final)
ax2.set_title('Processed Audio')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Sample')

plt.tight_layout()
plt.savefig('waveform_comparison.png')
```

## Troubleshooting

### No Input Files

```
ERROR - No audio files found in ./input
```

**Solution**: Add audio files to the `input/` directory

### FFmpeg Not Found

```
ERROR - ffmpeg conversion failed
```

**Solution**: Install ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Stage Not Found

```
WARNING - Stage xyz not found, skipping
```

**Solution**: Check available stages in config and ensure they match the processor implementation

### Out of Memory

For very large files, the script may run out of memory.

**Solution**: Process shorter audio clips or increase available RAM

## Advanced Usage

### Custom Output Directory

```bash
python test_audio_pipeline.py \
  --input ./input/test.wav \
  --output-dir ./my_results
```

### Integration with CI/CD

```bash
#!/bin/bash
# test_audio_quality.sh

# Run test
python test_audio_pipeline.py --config test_config.json

# Check if successful
if [ $? -eq 0 ]; then
  echo "✓ Audio processing test passed"
  exit 0
else
  echo "✗ Audio processing test failed"
  exit 1
fi
```

### Automated Regression Testing

```python
#!/usr/bin/env python3
# regression_test.py

import json
import subprocess
from pathlib import Path

def run_test(input_file, config_file):
    """Run test and return metrics"""
    result = subprocess.run(
        ['python', 'test_audio_pipeline.py',
         '--input', input_file,
         '--config', config_file],
        capture_output=True
    )

    # Find latest output directory
    output_dirs = sorted(Path('output').glob('run_*'))
    latest = output_dirs[-1] if output_dirs else None

    if latest:
        with open(latest / 'processing_results.json') as f:
            return json.load(f)

    return None

# Run tests
results = run_test('./input/test.wav', 'test_config.json')

# Validate metrics
assert results['success'] == True
assert results['final_metrics']['gain_change_db'] < 6.0  # Not too loud
assert results['final_metrics']['gain_change_db'] > -3.0  # Not too quiet

print("✓ All regression tests passed")
```

## Tips

1. **Start Simple**: Test with a single stage first, then gradually add more
2. **Compare Outputs**: Always listen to intermediate outputs to understand what each stage does
3. **Use Metrics**: The JSON output provides objective measurements to complement listening tests
4. **Test Edge Cases**: Try very quiet audio, very loud audio, noisy audio, etc.
5. **Document Findings**: Keep notes about which configurations work best for your use case

## Examples

See the `examples/` directory for:
- Sample audio files
- Common configurations
- Analysis scripts
- Best practices

## Getting Help

If you encounter issues:

1. Check the logs in the console output
2. Examine the `processing_results.json` for error details
3. Test with a simpler configuration
4. Verify your audio file is valid
5. Check the CLAUDE.md files in the service directories

## Related Documentation

- [Audio Processing Pipeline](./src/audio/README.md)
- [Configuration Guide](./src/audio/config.py)
- [Orchestration Service CLAUDE.md](./CLAUDE.md)
