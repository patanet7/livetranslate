# Audio Processing Configuration Examples

This directory contains ready-to-use configuration examples for common audio processing scenarios.

## Quick Start

```bash
# Test with a configuration
python test_audio_pipeline.py --config config_examples/meeting.json

# Test with your own audio file
python test_audio_pipeline.py --config config_examples/podcast.json --input ./input/my_voice.wav
```

## Available Configurations

### 1. minimal.json
**Use case**: Low latency, basic cleanup
**Processing**: VAD + Limiter only
**Latency**: < 10ms
**Best for**: Real-time streaming where latency is critical

```bash
python test_audio_pipeline.py --config config_examples/minimal.json
```

### 2. meeting.json
**Use case**: Online meetings, video conferencing
**Processing**: VAD, Voice Filter, Noise Reduction, Voice Enhancement, AGC, Compression, Limiter
**Latency**: 20-30ms
**Best for**: Zoom/Teams meetings, multiple speakers, moderate background noise

```bash
python test_audio_pipeline.py --config config_examples/meeting.json
```

### 3. noisy.json
**Use case**: Very noisy environments
**Processing**: Full pipeline with aggressive noise reduction
**Latency**: 50-80ms
**Best for**: Coffee shops, airports, construction sites, street recordings

```bash
python test_audio_pipeline.py --config config_examples/noisy.json
```

### 4. podcast.json
**Use case**: Professional voice recording
**Processing**: Voice optimization + LUFS normalization + EQ
**Latency**: 30-50ms
**Best for**: Podcasts, voiceovers, interviews, audiobooks

```bash
python test_audio_pipeline.py --config config_examples/podcast.json
```

### 5. broadcast.json
**Use case**: Professional broadcasting
**Processing**: Broadcast-grade processing with EBU R128 normalization
**Latency**: 40-60ms
**Best for**: Radio, TV, professional streaming

```bash
python test_audio_pipeline.py --config config_examples/broadcast.json
```

### 6. debug_noise.json
**Use case**: Testing noise reduction in isolation
**Processing**: Noise reduction only
**Best for**: Debugging, comparing different noise reduction settings

```bash
python test_audio_pipeline.py --config config_examples/debug_noise.json
```

## Customizing Configurations

### Basic Customization

1. Copy an example config:
```bash
cp config_examples/meeting.json my_custom_config.json
```

2. Edit parameters:
```json
{
  "enabled_stages": ["vad", "noise_reduction", "limiter"],
  "noise_reduction": {
    "strength": 0.5  // Reduce strength
  }
}
```

3. Test:
```bash
python test_audio_pipeline.py --config my_custom_config.json
```

### Parameter Tuning Guide

See [AUDIO_PARAMETERS_REFERENCE.md](../AUDIO_PARAMETERS_REFERENCE.md) for complete parameter documentation.

**Common adjustments**:

- **More noise reduction**: Increase `noise_reduction.strength` (0.7 → 0.9)
- **Less processing**: Remove stages from `enabled_stages`
- **Louder output**: Increase `agc.target_level` (-18 → -12 dB)
- **Preserve dynamics**: Reduce `compression.ratio` (4.0 → 2.0)
- **Brighter sound**: Add high frequency boost in `equalizer`

## Comparison Testing

Test multiple configurations on the same audio:

```bash
#!/bin/bash
# Compare all configurations

INPUT="./input/test_audio.wav"

for config in config_examples/*.json; do
    echo "Testing: $config"
    python test_audio_pipeline.py --config "$config" --input "$INPUT"
done

# Results will be in output/run_*/
```

## Performance Characteristics

| Config | Latency | CPU | Quality | Best For |
|--------|---------|-----|---------|----------|
| minimal | < 10ms | Low | Good | Real-time streaming |
| meeting | 20-30ms | Medium | Excellent | Video calls |
| noisy | 50-80ms | High | Excellent | Noisy environments |
| podcast | 30-50ms | Medium | Excellent | Voice recording |
| broadcast | 40-60ms | High | Professional | Broadcasting |

## Tips

1. **Start with an example** - Don't write configs from scratch
2. **Test incrementally** - Enable one stage at a time
3. **Compare outputs** - A/B test different settings
4. **Monitor metrics** - Check the JSON output for gain changes
5. **Listen critically** - Your ears are the best judge

## Troubleshooting

### Audio sounds muffled
- Reduce `noise_reduction.strength`
- Disable `voice_filter` or increase `high_freq_rolloff`
- Check `equalizer` isn't cutting highs

### Too much noise remains
- Increase `noise_reduction.strength`
- Enable `spectral_denoising`
- Use `noisy.json` as starting point

### Audio too quiet
- Increase `agc.target_level`
- Add `compression.makeup_gain`
- Check `gain_out` values

### Audio too loud/clipping
- Reduce `agc.target_level`
- Lower `limiter.threshold`
- Reduce individual `gain_out` values

### Unnatural sound
- Reduce processing intensity
- Use `soft_knee` compression
- Enable `soft_clip` on limiter
- Reduce `noise_reduction.strength`

## Related Documentation

- [Audio Parameters Reference](../AUDIO_PARAMETERS_REFERENCE.md) - Complete parameter documentation
- [Audio Testing Guide](../README_AUDIO_TESTING.md) - Testing workflow and analysis
- [Main README](../README.md) - Service overview
