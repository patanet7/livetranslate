# Audio Processing Effects - Complete Reference

**Status**: ✅ Production Ready
**Last Updated**: 2025-10-20
**Version**: 2.0 (Enhanced Stages)

This document provides comprehensive documentation for all audio processing effects available in the LiveTranslate orchestration service.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Effect Descriptions](#effect-descriptions)
3. [Configuration Guide](#configuration-guide)
4. [Usage Examples](#usage-examples)
5. [Performance Characteristics](#performance-characteristics)
6. [Troubleshooting](#troubleshooting)

---

## Quick Reference

| Effect | Purpose | Key Parameters | Typical Use Case |
|--------|---------|----------------|------------------|
| **LUFS Normalization** | Loudness standardization | `target_lufs`, `mode` | Broadcast, streaming |
| **Compression** | Dynamic range control | `threshold`, `ratio`, `attack`, `release` | Voice leveling |
| **Limiter** | Peak limiting | `threshold`, `release_time` | Prevent clipping |
| **AGC** | Automatic gain control | `target_level`, `mode` | Meeting audio |
| **Equalizer** | Frequency shaping | `bands`, `gains` | Voice clarity |
| **Noise Reduction** | Background noise removal | `mode`, `strength` | Noisy environments |
| **Voice Enhancement** | Speech intelligibility | `enhancement_level` | Clear speech |
| **VAD** | Voice activity detection | `mode`, `sensitivity` | Silence suppression |

---

## Effect Descriptions

### 1. LUFS Normalization ✅ **ENHANCED**

**Library**: `pyloudnorm` (ITU-R BS.1770-4 compliant)
**Performance**: 4.04ms avg (63% faster than custom)
**Status**: Production-ready

#### Description
Normalizes audio to a target loudness using the ITU-R BS.1770-4 standard. This is the industry-standard method for measuring and controlling loudness across broadcast, streaming, and podcast applications.

#### Parameters

```python
class LUFSNormalizationConfig:
    enabled: bool = True
    mode: LUFSNormalizationMode = LUFSNormalizationMode.STREAMING
    target_lufs: float = -14.0        # Target integrated loudness (LUFS)
    true_peak_limiting: bool = True   # Enable true peak limiting
    max_true_peak: float = -1.0       # Maximum true peak level (dB)
    gain_in: float = 0.0              # Input gain (dB)
    gain_out: float = 0.0             # Output gain (dB)
```

#### Modes & Presets

| Mode | Target LUFS | Use Case |
|------|-------------|----------|
| `STREAMING` | -14 LUFS | Spotify, YouTube, Apple Music |
| `BROADCAST_TV` | -23 LUFS | Television broadcast (EBU R128) |
| `BROADCAST_RADIO` | -18 LUFS | Radio broadcast |
| `PODCAST` | -16 LUFS | Podcast platforms |
| `YOUTUBE` | -14 LUFS | YouTube content |
| `NETFLIX` | -27 LUFS | Netflix delivery spec |

#### Quality Metrics
- **Accuracy**: ±0.1 LUFS (vs ±0.5 LUFS custom)
- **ITU-R Compliance**: Full BS.1770-4 implementation
- **K-weighting**: Accurate filter implementation
- **Gating**: Spec-compliant algorithm

#### Example

```python
from audio.config import LUFSNormalizationConfig, LUFSNormalizationMode

config = LUFSNormalizationConfig(
    enabled=True,
    mode=LUFSNormalizationMode.STREAMING,  # -14 LUFS for streaming
    true_peak_limiting=True,
    max_true_peak=-1.0
)
```

---

### 2. Compression ✅ **ENHANCED**

**Library**: `pedalboard` (Spotify's audio library)
**Performance**: 0.80ms avg (99.4% faster than custom)
**Status**: Production-ready

#### Description
Dynamic range compression reduces the difference between loud and quiet parts of audio. Uses Spotify's professional-grade algorithm for transparent, musical compression.

#### Parameters

```python
class CompressionConfig:
    enabled: bool = True
    mode: CompressionMode = CompressionMode.SOFT_KNEE
    threshold: float = -20.0          # Compression threshold (dB)
    ratio: float = 3.0                # Compression ratio (e.g., 3:1)
    attack_time: float = 5.0          # Attack time (ms)
    release_time: float = 100.0       # Release time (ms)
    knee: float = 3.0                 # Knee width (dB, 0 = hard knee)
    makeup_gain: float = 0.0          # Makeup gain (dB)
    gain_in: float = 0.0              # Input gain (dB)
    gain_out: float = 0.0             # Output gain (dB)
```

#### Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `SOFT_KNEE` | Gentle, musical compression | Music, podcasts |
| `HARD_KNEE` | Aggressive, precise compression | Voice leveling |
| `VOICE_OPTIMIZED` | Optimized for speech | Meetings, calls |
| `ADAPTIVE` | Adapts to input dynamics | Variable content |

#### Typical Settings

**Voice Leveling**:
```python
CompressionConfig(
    threshold=-20.0,
    ratio=3.0,
    attack_time=5.0,
    release_time=100.0,
    mode=CompressionMode.VOICE_OPTIMIZED
)
```

**Podcast/Broadcasting**:
```python
CompressionConfig(
    threshold=-18.0,
    ratio=2.5,
    attack_time=10.0,
    release_time=150.0,
    makeup_gain=3.0
)
```

---

### 3. Limiter ✅ **ENHANCED**

**Library**: `pedalboard` (Spotify's audio library)
**Performance**: 1.41ms avg (98.8% faster than custom)
**Status**: Production-ready

#### Description
Brick-wall peak limiting prevents audio from exceeding a specified threshold. Essential for preventing clipping and distortion. Uses true peak detection to prevent inter-sample peaks.

#### Parameters

```python
class LimiterConfig:
    enabled: bool = True
    threshold: float = -1.0           # Limiting threshold (dB)
    release_time: float = 50.0        # Release time (ms)
    soft_clip: bool = True            # Enable soft clipping
    gain_in: float = 0.0              # Input gain (dB)
    gain_out: float = 0.0             # Output gain (dB)
```

#### Features
- **True Peak Detection**: Prevents inter-sample peaks
- **Zero Overshoot**: Guaranteed no peaks above threshold
- **Transparent**: Minimal distortion with soft clipping
- **Fast Release**: Prevents pumping artifacts

#### Example

```python
LimiterConfig(
    threshold=-1.0,      # Limit to -1.0 dB
    release_time=50.0,   # Fast release
    soft_clip=True       # Enable soft clipping for transparency
)
```

---

### 4. AGC (Automatic Gain Control)

**Implementation**: Custom
**Performance**: ~12ms avg
**Status**: Production-ready

#### Description
Automatically adjusts gain to maintain a target output level. Ideal for varying input levels in meetings and calls.

#### Parameters

```python
class AGCConfig:
    enabled: bool = True
    mode: AGCMode = AGCMode.MEDIUM
    target_level: float = -20.0       # Target output level (dB)
    max_gain: float = 30.0            # Maximum gain boost (dB)
    min_gain: float = -10.0           # Minimum gain reduction (dB)
    attack_time: float = 10.0         # Attack time (ms)
    release_time: float = 200.0       # Release time (ms)
```

#### Modes

| Mode | Attack/Release | Use Case |
|------|----------------|----------|
| `FAST` | Fast response | Rapidly changing levels |
| `MEDIUM` | Balanced | General purpose |
| `SLOW` | Slow, natural | Music, minimal pumping |
| `ADAPTIVE` | Dynamic adjustment | Variable content |

---

### 5. Equalizer

**Implementation**: Custom
**Performance**: ~12ms avg
**Status**: Production-ready

#### Description
Frequency-based audio shaping using parametric EQ bands. Enhances voice clarity, reduces mud, and shapes tone.

#### Parameters

```python
class EqualizerConfig:
    enabled: bool = True
    bands: List[EqualizerBand] = [...]  # EQ bands
    gain_in: float = 0.0
    gain_out: float = 0.0

class EqualizerBand:
    frequency: float = 1000.0         # Center frequency (Hz)
    gain: float = 0.0                 # Gain adjustment (dB)
    q_factor: float = 1.0             # Bandwidth (Q)
    filter_type: str = "peak"         # peak, lowshelf, highshelf
```

#### Common Presets

**Voice Clarity**:
```python
bands = [
    EqualizerBand(frequency=80,   gain=-6.0, q_factor=0.7, filter_type="lowshelf"),   # Reduce rumble
    EqualizerBand(frequency=200,  gain=-3.0, q_factor=1.0),  # Reduce mud
    EqualizerBand(frequency=3000, gain=+3.0, q_factor=2.0),  # Boost presence
    EqualizerBand(frequency=8000, gain=+2.0, q_factor=1.0),  # Add air
]
```

**Telephone Effect**:
```python
bands = [
    EqualizerBand(frequency=300,  gain=-12.0, filter_type="highpass"),
    EqualizerBand(frequency=3400, gain=-12.0, filter_type="lowpass"),
]
```

---

### 6. Noise Reduction

**Implementation**: Custom (Spectral Subtraction)
**Performance**: ~15ms avg
**Status**: Production-ready

#### Description
Removes background noise while preserving speech. Uses spectral subtraction and Wiener filtering.

#### Parameters

```python
class NoiseReductionConfig:
    enabled: bool = True
    mode: NoiseReductionMode = NoiseReductionMode.MODERATE
    noise_floor_db: float = -40.0     # Noise floor estimate (dB)
    reduction_db: float = 12.0        # Max noise reduction (dB)
    smoothing: float = 0.8            # Spectral smoothing (0-1)
```

#### Modes

| Mode | Reduction | Artifacts | Use Case |
|------|-----------|-----------|----------|
| `LIGHT` | 6 dB | Minimal | Clean environments |
| `MODERATE` | 12 dB | Low | Office, home |
| `AGGRESSIVE` | 20 dB | Moderate | Very noisy |
| `ADAPTIVE` | Dynamic | Variable | Changing noise |

---

### 7. Voice Enhancement

**Implementation**: Custom
**Performance**: ~10ms avg
**Status**: Production-ready

#### Description
Enhances speech intelligibility through spectral shaping, formant enhancement, and dynamic processing.

#### Parameters

```python
class VoiceEnhancementConfig:
    enabled: bool = True
    enhancement_level: float = 0.5    # Enhancement strength (0-1)
    formant_shift: float = 0.0        # Formant frequency shift (semitones)
    clarity_boost: float = 3.0        # Clarity boost (dB)
```

---

### 8. VAD (Voice Activity Detection)

**Implementation**: Custom
**Performance**: ~5ms avg
**Status**: Production-ready

#### Description
Detects when speech is present in audio. Used for silence suppression and efficient processing.

#### Parameters

```python
class VADConfig:
    enabled: bool = True
    mode: VADMode = VADMode.BASIC
    sensitivity: float = 0.5          # Detection sensitivity (0-1)
    min_speech_duration: float = 0.3  # Minimum speech duration (seconds)
    min_silence_duration: float = 0.5 # Minimum silence duration (seconds)
```

#### Modes

| Mode | Accuracy | Latency | Use Case |
|------|----------|---------|----------|
| `BASIC` | Good | Low | General purpose |
| `AGGRESSIVE` | High sensitivity | Low | Quiet speech |
| `SILERO` | Best accuracy | Medium | Critical applications |
| `WEBRTC` | Balanced | Very low | Real-time |

---

## Configuration Guide

### Using Presets

```python
from audio.config import AudioProcessingConfig, create_audio_config_manager

# Get preset configurations
config_manager = create_audio_config_manager()

# Available presets
presets = [
    "default",              # Balanced processing
    "voice",                # Voice-optimized
    "noisy",                # Aggressive noise reduction
    "minimal",              # Light processing
    "broadcast",            # Broadcast quality
    "conference",           # Meeting/conference
]

# Load a preset
config = config_manager.get_preset_config("broadcast")
```

### Custom Configuration

```python
config = AudioProcessingConfig(
    enabled_stages=[
        "vad",
        "noise_reduction",
        "voice_enhancement",
        "equalizer",
        "lufs_normalization",
        "compression",
        "limiter"
    ],

    lufs_normalization=LUFSNormalizationConfig(
        mode=LUFSNormalizationMode.STREAMING,
        target_lufs=-14.0
    ),

    compression=CompressionConfig(
        threshold=-20.0,
        ratio=3.0,
        mode=CompressionMode.VOICE_OPTIMIZED
    ),

    limiter=LimiterConfig(
        threshold=-1.0,
        soft_clip=True
    )
)
```

---

## Usage Examples

### Testing with Audio Files

```bash
# Test complete pipeline
python test_audio_pipeline.py \
    --config config_examples/broadcast.json \
    --input ./input/test.wav

# Test specific stages
python test_audio_pipeline.py \
    --config config_examples/voice.json \
    --input ./input/speech.wav

# List available presets
python test_audio_pipeline.py --list-presets
```

### API Integration

```python
from audio.audio_processor import AudioPipelineProcessor
from audio.config import AudioProcessingConfig

# Create processor
config = AudioProcessingConfig()
processor = AudioPipelineProcessor(config, sample_rate=16000)

# Process audio chunk
import numpy as np
audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
processed, metadata = processor.process_audio_chunk(audio)

# Check results
print(f"Stages applied: {metadata['stages_processed']}")
print(f"Processing time: {metadata['total_processing_time_ms']}ms")
```

---

## Performance Characteristics

### Processing Times (5-second audio @ 16kHz)

| Stage | Time (ms) | CPU Usage | Notes |
|-------|-----------|-----------|-------|
| LUFS Normalization | 4.04 | Low | ✅ Enhanced (pyloudnorm) |
| Compression | 0.80 | Very Low | ✅ Enhanced (pedalboard) |
| Limiter | 1.41 | Very Low | ✅ Enhanced (pedalboard) |
| AGC | ~12 | Low | Custom implementation |
| Equalizer | ~12 | Low | Custom implementation |
| Noise Reduction | ~15 | Medium | Spectral processing |
| Voice Enhancement | ~10 | Low | Custom implementation |
| VAD | ~5 | Very Low | Custom implementation |

**Total Pipeline**: ~50-60ms for all stages enabled

### Quality Improvements (Enhanced Stages)

- **LUFS**: ±0.1 LUFS accuracy (vs ±0.5 LUFS)
- **Compression**: 0.964 correlation, minimal distortion
- **Limiter**: Zero overshoot guarantee, true peak limiting

---

## Troubleshooting

### Common Issues

**Audio too quiet/loud**:
- Check LUFS normalization `target_lufs` setting
- Verify compression `makeup_gain` is appropriate
- Check limiter `threshold` isn't too low

**Distortion/clipping**:
- Lower compression `ratio`
- Increase limiter `release_time`
- Enable limiter `soft_clip`

**Too much noise reduction artifacts**:
- Use `MODERATE` instead of `AGGRESSIVE` mode
- Reduce `reduction_db` parameter
- Increase `smoothing` parameter

**Pumping/breathing artifacts**:
- Increase compression `attack_time` and `release_time`
- Lower compression `ratio`
- Use AGC `SLOW` mode instead of `FAST`

---

## References

- **ITU-R BS.1770-4**: https://www.itu.int/rec/R-REC-BS.1770/
- **EBU R128**: https://tech.ebu.ch/loudness
- **pyloudnorm**: https://github.com/csteinmetz1/pyloudnorm
- **pedalboard**: https://github.com/spotify/pedalboard

---

**Questions?** See `src/audio/stages_enhanced/README.md` for implementation details.
