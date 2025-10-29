# Audio Processing Parameters Reference

**Complete reference for all audio processing stage parameters**

---

## Table of Contents

1. [Overview](#overview)
2. [Common Parameters](#common-parameters)
3. [Stage Reference](#stage-reference)
   - [VAD (Voice Activity Detection)](#vad-voice-activity-detection)
   - [Voice Filter](#voice-filter)
   - [Noise Reduction](#noise-reduction)
   - [Spectral Denoising](#spectral-denoising)
   - [Conventional Denoising](#conventional-denoising)
   - [Voice Enhancement](#voice-enhancement)
   - [Equalizer](#equalizer)
   - [LUFS Normalization](#lufs-normalization)
   - [AGC (Auto Gain Control)](#agc-auto-gain-control)
   - [Compression](#compression)
   - [Limiter](#limiter)
4. [Presets](#presets)
5. [Example Configurations](#example-configurations)

---

## Overview

The audio processing pipeline consists of 11 configurable stages that can be enabled/disabled and tuned independently. Each stage has specific parameters that control its behavior.

### Processing Order (Recommended)

1. **VAD** - Detect voice activity
2. **Voice Filter** - Filter voice frequencies
3. **Noise Reduction** - Remove background noise
4. **Spectral Denoising** - Advanced spectral noise removal (optional)
5. **Conventional Denoising** - Time-domain denoising (optional)
6. **Voice Enhancement** - Enhance voice clarity
7. **Equalizer** - Frequency shaping (optional)
8. **LUFS Normalization** - Loudness standardization (optional)
9. **AGC** - Automatic gain control
10. **Compression** - Dynamic range control
11. **Limiter** - Final peak limiting

---

## Common Parameters

All stages support these common parameters:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | varies | Enable/disable this stage |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain adjustment |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain adjustment |

---

## Stage Reference

### VAD (Voice Activity Detection)

Detects voice activity and filters out silence/noise segments.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable VAD |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "webrtc" | VAD algorithm |
| `aggressiveness` | int | 0 to 3 | 2 | WebRTC VAD aggressiveness |
| `energy_threshold` | float | 0.0 to 1.0 | 0.01 | Energy threshold for voice detection |
| `voice_freq_min` | float | Hz | 85 | Minimum voice frequency |
| `voice_freq_max` | float | Hz | 300 | Maximum voice frequency |
| `frame_duration_ms` | int | 10, 20, 30 | 30 | Frame duration for VAD |
| `sensitivity` | float | 0.0 to 1.0 | 0.5 | Detection sensitivity |

#### VAD Modes

- `disabled` - No VAD processing
- `basic` - Simple energy-based VAD
- `aggressive` - More aggressive voice detection
- `silero` - ML-based Silero VAD
- `webrtc` - WebRTC VAD (recommended)

#### Example

```json
{
  "vad": {
    "enabled": true,
    "mode": "webrtc",
    "aggressiveness": 2,
    "energy_threshold": 0.01,
    "sensitivity": 0.5
  }
}
```

---

### Voice Filter

Filters audio to voice frequency ranges.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable voice filtering |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `fundamental_min` | float | Hz | 85 | Minimum fundamental frequency |
| `fundamental_max` | float | Hz | 300 | Maximum fundamental frequency |
| `formant1_min` | float | Hz | 200 | First formant min |
| `formant1_max` | float | Hz | 1000 | First formant max |
| `formant2_min` | float | Hz | 900 | Second formant min |
| `formant2_max` | float | Hz | 3000 | Second formant max |
| `preserve_formants` | boolean | true/false | true | Preserve formant structure |
| `voice_band_gain` | float | 0.1 to 3.0 | 1.1 | Boost for voice frequencies |
| `high_freq_rolloff` | float | Hz | 8000 | Roll off above this frequency |

#### Example

```json
{
  "voice_filter": {
    "enabled": true,
    "fundamental_min": 85,
    "fundamental_max": 300,
    "voice_band_gain": 1.1,
    "high_freq_rolloff": 8000
  }
}
```

---

### Noise Reduction

General-purpose noise reduction.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable noise reduction |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "moderate" | Noise reduction mode |
| `strength` | float | 0.0 to 1.0 | 0.7 | Reduction strength |
| `voice_protection` | boolean | true/false | true | Protect voice frequencies |
| `stationary_noise_reduction` | float | 0.0 to 1.0 | 0.8 | For stationary noise |
| `non_stationary_noise_reduction` | float | 0.0 to 1.0 | 0.5 | For non-stationary noise |
| `noise_floor_db` | float | dB | -40 | Noise floor threshold |
| `adaptation_rate` | float | 0.01 to 1.0 | 0.1 | Adaptation speed |

#### Noise Reduction Modes

- `disabled` - No noise reduction
- `light` - Light noise reduction (0.3 strength)
- `moderate` - Moderate reduction (0.5-0.7 strength)
- `aggressive` - Aggressive reduction (0.8-0.9 strength)
- `adaptive` - Adaptive based on noise profile

#### Example

```json
{
  "noise_reduction": {
    "enabled": true,
    "mode": "moderate",
    "strength": 0.7,
    "voice_protection": true,
    "stationary_noise_reduction": 0.8,
    "non_stationary_noise_reduction": 0.5
  }
}
```

---

### Spectral Denoising

Advanced frequency-domain noise reduction.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | false | Enable spectral denoising |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "wiener_filter" | Denoising algorithm |
| `reduction_strength` | float | 0.0 to 1.0 | 0.7 | Reduction strength |
| `spectral_floor` | float | 0.01 to 1.0 | 0.1 | Minimum spectral gain |
| `fft_size` | int | 256 to 4096 (power of 2) | 1024 | FFT size |
| `smoothing_factor` | float | 0.0 to 0.99 | 0.8 | Temporal smoothing |
| `noise_update_rate` | float | 0.01 to 0.5 | 0.1 | Noise estimation update rate |
| `vad_threshold` | float | 1.0 to 10.0 | 2.0 | VAD threshold |
| `noise_floor_db` | float | -80.0 to -20.0 dB | -60.0 | Noise floor |

#### Spectral Denoising Modes

- `minimal` - Minimal processing
- `spectral_subtraction` - Spectral subtraction method
- `wiener_filter` - Wiener filtering (recommended)
- `adaptive` - Adaptive spectral processing

#### Example

```json
{
  "spectral_denoising": {
    "enabled": true,
    "mode": "wiener_filter",
    "reduction_strength": 0.7,
    "fft_size": 1024,
    "smoothing_factor": 0.8
  }
}
```

---

### Conventional Denoising

Time-domain denoising algorithms.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable conventional denoising |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "median_filter" | Denoising algorithm |
| `strength` | float | 0.0 to 1.0 | 0.5 | Denoising strength |
| `window_size` | int | 3 to 21 | 5 | Filter window size |
| `threshold` | float | 0.0 to 1.0 | 0.1 | Noise detection threshold |
| `adaptation_rate` | float | 0.01 to 0.9 | 0.1 | Adaptation speed |
| `preserve_transients` | boolean | true/false | true | Preserve attack/transients |
| `high_freq_emphasis` | float | 0.0 to 1.0 | 0.3 | High frequency emphasis |

#### Median Filter Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `median_kernel_size` | int | 3 to 15 (odd) | 3 | Median filter kernel |

#### Gaussian Filter Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `gaussian_sigma` | float | 0.1 to 5.0 | 1.0 | Standard deviation |

#### Bilateral Filter Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `bilateral_sigma_color` | float | 0.01 to 1.0 | 0.1 | Color sigma |
| `bilateral_sigma_space` | float | 0.1 to 5.0 | 1.0 | Spatial sigma |

#### Wavelet Denoising Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `wavelet_type` | string | wavelet types | "db4" | Wavelet type |
| `wavelet_levels` | int | 1 to 6 | 4 | Decomposition levels |
| `wavelet_threshold_mode` | string | "soft"/"hard" | "soft" | Thresholding mode |

#### Conventional Denoising Modes

- `disabled` - No denoising
- `median_filter` - Median filtering
- `gaussian_filter` - Gaussian smoothing
- `bilateral_filter` - Edge-preserving bilateral filter
- `wavelet_denoising` - Wavelet-based denoising
- `adaptive_filter` - Adaptive filtering
- `rnr_filter` - Reduce Noise and Reverb

#### Example

```json
{
  "conventional_denoising": {
    "enabled": true,
    "mode": "median_filter",
    "strength": 0.5,
    "window_size": 5,
    "preserve_transients": true
  }
}
```

---

### Voice Enhancement

Enhances voice clarity and presence.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable voice enhancement |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `normalize` | boolean | true/false | false | Normalize output |
| `clarity_enhancement` | float | 0.0 to 1.0 | 0.2 | Clarity boost |
| `presence_boost` | float | 0.0 to 1.0 | 0.1 | Presence boost |
| `warmth_adjustment` | float | -1.0 to 1.0 | 0.0 | Warmth control |
| `brightness_adjustment` | float | -1.0 to 1.0 | 0.0 | Brightness control |
| `sibilance_control` | float | 0.0 to 1.0 | 0.1 | Control harsh sibilants |

#### Example

```json
{
  "voice_enhancement": {
    "enabled": true,
    "clarity_enhancement": 0.2,
    "presence_boost": 0.1,
    "sibilance_control": 0.1
  }
}
```

---

### Equalizer

Multi-band parametric equalizer.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | false | Enable equalizer |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `bands` | array | see below | 5 bands | EQ bands configuration |
| `preset_name` | string | preset name | "flat" | EQ preset |

#### Equalizer Band Parameters

Each band has:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | varies | Enable this band |
| `frequency` | float | 20 to 20000 Hz | varies | Center frequency |
| `gain` | float | -20.0 to 20.0 dB | 0.0 | Gain adjustment |
| `bandwidth` | float | 0.1 to 5.0 octaves | 1.0 | Filter bandwidth |
| `filter_type` | string | see types below | "peaking" | Filter type |

#### Filter Types

- `peaking` - Peak/dip at frequency
- `low_shelf` - Low frequency shelf
- `high_shelf` - High frequency shelf
- `low_pass` - Low pass filter
- `high_pass` - High pass filter

#### Example

```json
{
  "equalizer": {
    "enabled": true,
    "bands": [
      {
        "enabled": true,
        "frequency": 80,
        "gain": -3.0,
        "bandwidth": 1.0,
        "filter_type": "high_pass"
      },
      {
        "enabled": true,
        "frequency": 200,
        "gain": 2.0,
        "bandwidth": 1.0,
        "filter_type": "peaking"
      },
      {
        "enabled": true,
        "frequency": 3200,
        "gain": 3.0,
        "bandwidth": 1.0,
        "filter_type": "peaking"
      },
      {
        "enabled": true,
        "frequency": 8000,
        "gain": 1.0,
        "bandwidth": 1.0,
        "filter_type": "high_shelf"
      }
    ]
  }
}
```

---

### LUFS Normalization

Loudness standardization (EBU R128).

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | false | Enable LUFS normalization |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "streaming" | Normalization mode |
| `target_lufs` | float | -60.0 to 0.0 LUFS | -14.0 | Target loudness |
| `max_peak_db` | float | -6.0 to 0.0 dB | -1.0 | Max peak to prevent clipping |
| `gating_threshold` | float | -80.0 to -40.0 dB | -70.0 | Gating threshold |
| `measurement_window` | float | 0.1 to 10.0 s | 3.0 | Measurement window |
| `short_term_window` | float | 0.1 to 10.0 s | 3.0 | Short-term window |
| `momentary_window` | float | 0.1 to 1.0 s | 0.4 | Momentary window |
| `lookahead_time` | float | 0.01 to 1.0 s | 0.1 | Lookahead time |
| `adjustment_speed` | float | 0.01 to 1.0 | 0.5 | Adjustment speed |
| `true_peak_limiting` | boolean | true/false | true | Enable true peak limiting |
| `lufs_tolerance` | float | 0.1 to 3.0 LUFS | 0.5 | Acceptable deviation |
| `peak_tolerance` | float | 0.01 to 1.0 dB | 0.1 | Peak deviation |

#### LUFS Modes

- `disabled` - No normalization
- `streaming` - -14 LUFS (Spotify, Apple Music)
- `broadcast_tv` - -23 LUFS (EBU R128)
- `broadcast_radio` - -16 LUFS
- `podcast` - -18 LUFS
- `youtube` - -14 LUFS
- `netflix` - -27 LUFS (cinema style)
- `custom` - User-defined target

#### Example

```json
{
  "lufs_normalization": {
    "enabled": true,
    "mode": "podcast",
    "target_lufs": -18.0,
    "max_peak_db": -1.0,
    "true_peak_limiting": true
  }
}
```

---

### AGC (Auto Gain Control)

Automatic gain adjustment for consistent levels.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable AGC |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "medium" | AGC mode |
| `target_level` | float | -40.0 to 0.0 dB | -18.0 | Target output level |
| `max_gain` | float | 0.0 to 40.0 dB | 12.0 | Maximum gain |
| `min_gain` | float | -40.0 to 0.0 dB | -12.0 | Minimum gain |
| `attack_time` | float | 0.1 to 100.0 ms | 10.0 | Attack time |
| `release_time` | float | 1.0 to 1000.0 ms | 100.0 | Release time |
| `hold_time` | float | 0.0 to 500.0 ms | 50.0 | Hold time |
| `knee_width` | float | 0.0 to 10.0 dB | 2.0 | Soft knee width |
| `lookahead_time` | float | 0.0 to 20.0 ms | 5.0 | Lookahead time |
| `adaptation_rate` | float | 0.01 to 1.0 | 0.1 | Adaptation rate |
| `noise_gate_threshold` | float | -80.0 to -20.0 dB | -60.0 | Noise gate threshold |

#### AGC Modes

- `disabled` - No AGC
- `fast` - Fast response (10ms attack)
- `medium` - Medium response (50ms attack)
- `slow` - Slow response (100ms attack)
- `adaptive` - Adaptive based on signal

#### Example

```json
{
  "agc": {
    "enabled": true,
    "mode": "medium",
    "target_level": -18.0,
    "max_gain": 12.0,
    "attack_time": 10.0,
    "release_time": 100.0
  }
}
```

---

### Compression

Dynamic range compression.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable compression |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `mode` | string | see modes below | "soft_knee" | Compression mode |
| `threshold` | float | dB | -20 | Compression threshold |
| `ratio` | float | 1.0 to 20.0 | 3.0 | Compression ratio |
| `knee` | float | 0.0 to 10.0 dB | 2.0 | Soft knee width |
| `attack_time` | float | 0.1 to 100.0 ms | 5.0 | Attack time |
| `release_time` | float | 1.0 to 1000.0 ms | 100.0 | Release time |
| `makeup_gain` | float | dB | 0.0 | Makeup gain |
| `lookahead` | float | 0.0 to 20.0 ms | 5.0 | Lookahead time |

#### Compression Modes

- `disabled` - No compression
- `soft_knee` - Soft knee (smooth)
- `hard_knee` - Hard knee (precise)
- `adaptive` - Adaptive compression
- `voice_optimized` - Optimized for voice

#### Example

```json
{
  "compression": {
    "enabled": true,
    "mode": "soft_knee",
    "threshold": -20,
    "ratio": 3.0,
    "knee": 2.0,
    "attack_time": 5.0,
    "release_time": 100.0
  }
}
```

---

### Limiter

Final peak limiting to prevent clipping.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `enabled` | boolean | true/false | true | Enable limiter |
| `gain_in` | float | -20.0 to 20.0 dB | 0.0 | Input gain |
| `gain_out` | float | -20.0 to 20.0 dB | 0.0 | Output gain |
| `threshold` | float | dB | -1.0 | Limiting threshold |
| `release_time` | float | 1.0 to 500.0 ms | 50.0 | Release time |
| `lookahead` | float | 0.0 to 20.0 ms | 5.0 | Lookahead time |
| `soft_clip` | boolean | true/false | true | Soft clipping |

#### Example

```json
{
  "limiter": {
    "enabled": true,
    "threshold": -1.0,
    "release_time": 50.0,
    "soft_clip": true
  }
}
```

---

## Presets

### Available Presets

1. **default** - Balanced processing
2. **voice_optimized** - Optimized for voice clarity
3. **noisy_environment** - Aggressive noise reduction
4. **music_content** - Preserves musical content
5. **minimal_processing** - Light processing
6. **aggressive_processing** - Maximum processing
7. **conference_call** - Optimized for meetings
8. **broadcast_quality** - Professional broadcast

### Using Presets

```json
{
  "preset_name": "meeting_optimized"
}
```

---

## Example Configurations

### Minimal Processing (Low Latency)

```json
{
  "enabled_stages": ["vad", "limiter"],
  "vad": {
    "enabled": true,
    "mode": "basic",
    "aggressiveness": 1
  },
  "limiter": {
    "enabled": true,
    "threshold": -1.0
  }
}
```

### Noisy Environment (Aggressive)

```json
{
  "enabled_stages": [
    "vad",
    "voice_filter",
    "noise_reduction",
    "spectral_denoising",
    "voice_enhancement",
    "agc",
    "compression",
    "limiter"
  ],
  "vad": {
    "mode": "aggressive",
    "aggressiveness": 3
  },
  "noise_reduction": {
    "mode": "aggressive",
    "strength": 0.9,
    "stationary_noise_reduction": 0.95,
    "non_stationary_noise_reduction": 0.8
  },
  "spectral_denoising": {
    "enabled": true,
    "mode": "wiener_filter",
    "reduction_strength": 0.85,
    "fft_size": 2048
  }
}
```

### Voice Clarity (Podcasting)

```json
{
  "enabled_stages": [
    "vad",
    "voice_filter",
    "noise_reduction",
    "voice_enhancement",
    "equalizer",
    "lufs_normalization",
    "compression",
    "limiter"
  ],
  "voice_enhancement": {
    "clarity_enhancement": 0.4,
    "presence_boost": 0.3,
    "sibilance_control": 0.2
  },
  "equalizer": {
    "enabled": true,
    "bands": [
      {"enabled": true, "frequency": 80, "gain": -6, "filter_type": "high_pass"},
      {"enabled": true, "frequency": 250, "gain": 2, "filter_type": "peaking"},
      {"enabled": true, "frequency": 3500, "gain": 4, "filter_type": "peaking"},
      {"enabled": true, "frequency": 8000, "gain": 2, "filter_type": "high_shelf"}
    ]
  },
  "lufs_normalization": {
    "enabled": true,
    "mode": "podcast",
    "target_lufs": -18.0
  }
}
```

### Music Preservation

```json
{
  "enabled_stages": [
    "vad",
    "conventional_denoising",
    "agc",
    "limiter"
  ],
  "vad": {
    "mode": "basic",
    "aggressiveness": 0
  },
  "conventional_denoising": {
    "mode": "bilateral_filter",
    "strength": 0.3,
    "preserve_transients": true
  },
  "agc": {
    "mode": "slow",
    "target_level": -14.0
  }
}
```

### Broadcasting (Professional)

```json
{
  "enabled_stages": [
    "vad",
    "voice_filter",
    "noise_reduction",
    "voice_enhancement",
    "equalizer",
    "lufs_normalization",
    "compression",
    "limiter"
  ],
  "lufs_normalization": {
    "enabled": true,
    "mode": "broadcast_tv",
    "target_lufs": -23.0,
    "true_peak_limiting": true
  },
  "compression": {
    "mode": "voice_optimized",
    "threshold": -18,
    "ratio": 4.0,
    "knee": 3.0
  }
}
```

### Debug Single Stage

```json
{
  "enabled_stages": ["noise_reduction"],
  "noise_reduction": {
    "enabled": true,
    "mode": "moderate",
    "strength": 0.7,
    "stationary_noise_reduction": 0.8,
    "non_stationary_noise_reduction": 0.5,
    "noise_floor_db": -40,
    "adaptation_rate": 0.1
  }
}
```

---

## Parameter Tuning Tips

### Noise Reduction

- **Light environments**: `strength: 0.3-0.5`
- **Moderate noise**: `strength: 0.5-0.7`
- **Heavy noise**: `strength: 0.7-0.9`
- **Preserve voice**: `voice_protection: true`

### Compression

- **Natural sound**: `ratio: 2.0-3.0, soft_knee`
- **Controlled dynamics**: `ratio: 4.0-6.0, hard_knee`
- **Heavy compression**: `ratio: 8.0-12.0`

### AGC

- **Fast response**: `attack_time: 5-10ms`
- **Natural response**: `attack_time: 20-50ms`
- **Slow response**: `attack_time: 100ms+`

### LUFS Normalization

- **Streaming platforms**: `-14 LUFS`
- **Podcasts**: `-16 to -18 LUFS`
- **Broadcast TV**: `-23 LUFS`
- **Cinema**: `-27 LUFS`

---

## Validation Rules

All parameters are automatically validated and clamped to their valid ranges. Invalid values will be corrected to the nearest valid value.

### Common Validations

- Gain values: -20 to +20 dB
- Ratio values: 0.0 to 1.0
- Time values: Positive milliseconds
- Frequency values: 20 to 20000 Hz
- Boolean values: true or false

---

## Further Reading

- [Audio Testing Guide](./README_AUDIO_TESTING.md)
- [Audio Processing Pipeline](./src/audio/README.md)
- [Configuration System](./src/audio/config.py)
- [Orchestration Service CLAUDE.md](./CLAUDE.md)
