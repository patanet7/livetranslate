# Modular Audio Processing Pipeline - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Modular Architecture](#modular-architecture)
3. [Processing Stages](#processing-stages)
4. [Pipeline Configuration](#pipeline-configuration)
5. [Frontend Integration](#frontend-integration)
6. [Real-time Monitoring](#real-time-monitoring)
7. [Parameter Reference](#parameter-reference)
8. [Implementation Libraries](#implementation-libraries)
9. [Performance Metrics](#performance-metrics)
10. [API Endpoints](#api-endpoints)
11. [Configuration Management](#configuration-management)

---

## Overview

The LiveTranslate Orchestration Service implements a **fully modular audio processing pipeline** designed for real-time speech recognition optimization. The system allows **unlimited instances** of any stage type to be placed anywhere in the pipeline, with individual **gain controls** and **configurable parameters** for each instance.

### Key Features
- **Fully Modular Design**: Add multiple instances of any stage type anywhere in the pipeline
- **Individual Gain Controls**: Input and output gain adjustments (-20dB to +20dB) for every stage
- **Dynamic Pipeline Management**: Add, remove, move, and duplicate stages in real-time
- **Unique Instance IDs**: Each stage instance has its own identifier and configuration
- **Real-time Processing**: < 100ms latency target for entire pipeline
- **Performance Monitoring**: Per-instance latency tracking and quality metrics
- **Professional Quality**: Broadcast-grade audio processing algorithms
- **Frontend Integration**: Complete modular control from web interface
- **Preset Management**: Save and load complete pipeline configurations
- **Hot Configuration**: Real-time parameter updates without restart

### Libraries Used
- **NumPy**: Core numerical operations and array processing
- **SciPy**: Signal processing filters and transforms
- **SciPy.signal**: Butter filters, filtfilt, windowing functions
- **SciPy.fft**: Fast Fourier Transform for frequency domain analysis
- **Python asyncio**: Asynchronous processing coordination
- **Pydantic**: Configuration validation and serialization

---

## Modular Architecture

The pipeline is fully modular, allowing unlimited instances of any stage type to be placed anywhere in the processing chain. Each stage instance has individual gain controls and configuration.

### Example Modular Pipeline Configuration
```
┌─────────────────────────────────────────────────────────────────┐
│              Modular Audio Processing Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│  Input Audio Chunk (16kHz, 16-bit)                             │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "vad_main" (VAD)                                  │ │
│  │ • Gain In: +2dB | Gain Out: 0dB                           │ │
│  │ • WebRTC VAD / Energy-based detection                     │ │
│  │ • Library: Custom implementation with SciPy               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "voice_filter_main" (Voice Filter)               │ │
│  │ • Gain In: 0dB | Gain Out: +1dB                           │ │
│  │ • Fundamental frequency enhancement (85-300Hz)             │ │
│  │ • Library: SciPy.signal.butter, filtfilt                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "nr_light" (Noise Reduction)                     │ │
│  │ • Gain In: +1dB | Gain Out: 0dB                           │ │
│  │ • Light spectral subtraction                               │ │
│  │ • Library: SciPy.fft (rfft/irfft), NumPy                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "eq_voice" (Equalizer)                           │ │
│  │ • Gain In: 0dB | Gain Out: -1dB                           │ │
│  │ • Voice enhancement preset: 1-3kHz boost                   │ │
│  │ • Library: SciPy.signal (IIR filters)                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "nr_aggressive" (Noise Reduction)                │ │
│  │ • Gain In: 0dB | Gain Out: +2dB                           │ │
│  │ • Aggressive noise reduction for harsh environments        │ │
│  │ • Library: SciPy.fft (rfft/irfft), NumPy                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "agc_main" (Auto Gain Control)                   │ │
│  │ • Gain In: 0dB | Gain Out: 0dB                            │ │
│  │ • Adaptive level control with lookahead                    │ │
│  │ • Library: NumPy, custom DSP algorithms                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "eq_broadcast" (Equalizer)                       │ │
│  │ • Gain In: -1dB | Gain Out: 0dB                           │ │
│  │ • Broadcast preset: Professional radio curve               │ │
│  │ • Library: SciPy.signal (IIR filters)                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "comp_main" (Compression)                        │ │
│  │ • Gain In: 0dB | Gain Out: +1dB                           │ │
│  │ • Soft knee compression with makeup gain                   │ │
│  │ • Library: NumPy mathematical operations                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Instance: "limiter_final" (Limiter)                        │ │
│  │ • Gain In: 0dB | Gain Out: -2dB                           │ │
│  │ • Transparent limiting with lookahead                      │ │
│  │ • Library: NumPy with custom algorithms                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│  Processed Audio Output → Whisper Service                      │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Management Operations
```python
# Add multiple equalizer instances
pipeline.add_stage("equalizer", "eq_voice", position=3)
pipeline.add_stage("equalizer", "eq_broadcast", position=7)

# Add multiple noise reduction stages
pipeline.add_stage("noise_reduction", "nr_light", position=2)  
pipeline.add_stage("noise_reduction", "nr_aggressive", position=5)

# Configure individual gain controls
eq_voice = pipeline.get_stage("eq_voice")
eq_voice.config.gain_in = 0.0    # dB
eq_voice.config.gain_out = -1.0  # dB

# Move, duplicate, enable/disable stages
pipeline.move_stage("eq_voice", new_position=6)
pipeline.duplicate_stage("nr_light", "nr_light_copy")
pipeline.enable_stage("eq_broadcast", False)
```

---

## Pipeline Configuration

### Modular Pipeline Configuration System

The audio processing pipeline uses a completely modular configuration system that supports unlimited stage instances with individual control.

#### Core Configuration Classes

##### StageInstance
Each stage in the pipeline is represented by a `StageInstance`:
```python
@dataclass
class StageInstance:
    stage_type: str      # "vad", "equalizer", "noise_reduction", etc.
    instance_id: str     # Unique identifier: "eq_voice", "nr_aggressive"
    enabled: bool = True # Enable/disable this specific instance
    config: Any = None   # Stage-specific configuration object
```

##### ModularPipelineConfig  
The main pipeline configuration supports dynamic management:
```python
@dataclass
class ModularPipelineConfig:
    # Ordered list of stage instances
    pipeline: List[StageInstance] = field(default_factory=list)
    
    # Global settings
    sample_rate: int = 16000
    buffer_size: int = 1024
    processing_block_size: int = 512
    
    # Performance settings
    real_time_priority: bool = True
    cpu_usage_limit: float = 0.8
    processing_timeout: float = 100.0  # ms
    
    # Metadata
    preset_name: str = "default"
    version: str = "2.0"
```

#### Individual Gain Controls

**Every stage instance** includes input and output gain controls:
```python
# All stage configurations include:
gain_in: float = 0.0   # dB - Input gain adjustment (-20dB to +20dB)
gain_out: float = 0.0  # dB - Output gain adjustment (-20dB to +20dB)
```

**Gain Application Order:**
1. Apply `gain_in` to input audio
2. Process audio through stage algorithm  
3. Apply `gain_out` to processed audio
4. Pass to next stage in pipeline

#### Dynamic Pipeline Management

##### Adding Stages
```python
# Add single instance
instance_id = config.add_stage("equalizer", "eq_main")

# Add at specific position
instance_id = config.add_stage("noise_reduction", "nr_light", position=2)

# Add with custom configuration
custom_eq_config = EqualizerConfig(gain_in=2.0, gain_out=-1.0)
instance_id = config.add_stage("equalizer", "eq_custom", config=custom_eq_config)
```

##### Removing and Moving Stages
```python
# Remove stage instance
success = config.remove_stage("eq_custom")

# Move stage to new position
success = config.move_stage("nr_light", new_position=5)

# Enable/disable specific instance
config.enable_stage("eq_main", False)
```

##### Duplicating Stages
```python
# Duplicate with automatic ID generation
new_id = config.duplicate_stage("eq_main")  # Creates "eq_main_copy_1"

# Duplicate with custom ID
new_id = config.duplicate_stage("eq_main", "eq_voice_optimized")
```

#### Preset Management

##### Saving Pipeline Configurations
```python
# Save complete pipeline as preset
preset_data = config.save_preset("my_voice_chain")

# Preset data includes:
# - All stage instances with configurations
# - Pipeline order and positioning
# - Individual gain settings
# - Global pipeline settings
```

##### Example Saved Preset
```json
{
  "name": "broadcast_voice_chain",
  "version": "2.0",
  "pipeline": [
    {
      "stage_type": "vad",
      "instance_id": "vad_main",
      "enabled": true,
      "config": {
        "enabled": true,
        "gain_in": 2.0,
        "gain_out": 0.0,
        "mode": "webrtc",
        "aggressiveness": 2
      }
    },
    {
      "stage_type": "equalizer", 
      "instance_id": "eq_voice",
      "enabled": true,
      "config": {
        "enabled": true,
        "gain_in": 0.0,
        "gain_out": -1.0,
        "preset_name": "voice_enhance",
        "bands": [...]
      }
    }
  ],
  "settings": {
    "sample_rate": 16000,
    "real_time_priority": true
  }
}
```

#### Frontend Integration

The modular system enables complete frontend control:

##### Pipeline Editor
- **Drag-and-drop interface** for stage arrangement
- **Add/remove stages** with visual feedback
- **Instance-specific controls** for each stage
- **Real-time pipeline visualization**

##### Stage Configuration
- **Individual gain controls** (-20dB to +20dB sliders)
- **Stage-specific parameters** (EQ bands, compression ratios, etc.)
- **Enable/disable toggles** per instance
- **Preset selection** per stage type

##### Configuration Management
- **Save pipeline presets** with custom names
- **Load saved configurations** instantly
- **Export/import** preset files
- **A/B testing** between different configurations

#### Gain Control Implementation

Every stage instance includes professional-grade gain controls applied at precise points in the processing chain.

##### Gain Application Order
```python
def process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    # 1. Apply input gain
    if abs(self.config.gain_in) > 0.1:
        input_gain_linear = 10 ** (self.config.gain_in / 20)
        processed = audio_data * input_gain_linear
    else:
        processed = audio_data.copy()
    
    # 2. Stage-specific processing (EQ, compression, etc.)
    processed = self._apply_stage_processing(processed)
    
    # 3. Apply output gain  
    if abs(self.config.gain_out) > 0.1:
        output_gain_linear = 10 ** (self.config.gain_out / 20)
        processed = processed * output_gain_linear
    
    return processed, metadata
```

##### Gain Control Features
- **Range**: -20dB to +20dB (prevents extreme adjustments)
- **Resolution**: 0.1dB precision for professional control
- **Bypass Threshold**: Gains below 0.1dB are bypassed for efficiency
- **Linear Conversion**: Proper dB to linear conversion (10^(dB/20))
- **Metadata Tracking**: Gain values included in processing metadata

##### Frontend Gain Controls
- **Dual Sliders**: Separate input/output gain sliders per instance
- **Real-time Updates**: Instant parameter updates without audio dropouts
- **Visual Feedback**: Level meters showing gain effect
- **Preset Integration**: Gain settings saved with pipeline presets
- **Range Indicators**: Clear min/max dB markings

##### Use Cases for Individual Gain Controls

**Input Gain (`gain_in`)**:
- **Level Matching**: Compensate for varying input levels between sources
- **Stage Preparation**: Optimize signal level for specific processing algorithms
- **Headroom Management**: Prevent clipping in high-gain processing stages
- **Signal Conditioning**: Prepare weak signals for noise reduction

**Output Gain (`gain_out`)**:
- **Level Compensation**: Restore levels after processing-induced changes
- **Stage Balancing**: Match output levels between parallel processing chains  
- **Makeup Gain**: Compensate for level reduction from compression/limiting
- **Final Trimming**: Fine-tune overall pipeline output level

**Example Gain Strategy**:
```
VAD: gain_in=+2dB (boost weak input), gain_out=0dB
EQ_Voice: gain_in=0dB, gain_out=-1dB (reduce presence boost artifacts)
Noise_Reduction: gain_in=+1dB, gain_out=+2dB (compensate for NR attenuation)
Compressor: gain_in=0dB, gain_out=+3dB (makeup gain for compression)
Limiter: gain_in=0dB, gain_out=-2dB (final level trim)
```

---

## Processing Stages

The modular pipeline currently supports **11 professional audio processing stages**, each with individual gain controls and advanced algorithms.

### Stage 1: Voice Activity Detection (VAD)
**Purpose**: Identify speech segments and skip processing silence
**Library**: Custom implementation with SciPy support
**Processing Location**: `src/audio/stages/vad_stage.py`
**Performance Target**: 5.0ms (max 10.0ms)

#### Implementation Details
- **WebRTC VAD Simulation**: Energy-based detection with frequency analysis
- **Adaptive Noise Floor**: Dynamically adjusts to background noise level
- **Confidence Scoring**: Returns voice probability (0.0-1.0)
- **Voice Frequency Analysis**: Focuses on human speech range (85-300Hz fundamental)

#### Parameters
```python
@dataclass
class VADConfig:
    enabled: bool = True                    # Enable/disable VAD processing
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    mode: VADMode = VADMode.WEBRTC         # BASIC, WEBRTC, AGGRESSIVE, SILERO
    aggressiveness: int = 2                 # 0-3, WebRTC VAD aggressiveness
    energy_threshold: float = 0.01          # Energy threshold for voice detection
    voice_freq_min: float = 85              # Hz, minimum voice frequency
    voice_freq_max: float = 300             # Hz, maximum voice frequency
    frame_duration_ms: int = 30             # Frame duration (10, 20, or 30ms)
    sensitivity: float = 0.5                # 0.0-1.0, detection sensitivity
```

#### Frontend Display Requirements
- **Voice Detection Status**: Real-time voice/silence indicator
- **Confidence Meter**: Visual confidence level (0-100%)
- **Frequency Analysis**: Voice frequency range visualization
- **Threshold Controls**: Interactive sensitivity adjustment
- **Mode Selection**: Dropdown for VAD algorithm selection

#### Latency Impact
- **Target**: < 5ms per chunk
- **Factors**: Frame duration, frequency analysis complexity
- **Optimization**: Vectorized NumPy operations, minimal memory allocation

---

### Stage 2: Voice Frequency Filtering
**Purpose**: Enhance speech frequencies while preserving natural voice characteristics
**Library**: SciPy.signal (butter filters, filtfilt)
**Processing Location**: `src/audio/stages/voice_filter_stage.py`
**Performance Target**: 8.0ms (max 15.0ms)

#### Implementation Details
- **Fundamental Frequency Enhancement**: Boosts primary voice frequencies (85-300Hz)
- **Formant Preservation**: Maintains F1 (200-1000Hz) and F2 (900-3000Hz) formants
- **High Frequency Rolloff**: Reduces frequencies above 8kHz
- **Voice Band Gain**: Selective amplification of speech-critical frequencies

#### Parameters
```python
@dataclass
class VoiceFilterConfig:
    enabled: bool = True                    # Enable/disable voice filtering
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    fundamental_min: float = 85             # Hz, fundamental frequency minimum
    fundamental_max: float = 300            # Hz, fundamental frequency maximum
    formant1_min: float = 200              # Hz, first formant minimum
    formant1_max: float = 1000             # Hz, first formant maximum
    formant2_min: float = 900              # Hz, second formant minimum
    formant2_max: float = 3000             # Hz, second formant maximum
    preserve_formants: bool = True          # Enable formant preservation
    voice_band_gain: float = 1.1           # Voice frequency band gain (0.1-3.0)
    high_freq_rolloff: float = 8000        # Hz, high frequency rolloff point
```

#### Frontend Display Requirements
- **Frequency Response Graph**: Visual EQ curve showing filter response
- **Formant Indicators**: F1/F2 frequency range markers
- **Gain Meters**: Real-time gain visualization per frequency band
- **Filter Controls**: Sliders for frequency ranges and gain adjustments
- **Bypass Toggle**: A/B comparison with unfiltered audio

#### Latency Impact
- **Target**: < 8ms per chunk
- **Factors**: Filter order, filtfilt processing, frequency band count
- **Optimization**: Pre-calculated filter coefficients, efficient convolution

---

### Stage 3: Noise Reduction ✅ FIXED
**Purpose**: Remove background noise while preserving speech intelligibility
**Library**: SciPy.fft (rfft/irfft), NumPy
**Processing Location**: `src/audio/stages/noise_reduction_stage.py`
**Performance Target**: 15.0ms (max 25.0ms)

#### Implementation Details
- **Spectral Subtraction**: Frequency-domain noise reduction
- **Voice Protection**: Preserves speech frequencies during noise reduction (FIXED array indexing)
- **Adaptive Noise Profile**: Continuously updates noise floor estimation
- **Musical Noise Suppression**: Reduces processing artifacts

#### Parameters
```python
@dataclass
class NoiseReductionConfig:
    enabled: bool = True                    # Enable/disable noise reduction
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    mode: NoiseReductionMode = NoiseReductionMode.MODERATE  # LIGHT, MODERATE, AGGRESSIVE, ADAPTIVE
    strength: float = 0.7                   # 0.0-1.0, noise reduction strength
    voice_protection: bool = True           # Protect voice frequencies
    stationary_noise_reduction: float = 0.8 # 0.0-1.0, stationary noise reduction
    non_stationary_noise_reduction: float = 0.5  # 0.0-1.0, non-stationary noise reduction
    noise_floor_db: float = -40             # dB, noise floor threshold
    adaptation_rate: float = 0.1            # 0.01-1.0, adaptation rate
```

#### Frontend Display Requirements
- **Noise Profile Graph**: Visual noise floor estimation
- **Spectral Display**: Before/after frequency domain comparison
- **Reduction Meters**: Real-time noise reduction amount per frequency band
- **Voice Protection Indicator**: Shows protected frequency ranges
- **Adaptation Status**: Noise profile learning progress

#### Latency Impact
- **Target**: < 15ms per chunk
- **Factors**: FFT size, overlap percentage, spectral processing complexity
- **Optimization**: Efficient FFT implementation, vectorized operations

---

### Stage 4: Voice Enhancement
**Purpose**: Improve speech clarity, presence, and intelligibility
**Library**: SciPy.signal filtering, NumPy processing
**Processing Location**: `src/audio/stages/voice_enhancement_stage.py`
**Performance Target**: 10.0ms (max 20.0ms)

#### Implementation Details
- **Clarity Enhancement**: Harmonic amplification and transient sharpening
- **Presence Boost**: 2-5kHz frequency range enhancement
- **Warmth/Brightness Control**: Tonal character adjustment
- **Sibilance Control**: Reduces harsh 's' and 't' sounds

#### Parameters
```python
@dataclass
class VoiceEnhancementConfig:
    enabled: bool = True                    # Enable/disable voice enhancement
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    normalize: bool = False                 # Normalize output level
    clarity_enhancement: float = 0.2        # 0.0-1.0, clarity enhancement amount
    presence_boost: float = 0.1            # 0.0-1.0, presence boost amount
    warmth_adjustment: float = 0.0         # -1.0 to 1.0, warmth adjustment
    brightness_adjustment: float = 0.0     # -1.0 to 1.0, brightness adjustment
    sibilance_control: float = 0.1         # 0.0-1.0, sibilance control amount
```

#### Frontend Display Requirements
- **Enhancement Meters**: Real-time clarity, presence, warmth, brightness levels
- **Sibilance Monitor**: Sibilance detection and control visualization
- **Spectral Comparison**: Before/after frequency analysis
- **Harmonic Analyzer**: Harmonic content visualization
- **Quality Metrics**: Speech intelligibility scoring

#### Latency Impact
- **Target**: < 10ms per chunk
- **Factors**: Enhancement algorithm complexity, filter count
- **Optimization**: Efficient harmonic analysis, parallel processing

---

### Stage 5: Parametric Equalizer ✅ NEW
**Purpose**: Multi-band frequency response shaping and tonal adjustment
**Library**: SciPy.signal (IIR filters), NumPy processing
**Processing Location**: `src/audio/stages/equalizer_stage.py:EqualizerStage`

#### Implementation Details
- **Multi-band EQ**: Up to 10 configurable bands with independent controls
- **Filter Types**: Peaking, high/low shelf, high/low pass filters
- **Professional Presets**: Voice enhancement, broadcast, flat response, etc.
- **Real-time Control**: Hot-swappable EQ curves without audio dropouts

#### Parameters
```python
@dataclass
class EqualizerBand:
    enabled: bool = True                    # Enable/disable this band
    frequency: float = 1000.0              # Hz, center frequency (20-20000)
    gain: float = 0.0                      # dB, gain adjustment (-20 to +20)
    bandwidth: float = 1.0                 # Octaves, bandwidth for peaking (0.1-5.0)
    filter_type: str = "peaking"           # peaking, low_shelf, high_shelf, low_pass, high_pass

@dataclass  
class EqualizerConfig:
    enabled: bool = False                   # Enable/disable equalizer (optional stage)
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    bands: List[EqualizerBand]             # List of EQ bands (default: 5-band)
    preset_name: str = "flat"              # EQ preset name
```

#### Frontend Integration
The equalizer provides extensive frontend controls:
- **Band Controls**: Frequency, gain, bandwidth, filter type for each band
- **Graphical EQ**: Visual frequency response curve with real-time updates
- **Preset Management**: Professional presets with one-click application
- **A/B Testing**: Compare different EQ settings in real-time

#### Processing Flow
1. **Filter Design**: Generate IIR filter coefficients for each enabled band
2. **Serial Processing**: Apply each filter band sequentially to audio
3. **State Management**: Maintain filter memory for continuous processing
4. **Gain Application**: Apply overall gain to final output

#### Performance Considerations
- **Target Latency**: 12ms (configurable, max 22ms)
- **CPU Usage**: Moderate (proportional to number of enabled bands)
- **Memory Usage**: Low (filter state storage only)
- **Factors**: Number of bands, filter complexity, sample rate

#### Common Presets
- **Flat**: Neutral response, all bands disabled
- **Voice Enhance**: Boost presence (1-3kHz), reduce low frequencies
- **Broadcast**: Professional broadcast EQ curve
- **Bass Boost**: Enhanced low frequency response
- **Treble Boost**: Enhanced high frequency response

---

### Stage 6: Spectral Denoising ✅ NEW
**Purpose**: Advanced frequency-domain noise reduction with multiple algorithms
**Library**: SciPy.fft (rfft/irfft), NumPy
**Processing Location**: `src/audio/stages/spectral_denoising_stage.py`
**Performance Target**: 20.0ms (max 35.0ms)

#### Implementation Details
- **FFT-based Processing**: Overlap-add reconstruction for continuous processing
- **Multiple Algorithms**: Spectral subtraction, Wiener filtering, adaptive denoising
- **Phase Preservation**: Maintains audio phase relationships for natural sound
- **Artifact Reduction**: Minimizes musical noise and processing artifacts

#### Parameters
```python
@dataclass
class SpectralDenoisingConfig:
    enabled: bool = False                   # Enable/disable spectral denoising (optional stage)
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    mode: SpectralDenoisingMode = SpectralDenoisingMode.MINIMAL  # Algorithm selection
    noise_reduction_factor: float = 0.5     # 0.0-1.0, reduction strength
    noise_floor_estimate: float = 0.01      # Noise floor estimation
    spectral_floor: float = 0.1            # 0.0-1.0, spectral floor for artifact reduction
    smoothing_factor: float = 0.8          # 0.0-1.0, temporal smoothing
    overlap_factor: float = 0.75           # 0.5-0.875, FFT overlap factor
```

#### Algorithm Modes
- **MINIMAL**: Light spectral processing with minimal artifacts
- **SPECTRAL_SUBTRACTION**: Classic spectral subtraction algorithm
- **WIENER_FILTER**: Wiener filtering for optimal noise reduction
- **ADAPTIVE**: Adaptive algorithm selection based on content analysis

---

### Stage 7: Conventional Denoising ✅ NEW
**Purpose**: Time-domain denoising with various traditional filters
**Library**: SciPy.signal, SciPy.ndimage, PyWavelets
**Processing Location**: `src/audio/stages/conventional_denoising_stage.py`
**Performance Target**: 8.0ms (max 15.0ms)

#### Implementation Details
- **Multiple Filter Types**: Median, Gaussian, bilateral, wavelet, RNR filters
- **Time-domain Processing**: Fast processing without FFT overhead
- **Adaptive Parameters**: Self-adjusting filter parameters based on content
- **Edge Preservation**: Maintains speech transients and detail

#### Parameters
```python
@dataclass
class ConventionalDenoisingConfig:
    enabled: bool = False                   # Enable/disable conventional denoising (optional)
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    mode: ConventionalDenoisingMode = ConventionalDenoisingMode.MEDIAN_FILTER
    filter_strength: float = 0.3           # 0.0-1.0, filter strength
    preserve_transients: bool = True        # Preserve speech transients
    adaptation_rate: float = 0.1           # 0.0-1.0, parameter adaptation rate
```

#### Filter Types
- **MEDIAN_FILTER**: Median filtering for impulse noise removal
- **GAUSSIAN_FILTER**: Gaussian smoothing for general noise reduction
- **BILATERAL_FILTER**: Edge-preserving bilateral filtering
- **WAVELET_DENOISING**: Wavelet-based denoising with soft thresholding

---

### Stage 8: LUFS Normalization ✅ NEW
**Purpose**: Professional loudness normalization according to broadcast standards
**Library**: ITU-R BS.1770-4 implementation, NumPy, SciPy
**Processing Location**: `src/audio/stages/lufs_normalization_stage.py`
**Performance Target**: 18.0ms (max 30.0ms)

#### Implementation Details
- **ITU-R BS.1770-4 Compliance**: Official broadcast loudness standard
- **K-weighting Filter**: Psychoacoustic frequency weighting
- **Gating Algorithm**: Relative gating for accurate loudness measurement
- **Multiple Targets**: Support for various broadcast standards

#### Parameters
```python
@dataclass
class LUFSNormalizationConfig:
    enabled: bool = False                   # Enable/disable LUFS normalization (optional)
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    target_lufs: float = -23.0             # LUFS, target loudness level
    max_adjustment: float = 12.0           # dB, maximum adjustment allowed
    gating_threshold: float = -70.0        # dB, relative gating threshold
    measurement_window: float = 3.0        # seconds, loudness measurement window
    adaptation_rate: float = 0.1           # 0.0-1.0, adaptation rate for target tracking
```

#### Broadcast Standards
- **EBU R128**: -23 LUFS (European Broadcasting Union)
- **ATSC A/85**: -24 LUFS (US broadcast standard)
- **Spotify**: -14 LUFS (music streaming)
- **YouTube**: -14 LUFS (video platform)
- **Custom**: User-defined target LUFS value

---

### Stage 9: Auto Gain Control (AGC) ✅
**Purpose**: Maintain consistent audio levels automatically
**Library**: NumPy, custom DSP algorithms
**Processing Location**: `src/audio/stages/agc_stage.py`
**Performance Target**: 12.0ms (max 20.0ms)

#### Implementation Details
- **Adaptive Level Control**: Multiple control modes (Fast, Medium, Slow, Adaptive)
- **Noise Gating**: Prevents processing of low-level background noise
- **Lookahead Peak Detection**: Anticipates level changes for smooth control
- **Attack/Release/Hold**: Professional time constants for natural response

#### Parameters
```python
@dataclass
class AGCConfig:
    enabled: bool = True                    # Enable/disable AGC
    mode: AGCMode = AGCMode.MEDIUM         # DISABLED, FAST, MEDIUM, SLOW, ADAPTIVE
    target_level: float = -18.0            # dB, target output level
    max_gain: float = 12.0                 # dB, maximum gain allowed
    min_gain: float = -12.0                # dB, minimum gain allowed
    attack_time: float = 10.0              # ms, attack time constant
    release_time: float = 100.0            # ms, release time constant
    hold_time: float = 50.0                # ms, hold time after gain reduction
    knee_width: float = 2.0                # dB, soft knee width
    lookahead_time: float = 5.0            # ms, lookahead time
    adaptation_rate: float = 0.1           # 0.0-1.0, adaptation rate
    noise_gate_threshold: float = -60.0    # dB, noise gate threshold
```

#### Frontend Display Requirements
- **Level Meters**: Input/output level meters with target level indicator
- **Gain Reduction Meter**: Real-time gain adjustment visualization
- **Mode Indicator**: Current AGC mode and adaptation status
- **Threshold Display**: Noise gate threshold and target level markers
- **Time Constant Controls**: Attack/release/hold time adjustments
- **Lookahead Buffer**: Lookahead processing status

#### Latency Impact
- **Target**: < 12ms per chunk
- **Factors**: Lookahead buffer size, RMS calculation window, adaptation algorithm
- **Optimization**: Efficient level detection, minimal buffer delays

---

### Stage 10: Dynamic Range Compression ✅
**Purpose**: Control audio dynamics for consistent whisper processing
**Library**: NumPy mathematical operations
**Processing Location**: `src/audio/stages/compression_stage.py`
**Performance Target**: 8.0ms (max 15.0ms)

#### Implementation Details
- **Soft/Hard Knee Compression**: Smooth or abrupt compression onset
- **Variable Ratio**: Adjustable compression ratio (1:1 to 20:1)
- **Attack/Release Control**: Timing control for natural dynamics
- **Makeup Gain**: Automatic level compensation

#### Parameters
```python
@dataclass
class CompressionConfig:
    enabled: bool = True                    # Enable/disable compression
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    mode: CompressionMode = CompressionMode.SOFT_KNEE  # SOFT_KNEE, HARD_KNEE, ADAPTIVE, VOICE_OPTIMIZED
    threshold: float = -20                  # dB, compression threshold
    ratio: float = 3.0                     # 1.0-20.0, compression ratio
    knee: float = 2.0                      # dB, knee width
    attack_time: float = 5.0               # ms, attack time
    release_time: float = 100.0            # ms, release time
    makeup_gain: float = 0.0               # dB, makeup gain
    lookahead: float = 5.0                 # ms, lookahead time
```

#### Frontend Display Requirements
- **Compression Curve**: Threshold, ratio, and knee visualization
- **Gain Reduction Meter**: Real-time compression amount
- **Input/Output Meters**: Before/after level comparison
- **Dynamics Display**: Dynamic range visualization
- **Timing Controls**: Attack/release time adjustment

#### Latency Impact
- **Target**: < 8ms per chunk
- **Factors**: Lookahead buffer, envelope detection, gain smoothing
- **Optimization**: Efficient envelope follower, vectorized gain application

---

### Stage 11: Final Limiting ✅
**Purpose**: Prevent digital clipping and ensure consistent peak levels
**Library**: NumPy with custom limiting algorithms
**Processing Location**: `src/audio/stages/limiter_stage.py`
**Performance Target**: 6.0ms (max 12.0ms)

#### Implementation Details
- **Brick Wall Limiting**: Absolute peak level control
- **Soft Clipping**: Harmonic saturation instead of hard clipping
- **Lookahead Delay**: Transparent peak detection and control
- **Release Control**: Smooth gain recovery

#### Parameters
```python
@dataclass
class LimiterConfig:
    enabled: bool = True                    # Enable/disable limiting
    gain_in: float = 0.0                   # dB - Input gain adjustment (-20 to +20)
    gain_out: float = 0.0                  # dB - Output gain adjustment (-20 to +20)
    threshold: float = -1.0                 # dB, limiting threshold
    release_time: float = 50.0              # ms, release time
    lookahead: float = 5.0                  # ms, lookahead time
    soft_clip: bool = True                  # Enable soft clipping
```

#### Frontend Display Requirements
- **Peak Meter**: Real-time peak level monitoring
- **Limiting Indicator**: Active limiting status
- **Clip Detection**: Digital clipping warning
- **Lookahead Buffer**: Delay compensation display
- **Soft Clipping Indicator**: Harmonic saturation visualization

#### Latency Impact
- **Target**: < 6ms per chunk
- **Factors**: Lookahead buffer size, peak detection algorithm
- **Optimization**: Efficient peak detection, minimal delay buffer

---

## Real-time Monitoring

### Performance Metrics Structure
```python
processing_metadata = {
    "total_processing_time_ms": 0.0,        # Total pipeline latency
    "stage_timings": {                      # Per-instance timing breakdown
        "vad_main": 0.0,                    # VAD instance processing time (ms)
        "voice_filter_main": 0.0,           # Voice filter instance time (ms)  
        "nr_light": 0.0,                    # Light noise reduction time (ms)
        "eq_voice": 0.0,                    # Voice EQ instance time (ms)
        "nr_aggressive": 0.0,               # Aggressive noise reduction time (ms)
        "voice_enhancement_main": 0.0,      # Voice enhancement time (ms)
        "eq_broadcast": 0.0,                # Broadcast EQ instance time (ms)
        "agc_main": 0.0,                    # AGC instance time (ms)
        "compression_main": 0.0,            # Compression instance time (ms)
        "limiter_final": 0.0                # Final limiter time (ms)
    },
    "stages_applied": [],                   # List of enabled stages
    "quality_metrics": {                    # Audio quality analysis
        "input_quality": 0.0,               # Input quality score (0-1)
        "output_quality": 0.0,              # Output quality score (0-1)
        "quality_improvement": 0.0,         # Quality delta
        "snr_improvement": 0.0,             # SNR improvement (dB)
        "dynamic_range": 0.0                # Dynamic range (dB)
    },
    "vad_result": {                         # VAD-specific results
        "voice_detected": True,             # Voice detection status
        "confidence": 0.0                   # Detection confidence (0-1)
    },
    "bypassed": False,                      # Processing bypass status
    "bypass_reason": None                   # Bypass reason (if applicable)
}
```

### Frontend Real-time Display Requirements

#### 1. Pipeline Overview Dashboard
- **Processing Time Graph**: Real-time total processing latency
- **Stage Breakdown**: Per-stage timing visualization
- **Quality Metrics**: Input/output quality trending
- **Bypass Status**: Processing bypass indicators

#### 2. Individual Stage Monitors
- **VAD Monitor**: Voice detection status, confidence meter
- **Filter Monitor**: Frequency response, gain visualization
- **Noise Reduction Monitor**: Noise floor, reduction amount
- **Enhancement Monitor**: Clarity, presence, warmth, brightness meters
- **AGC Monitor**: Level meters, gain reduction, mode status
- **Compression Monitor**: Gain reduction, dynamics visualization
- **Limiter Monitor**: Peak levels, limiting activity

#### 3. Performance Alerts
- **Latency Warnings**: When processing time exceeds targets
- **Quality Alerts**: When quality metrics drop below thresholds
- **Bypass Notifications**: When processing is bypassed
- **Error Indicators**: Processing failures or recovery

### WebSocket Integration
```javascript
// Real-time processing updates via WebSocket
const processingUpdate = {
    type: "audio:processing_update",
    data: {
        timestamp: Date.now(),
        processing_metadata: { /* metadata object */ },
        audio_chunk_id: "chunk_123",
        session_id: "session_456"
    }
};

// Frontend subscription
websocket.send(JSON.stringify({
    type: "subscribe",
    channels: ["audio:processing_updates", "audio:quality_metrics"]
}));
```

---

## API Endpoints

### Processing Control
```http
POST /api/audio/process
Content-Type: application/json

{
    "audio_data": "base64_encoded_audio",
    "config": { /* AudioProcessingConfig */ },
    "enable_monitoring": true,
    "return_metadata": true
}
```

### Configuration Management
```http
GET /api/audio/config                    # Get current configuration
POST /api/audio/config                   # Update configuration
GET /api/audio/config/schema             # Get configuration schema
POST /api/audio/config/preset/{name}     # Apply preset configuration
```

### Monitoring Endpoints
```http
GET /api/audio/metrics                   # Get processing metrics
GET /api/audio/metrics/latency          # Get latency statistics
GET /api/audio/metrics/quality          # Get quality metrics
POST /api/audio/metrics/reset           # Reset metrics counters
```

### Stage-specific Endpoints
```http
POST /api/audio/process/vad             # Process only VAD stage
POST /api/audio/process/filter          # Process only voice filter stage
POST /api/audio/process/noise_reduction # Process only noise reduction stage
POST /api/audio/process/enhancement     # Process only voice enhancement stage
POST /api/audio/process/agc             # Process only AGC stage
POST /api/audio/process/compression     # Process only compression stage
POST /api/audio/process/limiter         # Process only limiter stage
```

---

## Performance Targets

### Latency Targets (per chunk)
- **VAD**: < 5ms
- **Voice Filter**: < 8ms
- **Noise Reduction**: < 15ms
- **Voice Enhancement**: < 10ms
- **AGC**: < 12ms
- **Compression**: < 8ms
- **Limiter**: < 6ms
- **Total Pipeline**: < 65ms

### Quality Targets
- **SNR Improvement**: > 3dB
- **Quality Score**: > 0.8 (0-1 scale)
- **Dynamic Range**: 30-60dB
- **Frequency Response**: ±3dB (85Hz-8kHz)

### Throughput Targets
- **Real-time Factor**: < 0.1 (10% of real-time)
- **Concurrent Streams**: 100+ simultaneous
- **CPU Usage**: < 50% per stream
- **Memory Usage**: < 100MB per stream

---

## Implementation Status

### ✅ Completed
- [x] VAD Implementation
- [x] Voice Frequency Filtering
- [x] Noise Reduction
- [x] Voice Enhancement
- [x] Auto Gain Control (AGC)
- [x] Dynamic Range Compression
- [x] Final Limiting
- [x] Configuration System
- [x] Basic Monitoring

### 🔄 In Progress
- [ ] Real-time latency monitoring
- [ ] Comprehensive documentation
- [ ] Frontend integration
- [ ] Performance optimization

### 📋 Completed ✅
- [x] Equalizer stage ✅ COMPLETED (5-band parametric with professional presets)
- [x] Spectral denoising ✅ COMPLETED (4 algorithms: minimal, spectral subtraction, Wiener, adaptive)
- [x] Conventional denoising ✅ COMPLETED (6 filters: median, Gaussian, bilateral, adaptive, RNR, wavelet)
- [x] Pre/post gain controls ✅ COMPLETED (Individual gain controls per stage: -20dB to +20dB)
- [x] LUFS normalization ✅ COMPLETED (ITU-R BS.1770-4 compliance with K-weighting)
- [x] FFT analysis endpoint ✅ COMPLETED (POST /api/audio/analyze/fft)
- [x] LUFS metering endpoint ✅ COMPLETED (POST /api/audio/analyze/lufs)
- [x] Individual stage processing endpoints ✅ COMPLETED (POST /api/audio/process/stage/{stage_name})
- [x] Preset management API endpoints ✅ COMPLETED (7 built-in presets with comparison)

---

## 🎯 Advanced Audio Analysis APIs ✅ NEW

The orchestration service provides comprehensive audio analysis endpoints for professional audio processing and quality assessment.

### FFT Analysis Endpoint
**Endpoint**: `POST /api/audio/analyze/fft`
**Purpose**: Real-time frequency domain analysis with professional features

#### Features
- **Comprehensive Spectral Analysis**: Full frequency spectrum with peak detection
- **Professional Metrics**: THD, spectral centroid, bandwidth, rolloff, zero-crossing rate
- **Peak Detection**: Automatic identification of dominant frequencies
- **Real-time Processing**: Optimized for real-time audio analysis

#### Response Data
```json
{
  "frequencies": [0, 10.77, 21.53, ...],     // Frequency bins (Hz)
  "magnitudes": [0.001, 0.023, 0.045, ...], // Magnitude spectrum
  "phases": [0.12, -1.45, 2.34, ...],       // Phase spectrum  
  "peaks": {
    "frequencies": [150.5, 1240.3, 3420.1], // Peak frequencies
    "magnitudes": [0.45, 0.32, 0.28]        // Peak magnitudes
  },
  "spectral_features": {
    "centroid": 1247.8,                     // Spectral centroid (Hz)
    "bandwidth": 890.2,                     // Spectral bandwidth (Hz)
    "rolloff": 3456.7,                      // Spectral rolloff (Hz)
    "zero_crossing_rate": 0.023,            // Zero-crossing rate
    "spectral_flatness": 0.34,             // Spectral flatness measure
    "total_harmonic_distortion": 0.012      // THD percentage
  },
  "processing_time_ms": 8.4                 // Analysis time
}
```

### LUFS Metering Endpoint
**Endpoint**: `POST /api/audio/analyze/lufs`
**Purpose**: Professional loudness measurement according to broadcast standards

#### Features
- **ITU-R BS.1770-4 Compliance**: Official broadcast loudness standard
- **K-weighting Filter**: Psychoacoustic frequency weighting
- **Gating Algorithm**: Relative gating for accurate measurement
- **Multiple Standards**: Support for EBU R128, ATSC A/85, streaming platforms

#### Response Data
```json
{
  "integrated_lufs": -18.4,               // Integrated loudness (LUFS)
  "momentary_lufs": -16.8,               // Momentary loudness (LUFS)
  "short_term_lufs": -17.2,              // Short-term loudness (LUFS)
  "loudness_range": 3.4,                 // Loudness range (LU)
  "true_peak_dbtp": -2.1,                // True peak level (dBTP)
  "gating_block_count": 342,             // Number of gating blocks
  "relative_threshold": -28.4,           // Relative gating threshold (LUFS)
  "compliance": {
    "ebu_r128": {"target": -23, "compliant": true, "offset": 4.6},
    "atsc_a85": {"target": -24, "compliant": true, "offset": 5.6},
    "spotify": {"target": -14, "compliant": false, "offset": -4.4}
  },
  "processing_time_ms": 12.1             // Analysis time
}
```

### Individual Stage Processing
**Endpoint**: `POST /api/audio/process/stage/{stage_name}`
**Purpose**: Process audio through individual stages for testing and analysis

#### Supported Stages
- `vad` - Voice Activity Detection
- `voice_filter` - Voice Frequency Filtering  
- `noise_reduction` - Spectral Noise Reduction
- `voice_enhancement` - Voice Enhancement
- `equalizer` - Parametric Equalizer
- `spectral_denoising` - Advanced Spectral Denoising
- `conventional_denoising` - Time-domain Denoising
- `lufs_normalization` - LUFS Loudness Normalization
- `agc` - Auto Gain Control
- `compression` - Dynamic Range Compression
- `limiter` - Peak Limiting

#### Response Data
```json
{
  "processed_audio": "base64_encoded_audio_data",
  "processing_metadata": {
    "stage_name": "equalizer",
    "processing_time_ms": 12.3,
    "parameters_used": {...},
    "quality_metrics": {
      "input_rms": 0.023,
      "output_rms": 0.031,
      "level_change_db": 2.8,
      "estimated_snr_db": 18.4
    }
  }
}
```

## 📁 Professional Preset Management System ✅ NEW

Complete preset management with built-in professional configurations and custom preset support.

### Built-in Presets (7 Professional Configurations)

#### 1. **Voice Optimized** (`voice_optimized`)
- **Use Case**: Speech recognition optimization, podcasts, voiceovers
- **Pipeline**: VAD + Voice Filter + Light NR + Voice EQ + AGC + Soft Compression
- **Characteristics**: Enhanced clarity, presence boost, minimal artifacts

#### 2. **Broadcast Quality** (`broadcast_quality`)  
- **Use Case**: Radio broadcast, professional streaming, live audio
- **Pipeline**: VAD + Voice Filter + Moderate NR + Broadcast EQ + AGC + Broadcast Compression + Limiter
- **Characteristics**: Industry-standard broadcast processing chain

#### 3. **Conference Call** (`conference_call`)
- **Use Case**: Video conferencing, meeting recordings, telephony
- **Pipeline**: Fast VAD + Telephony EQ + Aggressive NR + Fast AGC + Hard Compression
- **Characteristics**: Optimized for poor network conditions, maximum intelligibility

#### 4. **Noisy Environment** (`noisy_environment`)
- **Use Case**: Construction sites, traffic, crowded spaces
- **Pipeline**: Aggressive VAD + Heavy NR + Spectral Denoising + Voice EQ + Strong AGC + Limiting
- **Characteristics**: Maximum noise reduction, voice protection

#### 5. **Music Content** (`music_content`)
- **Use Case**: Music with vocals, singer-songwriter content, acoustic performances
- **Pipeline**: Music VAD + Flat EQ + Minimal NR + Stereo Enhancement + Gentle Compression
- **Characteristics**: Preserves musical content, natural dynamics

#### 6. **Minimal Processing** (`minimal_processing`)
- **Use Case**: High-quality studio recordings, clean speech, testing
- **Pipeline**: Basic VAD + Gentle EQ + Light Limiting
- **Characteristics**: Transparent processing, minimal coloration

#### 7. **High Quality** (`high_quality`)
- **Use Case**: Professional productions, archival recordings, mastering
- **Pipeline**: All stages optimized for maximum quality with conservative settings
- **Characteristics**: Best possible quality, latency not prioritized

### Preset Management API Endpoints

#### Load Preset
```bash
GET /api/audio/presets/{preset_name}
```

#### Save Custom Preset
```bash
POST /api/audio/presets/save
{
  "name": "my_custom_preset",
  "description": "Custom preset for my specific use case",
  "pipeline_config": {...}
}
```

#### List Available Presets
```bash
GET /api/audio/presets/list
```

#### Compare Presets
```bash
POST /api/audio/presets/compare
{
  "preset1": "voice_optimized",
  "preset2": "broadcast_quality", 
  "audio_sample": "base64_encoded_audio"
}
```

#### Delete Custom Preset
```bash
DELETE /api/audio/presets/{preset_name}
```

### Preset Comparison System

The preset comparison system allows A/B testing between different configurations:

```json
{
  "comparison_id": "comp_12345",
  "preset_results": {
    "voice_optimized": {
      "processed_audio": "base64_data_1",
      "quality_scores": {
        "speech_clarity": 8.7,
        "noise_reduction": 7.2,
        "naturalness": 9.1,
        "intelligibility": 8.9
      },
      "processing_time_ms": 45.2
    },
    "broadcast_quality": {
      "processed_audio": "base64_data_2", 
      "quality_scores": {
        "speech_clarity": 9.2,
        "noise_reduction": 8.8,
        "naturalness": 8.4,
        "intelligibility": 9.3
      },
      "processing_time_ms": 52.1
    }
  },
  "recommendation": "broadcast_quality",
  "recommendation_reason": "Higher overall scores for professional use"
}
```

---

## Modular System Benefits

### Professional Audio Production Flexibility

The fully modular architecture provides unprecedented flexibility for creating custom audio processing chains:

#### Complex Processing Chains
```
Input → VAD → Light NR → Voice EQ → Heavy NR → Broadcast EQ → AGC → Compressor → Final EQ → Limiter
```

Each stage instance can be:
- **Configured independently** with its own parameters
- **Gained individually** for precise level control  
- **Enabled/disabled** for A/B testing
- **Moved anywhere** in the pipeline
- **Duplicated** for parallel processing paths

#### Real-World Use Cases

**Broadcasting Chain**:
```
"vad_gate" → "eq_voice_enhance" → "nr_aggressive" → "eq_broadcast" → "agc_slow" → "comp_broadcast" → "limiter_final"
```

**Podcast Processing**:
```  
"vad_sensitive" → "nr_light" → "eq_warmth" → "voice_enhance" → "comp_gentle" → "eq_presence" → "limiter_transparent"
```

**Conference Call Optimization**:
```
"vad_fast" → "nr_adaptive" → "eq_telephony" → "agc_fast" → "comp_aggressive" → "limiter_hard"  
```

**Music Content Processing**:
```
"vad_music" → "eq_flat" → "nr_minimal" → "enhance_stereo" → "comp_multiband" → "eq_master" → "limiter_musical"
```

#### Frontend Implementation Benefits

**Drag-and-Drop Pipeline Editor**:
- Visual pipeline representation with stage flow
- Instant stage addition from component palette  
- Real-time parameter updates with audio preview
- Performance monitoring per instance

**Professional Mixing Console Experience**:
- Individual gain controls (-20dB to +20dB) per stage
- Real-time level meters and spectrum analyzers
- Preset management with instant recall
- A/B testing between different configurations

**Configuration Management**:
- Save complete processing chains as presets
- Export/import pipeline configurations
- Version control for different use cases
- Template library for common scenarios

### Technical Excellence

**Performance Optimized**:
- Each stage processes only when enabled
- Gain stages bypass when set to 0dB
- Efficient memory management per instance
- Real-time latency monitoring with targets

**Professional Quality**:
- Broadcast-grade algorithms in every stage
- Proper dB to linear conversions throughout
- Anti-aliasing and phase coherence maintained
- Professional parameter ranges and validation

**Scalable Architecture**:
- Unlimited stage instances supported
- Dynamic pipeline reconfiguration
- Hot-swappable configurations without dropouts
- Database-backed performance analytics

This modular audio processing system provides the **ultimate flexibility** for optimizing speech recognition across any acoustic environment, with the precision and control of professional audio production tools.

---

*Last Updated: 2025-07-15*  
*Version: 2.0 - Modular Architecture*  
*Orchestration Service - Modular Audio Processing Pipeline, 
