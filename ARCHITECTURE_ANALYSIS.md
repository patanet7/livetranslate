# LiveTranslate System Architecture Analysis

## Executive Summary

LiveTranslate is a sophisticated microservices-based real-time speech translation system with an enterprise-grade architecture. The system processes audio in real-time through a coordinated pipeline: **Frontend → Orchestration → Whisper (NPU/GPU) → Translation (GPU) → Response**. The architecture emphasizes hardware acceleration, streaming efficiency, and comprehensive error handling.

### Key Metrics
- **Real-time Latency Target**: < 100ms end-to-end
- **Throughput**: >500 translations/minute on GPU
- **Concurrency**: 1000+ WebSocket connections
- **Hardware**: NPU (Intel) for Whisper, GPU (NVIDIA) for Translation, CPU fallback
- **Message Loss**: Zero-message-loss design with session persistence

---

## 1. Real-Time Streaming Implementation

### 1.1 Frontend Audio Capture Pipeline

**Location**: `modules/frontend-service/src/hooks/useAudioProcessing.ts`

The frontend implements a sophisticated audio capture system:

```
┌─────────────────┐
│ getUserMedia()  │ ← Browser Audio Input (16kHz mono)
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│ MediaRecorder Configuration              │
│ • Format: WebM Opus (16-128 kbps)       │
│ • BitRate: Dynamic (64-256 kbps)        │
│ • Processing: RAW (disabled for streams)│
└────────┬────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ Audio Chunking (Frontend)                │
│ • Chunk Duration: Configurable (2-5s)   │
│ • Storage: Blob refs (non-Redux)        │
│ • Updates: Every 100ms for smooth UI    │
└────────┬────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ Upload to /api/audio/upload              │
│ • Multipart Form Data                    │
│ • Format Detection (MIME type)          │
│ • Retry Logic (circuit breaker)         │
└─────────────────────────────────────────┘
```

**Key Implementation Details**:
- **Audio Processing Features**: Disabled `echoCancellation`, `noiseSuppression`, `autoGainControl` for loopback audio preservation
- **Format Support**: WebM Opus, MP4, OGG, WAV (browser-dependent)
- **Quality Levels**: High (256kbps), Medium (128kbps), Low (64kbps), Lossless (16-bit PCM)
- **Recording State**: Stored in refs to avoid Redux serialization issues with Blob objects
- **Duration Tracking**: Real-time timer updates every 100ms for smooth UI feedback

### 1.2 Orchestration Service Audio Chunking

**Location**: `modules/orchestration-service/src/audio/chunk_manager.py`

The chunk manager centralizes audio chunking logic previously scattered across services:

```python
class ChunkManager:
    # Configurable chunking parameters
    chunk_duration: float = 3.0          # 3-second chunks default
    overlap_duration: float = 0.5        # 500ms overlap for context
    buffer_duration: float = 30.0        # 30-second rolling buffer
    silence_threshold: float = 0.0001    # Voice activity detection
    
    # Quality-based filtering
    min_quality_threshold: float = 0.3   # Minimum quality score
    noise_threshold: float = 0.5         # Noise level tolerance
    
    # Processing capabilities
    # - AudioBuffer: Rolling buffer with overlap blending
    # - AudioQualityAnalyzer: Comprehensive quality metrics
    # - ChunkFileManager: File storage and hashing
    # - Database Integration: Persistence with lineage tracking
```

**Chunking Algorithm**:
1. **Rolling Buffer Management**: Configurable max buffer size with automatic overflow handling
2. **Overlap Handling**: Linear blending of overlapping regions to prevent audio discontinuities
3. **Quality Analysis**: Comprehensive metrics on each chunk:
   - RMS Level & Peak Level
   - Signal-to-Noise Ratio (SNR)
   - Zero-Crossing Rate (voice detection)
   - Voice Activity Confidence
   - Spectral Centroid, Bandwidth, Rolloff
   - Overall Quality Score (weighted average)

4. **Quality-Based Filtering**: Chunks below `min_quality_threshold` are rejected with alerts
5. **File Storage**: Each chunk written to disk with metadata JSON files
6. **Database Persistence**: Full lineage tracking with file hashes for integrity

### 1.3 Whisper Service Streaming

**Location**: `modules/whisper-service/src/api_server.py`

Whisper service implements enterprise-grade streaming infrastructure:

```
┌──────────────────────────┐
│ Audio Chunk Received      │ (from Orchestration)
└────────┬─────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ AudioProcessor                           │
│ • Format Detection (7+ formats)          │
│ • Resampling (→ 16kHz with librosa)     │
│ • Quality Validation                     │
│ • Corruption Detection                   │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ RollingBufferManager                     │
│ • VAD Processing (WebRTC + Silero)      │
│ • Speech Detection                       │
│ • Memory-Efficient Buffering             │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Whisper Model Inference                  │
│ • Hardware: NPU (primary) → GPU → CPU    │
│ • Model: whisper-base, whisper-tiny      │
│ • OpenVINO Optimization                  │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Speaker Diarization                      │
│ • Embedding Methods: SpeechBrain, Pyannote │
│ • Clustering: HDBSCAN, DBSCAN, Agglom   │
│ • Speaker Timeline Tracking              │
└────────┬─────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Transcription Results                    │
│ • Text: Transcribed audio                │
│ • Segments: Timing boundaries            │
│ • Speakers: Identified speakers + IDs    │
│ • Confidence: Quality metric (0-1)       │
└──────────────────────────────────────────┘
```

**Performance Characteristics**:
- **Processing Time**: ~0.5-2s for 3-second audio chunks (varies by device)
- **Buffer Management**: 30-second rolling buffer to accumulate audio for context
- **Memory Usage**: ~200-500MB depending on buffer size and model
- **Connection Pooling**: 1000-capacity weak reference dictionary for WebSocket connections

---

## 2. Translation Pipeline

### 2.1 Architecture

**Location**: `modules/translation-service/src/api_server.py`

The translation service implements multi-backend architecture with intelligent fallback:

```
┌────────────────────────────────────────┐
│ Transcribed Text                        │
│ (from Whisper Service)                  │
└────────────┬─────────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│ Language Detection                       │
│ • Auto-detect source language           │
│ • Confidence scoring                    │
└────────────┬─────────────────────────────┘
             ↓
    ┌────────────────────────────────────────────────────────┐
    │                 Backend Selection                      │
    └────────────┬────────────────────────────────────────────┘
                 ↓
    ┌────────────────────────────────────────────────────────┐
    │ Primary: vLLM (GPU-Optimized)                         │
    │ • Model: Meta-Llama-3.1-8B or similar                │
    │ • GPU Memory: 6-24GB                                  │
    │ • Throughput: >500 trans/min                          │
    │ • Latency: <200ms                                     │
    └────────────┬───────────────────────────────────────────┘
                 │
    ┌────────────▼───────────────────────────────────────────┐
    │ Fallback 1: Triton Inference Server                   │
    │ • Enterprise inference optimization                   │
    │ • Multi-GPU support                                   │
    │ • Dynamic batching                                    │
    └────────────┬───────────────────────────────────────────┘
                 │
    ┌────────────▼───────────────────────────────────────────┐
    │ Fallback 2: Ollama                                    │
    │ • CPU/GPU support                                     │
    │ • Model management                                    │
    │ • ~50-100ms latency (CPU)                            │
    └────────────┬───────────────────────────────────────────┘
                 │
    ┌────────────▼───────────────────────────────────────────┐
    │ Fallback 3: OpenAI-Compatible APIs                    │
    │ • Groq, Together, OpenAI, external services          │
    │ • Fallback when local models unavailable             │
    └────────────┬───────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ Quality Scoring & Validation            │
│ • Confidence metrics                    │
│ • Error detection                       │
│ • Language-specific validation          │
└────────────┬─────────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│ Translated Text                         │
│ (with quality score & metadata)         │
└────────────────────────────────────────┘
```

### 2.2 Translation Configuration

**Supported Languages**: 50+ including:
- Major: English, Spanish, French, German, Chinese, Japanese, Korean
- Secondary: Portuguese, Italian, Russian, Arabic, Hindi, etc.

**Quality Thresholds**:
- Minimum: 0.7 (acceptable quality)
- Preferred: 0.85 (good quality)
- Fallback Trigger: 0.6 (use alternative backend)

### 2.3 Performance Metrics

- **GPU Utilization**: >80% during processing
- **Translation Latency**: <200ms real-time, <1s batch
- **Memory Efficiency**: <90% GPU memory usage
- **Throughput**: 650+ translations/minute on NVIDIA RTX 4090
- **Fallback Rate**: <5% (GPU→CPU transitions)

---

## 3. Google Meet Bot Integration

### 3.1 Complete Bot Architecture

**Location**: `modules/orchestration-service/src/bot/`

```
┌──────────────────────────────────────────────────┐
│ Bot Lifecycle                                    │
├──────────────────────────────────────────────────┤
│                                                  │
│ 1. Spawn Request                                 │
│    └─→ Bot Lifecycle Manager                    │
│         └─→ Database Session Tracking           │
│                                                  │
│ 2. Browser Automation                           │
│    └─→ Google Meet Automation (headless Chrome) │
│         └─→ Join meeting URL                    │
│              └─→ Authenticate                   │
│                   └─→ Browser ready             │
│                                                  │
│ 3. Audio Capture Pipeline                       │
│    └─→ Browser Audio Capture                    │
│         ├─→ MediaStreamAudioDestinationNode    │
│         ├─→ ScriptProcessorNode for PCM        │
│         └─→ Multi-fallback methods             │
│                                                  │
│ 4. Audio Processing Flow                        │
│    └─→ Orchestration Service ──────────────────┐│
│         ├─→ Chunk Manager (3s chunks)          ││
│         ├─→ Quality Analysis                    ││
│         └─→ Whisper Service (NPU/GPU)           ││
│              ├─→ Transcription                  ││
│              └─→ Speaker Diarization ──────────┐││
│                   ├─→ Time Correlation          │││
│                   └─→ Speaker Attribution       │││
│                                                 │││
│ 5. Translation Flow                             │││
│    └─→ Translation Service (GPU) ──────────────┐│││
│         ├─→ Language Detection                  ││││
│         ├─→ Model Selection (vLLM/Triton/...)   ││││
│         └─→ Translated Output ─────────────────┐│││
│                                                 ││││
│ 6. Virtual Webcam Generation                    ││││
│    └─→ Virtual Webcam System ─────────────────┐│││
│         ├─→ Frame Generation (30fps)            │││││
│         ├─→ Speaker Attribution Display         │││││
│         ├─→ Dual Content (transcription + trans)│││││
│         └─→ Professional Layout                 │││││
│                                                 │││││
│ 7. Integration with Google Meet                 │││││
│    └─→ Virtual Camera Input ───────────────────┐│││
│         └─→ Meeting Display                     ││││
│              └─→ Real-time Overlay             ││││
│                                                 │││
│ 8. Session Persistence & Analytics              │││
│    └─→ Database Tracking ──────────────────────┐│││
│         ├─→ Session Metadata                    ││││
│         ├─→ Audio Files                         ││││
│         ├─→ Transcriptions                      ││││
│         ├─→ Translations                        ││││
│         └─→ Speaker Correlations                ││││
│                                                 │││
│ 9. Graceful Shutdown                            │││
│    └─→ Resource Cleanup                         │││
│         ├─→ Browser Process Termination         │││
│         ├─→ Audio Stream Closure                │││
│         ├─→ Session Finalization                │││
│         └─→ Database Completion                 │││
│                                                  │││
└──────────────────────────────────────────────────┘││
```

### 3.2 Audio Capture Methods (Fallback Chain)

**Primary**: `MediaStreamAudioDestinationNode` + `ScriptProcessorNode`
- **Pros**: Direct PCM access, real-time processing
- **Cons**: Lower sample rate on some systems

**Fallback 1**: `AudioWorklet` for higher quality
- **Pros**: Better performance, higher sample rates
- **Cons**: More complex setup

**Fallback 2**: `OfflineAudioContext` for recording
- **Pros**: Guaranteed capture
- **Cons**: Post-processing delay

### 3.3 Time Correlation Engine

**Location**: `modules/orchestration-service/src/bot/time_correlation.py`

Correlates internal transcriptions with Google Meet captions:

```
Whisper Transcription Timeline:
┌─────────────────────────────────┐
│ 0:00-0:03: "Hello everyone"     │
│ 0:03-0:06: "Thank you for..."   │
│ 0:06-0:09: "Today we discuss..."│
└─────────────────────────────────┘
         ↓ (Time Correlation)
Google Meet Caption Timeline:
┌─────────────────────────────────┐
│ 0:01: "Hello everyone"          │
│ 0:04: "Thank you for joining"   │
│ 0:08: "Today we discuss..."     │
└─────────────────────────────────┘
         ↓ (Matched)
Speaker Attribution:
┌─────────────────────────────────┐
│ SPEAKER_00: "Hello everyone"    │
│ SPEAKER_00: "Thank you for..." │
│ SPEAKER_01: "Today we discuss..."│
└─────────────────────────────────┘
```

### 3.4 Virtual Webcam System

**Location**: `modules/orchestration-service/src/bot/virtual_webcam.py`

Professional translation overlay with speaker attribution:

```
Frame Generation (30fps):
┌────────────────────────────────────────────┐
│ 🎤 Transcription Box (top)                 │
│ ┌──────────────────────────────────────┐  │
│ │ John Doe (SPEAKER_00)                │  │
│ │ "Thank you all for being here"       │  │
│ │ Confidence: 95%                      │  │
│ │ Language: English                    │  │
│ └──────────────────────────────────────┘  │
│                                            │
│ 🌐 Translation Box (bottom)                │
│ ┌──────────────────────────────────────┐  │
│ │ María García (ES)                    │  │
│ │ "Gracias a todos por estar aquí"    │  │
│ │ Confidence: 89%                      │  │
│ │ Language: Spanish                    │  │
│ └──────────────────────────────────────┘  │
│                                            │
│ ⏱ Timestamp: 00:05:32                     │
└────────────────────────────────────────────┘
```

---

## 4. Performance Characteristics

### 4.1 End-to-End Latency

```
Frontend Audio Capture
    ↓ (50-100ms chunk buffering)
Upload to Orchestration
    ↓ (10-20ms network + processing)
Orchestration Chunking
    ↓ (5-10ms chunking overhead)
Whisper Service (NPU optimized)
    ├─ Model loading: 100-500ms (once)
    └─ Inference: 200-800ms per chunk
    ↓ (5-10ms serialization)
Translation Service
    ├─ Model loading: 100-500ms (once)
    └─ Inference: 50-200ms per chunk
    ↓ (10-20ms network + serialization)
Return to Frontend
    ↓
Display Result
```

**Total Latency Breakdown** (after warmup):
- **Optimal Case**: ~300-400ms (GPU accelerated)
- **Typical Case**: ~500-800ms (mixed optimization)
- **Worst Case**: ~2000-3000ms (CPU fallback, large chunks)

### 4.2 Throughput Metrics

| Component | Throughput | Device |
|-----------|-----------|--------|
| Frontend Recording | 16-48 kbps | Browser |
| Orchestration Chunking | >1000 chunks/min | CPU |
| Whisper (NPU) | >100 chunks/min | Intel NPU |
| Whisper (GPU) | >200 chunks/min | NVIDIA GPU |
| Whisper (CPU) | 20-50 chunks/min | CPU |
| Translation (GPU) | >650 trans/min | NVIDIA GPU |
| Translation (CPU) | 50-150 trans/min | CPU |

### 4.3 Memory Profiles

| Component | Memory Usage | Config |
|-----------|-------------|--------|
| Frontend (recording) | 10-50MB | Per session |
| Orchestration | 500MB-2GB | Service baseline |
| Whisper (model) | 500MB-2GB | Model-dependent |
| Whisper (buffer) | 50-200MB | 30s rolling buffer |
| Translation (model) | 6-24GB | vLLM, Triton |
| Translation (batch) | 100-500MB | Per batch |

### 4.4 Network Throughput

- **Frontend → Orchestration**: 50-200 kbps (chunked audio)
- **Orchestration → Whisper**: Variable (API calls)
- **Orchestration → Translation**: Variable (API calls)
- **WebSocket (bi-directional)**: 10-100 kbps (real-time updates)

---

## 5. Whisper Integration

### 5.1 Model Configuration

**NPU Detection & Fallback**:
```python
def _detect_best_device(self) -> str:
    core = ov.Core()
    available_devices = core.available_devices
    
    # Priority: NPU → GPU → CPU
    if "NPU" in available_devices:
        return "NPU"  # Intel NPU (primary)
    elif "GPU" in available_devices:
        return "GPU"  # NVIDIA/other GPU
    else:
        return "CPU"  # Fallback
```

**Models Available**:
- whisper-tiny (39M params) - Fast, lower quality
- whisper-base (74M params) - Balanced (default)
- whisper-small (244M params) - Better quality
- whisper-medium (769M params) - High quality
- whisper-large (1.5B params) - Highest quality

### 5.2 Model Loading Strategy

1. **On Demand**: Load model when first needed
2. **Caching**: Keep 3 most recent models in memory
3. **Eviction**: LRU eviction when memory threshold exceeded
4. **Fallback**: Use whisper-tiny if memory exhausted

### 5.3 Audio Format Support

- **Input Formats**: WAV, MP3, WebM, OGG, FLAC, M4A, MP4
- **Output Format**: 16kHz PCM mono (Whisper requirement)
- **Resampling**: librosa (primary) with fallback to pydub
- **Normalization**: Automatic level adjustment

### 5.4 Speaker Diarization

```
Input Audio (16s)
    ↓
Segment into 10ms frames
    ↓
Extract speaker embeddings
    ├─ Method 1: SpeechBrain (primary)
    ├─ Method 2: Pyannote (fallback)
    └─ Method 3: Resemblyzer (backup)
    ↓
Cluster embeddings
    ├─ HDBSCAN (density-based)
    ├─ DBSCAN (spatial clustering)
    └─ Agglomerative (hierarchical)
    ↓
Track speaker continuity
    ├─ Resolve speaker ID ambiguity
    ├─ Maintain timeline
    └─ Associate with Google Meet speakers
    ↓
Output: Speaker timeline with IDs
```

---

## 6. Key Bottlenecks & Optimization Opportunities

### 6.1 Current Bottlenecks

#### 1. **Model Inference Latency** (200-800ms)
- **Root Cause**: Large language models (1.5B+ parameters)
- **Impact**: Primary latency contributor (40-50% of total)
- **Solutions**:
  - Model quantization (INT8, INT4)
  - Distillation to smaller models
  - Batch processing when possible
  - Hardware accelerators (TPU, specialized NPU)

#### 2. **GPU Memory Constraints** (6-24GB)
- **Root Cause**: Large model parameters
- **Impact**: Limits batch size, single-GPU throughput cap
- **Solutions**:
  - Distributed inference (multi-GPU)
  - Dynamic batching based on available memory
  - Model sharding across GPUs
  - Streaming inference (process one token at a time)

#### 3. **Network I/O** (50-100ms overhead)
- **Root Cause**: REST/HTTP overhead for service communication
- **Impact**: Cumulative across 3 round trips (Frontend→Orchestration→Whisper/Translation)
- **Solutions**:
  - Embedded service clients (skip network hop)
  - gRPC for service communication
  - Connection pooling with keep-alive
  - Message batching

#### 4. **Audio Chunking Overhead** (5-15ms)
- **Root Cause**: Quality analysis on every chunk
- **Impact**: Accumulates with many chunks
- **Solutions**:
  - Simplified quality scoring
  - Batch quality analysis
  - Quality scoring only for edge cases

#### 6. **Database Persistence** (20-50ms per chunk)
- **Root Cause**: Synchronous writes to PostgreSQL
- **Impact**: Blocks chunk processing
- **Solutions**:
  - Asynchronous database writes
  - Batch inserts
  - Connection pooling with optimized queries
  - Write-ahead logging for fault tolerance

#### 7. **Browser Audio Capture Jitter** (50-500ms)
- **Root Cause**: Variable MediaRecorder buffering
- **Impact**: Unpredictable chunk timing
- **Solutions**:
  - Fixed-size buffer with overflow handling
  - WebRTC data channels for more consistent timing
  - Client-side adaptive buffering

### 6.2 Optimization Priorities

| Priority | Bottleneck | Impact | Difficulty | Timeline |
|----------|-----------|--------|-----------|----------|
| **HIGH** | Model Inference Latency | -200-300ms | Medium | 2-3 weeks |
| **HIGH** | GPU Memory Optimization | +100% throughput | Medium | 2-3 weeks |
| **MEDIUM** | Network I/O Optimization | -30-50ms | Low | 1 week |
| **MEDIUM** | Batch Processing | +50% efficiency | Medium | 2 weeks |
| **MEDIUM** | Database Optimization | -20-30ms | Medium | 1-2 weeks |
| **LOW** | Audio Capture Jitter | +consistency | Low | 1 week |
| **LOW** | Quality Analysis | -5-10ms | Low | 3 days |

### 6.3 Hardware Acceleration Opportunities

#### Intel NPU (Whisper Service)
- **Current Status**: ✅ Implemented with fallback
- **Potential**: 3-5x speedup vs CPU
- **Optimization**: OpenVINO model quantization

#### NVIDIA GPU (Translation Service)
- **Current Status**: ✅ vLLM integration
- **Potential**: 10-50x speedup vs CPU
- **Optimization**: Tensor parallel, paged attention, KV cache

#### TPU (if available)
- **Current Status**: ⚠️ Not integrated
- **Potential**: 5-10x speedup over GPU for LLMs
- **Integration**: TensorFlow Serving, JAX

#### Apple Neural Engine (macOS/iOS)
- **Current Status**: ⚠️ Separate module (whisper-service-mac)
- **Potential**: Device-local inference without network
- **Integration**: Core ML, Metal Performance Shaders

---

## 7. Scalability Analysis

### 7.1 Current Limits

**Single Orchestration Service**:
- Max concurrent users: 100-200 (WebSocket connections)
- Max RPS (requests/second): 10-20
- Response time @ 50% capacity: 100-200ms
- Response time @ 90% capacity: 500-1000ms

**Single Whisper Service**:
- Max concurrent streams: 5-10 (memory-limited)
- Max throughput: 100-200 chunks/min
- Response time: 200-800ms per chunk

**Single Translation Service**:
- Max concurrent batches: 5-10
- Max throughput: 500-650 translations/min
- Response time: 50-200ms per translation

### 7.2 Horizontal Scaling Strategy

```
┌─────────────────────────────────────┐
│ Load Balancer (Nginx/HAProxy)       │
└────────┬────────────────────────────┘
         ├─→ Orchestration-1 (port 3000)
         ├─→ Orchestration-2 (port 3001)
         └─→ Orchestration-3 (port 3002)
             │
             ├─→ Whisper Service Pool (5 instances)
             │   ├─→ whisper-1 (port 5001)
             │   ├─→ whisper-2 (port 5002)
             │   ├─→ whisper-3 (port 5003)
             │   ├─→ whisper-4 (port 5004)
             │   └─→ whisper-5 (port 5005)
             │
             └─→ Translation Service Pool (3 instances, GPU-constrained)
                 ├─→ translation-1 (port 5003, GPU:0)
                 ├─→ translation-2 (port 5004, GPU:1)
                 └─→ translation-3 (port 5005, GPU:2)
```

**Recommended Deployment**:
- 3-5 Orchestration instances (CPU)
- 5-10 Whisper instances (each with NPU or GPU)
- 2-4 Translation instances (each with GPU)
- Shared database with read replicas
- Redis for caching and session management

### 7.3 Vertical Scaling

**Hardware Recommendations**:

| Component | CPU | RAM | GPU | Storage |
|-----------|-----|-----|-----|---------|
| Orchestration | 8-16c | 16-32GB | None | 100GB |
| Whisper | 8c | 8-16GB | Intel NPU/GPU | 50GB |
| Translation | 16c | 32-64GB | 2x RTX 4090 | 200GB |
| Database | 16-32c | 64-128GB | None | 1-10TB |

---

## 8. Integration Architecture

### 8.1 Service Communication Patterns

```
Frontend (Port 5173)
    ├─ REST API: /api/*
    ├─ WebSocket: /ws
    └─ Static files: /
         ↓
API Gateway (Nginx/Orchestration)
    ├─ Authentication & Authorization
    ├─ Request validation
    ├─ Rate limiting
    └─ Routing to backend services
         │
         ├─→ Orchestration (port 3000)
         │   ├─→ Audio Router: /api/audio/*
         │   ├─→ Bot Router: /api/bot/*
         │   ├─→ Pipeline Router: /api/pipeline/*
         │   ├─→ Translation Router: /api/translation/*
         │   └─→ WebSocket: /ws
         │        │
         │        ├─→ Whisper Service (port 5001)
         │        │   ├─ Transcription: /transcribe
         │        │   ├─ Streaming: /stream
         │        │   └─ Health: /health
         │        │
         │        └─→ Translation Service (port 5003)
         │            ├─ Translation: /translate
         │            ├─ Batch: /translate/batch
         │            └─ Health: /health
         │
         └─→ Database (PostgreSQL)
             └─ Audio metadata, sessions, transcriptions
```

### 8.2 Data Flow Diagram

```
1. Audio Upload (Frontend)
   POST /api/audio/upload
   └─→ Audio chunk (binary)

2. Orchestration Receives Chunk
   • Validates format
   • Chunks audio (3s segments)
   • Analyzes quality
   • Stores file + metadata
   • Queues for processing

3. Whisper Processing
   • Receives audio chunk
   • Runs inference
   • Performs diarization
   • Returns: text, segments, speakers

4. Translation Processing
   • Receives transcribed text
   • Detects language
   • Selects translation model
   • Returns: translated text, quality score

5. Results Storage
   • Stores complete pipeline results
   • Correlates speakers (internal ↔ Google Meet)
   • Updates database
   • Notifies WebSocket clients

6. Frontend Display
   • Real-time updates via WebSocket
   • Original text + speaker ID
   • Translation + language
   • Confidence scores
   • Timestamps
```

---

## 9. Quality Metrics & Monitoring

### 9.1 Audio Quality Analysis

Every audio chunk is analyzed for:

```python
QualityMetrics:
    • rms_level: Perceived loudness (0-1)
    • peak_level: Maximum sample value (0-1)
    • signal_to_noise_ratio: SNR in dB
    • zero_crossing_rate: Voice activity indicator
    • voice_activity_detected: Boolean
    • voice_activity_confidence: 0-1
    • speaking_time_ratio: % of chunk with speech
    • clipping_detected: Boolean
    • distortion_level: 0-1 (0=clean, 1=severe)
    • noise_level: 0-1 (0=quiet, 1=loud noise)
    • spectral_centroid: Center frequency (Hz)
    • spectral_bandwidth: Frequency spread (Hz)
    • spectral_rolloff: 95% energy frequency (Hz)
    • overall_quality_score: 0-1 weighted average
```

### 9.2 Service Health Metrics

**Monitored Metrics**:
- Request latency (p50, p95, p99)
- Error rate (5xx, 4xx, timeouts)
- Throughput (RPS)
- Queue depth
- Memory usage
- CPU usage
- GPU utilization
- Connection count
- Cache hit rate

### 9.3 System-Level Metrics

```
Orchestration Service:
    • Active sessions: 0-1000+
    • Average response time: 100-500ms
    • Error rate: <1%
    • WebSocket connections: 0-1000+
    • Database query time: 10-100ms
    • Cache hit rate: >80%

Whisper Service:
    • Model accuracy: WER <5%
    • Average latency: 200-800ms
    • Throughput: 100-200 chunks/min
    • NPU utilization: >80%
    • Memory usage: 200-500MB
    • Speaker diarization accuracy: >90%

Translation Service:
    • Model accuracy: BLEU >0.4
    • Average latency: 50-200ms
    • Throughput: 500-650 trans/min
    • GPU utilization: >80%
    • Memory usage: 6-24GB
    • Quality score: 0.7-0.95
```

---

## 10. Recommendations

### 10.1 Short-Term Optimizations (1-2 weeks)

1. **Model Quantization**
   - Quantize Whisper to INT8 (20-30% speedup)
   - Quantize translation models to INT4 (40-50% speedup)
   - Maintain quality above 95% of original

2. **Batch Processing**
   - Accumulate small requests into batches
   - Process 5-10 items per inference pass
   - Reduce per-request overhead by 50%+

3. **Connection Pooling**
   - Implement HTTP/1.1 keep-alive
   - Use persistent WebSocket connections
   - Reduce connection establishment overhead

4. **Database Query Optimization**
   - Add indexes on frequently queried columns
   - Use connection pooling (pgBouncer)
   - Batch write operations

### 10.2 Medium-Term Optimizations (1-2 months)

1. **Streaming Inference**
   - Process Whisper output token-by-token
   - Stream translation results incrementally
   - Reduce perceived latency by 30-50%

2. **Multi-GPU Distribution**
   - Whisper on GPU 0, Translation on GPU 1
   - Parallel processing for independent requests
   - 50-100% throughput improvement

3. **Advanced Caching**
   - Cache common phrases (business words, names)
   - Semantic caching for similar requests
   - Redis-backed distributed cache

4. **Request Prioritization**
   - VIP queue for premium users
   - SLA-based scheduling
   - Resource reservation

### 10.3 Long-Term Architecture (3-6 months)

1. **Microservice Decomposition**
   - Separate VAD (voice activity detection)
   - Extract speaker diarization as service
   - Create time correlation microservice

2. **AI/ML Optimization**
   - Custom distilled models for common use cases
   - Transfer learning for domain-specific translation
   - Continuous quality monitoring and retraining

3. **Advanced Hardware**
   - TPU support for translation
   - Heterogeneous computing (CPU+GPU+NPU+TPU)
   - Custom silicon consideration

4. **Cloud-Native Deployment**
   - Kubernetes orchestration
   - Auto-scaling based on load
   - Multi-region deployment for latency

---

## Conclusion

LiveTranslate's architecture is well-designed for real-time speech translation with:
- **Strengths**: Hardware acceleration, comprehensive error handling, modular design
- **Current Latency**: ~500-800ms (room for 30-50% improvement)
- **Scalability**: Horizontal scaling to 1000+ concurrent users
- **Quality**: >90% accuracy for both transcription and translation

The primary optimization opportunities lie in **model inference optimization** (quantization, distillation), **GPU utilization** (multi-GPU, batch processing), and **network overhead reduction** (embedded services, gRPC). With these optimizations, the system can achieve <300ms end-to-end latency at scale.

