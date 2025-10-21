# Whisper Service Architecture - Phase 2 Requirements Analysis

**Created**: 2025-10-20
**Purpose**: Define what features belong in the discrete Whisper service vs Orchestration service
**Context**: Phase 2 SimulStreaming innovations implementation

---

## Executive Summary

The Whisper service must remain a **pure transcription microservice** focused on speech-to-text inference with model-specific optimizations. Coordination, business logic, and audio preprocessing should stay in the Orchestration service.

### Key Principle
> **"Whisper service owns the MODEL and INFERENCE. Orchestration service owns the PIPELINE and COORDINATION."**

---

## Architecture Decision Matrix

### ✅ WHISPER SERVICE - Core Responsibilities

These features MUST stay in the discrete Whisper service because they are **model-specific** and **inference-related**:

#### 1. Model Management & Loading
- **Whisper Large-v3 Model Loading** ✅ WHISPER
  - OpenVINO IR model loading
  - NPU/GPU/CPU device detection
  - Model caching and memory management
  - Automatic hardware fallback (NPU → GPU → CPU)
  - Model warmup and preloading

**Why**: Direct interaction with model binaries and hardware acceleration

#### 2. Beam Search Decoding
- **Beam Search Decoder** ✅ WHISPER
  - Beam width configuration (1, 3, 5, 10)
  - Hypothesis scoring and ranking
  - Length normalization
  - Temperature scaling
  - Greedy decoding fallback (beam_size=1)

**Why**: Core inference algorithm that operates on model logits

#### 3. AlignAtt Streaming Policy
- **Attention-Based Frame Limiting** ✅ WHISPER
  - Frame threshold enforcement
  - Attention mask generation
  - Incremental decoding state management
  - Streaming inference coordination

**Why**: Modifies model inference behavior and attention mechanisms

#### 4. In-Domain Prompting
- **Domain-Specific Prompt Injection** ✅ WHISPER
  - Medical/legal/technical terminology injection
  - Scrolling context window (448 tokens)
  - Custom terminology list management
  - Prompt formatting for Whisper input

**Why**: Directly affects model input and generation behavior

#### 5. Context Carryover
- **Previous Output Context** ✅ WHISPER
  - Context buffer management (last 223 tokens)
  - Prompt construction with history
  - Token limit enforcement
  - Context truncation strategy

**Why**: Part of Whisper's prompt-based conditioning mechanism

#### 6. Model-Specific Audio Processing
- **Audio Feature Extraction** ✅ WHISPER
  - 80-channel log-mel spectrogram generation
  - 16kHz resampling (if not done by orchestration)
  - Audio normalization for model input
  - Padding to 30-second chunks (model requirement)

**Why**: Model expects specific audio feature format

#### 7. Hardware Acceleration
- **NPU/GPU Optimization** ✅ WHISPER
  - OpenVINO optimization
  - INT8/FP16 quantization
  - Batch processing on GPU
  - NPU memory management
  - Thermal throttling awareness

**Why**: Direct hardware interaction for inference

#### 8. Model Output Processing
- **Transcription Result Formatting** ✅ WHISPER
  - Text extraction from logits
  - Language detection from model output
  - Confidence score calculation
  - Segment timestamp extraction
  - Hallucination detection (model-based patterns)

**Why**: Interpreting raw model outputs

---

### ⚙️ ORCHESTRATION SERVICE - Pipeline Responsibilities

These features should stay in Orchestration because they are **coordination** and **business logic**:

#### 1. Computationally Aware Chunking
- **Real-Time Factor (RTF) Monitoring** ⚙️ ORCHESTRATION
  - Track processing time vs audio duration
  - Dynamic chunk size adjustment (1-3s)
  - Performance-based adaptation
  - Jitter reduction strategy

**Why**: System-wide performance monitoring and coordination

#### 2. Silero VAD (Voice Activity Detection)
- **Audio Chunk Filtering** ⚙️ ORCHESTRATION
  - Silero VAD model inference
  - Speech vs non-speech classification
  - Chunk dropping for silence
  - Audio quality assessment

**Why**: Pre-processing before Whisper, reduces unnecessary inference

#### 3. CIF Word Boundaries
- **Chunk Boundary Optimization** ⚙️ ORCHESTRATION
  - CIF (Continuous Integrate-and-Fire) model inference
  - Word boundary detection
  - Smart chunk splitting
  - Overlap management

**Why**: Audio chunking strategy, happens before Whisper

#### 4. Audio Preprocessing Pipeline
- **Standard Audio Processing** ⚙️ ORCHESTRATION
  - Format detection and conversion (WAV, MP3, WebM)
  - Sample rate conversion (any → 16kHz)
  - Channel mixing (stereo → mono)
  - Volume normalization
  - Basic noise reduction (optional)

**Why**: Common preprocessing for all audio, not Whisper-specific

#### 5. Session & Chunk Management
- **Processing Coordination** ⚙️ ORCHESTRATION
  - Session lifecycle management
  - Chunk sequencing and tracking
  - Metadata management
  - Result aggregation across chunks
  - Duplicate detection and merging

**Why**: Business logic and state management

#### 6. Translation Integration
- **Post-Transcription Pipeline** ⚙️ ORCHESTRATION
  - Translation service coordination
  - Multi-language output management
  - Quality scoring
  - Caching strategy

**Why**: Separate service integration

#### 7. Database & Persistence
- **Data Storage** ⚙️ ORCHESTRATION
  - Chat history persistence
  - Session tracking
  - Analytics collection
  - User management

**Why**: Business data layer

#### 8. WebSocket & Real-Time Delivery
- **Client Communication** ⚙️ ORCHESTRATION
  - WebSocket connection management
  - Real-time event streaming
  - Progress updates
  - Error handling and recovery

**Why**: Client-facing coordination

---

### 🔄 SPLIT RESPONSIBILITIES - Shared Concerns

Some features need coordination between both services:

#### 1. Streaming Transcription
- **Whisper Side**: Incremental decoding with AlignAtt policy
- **Orchestration Side**: Chunk coordination, result merging, client delivery

#### 2. Performance Metrics
- **Whisper Side**: Inference time, model RTF, device utilization
- **Orchestration Side**: End-to-end latency, throughput, success rates

#### 3. Error Handling
- **Whisper Side**: Model errors, OOM, device failures, fallback logic
- **Orchestration Side**: Retry logic, alternative routing, client notifications

#### 4. Configuration
- **Whisper Side**: Model parameters (beam_size, temperature, language)
- **Orchestration Side**: System parameters (chunk_size, overlap, VAD threshold)

---

## Discrete Whisper Service API Contract

Based on this analysis, here's what the discrete Whisper service API should look like:

### Core Endpoints

#### 1. Transcribe Single Chunk (Synchronous)
```http
POST /api/transcribe
Content-Type: multipart/form-data

{
  "audio": <binary audio data>,
  "model": "whisper-large-v3",
  "language": "en",
  "beam_size": 5,
  "temperature": 0.0,
  "initial_prompt": "Medical consultation...",  # In-domain prompt
  "previous_context": "Patient reports...",     # Context carryover
  "domain": "medical",                          # Domain hint
  "custom_terms": ["diagnosis", "symptoms"]     # Custom terminology
}

Response:
{
  "text": "The patient reports severe headache...",
  "language": "en",
  "confidence": 0.94,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "The patient reports",
      "confidence": 0.96
    }
  ],
  "processing_time": 0.15,
  "device_used": "NPU",
  "model_used": "whisper-large-v3"
}
```

#### 2. Streaming Transcription (WebSocket)
```javascript
// Whisper-side WebSocket for streaming inference
ws://whisper-service:5001/ws/stream

// Client sends:
{
  "type": "start_stream",
  "config": {
    "model": "whisper-large-v3",
    "beam_size": 5,
    "language": "en",
    "streaming_policy": "alignatt",
    "frame_threshold_offset": 10
  }
}

// Client sends audio chunks:
{
  "type": "audio_chunk",
  "audio": <base64_encoded_audio>,
  "sequence": 0
}

// Whisper responds with incremental results:
{
  "type": "transcription_result",
  "text": "partial transcription",
  "is_final": false,
  "sequence": 0,
  "confidence": 0.89
}
```

#### 3. Model Management
```http
GET /api/models
Response:
{
  "available_models": [
    "whisper-tiny",
    "whisper-base",
    "whisper-large-v3"
  ],
  "loaded_models": ["whisper-base"],
  "default_model": "whisper-base"
}

POST /api/models/load
{
  "model_name": "whisper-large-v3",
  "device": "NPU"  # AUTO, NPU, GPU, CPU
}

POST /api/models/unload
{
  "model_name": "whisper-tiny"
}
```

#### 4. Configuration & Diagnostics
```http
GET /api/device-info
Response:
{
  "available_devices": ["NPU", "GPU", "CPU"],
  "current_device": "NPU",
  "npu_status": "available",
  "gpu_status": "fallback_ready",
  "memory": {
    "total": "8GB",
    "used": "2.1GB",
    "available": "5.9GB"
  }
}

GET /api/health
Response:
{
  "status": "healthy",
  "uptime": 3600,
  "requests_processed": 1250,
  "average_inference_time": 0.12,
  "current_load": 0.3
}
```

---

## Implementation Checklist for Phase 2

### Whisper Service Changes

- [ ] **Upgrade to Whisper Large-v3**
  - [ ] Download whisper-large-v3 OpenVINO model
  - [ ] Update ModelManager to support large-v3
  - [ ] Test NPU inference with large-v3
  - [ ] Benchmark performance vs whisper-base

- [ ] **Implement Beam Search Decoder**
  - [ ] Create BeamSearchDecoder class
  - [ ] Add beam_size parameter to API
  - [ ] Implement hypothesis ranking
  - [ ] Add greedy fallback (beam_size=1)
  - [ ] Test quality improvement (target: +20-30%)

- [ ] **Implement AlignAtt Streaming Policy**
  - [ ] Create AlignAttDecoder class
  - [ ] Add frame threshold enforcement
  - [ ] Implement attention masking
  - [ ] Add incremental decoding state
  - [ ] Test latency reduction (target: -30-50%)

- [ ] **Add In-Domain Prompting**
  - [ ] Create DomainPromptManager class
  - [ ] Add medical/legal/technical dictionaries
  - [ ] Implement scrolling context (448 tokens)
  - [ ] Add custom terminology injection
  - [ ] Test domain error reduction (target: -40-60%)

- [ ] **Implement Context Carryover**
  - [ ] Add context buffer management
  - [ ] Implement 223-token limit
  - [ ] Add context to initial_prompt
  - [ ] Test long-form quality (target: +25-40%)

- [ ] **Update API Endpoints**
  - [ ] Add beam_size parameter
  - [ ] Add initial_prompt parameter
  - [ ] Add previous_context parameter
  - [ ] Add domain parameter
  - [ ] Add custom_terms parameter
  - [ ] Add streaming_policy parameter

### Orchestration Service Changes

- [ ] **Implement Silero VAD**
  - [ ] Download Silero VAD model
  - [ ] Create VAD processor
  - [ ] Add speech detection before Whisper
  - [ ] Drop silent chunks
  - [ ] Test computation reduction (target: -30-50%)

- [ ] **Implement Computationally Aware Chunking**
  - [ ] Create RTF monitoring
  - [ ] Implement dynamic chunk sizing (1-3s)
  - [ ] Add performance adaptation
  - [ ] Test jitter reduction (target: -60%)

- [ ] **Implement CIF Word Boundaries**
  - [ ] Download CIF model
  - [ ] Create word boundary detector
  - [ ] Implement smart chunk splitting
  - [ ] Test re-translation reduction (target: -50%)

- [ ] **Update Audio Pipeline**
  - [ ] Integrate VAD before Whisper
  - [ ] Add CIF boundary detection
  - [ ] Implement dynamic chunking
  - [ ] Update Whisper service client calls

---

## Service Communication Flow

### Current Flow (Phase 1)
```
Frontend → Orchestration → Whisper Service
                ↓
         Translation Service
```

### Phase 2 Enhanced Flow
```
Frontend
   ↓
Orchestration Service
   ├─> Silero VAD (filter silence)
   ├─> CIF Word Boundaries (smart chunking)
   ├─> Computationally Aware Chunker (dynamic sizing)
   ↓
Whisper Service (Large-v3)
   ├─> AlignAtt Streaming Policy
   ├─> Beam Search Decoder (beam_size=5)
   ├─> In-Domain Prompting (medical/legal/tech)
   ├─> Context Carryover (223 tokens)
   ↓
   Return: text, segments, confidence
   ↓
Orchestration Service
   ├─> Result merging
   ├─> Translation coordination
   ├─> Database persistence
   ↓
Frontend (WebSocket)
```

---

## Testing Strategy

### Whisper Service Tests

1. **Model Loading Tests**
   - Test large-v3 loads on NPU/GPU/CPU
   - Test memory usage stays within limits
   - Test automatic fallback

2. **Beam Search Tests**
   - Test beam_size variations (1, 3, 5, 10)
   - Measure quality improvement vs greedy
   - Measure inference time increase

3. **AlignAtt Tests**
   - Test frame threshold enforcement
   - Test attention masking
   - Measure latency reduction

4. **Domain Prompt Tests**
   - Test medical terminology accuracy
   - Test custom term injection
   - Measure domain error reduction

5. **Context Carryover Tests**
   - Test long-form consistency
   - Test token limit enforcement
   - Measure quality improvement

### Integration Tests

1. **End-to-End Pipeline**
   - VAD → CIF → Dynamic Chunking → Whisper → Translation
   - Measure total latency (target: <400ms)
   - Measure P95/P99 latency

2. **Streaming Tests**
   - Test real-time streaming with all features
   - Measure jitter and stability
   - Test error recovery

3. **Performance Tests**
   - Load testing with concurrent requests
   - NPU/GPU resource utilization
   - Memory leak detection

---

## Critical Architectural Decisions

### ✅ APPROVED: Keep Whisper Service Discrete

**Reasons**:
1. **Clean separation of concerns** - Model inference vs coordination
2. **Independent scaling** - Scale Whisper horizontally without affecting orchestration
3. **Hardware flexibility** - Run Whisper on NPU machines, orchestration on CPU machines
4. **Testing isolation** - Test model changes without orchestration dependency
5. **Multi-model support** - Future: Run multiple Whisper instances with different models

### ✅ APPROVED: Orchestration Owns Preprocessing

**Reasons**:
1. **Reusability** - VAD/CIF preprocessing useful for other services
2. **Performance** - Avoid sending silent audio to Whisper
3. **Flexibility** - Change preprocessing without Whisper service changes
4. **Resource efficiency** - CPU-based preprocessing, save NPU/GPU for inference

### ✅ APPROVED: Whisper Owns In-Domain Prompting

**Reasons**:
1. **Model-specific** - Prompting is Whisper model feature
2. **Context management** - Needs access to model tokenizer
3. **Quality control** - Direct impact on model output quality
4. **Domain expertise** - Whisper service knows model capabilities

---

## Deployment Considerations

### Hardware Allocation

**Whisper Service (NPU/GPU-heavy)**:
- Intel Core Ultra with NPU
- Or NVIDIA GPU machine
- 16GB+ RAM
- Fast storage for model loading

**Orchestration Service (CPU-heavy)**:
- Standard CPU server
- 8GB+ RAM
- Redis for caching
- PostgreSQL for persistence

### Scaling Strategy

**Whisper Service**:
- Horizontal scaling with load balancer
- 2-3 instances for redundancy
- Sticky sessions for streaming
- Auto-scaling based on NPU utilization

**Orchestration Service**:
- Horizontal scaling
- Stateless design
- Redis for shared state
- Auto-scaling based on request queue

---

## Summary

### Whisper Service Responsibilities (DISCRETE)
✅ Model loading and inference
✅ Beam search decoding
✅ AlignAtt streaming policy
✅ In-domain prompting
✅ Context carryover
✅ Hardware acceleration (NPU/GPU)
✅ Model output processing

### Orchestration Service Responsibilities (COORDINATOR)
⚙️ Silero VAD preprocessing
⚙️ CIF word boundary detection
⚙️ Computationally aware chunking
⚙️ Audio format conversion
⚙️ Session management
⚙️ Translation coordination
⚙️ Database persistence
⚙️ WebSocket client delivery

### Key Interfaces
- **Whisper API**: `/api/transcribe` with beam_size, initial_prompt, previous_context
- **Orchestration → Whisper**: HTTP/gRPC calls with preprocessed audio chunks
- **Frontend → Orchestration**: WebSocket for real-time streaming

---

**Status**: Ready for implementation ✅
**Next Step**: Implement Whisper Large-v3 with beam search in discrete Whisper service
