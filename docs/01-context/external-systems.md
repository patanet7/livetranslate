# External Systems & Dependencies

## Core Dependencies

### AI/ML Services

**Ollama (Local LLM)**
- Purpose: Translation backend
- Protocol: HTTP REST API
- Deployment: Self-hosted
- Models: Mistral, Llama 3.1

**OpenAI API (Cloud LLM)**
- Purpose: Translation fallback
- Protocol: HTTPS REST API
- Deployment: Cloud SaaS
- Authentication: API key

**Whisper Models (OpenVINO)**
- Purpose: Speech-to-text
- Format: ONNX/OpenVINO IR
- Deployment: Local models directory
- Hardware: NPU/GPU/CPU

### Video Conferencing

**Google Meet**
- Integration: Official Google Meet API
- Purpose: Bot joins meetings, captures audio
- Authentication: OAuth 2.0

### Infrastructure

**PostgreSQL**
- Purpose: Session data, transcripts, translations
- Version: 14+
- Deployment: Self-hosted or cloud (RDS, Cloud SQL)

**BlackHole (macOS)**
- Purpose: System audio loopback
- Platform: macOS only
- Alternative: VoiceMeeter (Windows)

### Optional Services

**Prometheus**: Metrics collection
**Grafana**: Monitoring dashboards
**Triton Inference Server**: Advanced GPU inference
**vLLM**: High-performance LLM serving

## Integration Points

See [Container Architecture](../02-containers/communication-patterns.md) for details on how services communicate.
