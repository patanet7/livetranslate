# Level 1: System Context

## System Overview

**LiveTranslate** is a real-time speech-to-text transcription and translation system designed for multilingual communication scenarios. It captures audio, transcribes it in real-time, and provides instant translations to multiple languages.

## Context Diagram

```
                                    ┌─────────────────────┐
                                    │   LiveTranslate     │
                            ┌───────│      System         │───────┐
                            │       └─────────────────────┘       │
                            │                                     │
             Sends Audio    │                                     │  Provides
             Gets Subtitles │                                     │  Translated
                            │                                     │  Captions
                            ▼                                     ▼
              ┌──────────────────────┐               ┌──────────────────────┐
              │                      │               │                      │
              │   Meeting Hosts      │               │   Meeting Users      │
              │   & Participants     │               │   (Multi-lingual)    │
              │                      │               │                      │
              └──────────────────────┘               └──────────────────────┘
                            │                                     │
                            │                                     │
                            └─────────────┬───────────────────────┘
                                          │
                                          │ Integrates With
                                          ▼
                            ┌─────────────────────────────────┐
                            │                                 │
                            │      Google Meet / Zoom         │
                            │      (Video Conferencing)       │
                            │                                 │
                            └─────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    External Dependencies                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │   ┌────────────┐    ┌────────────┐    ┌──────────────┐         │
    │   │  Ollama    │    │  OpenAI    │    │  PostgreSQL  │         │
    │   │  (LLM)     │    │  (API)     │    │  (Database)  │         │
    │   └────────────┘    └────────────┘    └──────────────┘         │
    │                                                                   │
    │   ┌────────────┐    ┌────────────┐                              │
    │   │  Whisper   │    │  BlackHole │                              │
    │   │  Models    │    │  (Audio)   │                              │
    │   └────────────┘    └────────────┘                              │
    │                                                                   │
    └─────────────────────────────────────────────────────────────────┘
```

## Purpose & Goals

### Primary Use Case
Enable **real-time multilingual communication** during:
- International business meetings
- Cross-cultural conferences
- Educational webinars
- Content creation (podcasts, streams)

### Key Goals
1. **Real-time Performance**: < 100ms end-to-end latency
2. **High Accuracy**: Speaker diarization + context-aware translation
3. **Hardware Optimized**: NPU/GPU acceleration for efficiency
4. **Scalability**: Support 1000+ concurrent users
5. **Integration**: Seamless Google Meet bot deployment

## Key Features

### For End Users
- **Loopback Audio Capture**: Capture system audio for live translation
- **Multi-language Support**: Translate to/from 100+ languages
- **Real-time Subtitles**: Display translations as overlay or sidebar
- **Speaker Attribution**: Identify who said what
- **High Quality**: Production-grade transcription and translation

### For Developers
- **Microservices Architecture**: Independent, scalable services
- **Hardware Acceleration**: NPU (Intel), GPU (NVIDIA), CPU fallback
- **REST + WebSocket APIs**: Real-time and batch processing
- **Docker Deployment**: Container-ready for cloud deployment
- **Comprehensive Documentation**: C4 model + API references

### For DevOps/SRE
- **Monitoring**: Prometheus metrics + health checks
- **Logging**: Structured logging across all services
- **Database Integration**: PostgreSQL for session persistence
- **Configuration Management**: Centralized config sync
- **Deployment Flexibility**: Local, cloud, or hybrid

## External Systems & Integrations

### Video Conferencing Platforms
- **Google Meet**: Bot integration with official API
- **Zoom**: (Planned) Similar bot architecture
- **Microsoft Teams**: (Planned)

### AI/ML Services
- **Ollama**: Local LLM backend for translation
- **OpenAI API**: Cloud-based translation fallback
- **Whisper Models**: Speech-to-text (via OpenVINO)

### Infrastructure
- **PostgreSQL**: Session data, translations, analytics
- **BlackHole**: macOS audio loopback device
- **Docker**: Containerization and deployment
- **Kubernetes**: (Optional) Orchestration at scale

## Target Audience

### Primary Users
1. **Meeting Hosts**: Set up bots for multilingual meetings
2. **Participants**: Use loopback for personal subtitles
3. **Content Creators**: Podcast/stream translation
4. **Educators**: Multilingual lectures

### Technical Users
1. **Developers**: Extend and integrate LiveTranslate
2. **DevOps Engineers**: Deploy and operate the system
3. **Data Scientists**: Analyze translation quality and performance

## System Boundaries

### What LiveTranslate Does
✅ Real-time audio transcription
✅ Multi-language translation
✅ Google Meet bot management
✅ Virtual webcam subtitle overlay
✅ Session data persistence
✅ Configuration synchronization

### What LiveTranslate Does NOT Do
❌ Video processing (audio only)
❌ Recording/storage of meetings (privacy-focused)
❌ User authentication (delegates to platforms)
❌ Content moderation (trust-based)

## Success Metrics

1. **Latency**: < 100ms audio → subtitle display
2. **Accuracy**: > 95% transcription accuracy
3. **Translation Quality**: > 90% semantic accuracy
4. **Uptime**: 99.9% service availability
5. **Scalability**: Handle 1000+ concurrent sessions
6. **Hardware Efficiency**: 10x speedup with NPU/GPU

---

## Next Steps

- **Understand the Architecture**: See [Level 2: Container Architecture](../02-containers/README.md)
- **Get Started**: Read the [Quick Start Guide](../guides/quick-start.md)
- **Learn About Users**: See [Users & Personas](./users-and-personas.md)
