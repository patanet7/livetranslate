# ADR 0001: Microservices Architecture

## Status

Accepted

## Context

LiveTranslate requires a system that can handle:
- Real-time audio transcription using Whisper
- Multi-language translation using LLMs
- Low-latency WebSocket communication
- Different hardware optimization needs (NPU for Whisper, GPU for translation)
- Independent scaling of components

## Decision

We will use a microservices architecture with four core services:

1. **Whisper Service** - NPU-optimized transcription
2. **Translation Service** - GPU-optimized translation
3. **Orchestration Service** - API coordination and bot management
4. **Frontend Service** - React-based user interface

## Rationale

- **Hardware optimization**: Different services have different hardware requirements
- **Independent scaling**: Services can scale based on their specific load
- **Technology flexibility**: Each service can use the best technology stack
- **Fault isolation**: Failures in one service don't cascade
- **Development velocity**: Teams can work independently

## Consequences

### Positive
- Clear separation of concerns
- Independent deployment and scaling
- Technology flexibility per service
- Better fault isolation

### Negative
- Increased operational complexity
- Network latency between services
- Data consistency challenges
- More complex testing requirements

## Related Decisions
- ADR 0002: PDM for dependency management
