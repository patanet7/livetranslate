# Level 1: System Context

LiveTranslate is a real-time speech-to-text and translation platform for live meeting and streaming scenarios.

## Primary Users

- Meeting hosts managing multilingual sessions.
- Meeting participants consuming translated captions.
- Developers integrating or extending translation workflows.
- Operators maintaining service reliability and performance.

## External Systems

- Video conferencing platforms (currently Google Meet integrations).
- Local/remote model runtimes for transcription and translation.
- PostgreSQL and Redis for persistence and queue/cache workflows.

## Scope Boundary

LiveTranslate handles:

- Real-time transcription.
- Real-time translation.
- Session orchestration and service coordination.
- Bot/session data persistence.

LiveTranslate does not treat video rendering/transcoding as a primary concern.

## Related Context Documents

- [Users and Personas](./users-and-personas.md)
- [Use Cases](./use-cases.md)
- [External Systems](./external-systems.md)
- [Deployment Scenarios](./deployment-scenarios.md)

## Next Step

Continue to [Level 2: Container Architecture](../02-containers/README.md).
