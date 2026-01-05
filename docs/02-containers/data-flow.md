# Data Flow

## Primary Flow: Audio → Translation

```
1. [User Audio] 
   ↓ (Browser MediaRecorder API)
2. [Frontend Audio Capture]
   ↓ (HTTP POST /api/audio/upload, 5s chunks, WebM/WAV)
3. [Orchestration AudioCoordinator]
   ↓ (HTTP POST /transcribe, WAV 16kHz)
4. [Whisper Service]
   ↓ (Transcription result, JSON)
5. [Orchestration]
   ↓ (HTTP POST /api/translate, text + target languages)
6. [Translation Service]
   ↓ (Translated text, JSON)
7. [Orchestration]
   ↓ (Store in PostgreSQL)
8. [Database]
   ↓ (WebSocket broadcast)
9. [Frontend Display]
```

## Data Formats

**Audio Input**: WebM Opus, WAV PCM, MP3, OGG
**Processing**: WAV 16kHz mono (standardized)
**Transcription**: JSON `{text, language, confidence, segments}`
**Translation**: JSON `{translated_text, source_lang, target_lang, confidence}`
**Storage**: PostgreSQL JSONB columns

## Related
- [Communication Patterns](./communication-patterns.md)- [Service Overview](./service-overview.md)
