# Level 3: Component Details

Component-level documentation for each service.

## Orchestration Service Components
- [AudioCoordinator](./orchestration/audio-coordinator.md) - Audio processing coordination
- [BotManager](./orchestration/bot-manager.md) - Google Meet bot lifecycle
- [VirtualWebcam](./orchestration/virtual-webcam.md) - Subtitle overlay generation
- [DataPipeline](./orchestration/data-pipeline.md) - Complete audio → translation pipeline
- [ConfigSync](./orchestration/config-sync.md) - Configuration synchronization

## Whisper Service Components
- [TranscriptionEngine](./whisper/transcription-engine.md) - Core Whisper transcription
- [SpeakerDiarization](./whisper/speaker-diarization.md) - Speaker identification
- [VADDetector](./whisper/vad-detector.md) - Voice activity detection
- [SessionManager](./whisper/session-manager.md) - Session state management

## Translation Service Components
- [TranslationEngine](./translation/translation-engine.md) - Core translation logic
- [BackendManager](./translation/backend-manager.md) - Multi-backend (Ollama, OpenAI, vLLM)
- [QualityScorer](./translation/quality-scorer.md) - Translation quality assessment

## Frontend Service Components
- [AudioProcessing](./frontend/audio-processing.md) - useAudioProcessing hook
- [BotDashboard](./frontend/bot-dashboard.md) - Bot management UI
- [SettingsSync](./frontend/settings-sync.md) - Real-time settings

---

**Note**: Detailed component documentation is maintained in each service's `modules/[service]/docs/` directory. This index provides cross-service navigation.
