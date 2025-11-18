/**
 * Streaming Types
 *
 * Shared type definitions for streaming audio processing.
 * Consolidates duplicate interfaces from:
 * - StreamingProcessor/index.tsx
 * - MeetingTest/index.tsx
 * - TranscriptionTesting/index.tsx
 */

export interface StreamingChunk {
  id: string;
  audio: Blob;
  timestamp: number;
  duration: number;
}

export interface SpeakerInfo {
  speaker_id: string;
  speaker_name: string;
  start_time: number;
  end_time: number;
}

export interface TranscriptionResult {
  id: string;
  chunkId: string;
  text: string;
  confidence: number;
  language: string;
  speakers?: SpeakerInfo[];
  timestamp: number;
  processing_time: number;
}

export interface TranslationResult {
  id: string;
  transcriptionId: string;
  sourceText: string;
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
  confidence: number;
  timestamp: number;
  processing_time: number;
}

export interface StreamingStats {
  chunksStreamed: number;
  totalDuration: number;
  averageProcessingTime: number;
  errorCount: number;
}

export interface ProcessingConfig {
  enableTranscription: boolean;
  enableTranslation: boolean;
  enableDiarization: boolean;
  enableVAD: boolean;
  whisperModel: string;
  translationQuality: string;
  audioProcessing: boolean;
  noiseReduction: boolean;
  speechEnhancement: boolean;
}
