/**
 * Default Configuration Constants
 *
 * Centralized default values for audio processing configuration.
 * Consolidates duplicate initializations from:
 * - StreamingProcessor/index.tsx
 * - MeetingTest/index.tsx
 * - AudioTesting/index.tsx
 */

import { ProcessingConfig, StreamingStats } from '@/types/streaming';

/**
 * Default target languages for translation
 */
export const DEFAULT_TARGET_LANGUAGES = ['es', 'fr', 'de'] as const;

/**
 * Default processing configuration
 */
export const DEFAULT_PROCESSING_CONFIG: ProcessingConfig = {
  enableTranscription: true,
  enableTranslation: true,
  enableDiarization: true,
  enableVAD: true,
  whisperModel: 'whisper-base',
  translationQuality: 'balanced',
  audioProcessing: true,
  noiseReduction: false,
  speechEnhancement: true,
};

/**
 * Default streaming statistics
 */
export const DEFAULT_STREAMING_STATS: StreamingStats = {
  chunksStreamed: 0,
  totalDuration: 0,
  averageProcessingTime: 0,
  errorCount: 0,
};

/**
 * Default chunk duration in milliseconds
 */
export const DEFAULT_CHUNK_DURATION_MS = 2000;

/**
 * Available Whisper models
 */
export const WHISPER_MODELS = [
  'whisper-tiny',
  'whisper-base',
  'whisper-small',
  'whisper-medium',
  'whisper-large',
] as const;

/**
 * Translation quality options
 */
export const TRANSLATION_QUALITY_OPTIONS = [
  { value: 'fast', label: 'Fast' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'quality', label: 'Quality' },
] as const;
