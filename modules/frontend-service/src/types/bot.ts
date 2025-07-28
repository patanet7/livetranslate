// Detailed bot management types

// Bot enums matching backend
export enum BotPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum MeetingPlatform {
  GOOGLE_MEET = 'google_meet',
  ZOOM = 'zoom',
  TEAMS = 'teams',
  WEBEX = 'webex'
}

export enum WebcamDisplayMode {
  OVERLAY = 'overlay',
  SIDEBAR = 'sidebar',
  FULLSCREEN = 'fullscreen',
  PICTURE_IN_PICTURE = 'picture_in_picture'
}

export enum WebcamTheme {
  DARK = 'dark',
  LIGHT = 'light',
  AUTO = 'auto',
  CUSTOM = 'custom'
}

// Configuration interfaces matching backend structure
export interface MeetingInfo {
  meetingId: string;
  meetingTitle?: string;
  meetingUrl?: string;
  platform: MeetingPlatform;
  organizerEmail?: string;
  scheduledStart?: string; // ISO string
  scheduledDurationMinutes?: number;
  participantCount: number;
}

export interface AudioCaptureConfig {
  deviceId?: string;
  sampleRate: number;
  channels: number;
  chunkSize: number;
  enableNoiseSuppression: boolean;
  enableEchoCancellation: boolean;
  enableAutoGain: boolean;
}

export interface TranslationConfig {
  targetLanguages: string[];
  sourceLanguage?: string;
  enableAutoTranslation: boolean;
  translationQuality: 'fast' | 'balanced' | 'accurate';
  realTimeTranslation: boolean;
}

export interface WebcamConfig {
  width: number;
  height: number;
  fps: number;
  displayMode: WebcamDisplayMode;
  theme: WebcamTheme;
  maxTranslationsDisplayed: number;
  fontSize: number;
  backgroundOpacity: number;
}

export interface BotConfiguration {
  meetingInfo: MeetingInfo;
  audioCapture: AudioCaptureConfig;
  translation: TranslationConfig;
  webcam: WebcamConfig;
  priority: BotPriority;
  autoTerminateMinutes?: number;
  enableRecording: boolean;
  enableTranscription: boolean;
  enableSpeakerDiarization: boolean;
  enableVirtualWebcam: boolean;
}

export interface BotSpawnRequest {
  config: BotConfiguration;
  userId?: string;
  sessionId?: string;
  metadata?: Record<string, any>;
}

// Legacy interface for backward compatibility
export interface BotConfig {
  botId: string;
  botName: string;
  targetLanguages: string[];
  virtualWebcamEnabled: boolean;
  serviceEndpoints: {
    whisperService: string;
    translationService: string;
    orchestrationService: string;
  };
  audioConfig: {
    sampleRate: number;
    channels: number;
    chunkDuration: number;
    qualityThreshold: number;
  };
  captionConfig: {
    enableSpeakerDiarization: boolean;
    confidenceThreshold: number;
    languageDetection: boolean;
  };
  webcamConfig: {
    width: number;
    height: number;
    fps: number;
    displayMode: 'overlay' | 'sidebar' | 'fullscreen';
    theme: 'light' | 'dark' | 'auto';
    maxTranslationsDisplayed: number;
  };
}

// Statistics interfaces matching backend structure
export interface AudioCaptureStats {
  isCapturing: boolean;
  totalChunksCaptured: number;
  averageChunkSizeBytes: number;
  totalAudioDurationS: number;
  averageQualityScore: number;
  lastCaptureTimestamp: string; // ISO string
  deviceInfo: string;
  sampleRateActual: number;
  channelsActual: number;
}

export interface CaptionProcessorStats {
  totalCaptionsProcessed: number;
  totalSpeakers: number;
  speakerTimeline: Array<{
    speakerId: string;
    start: number;
    end: number;
  }>;
  averageConfidence: number;
  lastCaptionTimestamp: string; // ISO string
  processingLatencyMs: number;
}

export interface VirtualWebcamStats {
  isStreaming: boolean;
  framesGenerated: number;
  currentTranslations: Array<{
    language: string;
    text: string;
  }>;
  averageFps: number;
  webcamConfig: WebcamConfig;
  lastFrameTimestamp: string; // ISO string
}

export interface TimeCorrelationStats {
  totalCorrelations: number;
  successRate: number;
  averageTimingOffsetMs: number;
  lastCorrelationTimestamp: string; // ISO string
  correlationAccuracy: number;
}

export interface BotPerformanceStats {
  sessionDurationS: number;
  totalProcessingTimeS: number;
  cpuUsagePercent: number;
  memoryUsageMb: number;
  networkBytesSent: number;
  networkBytesReceived: number;
  averageLatencyMs: number;
  errorCount: number;
  lastError?: string;
}

export interface BotHealthMetrics {
  joinedLog: boolean;
  captionRegionDetected: boolean;
  segmentsCount: number;
  memoryUsageMb: number;
  domSelectorStatus: 'healthy' | 'degraded' | 'failed';
  lastCaptionTimestamp: number;
  audioStreamActive: boolean;
  webcamStreamActive: boolean;
  lastHealthCheck: number;
}

export interface BotError {
  errorId: string;
  errorType: 'connection' | 'audio' | 'caption' | 'webcam' | 'memory' | 'timeout';
  errorMessage: string;
  errorStack?: string;
  timestamp: number;
  recoverable: boolean;
  retryCount: number;
}

export interface BotSession {
  sessionId: string;
  botId: string;
  meetingId: string;
  startTime: number;
  endTime?: number;
  status: 'active' | 'completed' | 'terminated' | 'error';
  totalAudioCaptured: number;
  totalCaptionsProcessed: number;
  totalTranslationsGenerated: number;
  averageProcessingLatency: number;
  qualityMetrics: {
    audioQualityScore: number;
    captionAccuracy: number;
    translationAccuracy: number;
  };
  errors: BotError[];
}

export interface CaptionSegment {
  segmentId: string;
  text: string;
  speakerName: string;
  speakerId: string;
  startTime: number;
  endTime: number;
  confidence: number;
  language: string;
  timestamp: number;
}

export interface SpeakerProfile {
  speakerId: string;
  speakerName: string;
  utteranceCount: number;
  totalSpeakingTime: number;
  averageConfidence: number;
  voiceCharacteristics?: {
    fundamentalFrequency: number;
    spectralProfile: number[];
  };
  isActive: boolean;
  firstSeen: number;
  lastSeen: number;
}

export interface TimeCorrelationResult {
  correlationId: string;
  externalEventId: string;
  internalResultId: string;
  correlationType: 'exact' | 'inferred' | 'interpolated';
  correlationConfidence: number;
  timingOffset: number;
  speakerName: string;
  matchedText?: string;
}

export interface Translation {
  translationId: string;
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
  speakerName: string;
  speakerId: string;
  translationConfidence: number;
  timestamp: number;
}

export interface VirtualWebcamFrame {
  frameId: string;
  timestamp: number;
  frameData: string; // base64 encoded
  translations: Translation[];
  metadata: {
    width: number;
    height: number;
    format: string;
    size: number;
  };
}