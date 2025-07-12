// Detailed bot management types

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