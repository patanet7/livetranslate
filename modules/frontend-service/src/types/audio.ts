// Detailed audio processing types

export interface AudioStreamConfig {
  sampleRate: number;
  channels: number;
  bitDepth: number;
  bufferSize: number;
  latencyHint: "interactive" | "balanced" | "playback";
}

export interface AudioAnalysisResult {
  rmsLevel: number;
  peakLevel: number;
  zeroCrossingRate: number;
  spectralCentroid: number;
  spectralRolloff: number;
  mfcc: number[];
  voiceActivityDetection: boolean;
  qualityScore: number;
  clippingDetected: boolean;
  noiseLevel: number;
  snrEstimate: number;
}

export interface AudioSegment {
  id: string;
  startTime: number;
  endTime: number;
  audioData: ArrayBuffer;
  sampleRate: number;
  channels: number;
  qualityMetrics: AudioAnalysisResult;
  transcription?: string;
  confidence?: number;
  speakerId?: string;
}

export interface AudioProcessor {
  name: string;
  type: "filter" | "enhancement" | "analysis" | "codec";
  parameters: Record<string, any>;
  enabled: boolean;
}

export interface AudioPipeline {
  id: string;
  name: string;
  description: string;
  processors: AudioProcessor[];
  inputConfig: AudioStreamConfig;
  outputConfig: AudioStreamConfig;
}

export interface VoiceActivityDetection {
  isActive: boolean;
  confidence: number;
  segmentStart?: number;
  segmentEnd?: number;
  algorithm: "webrtc" | "silero" | "energy" | "spectral";
}

// Processing pipeline types
export interface ProcessingStage {
  id: string;
  name: string;
  description: string;
  status: "pending" | "processing" | "completed" | "error";
  progress: number;
  startTime?: number;
  endTime?: number;
  processingTime?: number;
  result?: any;
  error?: string;
  metrics?: Record<string, any>;
}

export interface ProcessingLog {
  level: "INFO" | "SUCCESS" | "WARNING" | "ERROR";
  message: string;
  timestamp: number;
}
