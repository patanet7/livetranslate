// Audio Types
export interface AudioDevice {
  deviceId: string;
  label: string;
  kind: "audioinput" | "audiooutput";
  groupId: string;
}

export interface AudioConfig {
  sampleRate: number;
  channels: number;
  dtype: string;
  blocksize: number;
  chunkDuration: number;
  qualityThreshold: number;
  // Additional properties for audio testing
  duration: number;
  deviceId: string;
  format: string;
  quality: string;
  autoStop: boolean;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  rawAudio: boolean;
  source: "microphone" | "file" | "sample";
}

export interface AudioQualityMetrics {
  // Basic Level Metrics
  rmsLevel: number;
  peakLevel: number;
  frequency?: number;
  clipping?: number;

  // Advanced Audio Metrics
  zeroCrossingRate?: number;
  snrEstimate?: number;
  signalToNoise?: number;
  clippingDetected?: boolean;

  // Meeting-Specific Metrics (Enhanced)
  voiceActivity?: number; // Voice activity percentage (0-1)
  spectralCentroid?: number; // Frequency brightness (Hz)
  dynamicRange?: number; // Peak - RMS difference (dB)
  speechClarity?: number; // Speech frequency prominence (0-1)
  backgroundNoise?: number; // Background noise level (0-1)

  // Quality Assessment
  qualityScore?: number; // Overall quality score (0-100)
  qualityAssessment?: "excellent" | "good" | "fair" | "poor";
  recommendations?: string[]; // Quality improvement suggestions
  issues?: string[]; // Detected audio issues
}

export interface RecordingState {
  isRecording: boolean;
  duration: number;
  maxDuration: number;
  autoStop: boolean;
  format: string;
  sampleRate: number;
  recordedBlobUrl: string | null; // ✅ Store serializable URL string instead of Blob
  status: "idle" | "recording" | "processing" | "completed" | "error";
  isPlaying: boolean;
  recordingStartTime?: number | null;
  sessionId?: string | null;
  // NOTE: DOM objects like MediaRecorder, MediaStream, HTMLAudioElement should not be stored in Redux
  // They are not serializable and can cause issues - use refs in components instead
  // Blob objects are stored in component refs, only URLs stored in Redux
}

export interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
}

export interface ProcessingState {
  currentStage: number;
  isProcessing: boolean;
  progress: number;
  results: Record<string, any>;
  preset: string;
}

export interface VisualizationState {
  audioLevel: number;
  frequencyData: number[];
  timeData: number[];
}

// Bot Management Types
export type BotStatus =
  | "spawning"
  | "joining"
  | "active"
  | "recording"
  | "processing"
  | "error"
  | "terminating"
  | "terminated";

export interface MeetingRequest {
  meetingId: string;
  meetingTitle: string;
  organizerEmail?: string;
  targetLanguages: string[];
  autoTranslation: boolean;
  priority: "low" | "medium" | "high";
}

export interface MeetingInfo {
  meetingId: string;
  meetingTitle: string;
  organizerEmail?: string;
  participantCount: number;
}

export interface BotInstance {
  id: string;
  botId: string;
  status: BotStatus;
  config: import("./bot").BotConfiguration;

  // Statistics matching backend structure
  audioCapture: import("./bot").AudioCaptureStats;
  captionProcessor: import("./bot").CaptionProcessorStats;
  virtualWebcam: import("./bot").VirtualWebcamStats;
  timeCorrelation: import("./bot").TimeCorrelationStats;
  performance: import("./bot").BotPerformanceStats;

  // Runtime information
  lastActiveAt: string; // ISO string
  errorMessages: string[];

  // Timestamps
  createdAt: string; // ISO string
  updatedAt: string; // ISO string
}

export interface SpeakerTimelineEvent {
  eventId: string;
  eventType: "speaking_start" | "speaking_end" | "join" | "leave";
  speakerId: string;
  speakerName: string;
  timestamp: number;
  confidence: number;
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

export interface WebcamConfig {
  width: number;
  height: number;
  fps: number;
  displayMode: "overlay" | "sidebar" | "fullscreen";
  theme: "light" | "dark" | "auto";
  maxTranslationsDisplayed: number;
  fontSize: number;
  backgroundOpacity: number;
}

export interface SystemStats {
  totalBotsSpawned: number;
  activeBots: number;
  completedSessions: number;
  errorRate: number;
  averageSessionDuration: number;
}

// API Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// Service Health Types
export interface ServiceHealth {
  serviceName: string;
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  uptime: number;
  lastCheck: number;
  details?: Record<string, any>;
}

export interface SystemHealth {
  overall: "healthy" | "degraded" | "unhealthy";
  services: ServiceHealth[];
  timestamp: number;
}

// Processing Pipeline Types
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

export interface ProcessingPreset {
  id: string;
  name: string;
  description: string;
  stages: string[];
  parameters: Record<string, any>;
}

export interface ProcessingLog {
  level: "INFO" | "SUCCESS" | "WARNING" | "ERROR";
  message: string;
  timestamp: number;
}

// UI State Types
export interface UIState {
  theme: "light" | "dark";
  sidebarOpen: boolean;
  activeTab: string;
  notifications: Notification[];
  loading: boolean;
  error: string | null;
}

export interface Notification {
  id: string;
  type: "info" | "success" | "warning" | "error";
  title: string;
  message: string;
  timestamp: number;
  autoHide: boolean;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

// Form Types
export interface FormFieldProps {
  name: string;
  label: string;
  type?: string;
  required?: boolean;
  placeholder?: string;
  helperText?: string;
  disabled?: boolean;
  options?: Array<{ value: string; label: string }>;
}

// Export specific types to avoid conflicts
export * from "./audio";
export * from "./bot";

// Export additional bot-related types and enums
export {
  BotPriority,
  MeetingPlatform,
  WebcamDisplayMode,
  WebcamTheme,
} from "./bot";

export type {
  BotConfiguration,
  BotSpawnRequest,
  AudioCaptureConfig,
  TranslationConfig,
  AudioCaptureStats,
  CaptionProcessorStats,
  VirtualWebcamStats,
  TimeCorrelationStats,
  BotPerformanceStats,
} from "./bot";

// Export WebSocket types selectively to avoid conflicts
export type {
  WebSocketEventType,
  WebSocketEventData,
  WebSocketMessage,
  WebSocketResponse,
  WebSocketStats,
  WebSocketConfig,
} from "./websocket";
