// WebSocket communication types

// Missing types that are referenced in WebSocket events
export interface AudioAnalysisResult {
  rms: number;
  peak: number;
  frequency: number;
  clipping: number;
  quality: number;
}

export interface CaptionSegment {
  text: string;
  speaker: string;
  timestamp: number;
  confidence: number;
}

export interface VirtualWebcamFrame {
  frameData: string; // base64 encoded frame
  width: number;
  height: number;
  timestamp: number;
}

export interface BotError {
  errorType: string;
  errorMessage: string;
  timestamp: number;
  stackTrace?: string;
}

export interface BotSession {
  sessionId: string;
  startTime: number;
  endTime: number;
  transcriptionCount: number;
  translationCount: number;
  audioFilesProcessed: number;
}

// Basic connection types
export interface ConnectionState {
  isConnected: boolean;
  connectionId: string | null;
  lastPingTime: number;
  reconnectAttempts: number;
  error: string | null;
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  autoReconnect: boolean;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  connectionTimeout: number;
}

// Import types from index.ts to avoid duplication
import type {
  ServiceHealth,
  Notification,
  Translation,
  AudioDevice,
} from "./index";

// Define types to avoid circular imports with index.ts
type BotStatus = "spawning" | "active" | "error" | "terminated";

export interface WebSocketEventMap {
  // Connection events
  "connection:established": { connectionId: string; serverTime: number };
  "connection:lost": { reason: string; code: number };
  "connection:reconnecting": { attempt: number; maxAttempts: number };
  "connection:error": { error: string; code: number };
  "connection:ping": { timestamp: number };
  "connection:pong": { timestamp: number; server_time: string };

  // Audio events
  "audio:chunk": {
    chunk: ArrayBuffer;
    timestamp: number;
    quality: AudioAnalysisResult;
  };
  "audio:transcription": {
    text: string;
    confidence: number;
    language: string;
    timestamp: number;
  };
  "audio:device_change": { devices: AudioDevice[] };
  "audio:error": { error: string; timestamp: number };

  // Bot events
  "bot:spawned": { botId: string; meetingId: string; status: BotStatus };
  "bot:status_change": { botId: string; status: BotStatus; data?: any };
  "bot:audio_capture": { botId: string; metrics: AudioAnalysisResult };
  "bot:caption_received": { botId: string; caption: CaptionSegment };
  "bot:translation_ready": { botId: string; translation: Translation };
  "bot:webcam_frame": { botId: string; frame: VirtualWebcamFrame };
  "bot:error": { botId: string; error: BotError };
  "bot:terminated": { botId: string; reason: string; sessionData: BotSession };

  // System events
  "system:health_update": { services: ServiceHealth[]; timestamp: number };
  "system:performance_metrics": {
    cpu: number;
    memory: number;
    timestamp: number;
  };
  "system:alert": {
    level: "info" | "warning" | "error";
    message: string;
    timestamp: number;
  };

  // Processing events
  "processing:stage_start": {
    stageId: string;
    stageName: string;
    timestamp: number;
  };
  "processing:stage_complete": {
    stageId: string;
    result: any;
    timestamp: number;
  };
  "processing:pipeline_complete": {
    pipelineId: string;
    results: any;
    timestamp: number;
  };
  "processing:error": { stageId: string; error: string; timestamp: number };

  // UI events
  "ui:notification": { notification: Notification };
  "ui:page_change": { page: string; timestamp: number };
  "ui:user_action": { action: string; data: any; timestamp: number };

  // System heartbeat and echo
  "system:heartbeat": { timestamp: number; server_time: string };
  "message:echo": { original_message: any; echo_from: string };
}

export type WebSocketEventType = keyof WebSocketEventMap;
export type WebSocketEventData<T extends WebSocketEventType> =
  WebSocketEventMap[T];

export interface WebSocketMessage<
  T extends WebSocketEventType = WebSocketEventType,
> {
  type: T;
  data: WebSocketEventData<T>;
  timestamp: number;
  messageId: string;
  correlationId?: string;
}

export interface WebSocketResponse {
  success: boolean;
  messageId: string;
  data?: any;
  error?: string;
  timestamp: number;
}

export interface WebSocketStats {
  connectionDuration: number;
  messagesSent: number;
  messagesReceived: number;
  reconnectCount: number;
  averageLatency: number;
  lastPingTime: number;
  bytesTransferred: number;
}
