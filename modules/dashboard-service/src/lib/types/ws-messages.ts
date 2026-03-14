export const PROTOCOL_VERSION = 1;

// Client → Server
export interface StartSessionMessage {
  type: 'start_session';
  sample_rate: number;
  channels: number;
  device_id?: string;
}

export interface EndSessionMessage {
  type: 'end_session';
}

export interface PromoteToMeetingMessage {
  type: 'promote_to_meeting';
}

export interface EndMeetingMessage {
  type: 'end_meeting';
}

// Client → Transcription Service
export interface ConfigMessage {
  type: 'config';
  model?: string;
  language?: string;
  initial_prompt?: string;
  glossary_terms?: string[];
}

export interface EndMessage {
  type: 'end';
}

export type ClientMessage =
  | StartSessionMessage
  | EndSessionMessage
  | PromoteToMeetingMessage
  | EndMeetingMessage
  | ConfigMessage
  | EndMessage;

// Server → Client
export interface ConnectedMessage {
  type: 'connected';
  protocol_version: number;
  session_id: string;
}

export interface SegmentMessage {
  type: 'segment';
  text: string;
  language: string;
  confidence: number;
  stable_text: string;
  unstable_text: string;
  is_final: boolean;
  speaker_id: string | null;
}

export interface InterimMessage {
  type: 'interim';
  text: string;
  confidence: number;
}

export interface TranslationMessage {
  type: 'translation';
  text: string;
  source_lang: string;
  target_lang: string;
  transcript_id: number;
  context_used: number;
}

export interface MeetingStartedMessage {
  type: 'meeting_started';
  session_id: string;
  started_at: string;
}

export interface RecordingStatusMessage {
  type: 'recording_status';
  recording: boolean;
  chunks_written: number;
}

export interface ServiceStatusMessage {
  type: 'service_status';
  transcription: 'up' | 'down';
  translation: 'up' | 'down';
}

// Transcription Service → Client
export interface LanguageDetectedMessage {
  type: 'language_detected';
  language: string;
  confidence: number;
}

export interface BackendSwitchedMessage {
  type: 'backend_switched';
  backend: string;
  model: string;
  language: string;
}

export type ServerMessage =
  | ConnectedMessage
  | SegmentMessage
  | InterimMessage
  | TranslationMessage
  | MeetingStartedMessage
  | RecordingStatusMessage
  | ServiceStatusMessage
  | LanguageDetectedMessage
  | BackendSwitchedMessage;

export function parseServerMessage(raw: string): ServerMessage | null {
  try {
    const data = JSON.parse(raw);
    if (!data || typeof data.type !== 'string') return null;
    const knownTypes = [
      'connected', 'segment', 'interim', 'translation',
      'meeting_started', 'recording_status', 'service_status',
      'language_detected', 'backend_switched',
    ];
    if (!knownTypes.includes(data.type)) return null;
    return data as ServerMessage;
  } catch {
    return null;
  }
}
