export const PROTOCOL_VERSION = 1;

/** Per-session LLM sampling tunables — mirror of Python LLMParameterOverrides.
 *
 *  Two-state contract: every field is either a concrete value or `null`
 *  (matching Python's `Optional[T]` semantics). `null` means "no override —
 *  the server will use the resolved connection's default for this call".
 *  Send a partial object with just the fields the user changed; merge happens
 *  server-side on top of the previous SessionConfig.llm snapshot.
 *  API key is intentionally NOT here — keys live on the server-side
 *  ai_connections table and never traverse this WS channel.
 */
export interface LLMOverridesMessage {
  connection_id: string | null;
  model: string | null;
  temperature: number | null;
  max_tokens: number | null;
  top_p: number | null;
  top_k: number | null;
  repetition_penalty: number | null;
  presence_penalty: number | null;
}

/** Partial patch shape — every field optional, used for ConfigMessage deltas.
 *
 *  A delta only carries the fields the user touched. Sent on every Toolbar
 *  input change. The full `LLMOverridesMessage` is reconstructed on the
 *  server from the previous snapshot + this patch.
 */
export type LLMOverridesPatch = Partial<LLMOverridesMessage>;

/** Per-session Whisper decoding tunables — mirror of Python WhisperParameterOverrides.
 *
 *  Same two-state contract as LLMOverridesMessage. Connection swap (via
 *  `connection_id`) re-resolves the active backend; sampling fields apply to
 *  subsequent transcription calls.
 *  API key is intentionally NOT here — keys live on the server-side
 *  whisper_connections table and never traverse this WS channel.
 */
export interface WhisperOverridesMessage {
  connection_id: string | null;
  model: string | null;
  temperature: number | null;
  beam_size: number | null;
  no_speech_threshold: number | null;
  compression_ratio_threshold: number | null;
  language_hint: string | null;
  initial_prompt: string | null;
  timeout_s: number | null;
}

export type WhisperOverridesPatch = Partial<WhisperOverridesMessage>;

// Client → Server
export interface StartSessionMessage {
  type: 'start_session';
  sample_rate: number;
  channels: number;
  encoding?: string; // default "float32" for browser Float32Array
  device_id?: string;
  source?: 'mic' | 'screencapture'; // Audio source type
  llm?: LLMOverridesPatch;
  whisper?: WhisperOverridesPatch;
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
  language?: string | null;
  target_language?: string;
  interpreter_languages?: [string, string] | null;
  initial_prompt?: string;
  glossary_terms?: string[];
  llm?: LLMOverridesPatch;
  whisper?: WhisperOverridesPatch;
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
  | EndMessage
  | ChatCommandMessage;

// Server → Client
export interface ConnectedMessage {
  type: 'connected';
  protocol_version: number;
  session_id: string;
}

export interface SegmentMessage {
  type: 'segment';
  segment_id: number;
  text: string;
  language: string;
  confidence: number;
  stable_text: string;
  unstable_text: string;
  /**
   * True when the segment text ends at a sentence boundary (punctuation).
   *
   * WARNING: Does NOT mean "last segment" or "will not be updated."
   * A segment with is_final=false can still be the definitive transcription
   * for its audio window. See ARCHITECTURE.md Draft/Final Protocol.
   */
  is_final: boolean;
  /**
   * True for first-pass VAC snapshot (non-destructive, stride/2 audio).
   *
   * Draft and final segments share the same segment_id. The final is a
   * second-pass with the full audio stride -- same model, more audio,
   * usually longer/more accurate text. The frontend replaces the draft
   * in-place when the final arrives.
   */
  is_draft: boolean;
  speaker_id: string | null;
  start_ms: number | null;
  end_ms: number | null;
}

export interface InterimMessage {
  type: 'interim';
  text: string;
  confidence: number;
}

export interface TranslationChunkMessage {
  type: 'translation_chunk';
  transcript_id: number;
  delta: string;
  source_lang: string;
  target_lang: string;
  is_draft?: boolean;
}

export interface TranslationMessage {
  type: 'translation';
  text: string;
  source_lang: string;
  target_lang: string;
  /** Matches segment_id on SegmentMessage. BIGSERIAL in DB — safe up to 2^53 in JS number. */
  transcript_id: number;
  context_used: number;
  is_draft?: boolean;
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

export interface ErrorMessage {
  type: 'error';
  message: string;
  recoverable: boolean;
}

export interface ChatCommandMessage {
  type: 'chat_command';
  command: string;
  sender: string;
}

export interface ChatResponseMessage {
  type: 'chat_response';
  text: string;
}

export interface ConfigChangedMessage {
  type: 'config_changed';
  changes: Record<string, unknown>;
}

export interface AudioLevelMessage {
  type: 'audio_level';
  rms: number;
  source?: 'screencapture' | 'mic';
  chunks?: number;
}

export type ServerMessage =
  | ConnectedMessage
  | SegmentMessage
  | InterimMessage
  | TranslationChunkMessage
  | TranslationMessage
  | MeetingStartedMessage
  | RecordingStatusMessage
  | ServiceStatusMessage
  | LanguageDetectedMessage
  | BackendSwitchedMessage
  | ErrorMessage
  | ChatResponseMessage
  | ConfigChangedMessage
  | AudioLevelMessage;

export function parseServerMessage(raw: string): ServerMessage | null {
  try {
    const data = JSON.parse(raw);
    if (!data || typeof data.type !== 'string') return null;
    const knownTypes = [
      'connected', 'segment', 'interim', 'translation', 'translation_chunk',
      'meeting_started', 'recording_status', 'service_status',
      'language_detected', 'backend_switched', 'error',
      'chat_response', 'config_changed', 'audio_level',
    ];
    if (!knownTypes.includes(data.type)) return null;
    // Unsafe cast: validates `type` discriminant only, not field presence/types.
    return data as ServerMessage;
  } catch {
    return null;
  }
}
