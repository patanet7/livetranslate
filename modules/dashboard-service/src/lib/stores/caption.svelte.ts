import { SvelteMap, SvelteSet } from 'svelte/reactivity';

import { SPEAKER_COLORS } from '$lib/theme';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';

export type CaptionSource = 'local' | 'screencapture' | 'fireflies';
export type DisplayMode = 'split' | 'subtitle' | 'interpreter' | 'transcript' | 'wire';
export type TranslationState = 'pending' | 'draft' | 'streaming' | 'complete';
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface UnifiedCaption {
  id: string;
  text: string;
  stableText: string;
  unstableText: string;
  translation: string | null;
  translationState: TranslationState;
  speaker: string | null;
  speakerColor: string;
  language: string;
  confidence: number;
  timestamp: number;
  isFinal: boolean;
  isDraft: boolean;
}

// CaptionEvent type for Fireflies WebSocket events
export interface CaptionEventCaption {
  id: string;
  text: string;
  original_text: string;
  translated_text: string;
  speaker_name: string;
  speaker_color: string;
  target_language: string;
  confidence: number;
  duration_seconds: number;
  created_at: string;
  expires_at: string;
  receivedAt?: number;
}

export type CaptionEvent =
  | { event: 'caption_added'; caption: CaptionEventCaption }
  | { event: 'caption_updated'; caption: CaptionEventCaption }
  | { event: 'caption_expired'; caption_id: string }
  | { event: 'session_cleared' };

const MAX_CAPTIONS = 5000;
const STORAGE_KEY = 'livetranslate:caption-config';

/** Per-session LLM sampling tunables. Sent to the orchestration service in
 *  `ConfigMessage.llm` / `StartSessionMessage.llm` so the LLM client applies
 *  them as overrides on top of the resolved connection.
 *
 *  API key is intentionally NOT here — keys live on the server-side
 *  `ai_connections` table and never traverse the WS or localStorage.
 */
export interface LLMOverrides {
  connectionId: string | null;
  model: string | null;
  temperature: number | null;
  maxTokens: number | null;
  topP: number | null;
  topK: number | null;
  repetitionPenalty: number | null;
  presencePenalty: number | null;
}

const DEFAULT_LLM_OVERRIDES: LLMOverrides = {
  connectionId: null,
  model: null,
  temperature: null,
  maxTokens: null,
  topP: null,
  topK: null,
  repetitionPenalty: null,
  presencePenalty: null,
};

/**
 * Per-session Whisper decoding overrides — mirror of Python
 * WhisperParameterOverrides. Persisted via localStorage and forwarded
 * over the WS `ConfigMessage.whisper` field. API key intentionally
 * NOT here — keys live on the server-side whisper_connections table.
 */
export interface WhisperOverrides {
  connectionId: string | null;
  model: string | null;
  temperature: number | null;
  beamSize: number | null;
  noSpeechThreshold: number | null;
  compressionRatioThreshold: number | null;
  languageHint: string | null;
  initialPrompt: string | null;
}

const DEFAULT_WHISPER_OVERRIDES: WhisperOverrides = {
  connectionId: null,
  model: null,
  temperature: null,
  beamSize: null,
  noSpeechThreshold: null,
  compressionRatioThreshold: null,
  languageHint: null,
  initialPrompt: null,
};

interface PersistedConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  displayMode: DisplayMode;
  captionSource: CaptionSource;
  interpreterLangA: string;
  interpreterLangB: string;
  llm: LLMOverrides;
  whisper: WhisperOverrides;
}

function createCaptionStore() {
  // State using Svelte 5 runes
  let captions = $state<UnifiedCaption[]>([]);
  let interimText = $state('');
  let interimConfidence = $state(0);
  let captionSource = $state<CaptionSource>('local');
  let connectionState = $state<ConnectionState>('disconnected');
  let firefliesSessionId = $state<string | null>(null);
  let displayMode = $state<DisplayMode>('split');
  let sourceLanguage = $state<string | null>(null);
  let targetLanguage = $state('zh');
  let detectedLanguage = $state<string | null>(null);
  let interpreterLangA = $state('zh');
  let interpreterLangB = $state('en');
  let llm = $state<LLMOverrides>({ ...DEFAULT_LLM_OVERRIDES });
  let whisper = $state<WhisperOverrides>({ ...DEFAULT_WHISPER_OVERRIDES });
  let transcriptionStatus = $state<'up' | 'down'>('down');
  let translationStatus = $state<'up' | 'down'>('down');
  let isCapturing = $state(false);
  let isRecording = $state(false);
  let recordingChunks = $state(0);
  let chunksSent = $state(0);
  let segmentsReceived = $state(0);
  let translationsReceived = $state(0);
  let lastError = $state<string | null>(null);
  let isMeetingActive = $state(false);
  let meetingSessionId = $state<string | null>(null);
  let meetingStartedAt = $state<string | null>(null);

  // SvelteMap/SvelteSet (vs plain Map/Set): mutations trigger Svelte 5
  // reactivity for consumers that read these collections. Without this,
  // a new speaker → new colour assignment wouldn't propagate until the
  // captions array reassignment incidentally re-ran derived state.
  const speakerColorMap = new SvelteMap<string, string>();
  const seenCaptionIds = new SvelteSet<string>();

  // O(1) lookup index for captions by id. Replaces captions.findIndex(...)
  // in the hot ingest paths. Kept in sync with the captions array — every
  // push / shift / replacement updates this map. Internal book-keeping
  // (never exposed); plain Map is fine here since nothing reads it from
  // inside a reactive context.
  const captionIndex = new Map<string, number>();

  function getSpeakerColor(speaker: string | null): string {
    if (!speaker) return SPEAKER_COLORS[0];
    if (!speakerColorMap.has(speaker)) {
      speakerColorMap.set(speaker, SPEAKER_COLORS[speakerColorMap.size % SPEAKER_COLORS.length]);
    }
    return speakerColorMap.get(speaker)!;
  }

  function persistConfig(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sourceLanguage,
        targetLanguage,
        displayMode,
        captionSource,
        interpreterLangA,
        interpreterLangB,
        llm,
        whisper,
      }));
    } catch { /* ignore */ }
  }

  function restoreConfig(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as Partial<PersistedConfig>;
      if (saved.sourceLanguage !== undefined) sourceLanguage = saved.sourceLanguage;
      if (saved.targetLanguage !== undefined) targetLanguage = saved.targetLanguage;
      if (saved.displayMode !== undefined) displayMode = saved.displayMode;
      if (saved.captionSource !== undefined) captionSource = saved.captionSource;
      if (saved.interpreterLangA !== undefined) interpreterLangA = saved.interpreterLangA;
      if (saved.interpreterLangB !== undefined) interpreterLangB = saved.interpreterLangB;
      if (saved.llm !== undefined && saved.llm !== null) {
        // Old payloads pre-Phase-10 won't have `llm` — start from defaults and
        // overlay whatever the user stored. Forward-compat with new fields.
        llm = { ...DEFAULT_LLM_OVERRIDES, ...saved.llm };
      }
      if (saved.whisper !== undefined && saved.whisper !== null) {
        whisper = { ...DEFAULT_WHISPER_OVERRIDES, ...saved.whisper };
      }
    } catch { /* ignore */ }
  }

  function updateLLMOverrides(patch: Partial<LLMOverrides>): void {
    llm = { ...llm, ...patch };
    persistConfig();
  }

  function updateWhisperOverrides(patch: Partial<WhisperOverrides>): void {
    whisper = { ...whisper, ...patch };
    persistConfig();
  }

  function resetWhisperOverrides(): void {
    whisper = { ...DEFAULT_WHISPER_OVERRIDES };
    persistConfig();
  }

  function resetLLMOverrides(): void {
    llm = { ...DEFAULT_LLM_OVERRIDES };
    persistConfig();
  }

  // Restore on creation
  if (typeof localStorage !== 'undefined') {
    restoreConfig();
  }

  function ingestSegment(msg: SegmentMessage): void {
    if (typeof msg.text !== 'string') return;
    segmentsReceived++;

    // Prefix ID for source isolation
    const id = `lb_${msg.segment_id}`;
    const existingIdx = captionIndex.get(id);
    const incomingDraft = msg.is_draft ?? false;
    const stableText = msg.stable_text ?? msg.text;
    const unstableText = msg.unstable_text ?? '';
    const text = [msg.stable_text, msg.unstable_text].filter(Boolean).join(' ') || msg.text;

    if (existingIdx !== undefined) {
      // Update existing caption in place. Mutating individual fields keeps
      // object identity and reactive scope tight: consumers that read e.g.
      // `cap.translation` don't re-run on a stable-text update.
      const existing = captions[existingIdx];
      if (!existing.isDraft && incomingDraft) return; // Don't overwrite final with draft
      existing.text = text;
      existing.stableText = stableText;
      existing.unstableText = unstableText;
      existing.speaker = msg.speaker_id;
      existing.speakerColor = getSpeakerColor(msg.speaker_id);
      existing.language = msg.language;
      existing.confidence = msg.confidence;
      existing.isFinal = msg.is_final;
      existing.isDraft = incomingDraft;
      // translation / translationState / timestamp preserved across updates
    } else {
      const caption: UnifiedCaption = {
        id,
        text,
        stableText,
        unstableText,
        translation: null,
        translationState: 'pending',
        speaker: msg.speaker_id,
        speakerColor: getSpeakerColor(msg.speaker_id),
        language: msg.language,
        confidence: msg.confidence,
        timestamp: Date.now(),
        isFinal: msg.is_final,
        isDraft: incomingDraft,
      };
      appendCappedCaption(caption);
    }

    if (msg.is_final) {
      interimText = '';
      interimConfidence = 0;
    }
    if (msg.language) {
      detectedLanguage = msg.language;
    }
  }

  /** Append a caption, evicting from the front if MAX_CAPTIONS is exceeded.
   *  Keeps captionIndex in sync. Uses push/shift on the reactive array — both
   *  are observed by Svelte 5 deep proxying. */
  function appendCappedCaption(caption: UnifiedCaption): void {
    captions.push(caption);
    captionIndex.set(caption.id, captions.length - 1);
    if (captions.length > MAX_CAPTIONS) {
      // Eviction is rare (only when n > 5000) — pay the reindex cost then.
      const removed = captions.shift();
      if (removed) captionIndex.delete(removed.id);
      for (let i = 0; i < captions.length; i++) {
        captionIndex.set(captions[i].id, i);
      }
    }
  }

  function ingestTranslation(msg: TranslationMessage): void {
    if (typeof msg.text !== 'string') return;
    translationsReceived++;

    const id = `lb_${msg.transcript_id}`;
    const idx = captionIndex.get(id);
    if (idx === undefined) return;

    const c = captions[idx];
    const isDraft = msg.is_draft ?? false;
    if (c.translationState === 'complete') return;
    if (isDraft && c.translationState !== 'pending' && c.translationState !== 'draft') return;

    c.translation = msg.text;
    c.translationState = isDraft ? 'draft' : 'complete';
  }

  // The streaming-chunk path is the burst hotspot — under load a single
  // translation arrives as 20-30 chunks within a few hundred ms. Two
  // property writes per chunk, instead of an O(n) array.map allocation,
  // is the difference between smooth UI and visible lockup at session scale.
  function ingestTranslationChunk(msg: TranslationChunkMessage): void {
    const id = `lb_${msg.transcript_id}`;
    const idx = captionIndex.get(id);
    if (idx === undefined) return;

    const c = captions[idx];
    if (c.translationState === 'complete') return;

    // Draft → streaming: discard the draft text and start fresh from the
    // first streaming chunk (the streaming pass replaces the draft entirely).
    const base = c.translationState === 'draft' ? '' : (c.translation ?? '');
    c.translation = base + msg.delta;
    c.translationState = 'streaming';
  }

  function ingestCaptionEvent(event: CaptionEvent): void {
    if (event.event === 'caption_added' || event.event === 'caption_updated') {
      const cap = event.caption;
      const id = `ff_${cap.id}`;

      // Skip duplicates on reconnect
      if (event.event === 'caption_added' && seenCaptionIds.has(id)) return;
      seenCaptionIds.add(id);

      const existingIdx = captionIndex.get(id);
      const stableText = cap.original_text || cap.text;
      const translation = cap.translated_text !== cap.original_text ? cap.translated_text : null;
      const translationState: TranslationState = cap.translated_text ? 'complete' : 'pending';

      if (existingIdx !== undefined) {
        const existing = captions[existingIdx];
        existing.text = stableText;
        existing.stableText = stableText;
        existing.translation = translation;
        existing.translationState = translationState;
        existing.speaker = cap.speaker_name;
        existing.speakerColor = cap.speaker_color || getSpeakerColor(cap.speaker_name);
        existing.language = cap.target_language || 'auto';
        existing.confidence = cap.confidence;
        // timestamp / isFinal / isDraft preserved
      } else {
        const caption: UnifiedCaption = {
          id,
          text: stableText,
          stableText,
          unstableText: '',
          translation,
          translationState,
          speaker: cap.speaker_name,
          speakerColor: cap.speaker_color || getSpeakerColor(cap.speaker_name),
          language: cap.target_language || 'auto',
          confidence: cap.confidence,
          timestamp: cap.receivedAt || Date.now(),
          isFinal: true,
          isDraft: false,
        };
        appendCappedCaption(caption);
        segmentsReceived++;
        if (caption.translation) translationsReceived++;
      }
    } else if (event.event === 'session_cleared') {
      captions.length = 0;
      captionIndex.clear();
      seenCaptionIds.clear();
    }
    // caption_expired: keep for history
  }

  function ingestInterim(text: string, confidence: number): void {
    interimText = text;
    interimConfidence = confidence;
  }

  function startMeeting(sessionId: string, startedAt: string): void {
    isMeetingActive = true;
    meetingSessionId = sessionId;
    meetingStartedAt = startedAt;
  }

  function endMeeting(): void {
    isMeetingActive = false;
    meetingSessionId = null;
    meetingStartedAt = null;
    isRecording = false;
    recordingChunks = 0;
  }

  function clear(): void {
    // Mutate the reactive array in place rather than reassigning. Same
    // observable result, no fresh allocation, and consistent with the
    // ingest paths that now mutate fields in place.
    captions.length = 0;
    captionIndex.clear();
    interimText = '';
    interimConfidence = 0;
    chunksSent = 0;
    segmentsReceived = 0;
    translationsReceived = 0;
    lastError = null;
    speakerColorMap.clear();
    seenCaptionIds.clear();
  }

  return {
    get captions() { return captions; },
    get interimText() { return interimText; },
    get interimConfidence() { return interimConfidence; },
    get captionSource() { return captionSource; },
    set captionSource(v: CaptionSource) { captionSource = v; persistConfig(); },
    get connectionState() { return connectionState; },
    set connectionState(v: ConnectionState) { connectionState = v; },
    get firefliesSessionId() { return firefliesSessionId; },
    set firefliesSessionId(v: string | null) { firefliesSessionId = v; },
    get displayMode() { return displayMode; },
    set displayMode(v: DisplayMode) { displayMode = v; persistConfig(); },
    get sourceLanguage() { return sourceLanguage; },
    set sourceLanguage(v: string | null) { sourceLanguage = v; persistConfig(); },
    get targetLanguage() { return targetLanguage; },
    set targetLanguage(v: string) { targetLanguage = v; persistConfig(); },
    get detectedLanguage() { return detectedLanguage; },
    set detectedLanguage(v: string | null) { detectedLanguage = v; },
    get interpreterLangA() { return interpreterLangA; },
    set interpreterLangA(v: string) { interpreterLangA = v; persistConfig(); },
    get interpreterLangB() { return interpreterLangB; },
    set interpreterLangB(v: string) { interpreterLangB = v; persistConfig(); },
    get transcriptionStatus() { return transcriptionStatus; },
    set transcriptionStatus(v: 'up' | 'down') { transcriptionStatus = v; },
    get translationStatus() { return translationStatus; },
    set translationStatus(v: 'up' | 'down') { translationStatus = v; },
    get isCapturing() { return isCapturing; },
    set isCapturing(v: boolean) { isCapturing = v; },
    get isRecording() { return isRecording; },
    set isRecording(v: boolean) { isRecording = v; },
    get recordingChunks() { return recordingChunks; },
    set recordingChunks(v: number) { recordingChunks = v; },
    get chunksSent() { return chunksSent; },
    set chunksSent(v: number) { chunksSent = v; },
    get segmentsReceived() { return segmentsReceived; },
    get translationsReceived() { return translationsReceived; },
    get lastError() { return lastError; },
    set lastError(v: string | null) { lastError = v; },
    get isMeetingActive() { return isMeetingActive; },
    get meetingSessionId() { return meetingSessionId; },
    get meetingStartedAt() { return meetingStartedAt; },
    get llm() { return llm; },
    updateLLMOverrides,
    resetLLMOverrides,
    get whisper() { return whisper; },
    updateWhisperOverrides,
    resetWhisperOverrides,
    getSpeakerColor,
    ingestSegment,
    ingestTranslation,
    ingestTranslationChunk,
    ingestCaptionEvent,
    ingestInterim,
    startMeeting,
    endMeeting,
    clear,
    restoreConfig,
  };
}

export const captionStore = createCaptionStore();
