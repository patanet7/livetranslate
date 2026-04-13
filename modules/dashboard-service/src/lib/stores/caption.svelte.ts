import { SPEAKER_COLORS } from '$lib/theme';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';

export type CaptionSource = 'local' | 'screencapture' | 'fireflies';
export type DisplayMode = 'split' | 'subtitle' | 'interpreter' | 'transcript';
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

interface PersistedConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  displayMode: DisplayMode;
  captionSource: CaptionSource;
  interpreterLangA: string;
  interpreterLangB: string;
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

  const speakerColorMap = new Map<string, string>();
  const seenCaptionIds = new Set<string>();

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
    } catch { /* ignore */ }
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
    const existingIdx = captions.findIndex(c => c.id === id);

    const caption: UnifiedCaption = {
      id,
      text: [msg.stable_text, msg.unstable_text].filter(Boolean).join(' ') || msg.text,
      stableText: msg.stable_text ?? msg.text,
      unstableText: msg.unstable_text ?? '',
      translation: existingIdx >= 0 ? captions[existingIdx].translation : null,
      translationState: existingIdx >= 0 ? captions[existingIdx].translationState : 'pending',
      speaker: msg.speaker_id,
      speakerColor: getSpeakerColor(msg.speaker_id),
      language: msg.language,
      confidence: msg.confidence,
      timestamp: existingIdx >= 0 ? captions[existingIdx].timestamp : Date.now(),
      isFinal: msg.is_final,
      isDraft: msg.is_draft ?? false,
    };

    if (existingIdx >= 0) {
      const existing = captions[existingIdx];
      if (!existing.isDraft && caption.isDraft) return; // Don't overwrite final with draft
      captions = captions.map((c, i) => i === existingIdx ? caption : c);
    } else {
      captions = [...captions.slice(-(MAX_CAPTIONS - 1)), caption];
    }

    if (msg.is_final) {
      interimText = '';
      interimConfidence = 0;
    }
    if (msg.language) {
      detectedLanguage = msg.language;
    }
  }

  function ingestTranslation(msg: TranslationMessage): void {
    if (typeof msg.text !== 'string') return;
    translationsReceived++;

    const id = `lb_${msg.transcript_id}`;
    const idx = captions.findIndex(c => c.id === id);
    if (idx < 0) return;

    const isDraft = msg.is_draft ?? false;
    if (captions[idx].translationState === 'complete') return;
    if (isDraft && captions[idx].translationState !== 'pending' && captions[idx].translationState !== 'draft') return;

    captions = captions.map((c, i) => i === idx ? { ...c, translation: msg.text, translationState: isDraft ? 'draft' : 'complete' } : c);
  }

  function ingestTranslationChunk(msg: TranslationChunkMessage): void {
    const id = `lb_${msg.transcript_id}`;
    const idx = captions.findIndex(c => c.id === id);
    if (idx < 0) return;
    if (captions[idx].translationState === 'complete') return;

    const base = captions[idx].translationState === 'draft' ? '' : (captions[idx].translation ?? '');
    captions = captions.map((c, i) => i === idx ? { ...c, translation: base + msg.delta, translationState: 'streaming' } : c);
  }

  function ingestCaptionEvent(event: CaptionEvent): void {
    if (event.event === 'caption_added' || event.event === 'caption_updated') {
      const cap = event.caption;
      const id = `ff_${cap.id}`;

      // Skip duplicates on reconnect
      if (event.event === 'caption_added' && seenCaptionIds.has(id)) return;
      seenCaptionIds.add(id);

      const existingIdx = captions.findIndex(c => c.id === id);

      const caption: UnifiedCaption = {
        id,
        text: cap.original_text || cap.text,
        stableText: cap.original_text || cap.text,
        unstableText: '',
        translation: cap.translated_text !== cap.original_text ? cap.translated_text : null,
        translationState: cap.translated_text ? 'complete' : 'pending',
        speaker: cap.speaker_name,
        speakerColor: cap.speaker_color || getSpeakerColor(cap.speaker_name),
        language: cap.target_language || 'auto',
        confidence: cap.confidence,
        timestamp: cap.receivedAt || Date.now(),
        isFinal: true,
        isDraft: false,
      };

      if (existingIdx >= 0) {
        captions = captions.map((c, i) => i === existingIdx ? caption : c);
      } else {
        captions = [...captions.slice(-(MAX_CAPTIONS - 1)), caption];
        segmentsReceived++;
        if (caption.translation) translationsReceived++;
      }
    } else if (event.event === 'session_cleared') {
      captions = [];
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
    captions = [];
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
