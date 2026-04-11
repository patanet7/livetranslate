/**
 * Reactive state for the loopback page using Svelte 5 runes.
 *
 * Manages: captions, translations, connection state, meeting state,
 * display mode, and audio source configuration.
 *
 * Persists UI config (source/target language, display mode, interpreter
 * languages) to localStorage so page refreshes don't lose settings.
 */

import type { SegmentMessage, InterimMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';
import { type DisplayMode, SPEAKER_COLORS } from '$lib/theme';

export type { DisplayMode };

const STORAGE_KEY = 'livetranslate:loopback-config';

interface PersistedConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  displayMode: DisplayMode;
  interpreterLangA: string;
  interpreterLangB: string;
}

function saveToLocalStorage(config: PersistedConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // localStorage unavailable (private browsing, quota exceeded) — silently fail
  }
}

function loadFromLocalStorage(): Partial<PersistedConfig> | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as Partial<PersistedConfig>;
  } catch {
    return null;
  }
}

export type TranslationState = 'pending' | 'draft' | 'streaming' | 'complete';

export interface CaptionEntry {
  id: number;
  segmentId: number;  // Server-side segment_id for translation matching
  text: string;
  stableText: string;      // Confirmed text (won't change)
  unstableText: string;    // Tentative text (may be refined)
  language: string;
  confidence: number;
  speakerId: string | null;
  isFinal: boolean;
  isDraft: boolean;        // True for fast first-pass captions
  translation: string | null;
  translationState: TranslationState;
  timestamp: number;
}

// I2: Cap captions to prevent unbounded growth in long sessions.
// ~2400 segments per 2-hour meeting at 1 segment/3s — 5000 gives ample headroom.
const MAX_CAPTIONS = 5000;

function createLoopbackStore() {
  let captions = $state<CaptionEntry[]>([]);
  let interimText = $state('');
  let interimConfidence = $state(0);
  let displayMode = $state<DisplayMode>('split');
  let connectionState = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  let isCapturing = $state(false);
  let isMeetingActive = $state(false);
  let meetingSessionId = $state<string | null>(null);
  let meetingStartedAt = $state<string | null>(null);
  let transcriptionStatus = $state<'up' | 'down'>('down');
  let translationStatus = $state<'up' | 'down'>('down');
  let isRecording = $state(false);
  let recordingChunks = $state(0);
  let sourceLanguage = $state<string | null>(null);
  let detectedLanguage = $state<string | null>(null);
  let targetLanguage = $state('zh');
  let interpreterLangA = $state('zh');
  let interpreterLangB = $state('en');
  let chunksSent = $state(0);
  let segmentsReceived = $state(0);
  let translationsReceived = $state(0);
  let lastError = $state<string | null>(null);
  let nextId = 0;
  const speakerColorMap = new Map<string, string>();

  function persistConfig(): void {
    saveToLocalStorage({
      sourceLanguage,
      targetLanguage,
      displayMode,
      interpreterLangA,
      interpreterLangB,
    });
  }

  function restoreFromLocalStorage(): void {
    const saved = loadFromLocalStorage();
    if (!saved) return;
    if (saved.sourceLanguage !== undefined) sourceLanguage = saved.sourceLanguage;
    if (saved.targetLanguage !== undefined) targetLanguage = saved.targetLanguage;
    if (saved.displayMode !== undefined) displayMode = saved.displayMode;
    if (saved.interpreterLangA !== undefined) interpreterLangA = saved.interpreterLangA;
    if (saved.interpreterLangB !== undefined) interpreterLangB = saved.interpreterLangB;
  }

  // Restore persisted settings on creation (browser only)
  if (typeof localStorage !== 'undefined') {
    restoreFromLocalStorage();
  }

  function getSpeakerColor(speakerId: string | null): string {
    if (!speakerId) return SPEAKER_COLORS[0];
    if (!speakerColorMap.has(speakerId)) {
      speakerColorMap.set(speakerId, SPEAKER_COLORS[speakerColorMap.size % SPEAKER_COLORS.length]);
    }
    return speakerColorMap.get(speakerId)!;
  }

  function addSegment(msg: SegmentMessage) {
    // I1: Guard against malformed server messages with missing fields
    if (typeof msg.text !== 'string' || typeof msg.confidence !== 'number') return;
    segmentsReceived++;

    // Check if this segment_id already exists (draft→final refinement, or re-send)
    const existingIdx = captions.findIndex(c => c.segmentId === msg.segment_id);

    const entry: CaptionEntry = {
      id: existingIdx >= 0 ? captions[existingIdx].id : nextId++,
      segmentId: msg.segment_id,
      text: [msg.stable_text, msg.unstable_text].filter(Boolean).join(' ') || msg.text,
      stableText: msg.stable_text ?? msg.text,
      unstableText: msg.unstable_text ?? '',
      language: msg.language,
      confidence: msg.confidence,
      speakerId: msg.speaker_id,
      isFinal: msg.is_final,
      isDraft: msg.is_draft ?? false,
      translation: existingIdx >= 0 ? captions[existingIdx].translation : null,
      translationState: existingIdx >= 0 ? captions[existingIdx].translationState : 'pending',
      timestamp: existingIdx >= 0 ? captions[existingIdx].timestamp : Date.now(),
    };

    if (existingIdx >= 0) {
      const existing = captions[existingIdx];

      // Guard 1: Don't overwrite a finalized segment with a stale draft
      if (!existing.isDraft && entry.isDraft) return;

      // Guard 2: If both are finals, the second is a new segment from a
      // duplicate segment_id (counter race). SegmentStore tracks lifecycle
      // so this should be rare — append as a new caption.
      if (!existing.isDraft && !entry.isDraft) {
        entry.id = nextId++;
        captions = [...captions.slice(-(MAX_CAPTIONS - 1)), entry];
      } else {
        // Normal: draft→final refinement — replace in-place (no layout shift)
        captions = captions.map((c, i) => i === existingIdx ? entry : c);
      }
    } else {
      // New segment: append
      // I2: Cap captions array to prevent unbounded growth
      captions = [...captions.slice(-(MAX_CAPTIONS - 1)), entry];
    }

    // Clear interim text when final segment arrives — the interim was
    // a preview of this segment, now replaced by the definitive version.
    if (msg.is_final) {
      interimText = '';
      interimConfidence = 0;
    }

    // Update detected language from segment data (display only — does NOT
    // change the user's source language dropdown selection).
    if (msg.language) {
      detectedLanguage = msg.language;
    }
  }

  function updateInterim(msg: InterimMessage) {
    interimText = msg.text;
    interimConfidence = msg.confidence;
  }

  function appendTranslationChunk(msg: TranslationChunkMessage) {
    const caption = captions.find(c => c.segmentId === msg.transcript_id);
    if (!caption) return;

    // Guard: ignore chunks after translation is already complete
    if (caption.translationState === 'complete') return;

    // Guard: first streaming chunk arriving on a draft translation — clear draft text
    if (caption.translationState === 'draft') {
      caption.translation = '';
    }

    caption.translationState = 'streaming';
    caption.translation = caption.translation === null ? msg.delta : caption.translation + msg.delta;
  }

  function addTranslation(msg: TranslationMessage) {
    // I1: Guard against malformed translation messages
    if (typeof msg.text !== 'string' || typeof msg.transcript_id !== 'number') return;
    translationsReceived++;

    const caption = captions.find(c => c.segmentId === msg.transcript_id);
    if (!caption) return;

    const isDraft = !!(msg.is_draft ?? false);

    // Last-writer-wins guard 1: complete blocks all further updates
    if (caption.translationState === 'complete') return;

    // Last-writer-wins guard 2: don't downgrade — draft can't overwrite streaming/final
    if (isDraft && caption.translationState !== 'pending' && caption.translationState !== 'draft') return;

    caption.translation = msg.text;
    caption.translationState = isDraft ? 'draft' : 'complete';
  }

  function startMeeting(sessionId: string, startedAt: string) {
    isMeetingActive = true;
    meetingSessionId = sessionId;
    meetingStartedAt = startedAt;
  }

  function endMeeting() {
    isMeetingActive = false;
    meetingSessionId = null;
    meetingStartedAt = null;
    isRecording = false;
    recordingChunks = 0;
  }

  function clear() {
    captions = [];
    interimText = '';
    interimConfidence = 0;
    chunksSent = 0;
    segmentsReceived = 0;
    translationsReceived = 0;
    lastError = null;
    nextId = 0;
    speakerColorMap.clear();  // M2: Clear speaker colors with captions
  }

  return {
    get captions() { return captions; },
    get interimText() { return interimText; },
    get interimConfidence() { return interimConfidence; },
    get displayMode() { return displayMode; },
    set displayMode(v: DisplayMode) { displayMode = v; persistConfig(); },
    get connectionState() { return connectionState; },
    set connectionState(v: typeof connectionState) { connectionState = v; },
    get isCapturing() { return isCapturing; },
    set isCapturing(v: boolean) { isCapturing = v; },
    get isMeetingActive() { return isMeetingActive; },
    get meetingSessionId() { return meetingSessionId; },
    get meetingStartedAt() { return meetingStartedAt; },
    get transcriptionStatus() { return transcriptionStatus; },
    set transcriptionStatus(v: 'up' | 'down') { transcriptionStatus = v; },
    get translationStatus() { return translationStatus; },
    set translationStatus(v: 'up' | 'down') { translationStatus = v; },
    get isRecording() { return isRecording; },
    set isRecording(v: boolean) { isRecording = v; },
    get recordingChunks() { return recordingChunks; },
    set recordingChunks(v: number) { recordingChunks = v; },
    get sourceLanguage() { return sourceLanguage; },
    set sourceLanguage(v: string | null) { sourceLanguage = v; persistConfig(); },
    get detectedLanguage() { return detectedLanguage; },
    set detectedLanguage(v: string | null) { detectedLanguage = v; },
    get targetLanguage() { return targetLanguage; },
    set targetLanguage(v: string) { targetLanguage = v; persistConfig(); },
    get interpreterLangA() { return interpreterLangA; },
    set interpreterLangA(v: string) { interpreterLangA = v; persistConfig(); },
    get interpreterLangB() { return interpreterLangB; },
    set interpreterLangB(v: string) { interpreterLangB = v; persistConfig(); },
    get chunksSent() { return chunksSent; },
    set chunksSent(v: number) { chunksSent = v; },
    get segmentsReceived() { return segmentsReceived; },
    get translationsReceived() { return translationsReceived; },
    get lastError() { return lastError; },
    set lastError(v: string | null) { lastError = v; },
    getSpeakerColor,
    addSegment,
    updateInterim,
    appendTranslationChunk,
    addTranslation,
    startMeeting,
    endMeeting,
    clear,
    restoreFromLocalStorage,
  };
}

export const loopbackStore = createLoopbackStore();
