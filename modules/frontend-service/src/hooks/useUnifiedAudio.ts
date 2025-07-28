/**
 * Unified Audio Hook
 * 
 * RTK Query-based replacement for unifiedAudioManager.ts
 * Provides centralized audio processing functionality with standardized patterns
 */

import { useCallback, useRef, useState } from 'react';
import { useAppDispatch } from '@/store';
import { addNotification } from '@/store/slices/uiSlice';
import { addProcessingLog } from '@/store/slices/audioSlice';
import {
  useUploadAudioFileMutation,
  useProcessAudioMutation,
  useTranslateTextMutation,
  useGetSystemHealthQuery,
} from '@/store/slices/apiSlice';

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface AudioProcessingConfig {
  whisperModel?: string;
  enableTranscription?: boolean;
  enableTranslation?: boolean;
  enableDiarization?: boolean;
  enableVAD?: boolean;
  targetLanguages?: string[];
  audioProcessing?: boolean;
  noiseReduction?: boolean;
  speechEnhancement?: boolean;
  translationQuality?: 'fast' | 'balanced' | 'high_quality';
  chunkDuration?: number;
  sessionId?: string;
}

export interface AudioProcessingResult {
  processing_result?: {
    id: string;
    text: string;
    confidence: number;
    language: string;
    processing_time: number;
    segments?: any[];
    speakers?: any[];
  };
  translations?: Record<string, {
    translated_text: string;
    confidence: number;
    source_language: string;
    target_language: string;
    processing_time: number;
  }>;
  pipeline_result?: {
    stage_results: any[];
    total_processing_time_ms: number;
    overall_quality_score: number;
  };
}

export interface TranscriptionOptions {
  model?: string;
  language?: string;
  enableDiarization?: boolean;
  enableVAD?: boolean;
  sessionId?: string;
}

export interface TranslationOptions {
  sourceLanguage?: string;
  targetLanguages: string[];
  quality?: 'fast' | 'balanced' | 'high_quality';
  sessionId?: string;
}

export interface StreamingSession {
  sessionId: string;
  isActive: boolean;
  config: AudioProcessingConfig;
  websocket?: WebSocket;
}

// ============================================================================
// Unified Audio Hook
// ============================================================================

export const useUnifiedAudio = () => {
  const dispatch = useAppDispatch();
  const activeSessionsRef = useRef<Map<string, StreamingSession>>(new Map());
  const requestTimeoutRef = useRef<number>(30000); // 30 second timeout

  // RTK Query hooks
  const [uploadAudioFile] = useUploadAudioFileMutation();
  const [processAudio] = useProcessAudioMutation();
  const [translateText] = useTranslateTextMutation();
  const { data: systemHealth } = useGetSystemHealthQuery();

  // Local state
  const [activeStreams, setActiveStreams] = useState<string[]>([]);

  // ============================================================================
  // Audio Upload and Processing
  // ============================================================================

  const uploadAndProcessAudio = useCallback(async (
    audioBlob: Blob,
    config: AudioProcessingConfig = {}
  ): Promise<AudioProcessingResult> => {
    try {
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Starting audio upload and processing...',
        timestamp: Date.now()
      }));

      const result = await uploadAudioFile({
        file: new File([audioBlob], 'audio.webm', { type: 'audio/webm' }),
        metadata: config
      }).unwrap();

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Audio upload and processing completed successfully',
        timestamp: Date.now()
      }));

      dispatch(addNotification({
        type: 'success',
        title: 'Audio Processed',
        message: 'Audio processing completed successfully',
        autoHide: true
      }));

      return result.data || result;

    } catch (error: any) {
      const errorMessage = error?.data?.message || error?.message || 'Unknown error occurred';
      
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Audio upload and processing failed: ${errorMessage}`,
        timestamp: Date.now()
      }));

      dispatch(addNotification({
        type: 'error',
        title: 'Audio Processing Error',
        message: `Audio processing failed: ${errorMessage}`,
        autoHide: false
      }));

      throw error;
    }
  }, [uploadAudioFile, dispatch]);

  // ============================================================================
  // Transcription Services
  // ============================================================================

  const transcribeAudio = useCallback(async (
    audioBlob: Blob,
    options: TranscriptionOptions = {}
  ): Promise<any> => {
    try {
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Starting audio transcription...',
        timestamp: Date.now()
      }));

      // Use the audio processing endpoint with transcription config
      const result = await processAudio({
        audioBlob,
        preset: options.model,
        stages: options.enableDiarization ? ['transcription', 'diarization'] : ['transcription']
      }).unwrap();

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Audio transcription completed successfully',
        timestamp: Date.now()
      }));

      return result.data || result;

    } catch (error: any) {
      const errorMessage = error?.data?.message || error?.message || 'Unknown error occurred';
      
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Audio transcription failed: ${errorMessage}`,
        timestamp: Date.now()
      }));

      throw error;
    }
  }, [processAudio, dispatch]);

  const transcribeWithModel = useCallback(async (
    audioBlob: Blob,
    modelName: string,
    options: TranscriptionOptions = {}
  ): Promise<any> => {
    return transcribeAudio(audioBlob, { ...options, model: modelName });
  }, [transcribeAudio]);

  // ============================================================================
  // Translation Services  
  // ============================================================================

  const translateTextContent = useCallback(async (
    text: string,
    options: TranslationOptions
  ): Promise<any> => {
    try {
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Starting text translation...',
        timestamp: Date.now()
      }));

      const result = await translateText({
        text,
        sourceLanguage: options.sourceLanguage || 'auto',
        targetLanguage: options.targetLanguages[0], // RTK Query endpoint expects single target
        context: options.sessionId
      }).unwrap();

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Text translation completed successfully',
        timestamp: Date.now()
      }));

      return result.data || result;

    } catch (error: any) {
      const errorMessage = error?.data?.message || error?.message || 'Unknown error occurred';
      
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Text translation failed: ${errorMessage}`,
        timestamp: Date.now()
      }));

      throw error;
    }
  }, [translateText, dispatch]);

  const translateFromTranscription = useCallback(async (
    transcriptionResult: any,
    targetLanguages: string[],
    options: Partial<TranslationOptions> = {}
  ): Promise<any> => {
    const text = transcriptionResult.text || transcriptionResult.transcription || '';
    const sourceLanguage = transcriptionResult.language || transcriptionResult.detected_language || 'auto';
    
    if (!text.trim()) {
      throw new Error('No text found in transcription result to translate');
    }

    return translateTextContent(text, {
      sourceLanguage,
      targetLanguages,
      ...options
    });
  }, [translateTextContent]);

  // ============================================================================
  // Streaming Audio Processing
  // ============================================================================

  const startStreamingSession = useCallback(async (
    config: AudioProcessingConfig
  ): Promise<StreamingSession> => {
    const sessionId = config.sessionId || `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const session: StreamingSession = {
      sessionId,
      isActive: true,
      config,
    };

    // Add to active sessions
    activeSessionsRef.current.set(sessionId, session);
    setActiveStreams(prev => [...prev, sessionId]);
    
    dispatch(addNotification({
      type: 'success',
      title: 'Streaming Started',
      message: `Audio streaming session ${sessionId} started`,
      autoHide: true
    }));

    dispatch(addProcessingLog({
      level: 'INFO',
      message: `Streaming session ${sessionId} initialized`,
      timestamp: Date.now()
    }));

    return session;
  }, [dispatch]);

  const sendAudioChunk = useCallback(async (
    sessionId: string,
    audioChunk: Blob,
    chunkId?: string
  ): Promise<AudioProcessingResult> => {
    const session = activeSessionsRef.current.get(sessionId);
    if (!session || !session.isActive) {
      throw new Error(`No active streaming session found: ${sessionId}`);
    }

    try {
      // Use the existing audio processing endpoint for chunks
      const result = await processAudio({
        audioBlob: audioChunk,
        stages: ['chunk_processing']
      }).unwrap();

      return result.data || result;

    } catch (error: any) {
      const errorMessage = error?.data?.message || error?.message || 'Unknown error occurred';
      
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Chunk processing failed: ${errorMessage}`,
        timestamp: Date.now()
      }));

      throw error;
    }
  }, [processAudio, dispatch]);

  const stopStreamingSession = useCallback(async (sessionId: string): Promise<void> => {
    const session = activeSessionsRef.current.get(sessionId);
    if (!session) {
      return; // Session already stopped or doesn't exist
    }

    // Close WebSocket if exists
    if (session.websocket) {
      session.websocket.close();
    }

    // Mark session as inactive
    session.isActive = false;

    // Remove from active sessions
    activeSessionsRef.current.delete(sessionId);
    setActiveStreams(prev => prev.filter(id => id !== sessionId));
    
    dispatch(addNotification({
      type: 'info',
      title: 'Streaming Stopped',
      message: `Audio streaming session ${sessionId} stopped`,
      autoHide: true
    }));

    dispatch(addProcessingLog({
      level: 'INFO',
      message: `Streaming session ${sessionId} terminated`,
      timestamp: Date.now()
    }));
  }, [dispatch]);

  // ============================================================================
  // Complete Audio Processing Workflows
  // ============================================================================

  const processAudioComplete = useCallback(async (
    audioBlob: Blob,
    config: AudioProcessingConfig = {}
  ): Promise<AudioProcessingResult> => {
    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Starting complete audio processing workflow...',
      timestamp: Date.now()
    }));

    try {
      // Use the unified upload endpoint which handles all processing
      const result = await uploadAndProcessAudio(audioBlob, config);
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Complete audio processing workflow finished successfully',
        timestamp: Date.now()
      }));

      return result;

    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Complete audio processing workflow failed: ${error}`,
        timestamp: Date.now()
      }));
      throw error;
    }
  }, [uploadAndProcessAudio, dispatch]);

  const processAudioWithTranscriptionAndTranslation = useCallback(async (
    audioBlob: Blob,
    targetLanguages: string[],
    config: Partial<AudioProcessingConfig> = {}
  ): Promise<AudioProcessingResult> => {
    const fullConfig: AudioProcessingConfig = {
      enableTranscription: true,
      enableTranslation: true,
      targetLanguages,
      ...config
    };

    return processAudioComplete(audioBlob, fullConfig);
  }, [processAudioComplete]);

  // ============================================================================
  // Service Health and Status
  // ============================================================================

  const getServiceStatus = useCallback(() => {
    return systemHealth?.data || null;
  }, [systemHealth]);

  // ============================================================================
  // Session Management
  // ============================================================================

  const getActiveStreamingSessions = useCallback((): string[] => {
    return activeStreams;
  }, [activeStreams]);

  const getStreamingSessionInfo = useCallback((sessionId: string): StreamingSession | undefined => {
    return activeSessionsRef.current.get(sessionId);
  }, []);

  const cleanupInactiveSessions = useCallback((): void => {
    const inactiveSessions: string[] = [];
    
    activeSessionsRef.current.forEach((session, sessionId) => {
      if (!session.isActive) {
        inactiveSessions.push(sessionId);
      }
    });

    inactiveSessions.forEach(sessionId => {
      activeSessionsRef.current.delete(sessionId);
    });

    if (inactiveSessions.length > 0) {
      setActiveStreams(prev => prev.filter(id => !inactiveSessions.includes(id)));
      
      dispatch(addProcessingLog({
        level: 'INFO',
        message: `Cleaned up ${inactiveSessions.length} inactive streaming sessions`,
        timestamp: Date.now()
      }));
    }
  }, [dispatch]);

  // ============================================================================
  // Configuration and Settings
  // ============================================================================

  const updateRequestTimeout = useCallback((timeoutMs: number): void => {
    requestTimeoutRef.current = Math.max(1000, Math.min(120000, timeoutMs)); // 1s to 2min
  }, []);

  const getRequestTimeout = useCallback((): number => {
    return requestTimeoutRef.current;
  }, []);

  // ============================================================================
  // Return Public API
  // ============================================================================

  return {
    // Core audio processing
    uploadAndProcessAudio,
    processAudioComplete,
    processAudioWithTranscriptionAndTranslation,

    // Transcription
    transcribeAudio,
    transcribeWithModel,

    // Translation
    translateText: translateTextContent,
    translateFromTranscription,

    // Streaming
    startStreamingSession,
    sendAudioChunk,
    stopStreamingSession,

    // Service status
    getServiceStatus,

    // Session management
    getActiveStreamingSessions,
    getStreamingSessionInfo,
    cleanupInactiveSessions,

    // Configuration
    updateRequestTimeout,
    getRequestTimeout,

    // State
    activeStreams,
    isHealthy: !!systemHealth?.data,
  };
};

export type UnifiedAudioHook = ReturnType<typeof useUnifiedAudio>;
export default useUnifiedAudio;