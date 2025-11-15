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

  // RTK Query hooks
  const [uploadAudioFile] = useUploadAudioFileMutation();
  const [processAudio] = useProcessAudioMutation();
  const [translateText] = useTranslateTextMutation();
  const { data: systemHealth } = useGetSystemHealthQuery();

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

      // Extract sessionId from config
      const { sessionId, ...restConfig } = config;

      const result = await uploadAudioFile({
        audio: audioBlob,  // Fixed: use 'audio' parameter name
        config: restConfig,  // Fixed: use 'config' parameter name (not 'metadata')
        sessionId,  // Fixed: pass sessionId separately
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
  // Service Health and Status
  // ============================================================================

  const getServiceStatus = useCallback(() => {
    return systemHealth?.data || null;
  }, [systemHealth]);


  // ============================================================================
  // Return Public API
  // ============================================================================

  return {
    // Core audio processing
    uploadAndProcessAudio,

    // Transcription
    transcribeAudio,

    // Translation
    translateText: translateTextContent,
    translateFromTranscription,

    // Service status
    getServiceStatus,
    isHealthy: !!systemHealth?.data,
  };
};

export type UnifiedAudioHook = ReturnType<typeof useUnifiedAudio>;
export default useUnifiedAudio;