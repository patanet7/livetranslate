/**
 * useAudioStreaming Hook
 *
 * Shared hook for audio streaming with chunked upload.
 * Consolidates duplicate implementations from:
 * - StreamingProcessor/index.tsx (~200 lines)
 * - MeetingTest/index.tsx (~200 lines)
 * - TranscriptionTesting/index.tsx (~150 lines)
 *
 * Total savings: ~550 lines of duplicate code
 */

import { useState, useCallback, useRef } from 'react';
import { useAppDispatch } from '@/store';
import { addProcessingLog } from '@/store/slices/audioSlice';
import { generateChunkId } from '@/utils/sessionUtils';
import type { ProcessingConfig, StreamingStats } from '@/types/streaming';

export interface AudioStreamingOptions {
  /**
   * Session ID for this streaming session
   */
  sessionId: string;

  /**
   * Audio stream to record from
   */
  audioStream: MediaStream | null;

  /**
   * Chunk duration in seconds (default: 3)
   */
  chunkDuration?: number;

  /**
   * Target languages for translation
   */
  targetLanguages?: string[];

  /**
   * Processing configuration
   */
  processingConfig?: Partial<ProcessingConfig>;

  /**
   * Callback when chunk is processed
   */
  onChunkProcessed?: (chunkId: string, response: any) => void;

  /**
   * Callback when streaming stats update
   */
  onStatsUpdate?: (stats: StreamingStats) => void;

  /**
   * Enable logging (default: true)
   */
  enableLogging?: boolean;
}

/**
 * Hook to manage audio streaming with chunked upload
 */
export const useAudioStreaming = (options: AudioStreamingOptions) => {
  const dispatch = useAppDispatch();
  const {
    sessionId,
    audioStream,
    chunkDuration = 3,
    targetLanguages = [],
    processingConfig = {},
    onChunkProcessed,
    onStatsUpdate,
    enableLogging = true
  } = options;

  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStats, setStreamingStats] = useState<StreamingStats>({
    chunksStreamed: 0,
    totalDuration: 0,
    averageProcessingTime: 0,
    errorCount: 0
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const activeChunksRef = useRef<Set<string>>(new Set());

  /**
   * Send audio chunk to orchestration service
   */
  const sendAudioChunk = useCallback(async (chunkId: string, audioBlob: Blob) => {
    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'chunk.webm');
      formData.append('chunk_id', chunkId);
      formData.append('session_id', sessionId);
      formData.append('target_languages', JSON.stringify(targetLanguages));

      // Add processing config
      if (processingConfig.enableTranscription !== undefined) {
        formData.append('enable_transcription', processingConfig.enableTranscription.toString());
      }
      if (processingConfig.enableTranslation !== undefined) {
        formData.append('enable_translation', processingConfig.enableTranslation.toString());
      }
      if (processingConfig.enableDiarization !== undefined) {
        formData.append('enable_diarization', processingConfig.enableDiarization.toString());
      }
      if (processingConfig.enableVAD !== undefined) {
        formData.append('enable_vad', processingConfig.enableVAD.toString());
      }
      if (processingConfig.whisperModel) {
        formData.append('whisper_model', processingConfig.whisperModel);
      }
      if (processingConfig.translationQuality) {
        formData.append('translation_quality', processingConfig.translationQuality);
      }
      if (processingConfig.audioProcessing !== undefined) {
        formData.append('audio_processing', processingConfig.audioProcessing.toString());
      }
      if (processingConfig.noiseReduction !== undefined) {
        formData.append('noise_reduction', processingConfig.noiseReduction.toString());
      }
      if (processingConfig.speechEnhancement !== undefined) {
        formData.append('speech_enhancement', processingConfig.speechEnhancement.toString());
      }

      const response = await fetch('/api/audio/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      const processingTime = Date.now() - startTime;

      // Update stats
      setStreamingStats(prev => {
        const newStats = {
          ...prev,
          chunksStreamed: prev.chunksStreamed + 1,
          totalDuration: prev.totalDuration + chunkDuration,
          averageProcessingTime: ((prev.averageProcessingTime * prev.chunksStreamed) + processingTime) / (prev.chunksStreamed + 1)
        };
        if (onStatsUpdate) {
          onStatsUpdate(newStats);
        }
        return newStats;
      });

      // Remove from active chunks
      activeChunksRef.current.delete(chunkId);

      // Call callback
      if (onChunkProcessed) {
        onChunkProcessed(chunkId, result);
      }

      if (enableLogging) {
        dispatch(addProcessingLog({
          level: 'SUCCESS',
          message: `Chunk ${chunkId} processed in ${processingTime}ms`,
          timestamp: Date.now()
        }));
      }
    } catch (error) {
      setStreamingStats(prev => ({
        ...prev,
        errorCount: prev.errorCount + 1
      }));

      activeChunksRef.current.delete(chunkId);

      if (enableLogging) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: `Failed to process chunk ${chunkId}: ${error}`,
          timestamp: Date.now()
        }));
      }
    }
  }, [sessionId, targetLanguages, processingConfig, chunkDuration, onChunkProcessed, onStatsUpdate, enableLogging, dispatch]);

  /**
   * Start streaming
   */
  const startStreaming = useCallback(async () => {
    if (!audioStream) {
      if (enableLogging) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: 'No audio stream available',
          timestamp: Date.now()
        }));
      }
      return;
    }

    try {
      setIsStreaming(true);

      const mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (recordingChunksRef.current.length > 0) {
          const audioBlob = new Blob(recordingChunksRef.current, { type: 'audio/webm' });
          const chunkId = generateChunkId();

          activeChunksRef.current.add(chunkId);
          await sendAudioChunk(chunkId, audioBlob);
          recordingChunksRef.current = [];
        }
      };

      // Start recording and set up interval for chunks
      mediaRecorder.start();

      chunkIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start();
        }
      }, chunkDuration * 1000);

      if (enableLogging) {
        dispatch(addProcessingLog({
          level: 'SUCCESS',
          message: `Started streaming with ${chunkDuration}s chunks`,
          timestamp: Date.now()
        }));
      }
    } catch (error) {
      setIsStreaming(false);
      if (enableLogging) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: `Failed to start streaming: ${error}`,
          timestamp: Date.now()
        }));
      }
    }
  }, [audioStream, chunkDuration, sendAudioChunk, enableLogging, dispatch]);

  /**
   * Stop streaming
   */
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);

    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    if (enableLogging) {
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Streaming stopped',
        timestamp: Date.now()
      }));
    }
  }, [enableLogging, dispatch]);

  /**
   * Reset streaming stats
   */
  const resetStats = useCallback(() => {
    setStreamingStats({
      chunksStreamed: 0,
      totalDuration: 0,
      averageProcessingTime: 0,
      errorCount: 0
    });
  }, []);

  return {
    isStreaming,
    streamingStats,
    startStreaming,
    stopStreaming,
    resetStats,
    activeChunks: activeChunksRef.current
  };
};

export default useAudioStreaming;
