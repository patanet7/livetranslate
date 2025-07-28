/**
 * Pipeline Processing Hook
 * 
 * Provides real-time audio processing capabilities for the Pipeline Studio
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSnackbar } from 'notistack';
import {
  useProcessPipelineMutation,
  useProcessSingleStageMutation,
  useGetFFTAnalysisMutation,
  useGetLUFSAnalysisMutation,
  useStartRealtimeSessionMutation,
  useGetPresetsQuery,
  useSaveProcessingPresetMutation,
} from '@/store/slices/apiSlice';

interface ProcessingMetrics {
  totalLatency: number;
  stageLatencies: Record<string, number>;
  qualityMetrics: {
    snr: number;
    thd: number;
    lufs: number;
    rms: number;
  };
  cpuUsage: number;
  chunksProcessed: number;
  averageLatency: number;
  qualityScore: number;
}

interface AudioAnalysis {
  fft?: {
    frequencies: number[];
    magnitudes: number[];
    spectralFeatures: any;
    voiceCharacteristics: any;
  };
  lufs?: {
    integratedLoudness: number;
    loudnessRange: number;
    truePeak: number;
    complianceCheck: any;
  };
}

interface RealTimeProcessingSession {
  sessionId: string;
  pipelineId: string;
  status: 'initializing' | 'running' | 'paused' | 'stopped' | 'error';
  metrics: {
    chunksProcessed: number;
    averageLatency: number;
    qualityScore: number;
  };
}

interface PipelineProcessingRequest {
  pipelineConfig: {
    id: string;
    name: string;
    stages: any[];
    connections: any[];
  };
  audioData?: Blob | string;
  processingMode: 'realtime' | 'batch' | 'preview';
  outputFormat?: 'wav' | 'mp3' | 'base64';
  metadata?: any;
}

export const usePipelineProcessing = () => {
  const { enqueueSnackbar } = useSnackbar();
  
  // RTK Query mutations
  const [processPipelineAPI] = useProcessPipelineMutation();
  const [processSingleStageAPI] = useProcessSingleStageMutation();
  const [getFFTAnalysisAPI] = useGetFFTAnalysisMutation();
  const [getLUFSAnalysisAPI] = useGetLUFSAnalysisMutation();
  const [startRealtimeSessionAPI] = useStartRealtimeSessionMutation();
  const [savePresetAPI] = useSaveProcessingPresetMutation();
  
  // RTK Query queries
  const { data: presetsData } = useGetPresetsQuery();
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processedAudio, setProcessedAudio] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ProcessingMetrics | null>(null);
  const [audioAnalysis, setAudioAnalysis] = useState<AudioAnalysis>({});
  
  // Real-time session state
  const [realtimeSession, setRealtimeSession] = useState<RealTimeProcessingSession | null>(null);
  const [isRealtimeActive, setIsRealtimeActive] = useState(false);
  const websocketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  
  // Error handling
  const [error, setError] = useState<string | null>(null);

  /**
   * Process audio through a complete pipeline (batch mode)
   */
  const processPipeline = useCallback(async (
    request: PipelineProcessingRequest
  ): Promise<any | null> => {
    try {
      setIsProcessing(true);
      setError(null);
      setProcessingProgress(0);

      const response = await processPipelineAPI(request).unwrap();
      
      if (response.data?.success !== false) {
        const responseData = response.data || response;
        setProcessedAudio(responseData.processedAudio || null);
        setMetrics({
          totalLatency: responseData.metrics?.totalLatency || 0,
          stageLatencies: responseData.metrics?.stageLatencies || {},
          qualityMetrics: responseData.metrics?.qualityMetrics || {},
          cpuUsage: responseData.metrics?.cpuUsage || 0,
          chunksProcessed: 0,
          averageLatency: responseData.metrics?.totalLatency || 0,
          qualityScore: calculateQualityScore(responseData.metrics?.qualityMetrics || {}),
        });
        
        enqueueSnackbar('Pipeline processing completed successfully', { variant: 'success' });
        return response;
      } else {
        throw new Error(response.data?.errors?.join(', ') || 'Processing failed');
      }
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Processing failed: ${errorMessage}`, { variant: 'error' });
      return null;
    } finally {
      setIsProcessing(false);
      setProcessingProgress(100);
    }
  }, [processPipelineAPI, enqueueSnackbar]);

  /**
   * Process audio through a single stage for testing
   */
  const processSingleStage = useCallback(async (
    stageType: string,
    audioData: Blob | string,
    stageConfig: Record<string, any>
  ) => {
    try {
      setIsProcessing(true);
      setError(null);

      const response = await processSingleStageAPI({
        stageType,
        audioData,
        stageConfig
      }).unwrap();

      const responseData = response.data || response;
      setProcessedAudio(responseData.processedAudio);
      
      enqueueSnackbar(`${stageType} processing completed`, { variant: 'success' });
      return response;
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Stage processing failed: ${errorMessage}`, { variant: 'error' });
      return null;
    } finally {
      setIsProcessing(false);
    }
  }, [processSingleStageAPI, enqueueSnackbar]);

  /**
   * Analyze audio with FFT
   */
  const analyzeFFT = useCallback(async (audioData: Blob | string) => {
    try {
      const response = await getFFTAnalysisAPI(audioData).unwrap();
      const analysis = response.data || response;
      setAudioAnalysis(prev => ({ ...prev, fft: analysis }));
      return analysis;
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      enqueueSnackbar(`FFT analysis failed: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [getFFTAnalysisAPI, enqueueSnackbar]);

  /**
   * Analyze audio with LUFS metering
   */
  const analyzeLUFS = useCallback(async (audioData: Blob | string) => {
    try {
      const response = await getLUFSAnalysisAPI(audioData).unwrap();
      const analysis = response.data || response;
      setAudioAnalysis(prev => ({ ...prev, lufs: analysis }));
      return analysis;
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      enqueueSnackbar(`LUFS analysis failed: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [getLUFSAnalysisAPI, enqueueSnackbar]);

  /**
   * Start real-time processing session
   */
  const startRealtimeProcessing = useCallback(async (
    pipelineConfig: PipelineProcessingRequest['pipelineConfig']
  ) => {
    try {
      setError(null);
      
      // Start backend session
      const response = await startRealtimeSessionAPI({ pipelineConfig }).unwrap();
      const session = response.data || response;
      setRealtimeSession(session);

      // Note: WebSocket functionality would need to be implemented separately
      // as RTK Query doesn't handle WebSocket connections
      setIsRealtimeActive(true);
      
      enqueueSnackbar('Real-time processing session started', { variant: 'success' });
      return session;
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Failed to start real-time processing: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [startRealtimeSessionAPI, enqueueSnackbar]);

  /**
   * Start microphone capture for real-time processing
   */
  const startMicrophoneCapture = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false, // Let our pipeline handle it
          autoGainControl: false,  // Let our pipeline handle it
        },
      });

      // Initialize Web Audio API
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });

      // Create MediaRecorder for chunk-based processing
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0 && websocketRef.current) {
          // Note: Audio chunk sending would need to be implemented separately
          // as RTK Query doesn't handle WebSocket connections
          console.log('Audio chunk ready for processing:', event.data.size, 'bytes');
        }
      };

      // Record in 100ms chunks for real-time processing
      mediaRecorderRef.current.start(100);
      
      enqueueSnackbar('Microphone capture started', { variant: 'success' });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Failed to start microphone: ${errorMessage}`, { variant: 'error' });
    }
  }, [enqueueSnackbar]);

  /**
   * Stop real-time processing
   */
  const stopRealtimeProcessing = useCallback(() => {
    // Stop microphone
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Close WebSocket
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }

    setIsRealtimeActive(false);
    setRealtimeSession(null);
    
    enqueueSnackbar('Real-time processing stopped', { variant: 'info' });
  }, [enqueueSnackbar]);

  /**
   * Update pipeline configuration in real-time
   */
  const updateRealtimeConfig = useCallback((stageId: string, parameters: Record<string, any>) => {
    if (websocketRef.current && isRealtimeActive && websocketRef.current.readyState === WebSocket.OPEN) {
      try {
        websocketRef.current.send(JSON.stringify({
          type: 'update_stage',
          stage_id: stageId,
          parameters: parameters,
        }));
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        enqueueSnackbar(`Failed to update configuration: ${errorMessage}`, { variant: 'error' });
      }
    }
  }, [isRealtimeActive, enqueueSnackbar]);

  /**
   * Get audio from processed result (for playback)
   */
  const getProcessedAudioBlob = useCallback(async (): Promise<Blob | null> => {
    if (!processedAudio) return null;

    try {
      // Convert base64 to blob
      const byteCharacters = atob(processedAudio);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], { type: 'audio/wav' });
    } catch (err) {
      console.error('Failed to convert processed audio:', err);
      return null;
    }
  }, [processedAudio]);

  /**
   * Get available presets
   */
  const getPresets = useCallback(() => {
    return presetsData || [];
  }, [presetsData]);

  /**
   * Save custom preset
   */
  const savePreset = useCallback(async (
    name: string,
    pipelineConfig: PipelineProcessingRequest['pipelineConfig'],
    metadata: {
      description: string;
      category: string;
      tags: string[];
    }
  ) => {
    try {
      await savePresetAPI({
        name,
        pipelineConfig,
        metadata
      }).unwrap();
      
      enqueueSnackbar(`Preset '${name}' saved successfully`, { variant: 'success' });
    } catch (err: any) {
      const errorMessage = err?.data?.message || err?.message || 'Unknown error occurred';
      enqueueSnackbar(`Failed to save preset: ${errorMessage}`, { variant: 'error' });
      throw err;
    }
  }, [savePresetAPI, enqueueSnackbar]);

  /**
   * Calculate overall quality score from metrics
   */
  const calculateQualityScore = (qualityMetrics: Partial<ProcessingMetrics['qualityMetrics']>): number => {
    if (!qualityMetrics.snr || !qualityMetrics.rms) return 0;
    
    // Simple quality scoring based on SNR, RMS level, and distortion
    const snrScore = Math.min(qualityMetrics.snr / 20, 1) * 40; // 0-40 points
    const rmsScore = (1 - Math.abs(qualityMetrics.rms + 20) / 20) * 30; // 0-30 points  
    const thdScore = qualityMetrics.thd ? Math.max(0, (1 - qualityMetrics.thd) * 30) : 30; // 0-30 points
    
    return Math.round(snrScore + rmsScore + thdScore); // 0-100 scale
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRealtimeProcessing();
    };
  }, [stopRealtimeProcessing]);

  return {
    // Processing state
    isProcessing,
    processingProgress,
    processedAudio,
    metrics,
    audioAnalysis,
    error,
    
    // Real-time state
    realtimeSession,
    isRealtimeActive,
    
    // Processing functions
    processPipeline,
    processSingleStage,
    analyzeFFT,
    analyzeLUFS,
    
    // Real-time functions
    startRealtimeProcessing,
    startMicrophoneCapture,
    stopRealtimeProcessing,
    updateRealtimeConfig,
    
    // Utility functions
    getProcessedAudioBlob,
    
    // Preset management
    getPresets,
    savePreset,
    
    // Actions
    clearError: () => setError(null),
    clearResults: () => {
      setProcessedAudio(null);
      setMetrics(null);
      setAudioAnalysis({});
    },
  };
};