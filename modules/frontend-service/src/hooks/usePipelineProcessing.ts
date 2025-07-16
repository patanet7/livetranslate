/**
 * Pipeline Processing Hook
 * 
 * Provides real-time audio processing capabilities for the Pipeline Studio
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSnackbar } from 'notistack';
import pipelineApiClient, { 
  PipelineProcessingRequest, 
  PipelineProcessingResponse,
  RealTimeProcessingSession 
} from '@/services/pipelineApiClient';

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

export const usePipelineProcessing = () => {
  const { enqueueSnackbar } = useSnackbar();
  
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
  ): Promise<PipelineProcessingResponse | null> => {
    try {
      setIsProcessing(true);
      setError(null);
      setProcessingProgress(0);

      const response = await pipelineApiClient.processPipeline(request);
      
      if (response.success) {
        setProcessedAudio(response.processedAudio || null);
        setMetrics({
          totalLatency: response.metrics.totalLatency,
          stageLatencies: response.metrics.stageLatencies,
          qualityMetrics: response.metrics.qualityMetrics,
          cpuUsage: response.metrics.cpuUsage,
          chunksProcessed: 0,
          averageLatency: response.metrics.totalLatency,
          qualityScore: calculateQualityScore(response.metrics.qualityMetrics),
        });
        
        enqueueSnackbar('Pipeline processing completed successfully', { variant: 'success' });
        return response;
      } else {
        throw new Error(response.errors?.join(', ') || 'Processing failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Processing failed: ${errorMessage}`, { variant: 'error' });
      return null;
    } finally {
      setIsProcessing(false);
      setProcessingProgress(100);
    }
  }, [enqueueSnackbar]);

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

      const response = await pipelineApiClient.processSingleStage(
        stageType,
        audioData,
        stageConfig
      );

      setProcessedAudio(response.processedAudio);
      
      enqueueSnackbar(`${stageType} processing completed`, { variant: 'success' });
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Stage processing failed: ${errorMessage}`, { variant: 'error' });
      return null;
    } finally {
      setIsProcessing(false);
    }
  }, [enqueueSnackbar]);

  /**
   * Analyze audio with FFT
   */
  const analyzeFFT = useCallback(async (audioData: Blob | string) => {
    try {
      const analysis = await pipelineApiClient.getFFTAnalysis(audioData);
      setAudioAnalysis(prev => ({ ...prev, fft: analysis }));
      return analysis;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      enqueueSnackbar(`FFT analysis failed: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [enqueueSnackbar]);

  /**
   * Analyze audio with LUFS metering
   */
  const analyzeLUFS = useCallback(async (audioData: Blob | string) => {
    try {
      const analysis = await pipelineApiClient.getLUFSAnalysis(audioData);
      setAudioAnalysis(prev => ({ ...prev, lufs: analysis }));
      return analysis;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      enqueueSnackbar(`LUFS analysis failed: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [enqueueSnackbar]);

  /**
   * Start real-time processing session
   */
  const startRealtimeProcessing = useCallback(async (
    pipelineConfig: PipelineProcessingRequest['pipelineConfig']
  ) => {
    try {
      setError(null);
      
      // Start backend session
      const session = await pipelineApiClient.startRealtimeSession(pipelineConfig);
      setRealtimeSession(session);

      // Connect WebSocket
      const ws = pipelineApiClient.connectRealtimeWebSocket(session.sessionId, {
        onMetrics: (metricsData) => {
          setMetrics(prev => ({
            ...prev,
            ...metricsData,
            qualityScore: calculateQualityScore(metricsData.qualityMetrics || {}),
          }));
        },
        onProcessedAudio: (audio) => {
          setProcessedAudio(audio);
        },
        onError: (errorMessage) => {
          setError(errorMessage);
          enqueueSnackbar(`Real-time processing error: ${errorMessage}`, { variant: 'error' });
        },
      });

      websocketRef.current = ws;
      setIsRealtimeActive(true);
      
      enqueueSnackbar('Real-time processing session started', { variant: 'success' });
      return session;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      enqueueSnackbar(`Failed to start real-time processing: ${errorMessage}`, { variant: 'error' });
      return null;
    }
  }, [enqueueSnackbar]);

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
          pipelineApiClient.sendAudioChunk(event.data);
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
      pipelineApiClient.disconnect();
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
    if (websocketRef.current && isRealtimeActive) {
      try {
        pipelineApiClient.updatePipelineConfig(stageId, parameters);
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
    
    // Actions
    clearError: () => setError(null),
    clearResults: () => {
      setProcessedAudio(null);
      setMetrics(null);
      setAudioAnalysis({});
    },
  };
};