/**
 * useAudioStreaming Hook
 *
 * Real-time audio streaming from microphone to Whisper via WebSocket.
 *
 * Architecture:
 *   Browser Mic → WebSocket → Orchestration → Whisper → Transcription → Browser
 *
 * This follows the same pattern as bot containers for consistency.
 *
 * Usage:
 * ```typescript
 * const {
 *   isConnected,
 *   isStreaming,
 *   segments,
 *   error,
 *   connect,
 *   startStreaming,
 *   stopStreaming,
 *   clearSegments
 * } = useAudioStreaming({
 *   model: 'whisper-base',
 *   language: 'en',
 *   enableVAD: true,
 *   enableDiarization: true
 * });
 * ```
 */

import { useEffect, useState, useCallback, useRef } from 'react';

export interface AudioStreamingConfig {
  model?: string;
  language?: string;
  enableVAD?: boolean;
  enableDiarization?: boolean;
  enableCIF?: boolean;
  enableRollingContext?: boolean;
  orchestrationUrl?: string;
}

export interface TranscriptionSegment {
  type: 'segment';
  text: string;
  speaker?: string;
  absolute_start_time: string;
  absolute_end_time: string;
  confidence: number;
  is_final: boolean;
  session_id: string;
  language?: string;
  timestamp?: string;
}

export interface TranslationSegment {
  type: 'translation';
  text: string;
  source_lang: string;
  target_lang: string;
  confidence: number;
  session_id: string;
  timestamp?: string;
}

export interface ErrorMessage {
  type: 'error';
  error: string;
  timestamp: string;
}

type StreamMessage = TranscriptionSegment | TranslationSegment | ErrorMessage;

export interface UseAudioStreamingReturn {
  isConnected: boolean;
  isStreaming: boolean;
  segments: TranscriptionSegment[];
  translations: TranslationSegment[];
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  startStreaming: () => Promise<void>;
  stopStreaming: () => void;
  clearSegments: () => void;
  clearTranslations: () => void;
  sessionId: string;
}

/**
 * Hook for WebSocket audio streaming
 */
export const useAudioStreaming = (
  config: AudioStreamingConfig = {}
): UseAudioStreamingReturn => {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [translations, setTranslations] = useState<TranslationSegment[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef<string>(`session-${Date.now()}`);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Configuration
  const orchestrationUrl = config.orchestrationUrl || 'ws://localhost:3001';
  const wsUrl = `${orchestrationUrl}/api/audio/stream`;

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('⚠️ Already connected');
      return;
    }

    console.log('🔌 Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('✅ WebSocket connected');
      setIsConnected(true);
      setError(null);

      // Start session automatically
      const sessionConfig = {
        type: 'start_session',
        session_id: sessionIdRef.current,
        config: {
          model: config.model || 'whisper-base',
          language: config.language || 'en',
          enable_vad: config.enableVAD !== false,
          enable_diarization: config.enableDiarization !== false,
          enable_cif: config.enableCIF !== false,
          enable_rolling_context: config.enableRollingContext !== false
        }
      };

      console.log('🎬 Starting session:', sessionConfig);
      ws.send(JSON.stringify(sessionConfig));
    };

    ws.onmessage = (event) => {
      try {
        const message: StreamMessage = JSON.parse(event.data);
        console.log('📨 Message received:', message.type);

        if (message.type === 'segment') {
          setSegments(prev => [...prev, message as TranscriptionSegment]);
        } else if (message.type === 'translation') {
          setTranslations(prev => [...prev, message as TranslationSegment]);
        } else if (message.type === 'error') {
          console.error('❌ Server error:', message.error);
          setError(message.error);
        } else if (message.type === 'session_started') {
          console.log('✅ Session started:', message);
        } else if (message.type === 'connected') {
          console.log('✅ Connection established:', message);
        } else if (message.type === 'authenticated') {
          console.log('✅ Authenticated:', message);
        }
      } catch (err) {
        console.error('❌ Error parsing message:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('❌ WebSocket error:', event);
      setError('WebSocket connection error');
      setIsConnected(false);
    };

    ws.onclose = (event) => {
      console.log('🔌 WebSocket closed:', event.code, event.reason);
      setIsConnected(false);
      setIsStreaming(false);

      // Auto-reconnect after 3 seconds (if not intentionally closed)
      if (event.code !== 1000) {
        console.log('🔄 Auto-reconnecting in 3 seconds...');
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      }
    };

    wsRef.current = ws;
  }, [wsUrl, config]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      console.log('🔌 Disconnecting WebSocket...');

      // Send end_session message
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'end_session',
          session_id: sessionIdRef.current
        }));
      }

      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }

    setIsConnected(false);
  }, []);

  /**
   * Start streaming microphone audio
   */
  const startStreaming = useCallback(async () => {
    // Ensure WebSocket is connected
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect();
      // Wait for connection
      await new Promise(resolve => setTimeout(resolve, 1000));

      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setError('WebSocket not connected. Please try again.');
        return;
      }
    }

    try {
      console.log('🎤 Requesting microphone access...');

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,           // Whisper expects 16kHz
          channelCount: 1,             // Mono audio
          echoCancellation: false,     // Disable for better capture
          noiseSuppression: false,     // Disable for consistency
          autoGainControl: false       // Disable for consistent levels
        }
      });

      streamRef.current = stream;
      console.log('✅ Microphone access granted');

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Convert Blob to base64
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64Audio = (reader.result as string).split(',')[1];

            // Send audio chunk via WebSocket
            wsRef.current?.send(JSON.stringify({
              type: 'audio_chunk',
              audio: base64Audio,
              timestamp: new Date().toISOString()
            }));
          };
          reader.readAsDataURL(event.data);
        }
      };

      // Start recording with 100ms chunks for real-time streaming
      mediaRecorder.start(100);
      mediaRecorderRef.current = mediaRecorder;
      setIsStreaming(true);

      console.log('🎙️ Audio streaming started (100ms chunks)');

    } catch (err) {
      console.error('❌ Failed to start streaming:', err);
      setError(`Failed to access microphone: ${err}`);
    }
  }, [connect]);

  /**
   * Stop streaming microphone audio
   */
  const stopStreaming = useCallback(() => {
    console.log('⏹️ Stopping audio streaming...');

    // Stop MediaRecorder
    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      mediaRecorderRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    setIsStreaming(false);
    console.log('✅ Audio streaming stopped');
  }, []);

  /**
   * Clear all segments
   */
  const clearSegments = useCallback(() => {
    setSegments([]);
  }, []);

  /**
   * Clear all translations
   */
  const clearTranslations = useCallback(() => {
    setTranslations([]);
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopStreaming();
      disconnect();
    };
  }, [stopStreaming, disconnect]);

  return {
    isConnected,
    isStreaming,
    segments,
    translations,
    error,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    clearSegments,
    clearTranslations,
    sessionId: sessionIdRef.current
  };
};

export default useAudioStreaming;
