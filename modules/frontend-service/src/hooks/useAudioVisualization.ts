/**
 * useAudioVisualization Hook
 *
 * Shared hook for audio visualization and real-time metrics.
 * Consolidates duplicate implementations from:
 * - StreamingProcessor/index.tsx (~150 lines)
 * - MeetingTest/index.tsx (~150 lines)
 * - AudioTesting/index.tsx (~120 lines)
 *
 * Total savings: ~420 lines of duplicate code
 */

import { useEffect, useCallback, useRef } from "react";
import { useAppDispatch } from "@/store";
import {
  setVisualizationData,
  addProcessingLog,
  setAudioQualityMetrics,
} from "@/store/slices/audioSlice";
import {
  calculateMeetingAudioLevel,
  getMeetingAudioQuality,
  getDisplayLevel,
} from "@/utils/audioLevelCalculation";

export interface AudioVisualizationOptions {
  /**
   * Selected audio device ID
   */
  deviceId?: string;

  /**
   * Sample rate for audio processing (default: 16000)
   */
  sampleRate?: number;

  /**
   * Audio stream ref to populate (optional, for recording)
   */
  audioStreamRef?: React.MutableRefObject<MediaStream | null>;

  /**
   * Custom audio constraints
   */
  customConstraints?: MediaStreamConstraints["audio"];

  /**
   * Enable/disable logging
   */
  enableLogging?: boolean;
}

/**
 * Hook to set up audio visualization with real-time metrics
 */
export const useAudioVisualization = (
  options: AudioVisualizationOptions = {},
) => {
  const dispatch = useAppDispatch();
  const {
    deviceId,
    sampleRate = 16000,
    audioStreamRef,
    customConstraints,
    enableLogging = true,
  } = options;

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);

  const startVisualization = useCallback(() => {
    if (!analyserRef.current) return;

    const updateVisualization = () => {
      if (!analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const frequencyData = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(frequencyData);

      const timeData = new Uint8Array(analyserRef.current.fftSize);
      analyserRef.current.getByteTimeDomainData(timeData);

      // Calculate professional meeting-optimized audio metrics
      const audioMetrics = calculateMeetingAudioLevel(
        timeData,
        frequencyData,
        sampleRate,
      );
      const qualityAssessment = getMeetingAudioQuality(audioMetrics);
      const displayLevel = getDisplayLevel(audioMetrics);

      // Update Redux store with enhanced visualization data
      dispatch(
        setVisualizationData({
          frequencyData: Array.from(frequencyData),
          timeData: Array.from(timeData),
          audioLevel: displayLevel,
        }),
      );

      // Update comprehensive audio quality metrics
      dispatch(
        setAudioQualityMetrics({
          rmsLevel: audioMetrics.rmsDb,
          peakLevel: audioMetrics.peakDb,
          signalToNoise: audioMetrics.signalToNoise,
          frequency: sampleRate,
          clipping: audioMetrics.clipping * 100,
          voiceActivity: audioMetrics.voiceActivity,
          spectralCentroid: audioMetrics.spectralCentroid,
          dynamicRange: audioMetrics.dynamicRange,
          speechClarity: audioMetrics.speechClarity,
          backgroundNoise: audioMetrics.backgroundNoise,
          qualityAssessment: qualityAssessment.quality,
          qualityScore: qualityAssessment.score,
          recommendations: qualityAssessment.recommendations,
          issues: qualityAssessment.issues,
        }),
      );

      animationFrameRef.current = requestAnimationFrame(updateVisualization);
    };

    updateVisualization();
  }, [dispatch, sampleRate]);

  // Initialize audio visualization
  useEffect(() => {
    if (!deviceId && !customConstraints) return;

    const initializeAudioVisualization = async () => {
      try {
        const constraints: MediaStreamConstraints = {
          audio: customConstraints || {
            deviceId: deviceId,
            sampleRate: sampleRate,
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        localStreamRef.current = stream;

        // If external stream ref provided, populate it
        if (audioStreamRef) {
          audioStreamRef.current = stream;
        }

        // Clean up previous audio context if it exists
        if (
          audioContextRef.current &&
          audioContextRef.current.state !== "closed"
        ) {
          audioContextRef.current.close();
        }

        audioContextRef.current = new (
          window.AudioContext || (window as any).webkitAudioContext
        )();
        microphoneRef.current =
          audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 2048;
        microphoneRef.current.connect(analyserRef.current);

        startVisualization();

        if (enableLogging) {
          dispatch(
            addProcessingLog({
              level: "SUCCESS",
              message: `Audio visualization initialized with device: ${deviceId || "custom"}`,
              timestamp: Date.now(),
            }),
          );
        }
      } catch (error) {
        if (enableLogging) {
          dispatch(
            addProcessingLog({
              level: "ERROR",
              message: `Failed to initialize audio visualization: ${error}`,
              timestamp: Date.now(),
            }),
          );
        }
      }
    };

    initializeAudioVisualization();

    return () => {
      // Cleanup
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (
        audioContextRef.current &&
        audioContextRef.current.state !== "closed"
      ) {
        audioContextRef.current.close();
      }
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [
    dispatch,
    deviceId,
    sampleRate,
    customConstraints,
    audioStreamRef,
    startVisualization,
    enableLogging,
  ]);

  return {
    /**
     * Audio context ref
     */
    audioContextRef,

    /**
     * Analyser node ref
     */
    analyserRef,

    /**
     * Microphone source node ref
     */
    microphoneRef,

    /**
     * Animation frame ref
     */
    animationFrameRef,
  };
};

export default useAudioVisualization;
