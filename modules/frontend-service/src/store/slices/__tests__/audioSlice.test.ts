import { describe, it, expect, beforeEach } from "vitest";
import audioSlice, {
  setDevices,
  startRecording,
  stopRecording,
  updateRecordingDuration,
  setRecordedBlobUrl,
  clearRecording,
  startPlayback,
  pausePlayback,
  updatePlaybackTime,
  setVisualizationData,
  updateQualityMetrics,
  startProcessing,
  completeProcessing,
  failProcessing,
  setError,
  clearError,
  addProcessingLog,
  resetAudioState,
} from "../audioSlice";
import { createMockAudioDevice } from "@/test/utils";
import { AudioQualityMetrics, ProcessingLog } from "@/types";

describe("audioSlice", () => {
  const initialState = audioSlice.getInitialState();

  beforeEach(() => {
    // Reset to initial state before each test
  });

  describe("initial state", () => {
    it("should have correct initial state", () => {
      expect(initialState).toEqual({
        devices: [],
        selectedInputDevice: null,
        selectedOutputDevice: null,
        devicePermissions: "pending",
        recording: {
          isRecording: false,
          duration: 0,
          maxDuration: 300,
          autoStop: true,
          format: "wav",
          sampleRate: 16000,
          recordedBlobUrl: null,
          status: "idle",
          isPlaying: false,
          recordingStartTime: null,
          sessionId: null,
        },
        playback: {
          isPlaying: false,
          currentTime: 0,
          duration: 0,
          volume: 1.0,
        },
        processing: {
          currentStage: 0,
          isProcessing: false,
          progress: 0,
          results: {},
          preset: "default",
          logs: [],
        },
        stages: [],
        presets: [],
        visualization: {
          audioLevel: 0,
          frequencyData: [],
          timeData: [],
        },
        config: expect.objectContaining({
          sampleRate: 16000,
          channels: 1,
          dtype: "float32",
          blocksize: 1024,
          chunkDuration: 1.0,
          qualityThreshold: 0.7,
        }),
        currentQualityMetrics: null,
        qualityHistory: [],
        stats: {
          totalRecordings: 0,
          totalProcessingTime: 0,
          averageQualityScore: 0,
          successfulProcessings: 0,
          failedProcessings: 0,
        },
        error: null,
        loading: false,
      });
    });
  });

  describe("device management", () => {
    it("should set audio devices", () => {
      const devices = [
        createMockAudioDevice({ deviceId: "device-1", label: "Microphone 1" }),
        createMockAudioDevice({ deviceId: "device-2", label: "Microphone 2" }),
      ];

      const state = audioSlice.reducer(initialState, setDevices(devices));

      expect(state.devices).toEqual(devices);
    });

    it("should update device permissions", () => {
      const state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setDevicePermissions("granted"),
      );

      expect(state.devicePermissions).toBe("granted");
    });

    it("should set selected input device", () => {
      const deviceId = "selected-device-id";
      const state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setSelectedInputDevice(deviceId),
      );

      expect(state.selectedInputDevice).toBe(deviceId);
    });
  });

  describe("recording management", () => {
    it("should start recording", () => {
      const recordingOptions = { maxDuration: 600, format: "mp3" };
      const state = audioSlice.reducer(
        initialState,
        startRecording(recordingOptions),
      );

      expect(state.recording.isRecording).toBe(true);
      expect(state.recording.status).toBe("recording");
      expect(state.recording.duration).toBe(0);
      expect(state.recording.maxDuration).toBe(600);
      expect(state.recording.format).toBe("mp3");
      expect(state.stats.totalRecordings).toBe(1);
    });

    it("should stop recording", () => {
      const recordingState = {
        ...initialState,
        recording: {
          ...initialState.recording,
          isRecording: true,
          status: "recording" as const,
        },
      };

      const state = audioSlice.reducer(recordingState, stopRecording());

      expect(state.recording.isRecording).toBe(false);
      expect(state.recording.status).toBe("completed");
    });

    it("should update recording duration", () => {
      const duration = 45.5;
      const state = audioSlice.reducer(
        initialState,
        updateRecordingDuration(duration),
      );

      expect(state.recording.duration).toBe(duration);
    });

    it("should set audio blob URL", () => {
      const blobUrl = "blob:http://localhost/123-456-789";
      const state = audioSlice.reducer(
        initialState,
        setRecordedBlobUrl(blobUrl),
      );

      expect(state.recording.recordedBlobUrl).toBe(blobUrl);
    });

    it("should clear recording", () => {
      const recordingState = {
        ...initialState,
        recording: {
          ...initialState.recording,
          isRecording: true,
          duration: 30,
          recordedBlobUrl: "blob:http://localhost/123-456-789",
          status: "completed" as const,
        },
      };

      const state = audioSlice.reducer(recordingState, clearRecording());

      expect(state.recording.isRecording).toBe(false);
      expect(state.recording.duration).toBe(0);
      expect(state.recording.recordedBlobUrl).toBe(null);
      expect(state.recording.status).toBe("idle");
      // Should preserve max duration and format
      expect(state.recording.maxDuration).toBe(
        recordingState.recording.maxDuration,
      );
      expect(state.recording.format).toBe(recordingState.recording.format);
    });
  });

  describe("playback management", () => {
    it("should start playback", () => {
      const state = audioSlice.reducer(initialState, startPlayback());

      expect(state.playback.isPlaying).toBe(true);
    });

    it("should pause playback", () => {
      const playingState = {
        ...initialState,
        playback: { ...initialState.playback, isPlaying: true },
      };

      const state = audioSlice.reducer(playingState, pausePlayback());

      expect(state.playback.isPlaying).toBe(false);
    });

    it("should update playback time", () => {
      const currentTime = 25.7;
      const state = audioSlice.reducer(
        initialState,
        updatePlaybackTime(currentTime),
      );

      expect(state.playback.currentTime).toBe(currentTime);
    });

    it("should set volume within bounds", () => {
      // Test normal volume
      let state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setVolume(0.5),
      );
      expect(state.playback.volume).toBe(0.5);

      // Test volume above 1 (should clamp to 1)
      state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setVolume(1.5),
      );
      expect(state.playback.volume).toBe(1.0);

      // Test negative volume (should clamp to 0)
      state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setVolume(-0.5),
      );
      expect(state.playback.volume).toBe(0);
    });
  });

  describe("processing management", () => {
    it("should start processing", () => {
      const processingOptions = { preset: "high-quality" };
      const state = audioSlice.reducer(
        initialState,
        startProcessing(processingOptions),
      );

      expect(state.processing.isProcessing).toBe(true);
      expect(state.processing.currentStage).toBe(0);
      expect(state.processing.progress).toBe(0);
      expect(state.processing.results).toEqual({});
      expect(state.processing.preset).toBe("high-quality");
    });

    it("should update processing progress", () => {
      const progress = { stage: 2, progress: 65 };
      const state = audioSlice.reducer(
        initialState,
        audioSlice.actions.updateProcessingProgress(progress),
      );

      expect(state.processing.currentStage).toBe(2);
      expect(state.processing.progress).toBe(65);
    });

    it("should complete processing", () => {
      const processingState = {
        ...initialState,
        processing: { ...initialState.processing, isProcessing: true },
      };

      const results = { transcription: "Hello world", confidence: 0.95 };
      const state = audioSlice.reducer(
        processingState,
        completeProcessing(results),
      );

      expect(state.processing.isProcessing).toBe(false);
      expect(state.processing.progress).toBe(100);
      expect(state.processing.results).toEqual(results);
      expect(state.stats.successfulProcessings).toBe(1);
    });

    it("should handle processing failure", () => {
      const processingState = {
        ...initialState,
        processing: { ...initialState.processing, isProcessing: true },
      };

      const errorMessage = "Processing failed due to network error";
      const state = audioSlice.reducer(
        processingState,
        failProcessing(errorMessage),
      );

      expect(state.processing.isProcessing).toBe(false);
      expect(state.error).toBe(errorMessage);
      expect(state.stats.failedProcessings).toBe(1);
    });

    it("should add processing log", () => {
      const log: ProcessingLog = {
        level: "INFO",
        message: "Processing started",
        timestamp: Date.now(),
      };

      const state = audioSlice.reducer(initialState, addProcessingLog(log));

      expect(state.processing.logs).toHaveLength(1);
      expect(state.processing.logs[0]).toEqual(log);
    });

    it("should limit processing logs to 1000 entries", () => {
      // Create state with 1000 logs
      const logsState = {
        ...initialState,
        processing: {
          ...initialState.processing,
          logs: Array.from({ length: 1000 }, (_, i) => ({
            level: "INFO" as const,
            message: `Log ${i}`,
            timestamp: Date.now() + i,
          })),
        },
      };

      const newLog: ProcessingLog = {
        level: "INFO",
        message: "New log entry",
        timestamp: Date.now() + 1000,
      };

      const state = audioSlice.reducer(logsState, addProcessingLog(newLog));

      expect(state.processing.logs).toHaveLength(1000);
      expect(state.processing.logs[0].message).toBe("Log 1"); // First log removed
      expect(state.processing.logs[999].message).toBe("New log entry"); // New log added at end
    });
  });

  describe("visualization", () => {
    it("should update visualization data", () => {
      const visualizationData = {
        audioLevel: 0.75,
        frequencyData: [0.1, 0.2, 0.3],
        timeData: [0.5, 0.6, 0.7],
      };

      const state = audioSlice.reducer(
        initialState,
        setVisualizationData(visualizationData),
      );

      expect(state.visualization.audioLevel).toBe(0.75);
      expect(state.visualization.frequencyData).toEqual(
        visualizationData.frequencyData,
      );
      expect(state.visualization.timeData).toEqual(visualizationData.timeData);
    });

    it("should update visualization data with partial data", () => {
      const partialData = { audioLevel: 0.5 };
      const state = audioSlice.reducer(
        initialState,
        setVisualizationData(partialData),
      );

      expect(state.visualization.audioLevel).toBe(0.5);
      expect(state.visualization.frequencyData).toEqual([]);
      expect(state.visualization.timeData).toEqual([]);
    });
  });

  describe("quality metrics", () => {
    it("should update quality metrics", () => {
      const qualityMetrics: AudioQualityMetrics = {
        rmsLevel: -20,
        peakLevel: -10,
        zeroCrossingRate: 0.1,
        snrEstimate: 15,
        signalToNoise: 15,
        voiceActivity: 0.85,
        qualityScore: 0.85,
        clippingDetected: false,
      };

      const state = audioSlice.reducer(
        initialState,
        updateQualityMetrics(qualityMetrics),
      );

      expect(state.currentQualityMetrics).toEqual(qualityMetrics);
      expect(state.qualityHistory).toHaveLength(1);
      expect(state.qualityHistory[0]).toEqual(qualityMetrics);
      expect(state.stats.averageQualityScore).toBe(0.85);
    });

    it("should limit quality history to 100 entries", () => {
      // Create state with 100 quality measurements
      const qualityHistoryState = {
        ...initialState,
        qualityHistory: Array.from({ length: 100 }, (_, i) => ({
          rmsLevel: -20 + i,
          peakLevel: -10 + i,
          qualityScore: 0.5 + i * 0.005,
        })),
      };

      const newMetrics: AudioQualityMetrics = {
        rmsLevel: -15,
        peakLevel: -5,
        qualityScore: 0.95,
      };

      const state = audioSlice.reducer(
        qualityHistoryState,
        updateQualityMetrics(newMetrics),
      );

      expect(state.qualityHistory).toHaveLength(100);
      expect(state.qualityHistory[0].rmsLevel).toBe(-19); // First entry removed
      expect(state.qualityHistory[99]).toEqual(newMetrics); // New entry added at end
    });
  });

  describe("error handling", () => {
    it("should set error", () => {
      const errorMessage = "Audio processing error";
      const state = audioSlice.reducer(initialState, setError(errorMessage));

      expect(state.error).toBe(errorMessage);
      expect(state.loading).toBe(false);
    });

    it("should clear error", () => {
      const errorState = { ...initialState, error: "Some error" };
      const state = audioSlice.reducer(errorState, clearError());

      expect(state.error).toBe(null);
    });

    it("should set loading state", () => {
      const state = audioSlice.reducer(
        initialState,
        audioSlice.actions.setLoading(true),
      );

      expect(state.loading).toBe(true);
    });
  });

  describe("state reset", () => {
    it("should reset to initial state", () => {
      const modifiedState = {
        ...initialState,
        devices: [createMockAudioDevice()],
        recording: { ...initialState.recording, isRecording: true },
        error: "Some error",
        loading: true,
      };

      const state = audioSlice.reducer(modifiedState, resetAudioState());

      expect(state).toEqual(initialState);
    });
  });

  describe("configuration management", () => {
    it("should update configuration", () => {
      const configUpdate = {
        sampleRate: 48000,
        qualityThreshold: 0.8,
        autoStop: false,
      };

      const state = audioSlice.reducer(
        initialState,
        audioSlice.actions.updateConfig(configUpdate),
      );

      expect(state.config.sampleRate).toBe(48000);
      expect(state.config.qualityThreshold).toBe(0.8);
      expect(state.config.autoStop).toBe(false);
      // Should preserve other config values
      expect(state.config.channels).toBe(1);
      expect(state.config.dtype).toBe("float32");
    });
  });
});
