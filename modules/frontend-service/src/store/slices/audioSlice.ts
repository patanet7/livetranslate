import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import {
  AudioDevice,
  RecordingState,
  PlaybackState,
  ProcessingState,
  VisualizationState,
  AudioQualityMetrics,
  ProcessingStage,
  ProcessingPreset,
  AudioConfig,
  ProcessingLog,
} from "@/types";

interface AudioState {
  // Device management
  devices: AudioDevice[];
  selectedInputDevice: string | null;
  selectedOutputDevice: string | null;
  devicePermissions: "granted" | "denied" | "pending";

  // Recording state
  recording: RecordingState;

  // Playback state
  playback: PlaybackState;

  // Processing state
  processing: ProcessingState & {
    logs: ProcessingLog[];
  };
  stages: ProcessingStage[];
  presets: ProcessingPreset[];

  // Visualization
  visualization: VisualizationState;

  // Configuration
  config: AudioConfig;

  // Quality metrics
  currentQualityMetrics: AudioQualityMetrics | null;
  qualityHistory: AudioQualityMetrics[];

  // Statistics
  stats: {
    totalRecordings: number;
    totalProcessingTime: number;
    averageQualityScore: number;
    successfulProcessings: number;
    failedProcessings: number;
  };

  // Error handling
  error: string | null;
  loading: boolean;
}

const initialState: AudioState = {
  devices: [],
  selectedInputDevice: null,
  selectedOutputDevice: null,
  devicePermissions: "pending",

  recording: {
    isRecording: false,
    duration: 0,
    maxDuration: 300, // 5 minutes default
    autoStop: true,
    format: "wav",
    sampleRate: 16000,
    recordedBlobUrl: null, // ✅ Store URL string instead of Blob
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

  config: {
    sampleRate: 16000,
    channels: 1,
    dtype: "float32",
    blocksize: 1024,
    chunkDuration: 1.0,
    qualityThreshold: 0.7,
    // Meeting-optimized defaults
    duration: 15, // Meeting-optimized: 15 seconds default
    deviceId: "",
    format: "audio/wav", // Meeting-optimized: WAV for highest quality
    quality: "lossless", // Meeting-optimized: Lossless quality
    autoStop: true,
    echoCancellation: false, // Meeting-optimized: Disabled for loopback audio
    noiseSuppression: false, // Meeting-optimized: Disabled to preserve content
    autoGainControl: false, // Meeting-optimized: Disabled for consistent levels
    rawAudio: true, // Meeting-optimized: Raw audio for best quality
    source: "microphone",
  },

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
};

const audioSlice = createSlice({
  name: "audio",
  initialState,
  reducers: {
    // Device management
    setDevices: (state, action: PayloadAction<AudioDevice[]>) => {
      state.devices = action.payload;
    },

    setSelectedInputDevice: (state, action: PayloadAction<string>) => {
      state.selectedInputDevice = action.payload;
    },

    setSelectedOutputDevice: (state, action: PayloadAction<string>) => {
      state.selectedOutputDevice = action.payload;
    },

    setDevicePermissions: (
      state,
      action: PayloadAction<"granted" | "denied" | "pending">,
    ) => {
      state.devicePermissions = action.payload;
    },

    // Recording actions
    startRecording: (
      state,
      action: PayloadAction<{ maxDuration?: number; format?: string }>,
    ) => {
      state.recording.isRecording = true;
      state.recording.status = "recording";
      state.recording.duration = 0;
      if (action.payload.maxDuration) {
        state.recording.maxDuration = action.payload.maxDuration;
      }
      if (action.payload.format) {
        state.recording.format = action.payload.format;
      }
      state.stats.totalRecordings += 1;
    },

    stopRecording: (state) => {
      state.recording.isRecording = false;
      state.recording.status = "completed";
    },

    updateRecordingDuration: (state, action: PayloadAction<number>) => {
      state.recording.duration = action.payload;
    },

    setRecordedBlobUrl: (state, action: PayloadAction<string | null>) => {
      // Store serializable URL string instead of Blob object
      state.recording.recordedBlobUrl = action.payload;
    },

    clearRecording: (state) => {
      // Clean up any existing blob URL
      if (state.recording.recordedBlobUrl) {
        URL.revokeObjectURL(state.recording.recordedBlobUrl);
      }
      state.recording = {
        ...initialState.recording,
        maxDuration: state.recording.maxDuration,
        format: state.recording.format,
      };
    },

    // Playback actions
    startPlayback: (state) => {
      state.playback.isPlaying = true;
    },

    pausePlayback: (state) => {
      state.playback.isPlaying = false;
    },

    stopPlayback: (state) => {
      state.playback.isPlaying = false;
      state.playback.currentTime = 0;
    },

    updatePlaybackTime: (state, action: PayloadAction<number>) => {
      state.playback.currentTime = action.payload;
    },

    setPlaybackDuration: (state, action: PayloadAction<number>) => {
      state.playback.duration = action.payload;
    },

    setVolume: (state, action: PayloadAction<number>) => {
      state.playback.volume = Math.max(0, Math.min(1, action.payload));
    },

    // Processing actions
    startProcessing: (state, action: PayloadAction<{ preset?: string }>) => {
      state.processing.isProcessing = true;
      state.processing.currentStage = 0;
      state.processing.progress = 0;
      state.processing.results = {};
      if (action.payload.preset) {
        state.processing.preset = action.payload.preset;
      }
    },

    updateProcessingProgress: (
      state,
      action: PayloadAction<{ stage: number; progress: number }>,
    ) => {
      state.processing.currentStage = action.payload.stage;
      state.processing.progress = action.payload.progress;
    },

    addProcessingResult: (
      state,
      action: PayloadAction<{ stageId: string; result: any }>,
    ) => {
      state.processing.results[action.payload.stageId] = action.payload.result;
    },

    completeProcessing: (state, action: PayloadAction<Record<string, any>>) => {
      state.processing.isProcessing = false;
      state.processing.progress = 100;
      state.processing.results = action.payload;
      state.stats.successfulProcessings += 1;
    },

    failProcessing: (state, action: PayloadAction<string>) => {
      state.processing.isProcessing = false;
      state.error = action.payload;
      state.stats.failedProcessings += 1;
    },

    // Visualization actions
    setVisualizationData: (
      state,
      action: PayloadAction<{
        audioLevel: number;
        frequencyData?: number[];
        timeData?: number[];
      }>,
    ) => {
      state.visualization.audioLevel = action.payload.audioLevel;
      if (action.payload.frequencyData) {
        state.visualization.frequencyData = action.payload.frequencyData;
      }
      if (action.payload.timeData) {
        state.visualization.timeData = action.payload.timeData;
      }
    },

    // Quality metrics
    updateQualityMetrics: (
      state,
      action: PayloadAction<AudioQualityMetrics>,
    ) => {
      state.currentQualityMetrics = action.payload;
      state.qualityHistory.push(action.payload);

      // Keep only last 100 measurements
      if (state.qualityHistory.length > 100) {
        state.qualityHistory.shift();
      }

      // Update average quality score
      const avgQuality =
        state.qualityHistory.reduce(
          (sum, metrics) => sum + (metrics.qualityScore || 0),
          0,
        ) / state.qualityHistory.length;
      state.stats.averageQualityScore = avgQuality;
    },

    // Configuration
    updateConfig: (state, action: PayloadAction<Partial<AudioConfig>>) => {
      state.config = { ...state.config, ...action.payload };
    },

    // Presets and stages
    setPresets: (state, action: PayloadAction<ProcessingPreset[]>) => {
      state.presets = action.payload;
    },

    setStages: (state, action: PayloadAction<ProcessingStage[]>) => {
      state.stages = action.payload;
    },

    updateStage: (
      state,
      action: PayloadAction<{
        stageId: string;
        updates: Partial<ProcessingStage>;
      }>,
    ) => {
      const stageIndex = state.stages.findIndex(
        (stage) => stage.id === action.payload.stageId,
      );
      if (stageIndex !== -1) {
        state.stages[stageIndex] = {
          ...state.stages[stageIndex],
          ...action.payload.updates,
        };
      }
    },

    // Error handling
    setError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
      state.loading = false;
    },

    clearError: (state) => {
      state.error = null;
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },

    // Additional actions for audio testing components
    setAudioDevices: (state, action: PayloadAction<AudioDevice[]>) => {
      state.devices = action.payload;
    },

    setRecordingState: (
      state,
      action: PayloadAction<Partial<RecordingState>>,
    ) => {
      // Clean up old blob URL if replacing with new one
      if (action.payload.recordedBlobUrl && state.recording.recordedBlobUrl) {
        URL.revokeObjectURL(state.recording.recordedBlobUrl);
      }
      state.recording = { ...state.recording, ...action.payload };
    },

    updateRecordingConfig: (
      state,
      action: PayloadAction<Partial<AudioConfig>>,
    ) => {
      state.config = { ...state.config, ...action.payload };
    },

    setProcessingStage: (state, action: PayloadAction<ProcessingStage>) => {
      const existingIndex = state.stages.findIndex(
        (stage) => stage.id === action.payload.id,
      );
      if (existingIndex !== -1) {
        state.stages[existingIndex] = action.payload;
      } else {
        state.stages.push(action.payload);
      }
    },

    updateProcessingStage: (
      state,
      action: PayloadAction<Partial<ProcessingStage> & { id: string }>,
    ) => {
      const stageIndex = state.stages.findIndex(
        (stage) => stage.id === action.payload.id,
      );
      if (stageIndex !== -1) {
        state.stages[stageIndex] = {
          ...state.stages[stageIndex],
          ...action.payload,
        };
      }
    },

    addProcessingLog: (state, action: PayloadAction<ProcessingLog>) => {
      state.processing.logs.push(action.payload);
      // Keep only last 1000 logs
      if (state.processing.logs.length > 1000) {
        state.processing.logs.shift();
      }
    },

    clearProcessingLogs: (state) => {
      state.processing.logs = [];
    },

    setAudioQualityMetrics: (
      state,
      action: PayloadAction<AudioQualityMetrics>,
    ) => {
      state.currentQualityMetrics = action.payload;
      state.qualityHistory.push(action.payload);

      // Keep only last 100 measurements
      if (state.qualityHistory.length > 100) {
        state.qualityHistory.shift();
      }
    },

    // Reset state
    resetAudioState: () => initialState,
  },
});

export const {
  setDevices,
  setSelectedInputDevice,
  setSelectedOutputDevice,
  setDevicePermissions,
  startRecording,
  stopRecording,
  updateRecordingDuration,
  setRecordedBlobUrl,
  clearRecording,
  startPlayback,
  pausePlayback,
  stopPlayback,
  updatePlaybackTime,
  setPlaybackDuration,
  setVolume,
  startProcessing,
  updateProcessingProgress,
  addProcessingResult,
  completeProcessing,
  failProcessing,
  setVisualizationData,
  updateQualityMetrics,
  updateConfig,
  setPresets,
  setStages,
  updateStage,
  setError,
  clearError,
  setLoading,
  setAudioDevices,
  setRecordingState,
  updateRecordingConfig,
  setProcessingStage,
  updateProcessingStage,
  addProcessingLog,
  clearProcessingLogs,
  setAudioQualityMetrics,
  resetAudioState,
} = audioSlice.actions;

export default audioSlice;
