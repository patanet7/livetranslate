import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import {
  BotInstance,
  MeetingRequest,
  BotStatus,
  SystemStats,
  BotConfig,
  BotHealthMetrics,
  Translation,
  WebcamConfig,
  AudioQualityMetrics,
  CaptionSegment,
  WebcamDisplayMode,
  WebcamTheme,
  MeetingPlatform,
  BotPriority,
} from "@/types";
import { getCurrentISOTimestamp } from "@/utils/dateTimeUtils";

interface BotState {
  // Bot instances
  bots: Record<string, BotInstance>;
  activeBotIds: string[];

  // Bot spawning
  spawnerConfig: {
    maxConcurrentBots: number;
    defaultTargetLanguages: string[];
    autoTranslationEnabled: boolean;
    virtualWebcamEnabled: boolean;
    defaultBotConfig: BotConfig;
  };

  // Meeting requests
  meetingRequests: Record<
    string,
    {
      requestId: string;
      meetingId: string;
      meetingTitle: string;
      organizerEmail?: string;
      targetLanguages: string[];
      autoTranslation: boolean;
      priority: "low" | "medium" | "high";
      status: "pending" | "processing" | "completed" | "failed";
      createdAt: number;
      botId?: string;
    }
  >;

  // System statistics
  systemStats: SystemStats;

  // Health monitoring
  healthMetrics: Record<string, BotHealthMetrics>;

  // Real-time data
  realtimeData: {
    audioCapture: Record<string, AudioQualityMetrics>;
    captions: Record<string, CaptionSegment[]>;
    translations: Record<string, Translation[]>;
    webcamFrames: Record<string, string>; // botId -> base64 frame
  };

  // UI state
  selectedBotId: string | null;
  dashboardView: "overview" | "detailed" | "performance";

  // Error handling
  error: string | null;
  loading: boolean;
}

const defaultBotConfig: BotConfig = {
  botId: "",
  botName: "",
  targetLanguages: ["en", "es", "fr"],
  virtualWebcamEnabled: true,
  serviceEndpoints: {
    whisperService: "http://localhost:5001",
    translationService: "http://localhost:5003",
    orchestrationService: "http://localhost:3000",
  },
  audioConfig: {
    sampleRate: 16000,
    channels: 1,
    chunkDuration: 1.0,
    qualityThreshold: 0.7,
  },
  captionConfig: {
    enableSpeakerDiarization: true,
    confidenceThreshold: 0.6,
    languageDetection: true,
  },
  webcamConfig: {
    width: 1280,
    height: 720,
    fps: 30,
    displayMode: WebcamDisplayMode.OVERLAY,
    theme: WebcamTheme.DARK,
    maxTranslationsDisplayed: 3,
    fontSize: 16,
    backgroundOpacity: 0.8,
  },
};

const initialState: BotState = {
  bots: {},
  activeBotIds: [],

  spawnerConfig: {
    maxConcurrentBots: 10,
    defaultTargetLanguages: ["en", "es", "fr"],
    autoTranslationEnabled: true,
    virtualWebcamEnabled: true,
    defaultBotConfig,
  },

  meetingRequests: {},

  systemStats: {
    totalBotsSpawned: 0,
    activeBots: 0,
    completedSessions: 0,
    errorRate: 0,
    averageSessionDuration: 0,
  },

  healthMetrics: {},

  realtimeData: {
    audioCapture: {},
    captions: {},
    translations: {},
    webcamFrames: {},
  },

  selectedBotId: null,
  dashboardView: "overview",

  error: null,
  loading: false,
};

const generateId = () =>
  `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const botSlice = createSlice({
  name: "bot",
  initialState,
  reducers: {
    // Bot lifecycle management
    spawnBot: (state, action: PayloadAction<MeetingRequest>) => {
      state.loading = true;
      const requestId = generateId();
      state.meetingRequests[requestId] = {
        requestId,
        ...action.payload,
        status: "pending",
        createdAt: Date.now(),
      };
    },

    spawnBotSuccess: (
      state,
      action: PayloadAction<{
        requestId: string;
        botId: string;
        botData: Partial<BotInstance>;
      }>,
    ) => {
      const { requestId, botId, botData } = action.payload;
      const request = state.meetingRequests[requestId];

      if (request) {
        request.status = "completed";
        request.botId = botId;

        // Create bot instance
        const newBot: BotInstance = {
          id: botId,
          botId,
          status: "spawning",
          config: {
            meetingInfo: {
              meetingId: request.meetingId,
              meetingTitle: request.meetingTitle,
              platform: MeetingPlatform.GOOGLE_MEET,
              organizerEmail: request.organizerEmail,
              participantCount: 0,
            },
            audioCapture: {
              sampleRate: 16000,
              channels: 1,
              chunkSize: 1024,
              enableNoiseSuppression: true,
              enableEchoCancellation: false,
              enableAutoGain: false,
            },
            translation: {
              targetLanguages: request.targetLanguages || ["en"],
              enableAutoTranslation: request.autoTranslation ?? true,
              translationQuality: "balanced",
              realTimeTranslation: true,
            },
            webcam: state.spawnerConfig.defaultBotConfig.webcamConfig,
            priority:
              request.priority === "high"
                ? BotPriority.HIGH
                : request.priority === "low"
                  ? BotPriority.LOW
                  : BotPriority.MEDIUM,
            enableRecording: true,
            enableTranscription: true,
            enableSpeakerDiarization: true,
            enableVirtualWebcam: true,
          },
          updatedAt: getCurrentISOTimestamp(),
          audioCapture: {
            isCapturing: false,
            totalChunksCaptured: 0,
            averageChunkSizeBytes: 0,
            totalAudioDurationS: 0,
            averageQualityScore: 0,
            lastCaptureTimestamp: getCurrentISOTimestamp(),
            deviceInfo: "",
            sampleRateActual: 16000,
            channelsActual: 1,
          },
          captionProcessor: {
            totalCaptionsProcessed: 0,
            totalSpeakers: 0,
            speakerTimeline: [],
            averageConfidence: 0,
            lastCaptionTimestamp: getCurrentISOTimestamp(),
            processingLatencyMs: 0,
          },
          virtualWebcam: {
            isStreaming: false,
            framesGenerated: 0,
            currentTranslations: [],
            averageFps: 0,
            webcamConfig: {
              ...state.spawnerConfig.defaultBotConfig.webcamConfig,
              fontSize:
                state.spawnerConfig.defaultBotConfig.webcamConfig.fontSize ||
                16,
              backgroundOpacity:
                state.spawnerConfig.defaultBotConfig.webcamConfig
                  .backgroundOpacity || 0.8,
            },
            lastFrameTimestamp: getCurrentISOTimestamp(),
          },
          timeCorrelation: {
            totalCorrelations: 0,
            successRate: 0,
            averageTimingOffsetMs: 0,
            lastCorrelationTimestamp: getCurrentISOTimestamp(),
            correlationAccuracy: 0,
          },
          performance: {
            sessionDurationS: 0,
            totalProcessingTimeS: 0,
            cpuUsagePercent: 0,
            memoryUsageMb: 0,
            networkBytesSent: 0,
            networkBytesReceived: 0,
            averageLatencyMs: 0,
            errorCount: 0,
          },
          errorMessages: [],
          createdAt: getCurrentISOTimestamp(),
          lastActiveAt: getCurrentISOTimestamp(),
          ...botData,
        };

        state.bots[botId] = newBot;
        state.activeBotIds.push(botId);
        state.systemStats.totalBotsSpawned += 1;
        state.systemStats.activeBots += 1;
      }

      state.loading = false;
    },

    spawnBotFailure: (
      state,
      action: PayloadAction<{ requestId: string; error: string }>,
    ) => {
      const { requestId, error } = action.payload;
      const request = state.meetingRequests[requestId];

      if (request) {
        request.status = "failed";
      }

      state.error = error;
      state.loading = false;
    },

    updateBotStatus: (
      state,
      action: PayloadAction<{ botId: string; status: BotStatus; data?: any }>,
    ) => {
      const { botId, status, data } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.status = status;
        bot.lastActiveAt = getCurrentISOTimestamp();

        if (data) {
          Object.assign(bot, data);
        }

        // Update session duration in seconds
        const createdTime = new Date(bot.createdAt).getTime();
        bot.performance.sessionDurationS = (Date.now() - createdTime) / 1000;
      }
    },

    terminateBot: (state, action: PayloadAction<string>) => {
      const botId = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.status = "terminated";
        state.activeBotIds = state.activeBotIds.filter((id) => id !== botId);
        state.systemStats.activeBots = Math.max(
          0,
          state.systemStats.activeBots - 1,
        );
        state.systemStats.completedSessions += 1;

        // Calculate final session duration in seconds
        const createdTime = new Date(bot.createdAt).getTime();
        bot.performance.sessionDurationS = (Date.now() - createdTime) / 1000;

        // Clean up realtime data
        delete state.realtimeData.audioCapture[botId];
        delete state.realtimeData.captions[botId];
        delete state.realtimeData.translations[botId];
        delete state.realtimeData.webcamFrames[botId];
        delete state.healthMetrics[botId];
      }
    },

    removeBotFromState: (state, action: PayloadAction<string>) => {
      const botId = action.payload;
      delete state.bots[botId];
      state.activeBotIds = state.activeBotIds.filter((id) => id !== botId);

      if (state.selectedBotId === botId) {
        state.selectedBotId = null;
      }
    },

    // Audio capture updates
    updateAudioCapture: (
      state,
      action: PayloadAction<{ botId: string; metrics: AudioQualityMetrics }>,
    ) => {
      const { botId, metrics } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.audioCapture.totalChunksCaptured += 1;
        bot.audioCapture.averageQualityScore =
          (bot.audioCapture.averageQualityScore + (metrics.qualityScore || 0)) /
          2;
        bot.audioCapture.lastCaptureTimestamp = getCurrentISOTimestamp();
        bot.lastActiveAt = getCurrentISOTimestamp();

        // Store realtime metrics
        state.realtimeData.audioCapture[botId] = metrics;
      }
    },

    setAudioCaptureStatus: (
      state,
      action: PayloadAction<{
        botId: string;
        isCapturing: boolean;
        deviceInfo?: string;
      }>,
    ) => {
      const { botId, isCapturing, deviceInfo } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.audioCapture.isCapturing = isCapturing;
        if (deviceInfo) {
          bot.audioCapture.deviceInfo = deviceInfo;
        }
      }
    },

    // Caption processing updates
    addCaption: (
      state,
      action: PayloadAction<{ botId: string; caption: CaptionSegment }>,
    ) => {
      const { botId, caption } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.captionProcessor.totalCaptionsProcessed += 1;
        bot.captionProcessor.lastCaptionTimestamp = new Date(
          caption.timestamp,
        ).toISOString();
        bot.lastActiveAt = getCurrentISOTimestamp();

        // Store in realtime data
        if (!state.realtimeData.captions[botId]) {
          state.realtimeData.captions[botId] = [];
        }
        state.realtimeData.captions[botId].push(caption);

        // Keep only last 50 captions
        if (state.realtimeData.captions[botId].length > 50) {
          state.realtimeData.captions[botId].shift();
        }

        // Update speaker count
        const speakerIds = new Set(
          state.realtimeData.captions[botId].map((c) => c.speakerId),
        );
        bot.captionProcessor.totalSpeakers = speakerIds.size;
      }
    },

    // Translation updates
    addTranslation: (
      state,
      action: PayloadAction<{ botId: string; translation: Translation }>,
    ) => {
      const { botId, translation } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        // Add to virtual webcam current translations (store as simple { language, text })
        const simpleTranslation = {
          language: translation.targetLanguage,
          text: translation.translatedText,
        };
        bot.virtualWebcam.currentTranslations.push(simpleTranslation);

        // Keep only last 3 translations for display
        if (bot.virtualWebcam.currentTranslations.length > 3) {
          bot.virtualWebcam.currentTranslations.shift();
        }

        // Store in realtime data
        if (!state.realtimeData.translations[botId]) {
          state.realtimeData.translations[botId] = [];
        }
        state.realtimeData.translations[botId].push(translation);

        // Keep only last 100 translations
        if (state.realtimeData.translations[botId].length > 100) {
          state.realtimeData.translations[botId].shift();
        }

        bot.lastActiveAt = getCurrentISOTimestamp();
      }
    },

    // Virtual webcam updates
    updateWebcamStatus: (
      state,
      action: PayloadAction<{
        botId: string;
        isStreaming: boolean;
        framesGenerated?: number;
      }>,
    ) => {
      const { botId, isStreaming, framesGenerated } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.virtualWebcam.isStreaming = isStreaming;
        if (framesGenerated !== undefined) {
          bot.virtualWebcam.framesGenerated = framesGenerated;
        }
      }
    },

    updateWebcamFrame: (
      state,
      action: PayloadAction<{ botId: string; frameBase64: string }>,
    ) => {
      const { botId, frameBase64 } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        bot.virtualWebcam.framesGenerated += 1;
        state.realtimeData.webcamFrames[botId] = frameBase64;
        bot.lastActiveAt = getCurrentISOTimestamp();
      }
    },

    updateWebcamConfig: (
      state,
      action: PayloadAction<{ botId: string; config: Partial<WebcamConfig> }>,
    ) => {
      const { botId, config } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        Object.assign(bot.virtualWebcam.webcamConfig, config);
      }
    },

    // Health monitoring
    updateHealthMetrics: (
      state,
      action: PayloadAction<{ botId: string; metrics: BotHealthMetrics }>,
    ) => {
      const { botId, metrics } = action.payload;
      state.healthMetrics[botId] = metrics;

      const bot = state.bots[botId];
      if (bot) {
        bot.lastActiveAt = getCurrentISOTimestamp();
      }
    },

    // Performance metrics
    updatePerformanceMetrics: (
      state,
      action: PayloadAction<{
        botId: string;
        metrics: Partial<BotInstance["performance"]>;
      }>,
    ) => {
      const { botId, metrics } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        Object.assign(bot.performance, metrics);
        bot.lastActiveAt = getCurrentISOTimestamp();
      }
    },

    // Time correlation updates
    updateTimeCorrelation: (
      state,
      action: PayloadAction<{
        botId: string;
        metrics: Partial<BotInstance["timeCorrelation"]>;
      }>,
    ) => {
      const { botId, metrics } = action.payload;
      const bot = state.bots[botId];

      if (bot) {
        Object.assign(bot.timeCorrelation, metrics);
        bot.lastActiveAt = getCurrentISOTimestamp();
      }
    },

    // System statistics
    updateSystemStats: (state, action: PayloadAction<Partial<SystemStats>>) => {
      Object.assign(state.systemStats, action.payload);
    },

    // Configuration updates
    updateSpawnerConfig: (
      state,
      action: PayloadAction<Partial<BotState["spawnerConfig"]>>,
    ) => {
      Object.assign(state.spawnerConfig, action.payload);
    },

    // UI state
    setSelectedBot: (state, action: PayloadAction<string | null>) => {
      state.selectedBotId = action.payload;
    },

    setDashboardView: (
      state,
      action: PayloadAction<"overview" | "detailed" | "performance">,
    ) => {
      state.dashboardView = action.payload;
    },

    // Error handling
    setBotError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
      state.loading = false;
    },

    clearBotError: (state) => {
      state.error = null;
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },

    // Additional actions for useBotManager hook
    setBots: (state, action: PayloadAction<Record<string, BotInstance>>) => {
      state.bots = action.payload;
    },

    setActiveBotIds: (state, action: PayloadAction<string[]>) => {
      state.activeBotIds = action.payload;
    },

    setSystemStats: (state, action: PayloadAction<SystemStats>) => {
      state.systemStats = action.payload;
    },

    addBot: (state, action: PayloadAction<BotInstance>) => {
      const bot = action.payload;
      state.bots[bot.botId] = bot;
      if (!state.activeBotIds.includes(bot.botId)) {
        state.activeBotIds.push(bot.botId);
      }
    },

    updateBot: (
      state,
      action: PayloadAction<{ botId: string; updates: Partial<BotInstance> }>,
    ) => {
      const { botId, updates } = action.payload;
      const bot = state.bots[botId];
      if (bot) {
        Object.assign(bot, updates);
      }
    },

    removeBot: (state, action: PayloadAction<string>) => {
      const botId = action.payload;
      delete state.bots[botId];
      state.activeBotIds = state.activeBotIds.filter((id) => id !== botId);
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    // Reset state
    resetBotState: () => initialState,
  },
});

export const {
  spawnBot,
  spawnBotSuccess,
  spawnBotFailure,
  updateBotStatus,
  terminateBot,
  removeBotFromState,
  updateAudioCapture,
  setAudioCaptureStatus,
  addCaption,
  addTranslation,
  updateWebcamStatus,
  updateWebcamFrame,
  updateWebcamConfig,
  updateHealthMetrics,
  updatePerformanceMetrics,
  updateTimeCorrelation,
  updateSystemStats,
  updateSpawnerConfig,
  setSelectedBot,
  setDashboardView,
  setBotError,
  clearBotError,
  setLoading,
  resetBotState,
  // Additional exports for useBotManager hook
  setBots,
  setActiveBotIds,
  setSystemStats,
  addBot,
  updateBot,
  removeBot,
  setError,
} = botSlice.actions;

export default botSlice;
