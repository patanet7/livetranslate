import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { 
  ApiResponse, 
  PaginatedResponse, 
  AudioDevice, 
  BotInstance, 
  MeetingRequest, 
  ServiceHealth,
  SystemHealth,
  ProcessingPreset,
  BotSession,
  Translation
} from '@/types';

// Enhanced base query with retry logic and error handling
const baseQuery = fetchBaseQuery({
  baseUrl: '/api',
  timeout: 30000, // 30 second timeout
  prepareHeaders: (headers) => {
    // Add any authentication headers here if needed
    headers.set('Content-Type', 'application/json');
    return headers;
  },
});

// Retry mechanism with exponential backoff
const baseQueryWithRetry = async (args: any, api: any, extraOptions: any) => {
  const maxRetries = 3;
  const baseDelay = 1000; // 1 second

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const result = await baseQuery(args, api, extraOptions);
    
    // If successful or it's the last attempt, return the result
    if (!result.error || attempt === maxRetries) {
      return result;
    }

    // Only retry on network errors or 5xx server errors
    const shouldRetry = 
      result.error.status === 'FETCH_ERROR' ||
      result.error.status === 'TIMEOUT_ERROR' ||
      (typeof result.error.status === 'number' && result.error.status >= 500);

    if (!shouldRetry) {
      return result;
    }

    // Exponential backoff delay
    const delay = baseDelay * Math.pow(2, attempt);
    console.warn(`API request failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms...`);
    
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  // This should never be reached, but TypeScript requires it
  return await baseQuery(args, api, extraOptions);
};

// API slice with all service endpoints
export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: baseQueryWithRetry,
  tagTypes: [
    'AudioDevice',
    'Bot',
    'BotSession', 
    'ServiceHealth',
    'SystemHealth',
    'ProcessingPreset',
    'Translation',
    'AudioFile',
    'SystemMetrics',
  ],
  endpoints: (builder) => ({
    // Audio API endpoints
    getAudioDevices: builder.query<ApiResponse<AudioDevice[]>, void>({
      query: () => 'audio/devices',
      providesTags: ['AudioDevice'],
    }),
    
    processAudio: builder.mutation<ApiResponse<any>, {
      audioBlob: Blob;
      preset?: string;
      stages?: string[];
    }>({
      query: ({ audioBlob, preset, stages }) => {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        if (preset) formData.append('preset', preset);
        if (stages) formData.append('stages', JSON.stringify(stages));
        
        return {
          url: 'audio/process',
          method: 'POST',
          body: formData,
        };
      },
    }),
    
    uploadAudioFile: builder.mutation<ApiResponse<any>, {
      audio: Blob | File;
      config?: Record<string, any>;
      sessionId?: string;
    }>({
      query: ({ audio, config, sessionId }) => {
        const formData = new FormData();
        // Backend expects 'audio' field name (not 'file')
        const audioFile = audio instanceof File ? audio : new File([audio], 'audio.webm', { type: 'audio/webm' });
        formData.append('audio', audioFile);
        // Backend expects 'config' field name (not 'metadata') as JSON string
        if (config) formData.append('config', JSON.stringify(config));
        if (sessionId) formData.append('session_id', sessionId);

        return {
          url: 'audio/upload',
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: ['AudioFile'],
    }),
    
    getAudioProcessingPresets: builder.query<ApiResponse<ProcessingPreset[]>, void>({
      query: () => 'audio/presets',
      providesTags: ['ProcessingPreset'],
    }),
    
    // Bot Management API endpoints
    getBots: builder.query<ApiResponse<BotInstance[]>, void>({
      query: () => 'bot',
      providesTags: ['Bot'],
    }),
    
    getBot: builder.query<ApiResponse<BotInstance>, string>({
      query: (botId) => `bot/${botId}`,
      providesTags: (_result, _error, botId) => [{ type: 'Bot', id: botId }],
    }),
    
    spawnBot: builder.mutation<ApiResponse<{ botId: string }>, MeetingRequest>({
      query: (meetingRequest) => ({
        url: 'bot/spawn',
        method: 'POST',
        body: meetingRequest,
      }),
      invalidatesTags: ['Bot'],
    }),
    
    terminateBot: builder.mutation<ApiResponse<void>, string>({
      query: (botId) => ({
        url: `bot/${botId}/terminate`,
        method: 'POST',
      }),
      invalidatesTags: ['Bot'],
    }),
    
    getBotStatus: builder.query<ApiResponse<BotInstance>, string>({
      query: (botId) => `bot/${botId}/status`,
      providesTags: (_result, _error, botId) => [{ type: 'Bot', id: botId }],
    }),
    
    getBotSessions: builder.query<ApiResponse<PaginatedResponse<BotSession>>, {
      page?: number;
      pageSize?: number;
      botId?: string;
      status?: string;
    }>({
      query: ({ page = 1, pageSize = 20, botId, status }) => ({
        url: 'bot/sessions',
        params: { page, pageSize, botId, status },
      }),
      providesTags: ['BotSession'],
    }),
    
    getBotSession: builder.query<ApiResponse<BotSession>, string>({
      query: (sessionId) => `bot/sessions/${sessionId}`,
      providesTags: (_result, _error, sessionId) => [{ type: 'BotSession', id: sessionId }],
    }),
    
    // Virtual Webcam API endpoints
    getWebcamFrame: builder.query<ApiResponse<{ frameBase64: string }>, string>({
      query: (botId) => `bot/${botId}/webcam/frame`,
    }),
    
    updateWebcamConfig: builder.mutation<ApiResponse<void>, {
      botId: string;
      config: Record<string, any>;
    }>({
      query: ({ botId, config }) => ({
        url: `bot/${botId}/webcam/config`,
        method: 'PATCH',
        body: config,
      }),
    }),
    
    // Translation API endpoints
    getTranslations: builder.query<ApiResponse<Translation[]>, {
      botId?: string;
      sessionId?: string;
      limit?: number;
    }>({
      query: ({ botId, sessionId, limit = 50 }) => ({
        url: 'translations',
        params: { botId, sessionId, limit },
      }),
      providesTags: ['Translation'],
    }),
    
    translateText: builder.mutation<ApiResponse<Translation>, {
      text: string;
      sourceLanguage: string;
      targetLanguage: string;
      context?: string;
    }>({
      query: (data) => ({
        url: 'translations/translate',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['Translation'],
    }),
    
    // Health and System API endpoints
    getSystemHealth: builder.query<ApiResponse<SystemHealth>, void>({
      query: () => 'system/health',
      providesTags: ['SystemHealth'],
    }),
    
    getServiceHealth: builder.query<ApiResponse<ServiceHealth[]>, void>({
      query: () => 'system/services',
      providesTags: ['ServiceHealth'],
    }),
    
    getSystemMetrics: builder.query<ApiResponse<any>, void>({
      query: () => 'system/metrics',
      providesTags: ['SystemMetrics'],
    }),
    
    // Configuration API endpoints
    getConfiguration: builder.query<ApiResponse<any>, void>({
      query: () => 'system/config',
    }),
    
    updateConfiguration: builder.mutation<ApiResponse<void>, Record<string, any>>({
      query: (config) => ({
        url: 'system/config',
        method: 'PATCH',
        body: config,
      }),
    }),
    
    // Logs API endpoints
    getLogs: builder.query<ApiResponse<any[]>, {
      level?: string;
      source?: string;
      limit?: number;
      offset?: number;
    }>({
      query: ({ level, source, limit = 100, offset = 0 }) => ({
        url: 'system/logs',
        params: { level, source, limit, offset },
      }),
    }),
    
    // Statistics API endpoints
    getStatistics: builder.query<ApiResponse<any>, {
      timeRange?: string;
      metrics?: string[];
    }>({
      query: ({ timeRange = '24h', metrics }) => ({
        url: 'system/statistics',
        params: { timeRange, metrics: metrics?.join(',') },
      }),
    }),
    
    // Bot testing and smoke tests
    runBotSmokeTest: builder.mutation<ApiResponse<any>, void>({
      query: () => ({
        url: 'bot/smoke-test',
        method: 'POST',
      }),
    }),
    
    getBotMemoryStats: builder.query<ApiResponse<any>, string>({
      query: (botId) => `bot/${botId}/memory-stats`,
    }),
    
    // Audio testing endpoints
    testAudioCapture: builder.mutation<ApiResponse<any>, {
      deviceId: string;
      duration: number;
    }>({
      query: (data) => ({
        url: 'audio/test-capture',
        method: 'POST',
        body: data,
      }),
    }),
    
    analyzeAudioQuality: builder.mutation<ApiResponse<any>, {
      audioBlob: Blob;
    }>({
      query: ({ audioBlob }) => {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        return {
          url: 'audio/analyze-quality',
          method: 'POST',
          body: formData,
        };
      },
    }),
    
    // WebSocket connection management
    getWebSocketInfo: builder.query<ApiResponse<any>, void>({
      query: () => 'websocket/info',
    }),
    
    // Feature flags
    getFeatureFlags: builder.query<ApiResponse<Record<string, boolean>>, void>({
      query: () => 'system/features',
    }),
    
    updateFeatureFlag: builder.mutation<ApiResponse<void>, {
      flag: string;
      enabled: boolean;
    }>({
      query: ({ flag, enabled }) => ({
        url: `system/features/${flag}`,
        method: 'PATCH',
        body: { enabled },
      }),
    }),

    // Pipeline Processing endpoints - standardized from pipelineApiClient.ts
    processPipeline: builder.mutation<any, {
      pipelineConfig: any;
      audioData?: Blob | string;
      processingMode: 'realtime' | 'batch' | 'preview';
      outputFormat?: 'wav' | 'mp3' | 'base64';
      metadata?: any;
    }>({
      query: ({ pipelineConfig, audioData, processingMode, outputFormat, metadata }) => {
        const formData = new FormData();
        formData.append('pipeline_config', JSON.stringify(pipelineConfig));
        
        if (audioData instanceof Blob) {
          formData.append('audio_file', audioData, 'audio.wav');
        } else if (audioData) {
          formData.append('audio_data', audioData);
        }
        
        formData.append('processing_mode', processingMode);
        formData.append('output_format', outputFormat || 'wav');
        
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }

        return {
          url: 'audio/pipeline/process',
          method: 'POST',
          body: formData,
        };
      },
    }),

    processSingleStage: builder.mutation<any, {
      stageType: string;
      audioData: Blob | string;
      stageConfig: Record<string, any>;
    }>({
      query: ({ stageType, audioData, stageConfig }) => {
        const formData = new FormData();
        
        if (audioData instanceof Blob) {
          formData.append('audio_file', audioData, 'audio.wav');
        } else {
          formData.append('audio_data', audioData);
        }
        
        formData.append('stage_config', JSON.stringify(stageConfig));

        return {
          url: `audio/process/stage/${stageType}`,
          method: 'POST',
          body: formData,
        };
      },
    }),

    getFFTAnalysis: builder.mutation<any, Blob | string>({
      query: (audioData) => {
        const formData = new FormData();
        
        if (audioData instanceof Blob) {
          formData.append('audio_file', audioData, 'audio.wav');
        } else {
          formData.append('audio_data', audioData);
        }

        return {
          url: 'audio/analyze/fft',
          method: 'POST',
          body: formData,
        };
      },
    }),

    getLUFSAnalysis: builder.mutation<any, Blob | string>({
      query: (audioData) => {
        const formData = new FormData();
        
        if (audioData instanceof Blob) {
          formData.append('audio_file', audioData, 'audio.wav');
        } else {
          formData.append('audio_data', audioData);
        }

        return {
          url: 'audio/analyze/lufs',
          method: 'POST',
          body: formData,
        };
      },
    }),

    startRealtimeSession: builder.mutation<any, {
      pipelineConfig: any;
      sessionConfig?: any;
    }>({
      query: ({ pipelineConfig, sessionConfig }) => ({
        url: 'pipeline/realtime/start',
        method: 'POST',
        body: {
          pipeline_config: pipelineConfig,
          session_config: sessionConfig || {
            chunkSize: 1024,
            sampleRate: 16000,
            channels: 1,
          },
        },
      }),
    }),

    // Processing Presets endpoints
    getProcessingPresets: builder.query<ApiResponse<any[]>, void>({
      query: () => 'audio/presets',
      providesTags: ['ProcessingPreset'],
    }),

    saveProcessingPreset: builder.mutation<ApiResponse<void>, {
      name: string;
      pipelineConfig: any;
      metadata: {
        description: string;
        category: string;
        tags: string[];
      };
    }>({
      query: ({ name, pipelineConfig, metadata }) => ({
        url: 'audio/presets/save',
        method: 'POST',
        body: {
          name,
          pipeline_config: pipelineConfig,
          metadata,
        },
      }),
      invalidatesTags: ['ProcessingPreset'],
    }),

    // Additional preset endpoints for pipeline processing
    getPresets: builder.query<any[], void>({
      query: () => 'audio/presets',
      providesTags: ['ProcessingPreset'],
    }),

    // Analytics endpoints to replace unifiedAnalyticsService.ts
    getAnalyticsOverview: builder.query<ApiResponse<any>, void>({
      query: () => 'analytics/overview',
      providesTags: ['SystemMetrics'],
    }),

    // Audio Analysis endpoints - connecting backend audio analysis APIs
    getAudioQualityAnalysis: builder.mutation<any, Blob | string>({
      query: (audioData) => {
        const formData = new FormData();

        if (audioData instanceof Blob) {
          formData.append('audio_file', audioData, 'audio.wav');
        } else {
          formData.append('audio_data', audioData);
        }

        return {
          url: 'audio/analyze/quality',
          method: 'POST',
          body: formData,
        };
      },
    }),

    getSpectrumAnalysis: builder.query<any, string>({
      query: (sessionId) => `audio/analyze/spectrum/${sessionId}`,
    }),

    // Audio Statistics
    getAudioStats: builder.query<any, void>({
      query: () => 'audio/stats',
      providesTags: ['SystemMetrics'],
    }),

    // Audio Models endpoint
    getAudioModels: builder.query<any, void>({
      query: () => 'audio/models',
    }),

    // Audio Processing Stages Info
    getAudioStagesInfo: builder.query<any, void>({
      query: () => 'audio/stages/info',
    }),

    getStageConfig: builder.query<any, string>({
      query: (stageName) => `audio/stages/${stageName}/config`,
    }),

    // Bot Analytics endpoints - connecting backend bot analytics APIs
    getBotAnalytics: builder.query<any, string>({
      query: (botId) => `bot/${botId}/analytics`,
      providesTags: (result, error, botId) => [{ type: 'Bot', id: botId }],
    }),

    getBotPerformance: builder.query<any, string>({
      query: (botId) => `bot/${botId}/performance`,
      providesTags: (result, error, botId) => [{ type: 'Bot', id: botId }],
    }),

    getBotQualityReport: builder.query<any, string>({
      query: (botId) => `bot/${botId}/quality-report`,
      providesTags: (result, error, botId) => [{ type: 'Bot', id: botId }],
    }),

    getSessionAnalytics: builder.query<any, void>({
      query: () => 'bot/analytics/sessions',
      providesTags: ['BotSession'],
    }),

    getQualityAnalytics: builder.query<any, void>({
      query: () => 'bot/analytics/quality',
      providesTags: ['SystemMetrics'],
    }),

    // System Analytics endpoints
    getTrendAnalysis: builder.query<any, void>({
      query: () => 'analytics/trends',
      providesTags: ['SystemMetrics'],
    }),

    getActiveAlerts: builder.query<any, void>({
      query: () => 'analytics/alerts',
      providesTags: ['SystemMetrics'],
    }),

    getAudioProcessingAnalytics: builder.query<any, void>({
      query: () => 'analytics/audio/processing',
      providesTags: ['SystemMetrics'],
    }),

    getWebSocketAnalytics: builder.query<any, void>({
      query: () => 'analytics/websocket/connections',
      providesTags: ['SystemMetrics'],
    }),

    // Translation Analytics
    getTranslationAnalytics: builder.query<any, void>({
      query: () => 'analytics/translation/performance',
      providesTags: ['SystemMetrics'],
    }),

    // ============================================
    // Chat History endpoints
    // ============================================

    // User management
    createUser: builder.mutation<any, { user_id: string; email: string; name?: string }>({
      query: (userData) => ({
        url: 'chat/users',
        method: 'POST',
        body: userData,
      }),
    }),

    getUser: builder.query<any, string>({
      query: (userId) => `chat/users/${userId}`,
    }),

    // Session management
    createSession: builder.mutation<any, { user_id: string; session_type?: string; session_title?: string; enable_translation?: boolean; target_languages?: string[] }>({
      query: (sessionData) => ({
        url: 'chat/sessions',
        method: 'POST',
        body: sessionData,
      }),
    }),

    getSession: builder.query<any, string>({
      query: (sessionId) => `chat/sessions/${sessionId}`,
    }),

    listSessions: builder.query<any, { user_id: string; start_date?: string; end_date?: string; session_type?: string; limit?: number; offset?: number }>({
      query: ({ user_id, start_date, end_date, session_type, limit = 50, offset = 0 }) => {
        const params = new URLSearchParams({
          user_id,
          limit: String(limit),
          offset: String(offset),
        });
        if (start_date) params.append('start_date', start_date);
        if (end_date) params.append('end_date', end_date);
        if (session_type) params.append('session_type', session_type);
        return `chat/sessions?${params.toString()}`;
      },
    }),

    deleteSession: builder.mutation<void, string>({
      query: (sessionId) => ({
        url: `chat/sessions/${sessionId}`,
        method: 'DELETE',
      }),
    }),

    // Message management
    createMessage: builder.mutation<any, { session_id: string; role: string; content: string; original_language?: string; translated_content?: Record<string, string> }>({
      query: (messageData) => ({
        url: 'chat/messages',
        method: 'POST',
        body: messageData,
      }),
    }),

    getMessages: builder.query<any, { session_id: string; limit?: number; offset?: number }>({
      query: ({ session_id, limit = 100, offset = 0 }) =>
        `chat/messages/${session_id}?limit=${limit}&offset=${offset}`,
    }),

    // Search
    searchMessages: builder.query<any, { user_id: string; query: string; limit?: number }>({
      query: ({ user_id, query, limit = 50 }) =>
        `chat/search?user_id=${user_id}&query=${encodeURIComponent(query)}&limit=${limit}`,
    }),

    // Export
    exportSession: builder.query<any, { session_id: string; format?: 'json' | 'txt' }>({
      query: ({ session_id, format = 'json' }) =>
        `chat/export/${session_id}?format=${format}`,
    }),

    // Statistics
    getSessionStatistics: builder.query<any, string>({
      query: (sessionId) => `chat/statistics/${sessionId}`,
    }),
  }),
});

// Export hooks for components to use
export const {
  // Audio hooks
  useGetAudioDevicesQuery,
  useProcessAudioMutation,
  useUploadAudioFileMutation,
  useGetAudioProcessingPresetsQuery,
  useTestAudioCaptureMutation,
  useAnalyzeAudioQualityMutation,
  
  // Bot management hooks
  useGetBotsQuery,
  useGetBotQuery,
  useSpawnBotMutation,
  useTerminateBotMutation,
  useGetBotStatusQuery,
  useGetBotSessionsQuery,
  useGetBotSessionQuery,
  useRunBotSmokeTestMutation,
  useGetBotMemoryStatsQuery,
  
  // Virtual webcam hooks
  useGetWebcamFrameQuery,
  useUpdateWebcamConfigMutation,
  
  // Translation hooks
  useGetTranslationsQuery,
  useTranslateTextMutation,
  
  // Health and system hooks
  useGetSystemHealthQuery,
  useGetServiceHealthQuery,
  useGetSystemMetricsQuery,
  useGetConfigurationQuery,
  useUpdateConfigurationMutation,
  useGetLogsQuery,
  useGetStatisticsQuery,
  
  // WebSocket hooks
  useGetWebSocketInfoQuery,
  
  // Feature flag hooks
  useGetFeatureFlagsQuery,
  useUpdateFeatureFlagMutation,

  // Pipeline Processing hooks - standardized API patterns
  useProcessPipelineMutation,
  useProcessSingleStageMutation,
  useGetFFTAnalysisMutation,
  useGetLUFSAnalysisMutation,
  useStartRealtimeSessionMutation,
  useGetProcessingPresetsQuery,
  useSaveProcessingPresetMutation,
  useGetPresetsQuery,
  useGetAnalyticsOverviewQuery,

  // Audio Analysis hooks
  useGetAudioQualityAnalysisMutation,
  useGetSpectrumAnalysisQuery,
  useGetAudioStatsQuery,
  useGetAudioModelsQuery,
  useGetAudioStagesInfoQuery,
  useGetStageConfigQuery,

  // Bot Analytics hooks
  useGetBotAnalyticsQuery,
  useGetBotPerformanceQuery,
  useGetBotQualityReportQuery,
  useGetSessionAnalyticsQuery,
  useGetQualityAnalyticsQuery,

  // System Analytics hooks
  useGetTrendAnalysisQuery,
  useGetActiveAlertsQuery,
  useGetAudioProcessingAnalyticsQuery,
  useGetWebSocketAnalyticsQuery,
  useGetTranslationAnalyticsQuery,

  // Chat History hooks
  useCreateUserMutation,
  useGetUserQuery,
  useCreateSessionMutation,
  useGetSessionQuery,
  useListSessionsQuery,
  useDeleteSessionMutation,
  useCreateMessageMutation,
  useGetMessagesQuery,
  useSearchMessagesQuery,
  useExportSessionQuery,
  useGetSessionStatisticsQuery,
} = apiSlice;