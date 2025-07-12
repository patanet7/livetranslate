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

// Base query with automatic error handling
const baseQuery = fetchBaseQuery({
  baseUrl: '/api',
  prepareHeaders: (headers, { getState }) => {
    // Add any authentication headers here if needed
    headers.set('Content-Type', 'application/json');
    return headers;
  },
});

// API slice with all service endpoints
export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery,
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
    
    uploadAudioFile: builder.mutation<ApiResponse<{ fileId: string }>, {
      file: File;
      metadata?: Record<string, any>;
    }>({
      query: ({ file, metadata }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) formData.append('metadata', JSON.stringify(metadata));
        
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
      providesTags: (result, error, botId) => [{ type: 'Bot', id: botId }],
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
      providesTags: (result, error, botId) => [{ type: 'Bot', id: botId }],
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
      providesTags: (result, error, sessionId) => [{ type: 'BotSession', id: sessionId }],
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
} = apiSlice;