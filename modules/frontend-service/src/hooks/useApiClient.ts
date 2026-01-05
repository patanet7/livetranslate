import { useCallback } from 'react';
import {
  WebSocketEventType,
  WebSocketEventData,
  BotSpawnRequest
} from '@/types';
import { useNotifications } from './useNotifications';

interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

export const useApiClient = (config?: Partial<ApiClientConfig>) => {
  const { notifySuccess, notifyError, notifyWarning, notifyInfo } = useNotifications();

  const defaultConfig: ApiClientConfig = {
    // Use relative path '/api' which works with both dev proxy and production
    // Vite dev server will proxy /api -> http://localhost:3000/api
    // Production build served from orchestration service uses same /api path
    baseUrl: '/api',
    timeout: 10000,
    retries: 2,
    ...config
  };

  // Generic API request function with retry logic
  const apiRequest = useCallback(async <T>(
    endpoint: string,
    options: RequestInit = {},
    retryCount = 0
  ): Promise<ApiResponse<T>> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), defaultConfig.timeout);

      const response = await fetch(`${defaultConfig.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return { success: true, data };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Retry logic for network errors
      if (retryCount < defaultConfig.retries && 
          (errorMessage.includes('fetch') || errorMessage.includes('network'))) {
        console.log(`API request failed, retrying... (${retryCount + 1}/${defaultConfig.retries})`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
        return apiRequest<T>(endpoint, options, retryCount + 1);
      }

      console.error(`API request failed: ${errorMessage}`);
      return { success: false, error: errorMessage };
    }
  }, [defaultConfig]);

  // Bot management API endpoints
  const spawnBot = useCallback(async (request: BotSpawnRequest) => {
    const response = await apiRequest<{ botId: string; status: string }>('/bot/spawn', {
      method: 'POST',
      body: JSON.stringify(request),
    });

    if (response.success) {
      notifySuccess('Bot Spawned', 'Bot spawned successfully via API');
    } else {
      notifyError('Bot Spawn Failed', response.error || 'Failed to spawn bot');
    }

    return response;
  }, [apiRequest, notifySuccess, notifyError]);

  const terminateBot = useCallback(async (botId: string) => {
    const response = await apiRequest<{ success: boolean }>(`/bot/${botId}/terminate`, {
      method: 'POST',
    });

    if (response.success) {
      notifyInfo('Bot Terminated', `Bot ${botId} terminated via API`);
    }

    return response;
  }, [apiRequest, notifyInfo]);

  const getBotStatus = useCallback(async (botId: string) => {
    return await apiRequest<any>(`/bot/${botId}/status`);
  }, [apiRequest]);

  const getActiveBots = useCallback(async () => {
    return await apiRequest<any[]>('/bot/active');
  }, [apiRequest]);

  // System health API endpoints
  const getSystemHealth = useCallback(async () => {
    const response = await apiRequest<any>('/health');
    
    if (!response.success) {
      notifyWarning('Health Check Failed', 'Unable to retrieve system health status');
    }

    return response;
  }, [apiRequest, notifyWarning]);

  const getServiceHealth = useCallback(async (serviceName: string) => {
    return await apiRequest<any>(`/health/${serviceName}`);
  }, [apiRequest]);

  // Audio processing API endpoints
  const uploadAudio = useCallback(async (audioBlob: Blob, options: any = {}) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    const response = await apiRequest<any>('/audio/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });

    if (response.success) {
      notifySuccess('Audio Processed', 'Audio processing completed via API');
    } else {
      notifyError('Audio Processing Failed', response.error || 'Failed to process audio');
    }

    return response;
  }, [apiRequest, notifySuccess, notifyError]);

  // Translation API endpoints  
  const translateText = useCallback(async (request: any) => {
    const response = await apiRequest<any>('/translate/', {
      method: 'POST',
      body: JSON.stringify({
        text: request.text,
        target_language: request.target_language || request.targetLanguage,
        source_language: request.source_language || request.sourceLanguage,
        model: request.model || 'default',
        quality: request.quality || 'balanced',
        prompt_id: request.prompt_id || request.promptId,
        session_id: request.session_id || request.sessionId,
      }),
    });

    return response;
  }, [apiRequest]);

  // Generic message sender that maps WebSocket events to API calls
  const sendMessage = useCallback(async <T extends WebSocketEventType>(
    type: T,
    data: WebSocketEventData<T>,
    _correlationId?: string
  ): Promise<ApiResponse> => {
    console.log(`API fallback: ${type}`, data);

    switch (type) {
      case 'bot:spawned':
        if ('botId' in data && 'meetingId' in data) {
          return await spawnBot(data as unknown as BotSpawnRequest);
        }
        break;

      case 'bot:terminated':
        if ('botId' in data) {
          return await terminateBot((data as any).botId);
        }
        break;

      case 'bot:status_change':
        if ('botId' in data) {
          return await getBotStatus((data as any).botId);
        }
        break;

      case 'system:health_update':
        return await getSystemHealth();

      case 'audio:chunk':
        if ('chunk' in data) {
          const audioBlob = new Blob([data.chunk as ArrayBuffer], { type: 'audio/webm' });
          return await uploadAudio(audioBlob, {});
        }
        break;

      case 'connection:ping':
        // Simulate ping with health check
        return await getSystemHealth();

      default:
        console.warn(`API fallback not implemented for event type: ${type}`);
        notifyWarning('Feature Unavailable', `${type} requires WebSocket connection`);
        return { success: false, error: 'Not supported in API mode' };
    }

    return { success: false, error: 'Invalid request' };
  }, [spawnBot, terminateBot, getBotStatus, getSystemHealth, uploadAudio, notifyWarning]);

  // Polling for real-time updates when in API mode
  const startPolling = useCallback((interval = 5000) => {
    const pollId = setInterval(async () => {
      try {
        const healthResponse = await getSystemHealth();
        if (healthResponse.success) {
          // Dispatch health updates similar to WebSocket
          // This would trigger the same state updates as WebSocket messages
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, interval);

    return () => clearInterval(pollId);
  }, [getSystemHealth]);

  return {
    // Direct API methods
    spawnBot,
    terminateBot,
    getBotStatus,
    getActiveBots,
    getSystemHealth,
    getServiceHealth,
    uploadAudio,
    translateText,
    
    // WebSocket replacement
    sendMessage,
    startPolling,
    
    // Low-level API access
    apiRequest,
    
    // Configuration
    config: defaultConfig,
  };
};

export type ApiClient = ReturnType<typeof useApiClient>;