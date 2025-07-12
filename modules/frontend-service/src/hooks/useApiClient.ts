import { useCallback } from 'react';
import { useAppDispatch } from '@/store';
import { addNotification } from '@/store/slices/uiSlice';
import { 
  WebSocketEventType, 
  WebSocketEventData,
  BotSpawnRequest,
  SystemHealthRequest 
} from '@/types';

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
  const dispatch = useAppDispatch();
  
  const defaultConfig: ApiClientConfig = {
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
      dispatch(addNotification({
        type: 'success',
        title: 'Bot Spawned',
        message: `Bot spawned successfully via API`,
        autoHide: true
      }));
    } else {
      dispatch(addNotification({
        type: 'error',
        title: 'Bot Spawn Failed',
        message: response.error || 'Failed to spawn bot',
        autoHide: false
      }));
    }

    return response;
  }, [apiRequest, dispatch]);

  const terminateBot = useCallback(async (botId: string) => {
    const response = await apiRequest<{ success: boolean }>(`/bot/${botId}/terminate`, {
      method: 'POST',
    });

    if (response.success) {
      dispatch(addNotification({
        type: 'info',
        title: 'Bot Terminated',
        message: `Bot ${botId} terminated via API`,
        autoHide: true
      }));
    }

    return response;
  }, [apiRequest, dispatch]);

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
      dispatch(addNotification({
        type: 'warning',
        title: 'Health Check Failed',
        message: 'Unable to retrieve system health status',
        autoHide: true
      }));
    }
    
    return response;
  }, [apiRequest, dispatch]);

  const getServiceHealth = useCallback(async (serviceName: string) => {
    return await apiRequest<any>(`/health/${serviceName}`);
  }, [apiRequest]);

  // Audio processing API endpoints
  const uploadAudio = useCallback(async (audioBlob: Blob, options: any = {}) => {
    const formData = new FormData();
    formData.append('file', audioBlob);
    
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    const response = await apiRequest<any>('/audio/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });

    if (response.success) {
      dispatch(addNotification({
        type: 'success',
        title: 'Audio Processed',
        message: 'Audio processing completed via API',
        autoHide: true
      }));
    } else {
      dispatch(addNotification({
        type: 'error',
        title: 'Audio Processing Failed',
        message: response.error || 'Failed to process audio',
        autoHide: false
      }));
    }

    return response;
  }, [apiRequest, dispatch]);

  // Translation API endpoints  
  const translateText = useCallback(async (text: string, options: any = {}) => {
    const response = await apiRequest<any>('/translate', {
      method: 'POST',
      body: JSON.stringify({
        text,
        ...options,
      }),
    });

    return response;
  }, [apiRequest]);

  // Generic message sender that maps WebSocket events to API calls
  const sendMessage = useCallback(async <T extends WebSocketEventType>(
    type: T,
    data: WebSocketEventData<T>,
    correlationId?: string
  ) => {
    console.log(`API fallback: ${type}`, data);

    switch (type) {
      case 'bot:spawn':
        return await spawnBot(data as BotSpawnRequest);
        
      case 'bot:terminate':
        if ('botId' in data) {
          return await terminateBot((data as any).botId);
        }
        break;
        
      case 'bot:get_status':
        if ('botId' in data) {
          return await getBotStatus((data as any).botId);
        }
        break;
        
      case 'system:health_check':
        return await getSystemHealth();
        
      case 'audio:upload':
        if ('audioBlob' in data) {
          return await uploadAudio((data as any).audioBlob, (data as any).options);
        }
        break;
        
      case 'translate:text':
        if ('text' in data) {
          return await translateText((data as any).text, (data as any).options);
        }
        break;
        
      case 'connection:ping':
        // Simulate ping with health check
        return await getSystemHealth();
        
      default:
        console.warn(`API fallback not implemented for event type: ${type}`);
        dispatch(addNotification({
          type: 'warning',
          title: 'Feature Unavailable',
          message: `${type} requires WebSocket connection`,
          autoHide: true
        }));
        return { success: false, error: 'Not supported in API mode' };
    }

    return { success: false, error: 'Invalid request' };
  }, [spawnBot, terminateBot, getBotStatus, getSystemHealth, uploadAudio, translateText, dispatch]);

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