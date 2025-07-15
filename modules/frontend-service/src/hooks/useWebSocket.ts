import { useEffect, useCallback, useRef, useState } from 'react';
import { useAppDispatch, useAppSelector } from '@/store';
import { 
  initializeConnection,
  connectionEstablished,
  connectionLost,
  reconnecting,
  reconnectFailed,
  messageReceived,
  messageSent,
  queueMessage,
  processMessageQueue,
  updateHeartbeat,
  addError,
  updateConfig
} from '@/store/slices/websocketSlice';
import {
  updateBotStatus,
  updateAudioCapture,
  addCaption,
  addTranslation,
  updateWebcamFrame,
} from '@/store/slices/botSlice';
import { updateSystemMetrics } from '@/store/slices/systemSlice';
import { addNotification } from '@/store/slices/uiSlice';
import { 
  WebSocketMessage, 
  WebSocketEventType, 
  WebSocketEventData,
  BotError 
} from '@/types';
import { useApiClient } from './useApiClient';

export const useWebSocket = () => {
  const dispatch = useAppDispatch();
  const { connection, config } = useAppSelector(state => state.websocket);
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const hasShownInitialConnectionRef = useRef(false);
  
  // API fallback
  const apiClient = useApiClient();
  const [useApiMode, setUseApiMode] = useState(false);
  const [wsFailureCount, setWsFailureCount] = useState(0);
  
  // Enhanced configuration with better defaults
  const enhancedConfig = {
    ...config,
    reconnectInterval: Math.min(config.reconnectInterval, 10000), // Max 10s
    maxReconnectAttempts: 3, // Only 3 attempts before fallback
    heartbeatInterval: Math.max(config.heartbeatInterval, 45000), // Min 45s
    debounceDelay: 2000, // 2s debounce
  };

  // Message handlers for different event types
  const handleMessage = useCallback((message: WebSocketMessage) => {
    dispatch(messageReceived(message));

    // Route messages based on type
    switch (message.type) {
      case 'connection:established':
        // Handled in connection event
        break;

      case 'bot:spawned': {
        const spawnData = message.data as { botId: string; meetingId: string; status: any };
        dispatch(updateBotStatus({
          botId: spawnData.botId,
          status: spawnData.status,
          data: spawnData
        }));
        dispatch(addNotification({
          type: 'success',
          title: 'Bot Spawned',
          message: `Bot for meeting ${spawnData.meetingId} has been spawned successfully`,
          autoHide: true
        }));
        break;
      }

      case 'bot:status_change': {
        const statusData = message.data as { botId: string; status: any; data?: any };
        dispatch(updateBotStatus({
          botId: statusData.botId,
          status: statusData.status,
          data: statusData.data
        }));
        break;
      }

      case 'bot:audio_capture': {
        const audioData = message.data as { botId: string; metrics: any };
        dispatch(updateAudioCapture({
          botId: audioData.botId,
          metrics: audioData.metrics
        }));
        break;
      }

      case 'bot:caption_received': {
        const captionData = message.data as { botId: string; caption: any };
        dispatch(addCaption({
          botId: captionData.botId,
          caption: captionData.caption
        }));
        break;
      }

      case 'bot:translation_ready': {
        const translationData = message.data as { botId: string; translation: any };
        dispatch(addTranslation({
          botId: translationData.botId,
          translation: translationData.translation
        }));
        break;
      }

      case 'bot:webcam_frame': {
        const frameData = message.data as { botId: string; frame: { frameData: string } };
        dispatch(updateWebcamFrame({
          botId: frameData.botId,
          frameBase64: frameData.frame.frameData
        }));
        break;
      }

      case 'bot:error':
        if ('botId' in message.data && 'error' in message.data) {
          const errorData = message.data as { botId: string; error: BotError };
          dispatch(addNotification({
            type: 'error',
            title: 'Bot Error',
            message: `Bot ${errorData.botId}: ${errorData.error.errorMessage}`,
            autoHide: false
          }));
        }
        break;

      case 'bot:terminated':
        if ('botId' in message.data && 'reason' in message.data) {
          const terminatedData = message.data as { botId: string; reason: string };
          dispatch(addNotification({
            type: 'info',
            title: 'Bot Terminated',
            message: `Bot ${terminatedData.botId} has been terminated: ${terminatedData.reason}`,
            autoHide: true
          }));
        }
        break;

      case 'system:health_update':
        if ('services' in message.data) {
          const healthData = message.data as { services: any; timestamp: number };
          // Handle both array and object formats for services
          let serviceHealth: any = {};
          if (Array.isArray(healthData.services)) {
            serviceHealth = healthData.services.reduce((acc: any, service: any) => {
              acc[service.serviceName?.toLowerCase() || service.name?.toLowerCase() || 'unknown'] = service;
              return acc;
            }, {});
          } else if (typeof healthData.services === 'object' && healthData.services !== null) {
            serviceHealth = healthData.services;
          }
          
          dispatch(updateSystemMetrics({
            serviceHealth
          }));
        }
        break;

      case 'system:performance_metrics':
        if ('cpu' in message.data && 'memory' in message.data) {
          const perfData = message.data as { cpu: number; memory: number; timestamp: number };
          dispatch(updateSystemMetrics({
            performance: {
              cpu: { usage: perfData.cpu, cores: 1 },
              memory: { used: 0, total: 1, percentage: perfData.memory }
            }
          }));
        }
        break;

      case 'system:alert':
        if ('level' in message.data && 'message' in message.data) {
          const alertData = message.data as { level: 'info' | 'warning' | 'error'; message: string; timestamp: number };
          dispatch(addNotification({
            type: alertData.level === 'error' ? 'error' : 
                  alertData.level === 'warning' ? 'warning' : 'info',
            title: 'System Alert',
            message: alertData.message,
            autoHide: alertData.level === 'info'
          }));
        }
        break;

      case 'ui:notification':
        if ('notification' in message.data) {
          const notificationData = message.data as { notification: any };
          dispatch(addNotification(notificationData.notification));
        }
        break;

      case 'system:heartbeat':
        // Server heartbeat - update last heartbeat time
        dispatch(updateHeartbeat());
        break;

      case 'connection:pong':
        // Response to our ping - update heartbeat
        dispatch(updateHeartbeat());
        break;

      case 'message:echo':
        // Echo response - can be ignored or logged
        console.debug('Received echo:', message.data);
        break;

      default:
        console.log('Unhandled WebSocket message type:', message.type);
    }
  }, [dispatch]);

  // Send message function
  const sendMessage = useCallback(<T extends WebSocketEventType>(
    type: T,
    data: WebSocketEventData<T>,
    correlationId?: string
  ) => {
    const message: WebSocketMessage<T> = {
      type,
      data,
      timestamp: Date.now(),
      messageId: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      correlationId,
    };

    if (websocketRef.current && connection.isConnected) {
      try {
        websocketRef.current.send(JSON.stringify(message));
        dispatch(messageSent(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        dispatch(queueMessage(message));
      }
    } else {
      dispatch(queueMessage(message));
    }
  }, [connection.isConnected, dispatch]);

  // Debounced connection function
  const debouncedConnect = useCallback(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, enhancedConfig.debounceDelay);
  }, [enhancedConfig.debounceDelay]);

  // Connection management with failure tracking
  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    
    // Check if we should use API mode
    if (wsFailureCount >= enhancedConfig.maxReconnectAttempts) {
      console.log('WebSocket failed 3+ times, switching to API mode');
      setUseApiMode(true);
      dispatch(addNotification({
        type: 'warning',
        title: 'WebSocket Unavailable',
        message: 'Switched to REST API mode for better stability',
        autoHide: true
      }));
      return;
    }
    
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      dispatch(initializeConnection({ url: enhancedConfig.url, protocols: enhancedConfig.protocols }));
      
      websocketRef.current = new WebSocket(enhancedConfig.url, enhancedConfig.protocols);

      websocketRef.current.onopen = () => {
        if (!mountedRef.current) return;
        
        const connectionId = `conn-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        dispatch(connectionEstablished({ 
          connectionId, 
          serverTime: Date.now() 
        }));
        
        // Reset failure count on successful connection
        setWsFailureCount(0);
        setUseApiMode(false);

        // Process any queued messages
        dispatch(processMessageQueue());

        // Start heartbeat with longer interval
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }
        heartbeatIntervalRef.current = setInterval(() => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            sendMessage('connection:ping', { timestamp: Date.now() });
            dispatch(updateHeartbeat());
            
            // Request health updates every few heartbeats
            if (Math.random() < 0.3) { // 30% chance each heartbeat
              sendMessage('system:health_request', { timestamp: Date.now() });
            }
          }
        }, enhancedConfig.heartbeatInterval);
        
        // Request initial health data
        setTimeout(() => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            sendMessage('system:health_request', { timestamp: Date.now() });
          }
        }, 1000);
        
        // Only show initial connection notification once per session
        if (!hasShownInitialConnectionRef.current) {
          hasShownInitialConnectionRef.current = true;
          dispatch(addNotification({
            type: 'success',
            title: 'WebSocket Connected',
            message: 'Real-time communication established',
            autoHide: true
          }));
        }
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          dispatch(addError({ 
            error: 'Failed to parse message', 
            context: 'message_parsing' 
          }));
        }
      };

      websocketRef.current.onclose = (event) => {
        if (!mountedRef.current) return;
        
        dispatch(connectionLost({ 
          reason: event.reason || 'Connection closed', 
          code: event.code 
        }));

        // Clear heartbeat
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
          heartbeatIntervalRef.current = null;
        }

        // Increment failure count for non-normal closures
        if (event.code !== 1000) {
          setWsFailureCount(prev => prev + 1);
        }

        // Attempt to reconnect with debouncing
        if (enhancedConfig.autoReconnect && connection.reconnectAttempts < enhancedConfig.maxReconnectAttempts) {
          const attemptNumber = connection.reconnectAttempts + 1;
          dispatch(reconnecting({ attempt: attemptNumber }));
          
          // Use exponential backoff
          const backoffDelay = Math.min(enhancedConfig.reconnectInterval * Math.pow(2, attemptNumber - 1), 30000);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              debouncedConnect();
            }
          }, backoffDelay);
        } else if (connection.reconnectAttempts >= enhancedConfig.maxReconnectAttempts) {
          dispatch(reconnectFailed());
          setWsFailureCount(prev => prev + 1);
        }
      };

      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        dispatch(addError({ 
          error: 'WebSocket connection error', 
          context: 'connection' 
        }));
        
        // Increment failure count on error
        setWsFailureCount(prev => prev + 1);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      dispatch(connectionLost({ 
        reason: 'Failed to create connection' 
      }));
      setWsFailureCount(prev => prev + 1);
    }
  }, [enhancedConfig, connection.reconnectAttempts, dispatch, handleMessage, sendMessage, wsFailureCount, debouncedConnect]);

  // Enhanced disconnect function
  const disconnect = useCallback(() => {
    mountedRef.current = false;
    
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
      debounceTimeoutRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'User disconnected');
      websocketRef.current = null;
    }
  }, []);

  // Manual connection retry function
  const retryConnection = useCallback(() => {
    setWsFailureCount(0);
    setUseApiMode(false);
    debouncedConnect();
  }, [debouncedConnect]);

  // Initialize connection on mount - single useEffect to prevent conflicts
  useEffect(() => {
    mountedRef.current = true;
    
    // Update config with enhanced settings
    dispatch(updateConfig({
      reconnectInterval: enhancedConfig.reconnectInterval,
      maxReconnectAttempts: enhancedConfig.maxReconnectAttempts,
      heartbeatInterval: enhancedConfig.heartbeatInterval,
    }));
    
    // Start initial connection
    debouncedConnect();

    return () => {
      disconnect();
    };
  }, []); // Empty dependency array to prevent reconnection loops

  // Enhanced send message with API fallback
  const sendMessageEnhanced = useCallback(<T extends WebSocketEventType>(
    type: T,
    data: WebSocketEventData<T>,
    correlationId?: string
  ) => {
    if (useApiMode) {
      // Use API client for critical operations
      return apiClient.sendMessage(type, data, correlationId);
    } else {
      return sendMessage(type, data, correlationId);
    }
  }, [useApiMode, apiClient, sendMessage]);

  return {
    sendMessage: sendMessageEnhanced,
    connect: debouncedConnect,
    disconnect,
    retryConnection,
    isConnected: connection.isConnected && !useApiMode,
    connectionId: connection.connectionId,
    reconnectAttempts: connection.reconnectAttempts,
    useApiMode,
    wsFailureCount,
    canRetryWebSocket: wsFailureCount < enhancedConfig.maxReconnectAttempts,
  };
};