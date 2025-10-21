import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { 
  WebSocketMessage, 
  WebSocketStats,
  WebSocketEventType
} from '@/types';
import { ConnectionState } from '@/types/websocket';

interface WebSocketState {
  // Connection state
  connection: ConnectionState;
  
  // Configuration
  config: {
    url: string;
    protocols: string[];
    autoReconnect: boolean;
    reconnectInterval: number;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
    connectionTimeout: number;
  };
  
  // Message handling
  messageQueue: WebSocketMessage[];
  lastMessage: WebSocketMessage | null;
  messageHistory: WebSocketMessage[];
  
  // Statistics
  stats: WebSocketStats;
  
  // Event subscriptions
  subscriptions: Record<WebSocketEventType, number>; // event type -> subscriber count
  
  // Error handling
  errors: Array<{
    timestamp: number;
    error: string;
    code?: number;
    context?: string;
  }>;
  
  // Status
  isInitialized: boolean;
  loading: boolean;
}

const initialState: WebSocketState = {
  connection: {
    isConnected: false,
    connectionId: null,
    lastPingTime: 0,
    reconnectAttempts: 0,
    error: null,
  },
  
  config: {
    // IMPORTANT: Use environment variable for WebSocket URL
    // Development: ws://localhost:5173/api/websocket/connect (uses Vite proxy to avoid CORS)
    // Production: ws://localhost:3000/api/websocket/connect (direct connection)
    url: `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:5173'}/api/websocket/connect`,
    protocols: [],
    autoReconnect: import.meta.env.VITE_WS_AUTO_RECONNECT !== 'false',
    reconnectInterval: Number(import.meta.env.VITE_WS_RECONNECT_INTERVAL) || 10000, // 10 seconds (was 5s - too aggressive)
    maxReconnectAttempts: Number(import.meta.env.VITE_WS_MAX_RECONNECT_ATTEMPTS) || 3,  // Only 3 attempts before API fallback (was 10)
    heartbeatInterval: Number(import.meta.env.VITE_WS_HEARTBEAT_INTERVAL) || 45000, // 45 seconds (was 30s - more conservative)
    connectionTimeout: 15000, // 15 seconds (was 10s)
  },
  
  messageQueue: [],
  lastMessage: null,
  messageHistory: [],
  
  stats: {
    connectionDuration: 0,
    messagesSent: 0,
    messagesReceived: 0,
    reconnectCount: 0,
    averageLatency: 0,
    lastPingTime: 0,
    bytesTransferred: 0,
  },
  
  subscriptions: {} as Record<WebSocketEventType, number>,
  
  errors: [],
  
  isInitialized: false,
  loading: false,
};

const websocketSlice = createSlice({
  name: 'websocket',
  initialState,
  reducers: {
    // Connection management
    initializeConnection: (state, action: PayloadAction<{ url?: string; protocols?: string[] }>) => {
      state.loading = true;
      if (action.payload.url) {
        state.config.url = action.payload.url;
      }
      if (action.payload.protocols) {
        state.config.protocols = action.payload.protocols;
      }
    },
    
    connectionEstablished: (state, action: PayloadAction<{ connectionId: string; serverTime: number }>) => {
      state.connection.isConnected = true;
      state.connection.connectionId = action.payload.connectionId;
      state.connection.error = null;
      state.connection.reconnectAttempts = 0;
      state.stats.connectionDuration = Date.now();
      state.isInitialized = true;
      state.loading = false;
    },
    
    connectionLost: (state, action: PayloadAction<{ reason: string; code?: number }>) => {
      state.connection.isConnected = false;
      state.connection.connectionId = null;
      state.connection.error = action.payload.reason;
      
      // Add to error history
      state.errors.push({
        timestamp: Date.now(),
        error: action.payload.reason,
        code: action.payload.code,
        context: 'connection_lost',
      });
      
      // Update stats
      if (state.stats.connectionDuration > 0) {
        state.stats.connectionDuration = Date.now() - state.stats.connectionDuration;
      }
      
      state.loading = false;
    },
    
    reconnecting: (state, action: PayloadAction<{ attempt: number }>) => {
      state.connection.reconnectAttempts = action.payload.attempt;
      state.loading = true;
    },
    
    reconnectFailed: (state) => {
      state.connection.isConnected = false;
      state.connection.error = 'Maximum reconnection attempts exceeded';
      state.loading = false;
      
      state.errors.push({
        timestamp: Date.now(),
        error: 'Reconnection failed after maximum attempts',
        context: 'reconnect_failed',
      });
    },
    
    // Message handling
    messageReceived: (state, action: PayloadAction<WebSocketMessage>) => {
      const message = action.payload;
      
      state.lastMessage = message;
      state.messageHistory.push(message);
      state.stats.messagesReceived += 1;
      
      // Keep message history limited
      if (state.messageHistory.length > 1000) {
        state.messageHistory.shift();
      }
      
      // Calculate latency if this is a pong message
      if (message.type === 'connection:pong' && state.connection.lastPingTime > 0) {
        const latency = Date.now() - state.connection.lastPingTime;
        state.stats.averageLatency = (state.stats.averageLatency + latency) / 2;
      }
    },
    
    messageSent: (state, action: PayloadAction<WebSocketMessage>) => {
      state.stats.messagesSent += 1;
      
      // Track ping time
      if (action.payload.type === 'connection:ping') {
        state.connection.lastPingTime = Date.now();
        state.stats.lastPingTime = Date.now();
      }
    },
    
    queueMessage: (state, action: PayloadAction<WebSocketMessage>) => {
      state.messageQueue.push(action.payload);
    },
    
    clearMessageQueue: (state) => {
      state.messageQueue = [];
    },
    
    processMessageQueue: (state) => {
      // This would trigger processing of queued messages
      // Implementation would be in middleware
      state.stats.messagesSent += state.messageQueue.length;
      state.messageQueue = [];
    },
    
    // Event subscriptions
    subscribe: (state, action: PayloadAction<WebSocketEventType>) => {
      const eventType = action.payload;
      state.subscriptions[eventType] = (state.subscriptions[eventType] || 0) + 1;
    },
    
    unsubscribe: (state, action: PayloadAction<WebSocketEventType>) => {
      const eventType = action.payload;
      if (state.subscriptions[eventType]) {
        state.subscriptions[eventType] -= 1;
        if (state.subscriptions[eventType] <= 0) {
          delete state.subscriptions[eventType];
        }
      }
    },
    
    // Configuration updates
    updateConfig: (state, action: PayloadAction<Partial<WebSocketState['config']>>) => {
      Object.assign(state.config, action.payload);
    },
    
    // Statistics updates
    updateStats: (state, action: PayloadAction<Partial<WebSocketStats>>) => {
      Object.assign(state.stats, action.payload);
    },
    
    incrementBytesTransferred: (state, action: PayloadAction<number>) => {
      state.stats.bytesTransferred += action.payload;
    },
    
    // Heartbeat
    updateHeartbeat: (state) => {
      state.connection.lastPingTime = Date.now();
      state.stats.lastPingTime = Date.now();
    },
    
    // Error handling
    addError: (state, action: PayloadAction<{ error: string; code?: number; context?: string }>) => {
      state.errors.push({
        timestamp: Date.now(),
        ...action.payload,
      });
      
      // Keep error history limited
      if (state.errors.length > 100) {
        state.errors.shift();
      }
    },
    
    clearErrors: (state) => {
      state.errors = [];
    },
    
    // Reset state
    resetWebSocketState: () => initialState,
    
    // Disconnect
    disconnect: (state) => {
      state.connection.isConnected = false;
      state.connection.connectionId = null;
      state.connection.error = null;
      state.messageQueue = [];
      state.loading = false;
      
      // Update connection duration
      if (state.stats.connectionDuration > 0) {
        state.stats.connectionDuration = Date.now() - state.stats.connectionDuration;
      }
    },
  },
});

export const {
  initializeConnection,
  connectionEstablished,
  connectionLost,
  reconnecting,
  reconnectFailed,
  messageReceived,
  messageSent,
  queueMessage,
  clearMessageQueue,
  processMessageQueue,
  subscribe,
  unsubscribe,
  updateConfig,
  updateStats,
  incrementBytesTransferred,
  updateHeartbeat,
  addError,
  clearErrors,
  resetWebSocketState,
  disconnect,
} = websocketSlice.actions;

export default websocketSlice;