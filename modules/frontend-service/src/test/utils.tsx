import React from 'react';
import { render as rtlRender, RenderOptions } from '@testing-library/react';
import { configureStore, PreloadedState } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { vi } from 'vitest';

// Import store slices
import audioSlice from '@/store/slices/audioSlice';
import botSlice from '@/store/slices/botSlice';
import websocketSlice from '@/store/slices/websocketSlice';
import uiSlice from '@/store/slices/uiSlice';
import systemSlice from '@/store/slices/systemSlice';
import { apiSlice } from '@/store/slices/apiSlice';
import { RootState } from '@/store';
import { theme } from '@/styles/theme';

// Mock data factories
export const createMockAudioDevice = (overrides = {}) => ({
  deviceId: 'mock-device-id',
  label: 'Mock Audio Device',
  kind: 'audioinput' as const,
  groupId: 'mock-group-id',
  ...overrides,
});

export const createMockBotInstance = (overrides = {}) => ({
  botId: 'mock-bot-id',
  status: 'active' as const,
  meetingInfo: {
    meetingId: 'mock-meeting-id',
    meetingTitle: 'Mock Meeting',
    organizerEmail: 'organizer@example.com',
    participantCount: 3,
  },
  audioCapture: {
    isCapturing: true,
    totalChunksCaptured: 100,
    averageQualityScore: 0.85,
    lastCaptureTimestamp: Date.now(),
    deviceInfo: 'Mock Audio Device',
  },
  captionProcessor: {
    totalCaptionsProcessed: 50,
    totalSpeakers: 3,
    speakerTimeline: [],
    lastCaptionTimestamp: Date.now(),
  },
  virtualWebcam: {
    isStreaming: true,
    framesGenerated: 1000,
    currentTranslations: [],
    webcamConfig: {
      width: 1280,
      height: 720,
      fps: 30,
      displayMode: 'overlay' as const,
      theme: 'dark' as const,
      maxTranslationsDisplayed: 5,
    },
  },
  timeCorrelation: {
    totalCorrelations: 25,
    successRate: 0.9,
    averageTimingOffset: 50,
    lastCorrelationTimestamp: Date.now(),
  },
  performance: {
    sessionDuration: 3600,
    totalProcessingTime: 1800,
    averageLatency: 150,
    errorCount: 2,
  },
  createdAt: Date.now() - 3600000,
  lastActiveAt: Date.now(),
  ...overrides,
});

export const createMockTranslation = (overrides = {}) => ({
  translationId: 'mock-translation-id',
  translatedText: 'This is a mock translation',
  sourceLanguage: 'en',
  targetLanguage: 'es',
  speakerName: 'Speaker 1',
  speakerId: 'speaker-1',
  translationConfidence: 0.95,
  timestamp: Date.now(),
  ...overrides,
});

export const createMockSystemStats = (overrides = {}) => ({
  totalBotsSpawned: 10,
  activeBots: 3,
  completedSessions: 7,
  errorRate: 0.05,
  averageSessionDuration: 3600,
  ...overrides,
});

// Create a test store
export function createTestStore(preloadedState?: PreloadedState<RootState>) {
  return configureStore({
    reducer: {
      audio: audioSlice.reducer,
      bot: botSlice.reducer,
      websocket: websocketSlice.reducer,
      ui: uiSlice.reducer,
      system: systemSlice.reducer,
      api: apiSlice.reducer,
    },
    preloadedState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: {
          ignoredActions: [
            'audio/setVisualizationData',
            'audio/setAudioBlob',
            'bot/updateAudioCapture',
            'websocket/messageReceived',
          ],
          ignoredActionsPaths: ['payload.blob', 'payload.frequencyData', 'payload.timeData'],
          ignoredPaths: [
            'audio.recording.blob',
            'audio.visualization.frequencyData',
            'audio.visualization.timeData',
          ],
        },
      }).concat(apiSlice.middleware),
  });
}

// Custom render function
interface ExtendedRenderOptions extends Omit<RenderOptions, 'queries'> {
  preloadedState?: PreloadedState<RootState>;
  store?: ReturnType<typeof createTestStore>;
  withRouter?: boolean;
  withTheme?: boolean;
}

export function render(
  ui: React.ReactElement,
  {
    preloadedState,
    store = createTestStore(preloadedState),
    withRouter = true,
    withTheme = true,
    ...renderOptions
  }: ExtendedRenderOptions = {}
) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    let wrapped = <Provider store={store}>{children}</Provider>;
    
    if (withRouter) {
      wrapped = <BrowserRouter>{wrapped}</BrowserRouter>;
    }
    
    if (withTheme) {
      wrapped = (
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {wrapped}
        </ThemeProvider>
      );
    }
    
    return wrapped;
  }

  return {
    store,
    ...rtlRender(ui, { wrapper: Wrapper, ...renderOptions }),
  };
}

// Mock API responses
export const mockApiResponse = (data: any, options: { delay?: number; shouldFail?: boolean } = {}) => {
  const { delay = 0, shouldFail = false } = options;
  
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (shouldFail) {
        reject(new Error('Mock API Error'));
      } else {
        resolve({
          ok: true,
          json: () => Promise.resolve(data),
          text: () => Promise.resolve(JSON.stringify(data)),
          blob: () => Promise.resolve(new Blob([JSON.stringify(data)])),
        });
      }
    }, delay);
  });
};

// Mock WebSocket
export const createMockWebSocket = () => {
  const eventListeners: Record<string, Array<(event: any) => void>> = {};
  
  return {
    send: vi.fn(),
    close: vi.fn(),
    addEventListener: vi.fn((event: string, listener: (event: any) => void) => {
      if (!eventListeners[event]) {
        eventListeners[event] = [];
      }
      eventListeners[event].push(listener);
    }),
    removeEventListener: vi.fn((event: string, listener: (event: any) => void) => {
      if (eventListeners[event]) {
        eventListeners[event] = eventListeners[event].filter(l => l !== listener);
      }
    }),
    dispatchEvent: vi.fn((event: string, data: any) => {
      if (eventListeners[event]) {
        eventListeners[event].forEach(listener => listener(data));
      }
    }),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  };
};

// User interaction helpers
export const createMockFileList = (files: File[]): FileList => {
  const fileList = {
    ...files,
    length: files.length,
    item: (index: number) => files[index] || null,
    [Symbol.iterator]: function* () {
      for (let i = 0; i < files.length; i++) {
        yield files[i];
      }
    },
  };
  
  return fileList as FileList;
};

export const createMockFile = (
  name: string,
  content: string,
  options: { type?: string; lastModified?: number } = {}
): File => {
  const { type = 'text/plain', lastModified = Date.now() } = options;
  
  return new File([content], name, {
    type,
    lastModified,
  });
};

// Audio mock helpers
export const createMockAudioBuffer = (
  sampleRate = 44100,
  numberOfChannels = 2,
  length = 44100
) => ({
  sampleRate,
  numberOfChannels,
  length,
  duration: length / sampleRate,
  getChannelData: vi.fn(() => new Float32Array(length)),
  copyFromChannel: vi.fn(),
  copyToChannel: vi.fn(),
});

export const createMockMediaStream = () => ({
  id: 'mock-stream-id',
  active: true,
  getTracks: vi.fn(() => []),
  getVideoTracks: vi.fn(() => []),
  getAudioTracks: vi.fn(() => [
    {
      id: 'mock-audio-track',
      kind: 'audio',
      label: 'Mock Audio Track',
      enabled: true,
      muted: false,
      readyState: 'live',
      stop: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
  ]),
  addTrack: vi.fn(),
  removeTrack: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
});

// Test data constants
export const MOCK_MEETING_REQUEST = {
  meetingId: 'abc-defg-hij',
  meetingTitle: 'Test Meeting',
  organizerEmail: 'test@example.com',
  targetLanguages: ['en', 'es', 'fr'],
  autoTranslation: true,
  priority: 'medium' as const,
};

export const MOCK_AUDIO_CONFIG = {
  sampleRate: 16000,
  channels: 1,
  dtype: 'float32',
  blocksize: 1024,
  chunkDuration: 1.0,
  qualityThreshold: 0.7,
  duration: 30,
  deviceId: 'mock-device-id',
  format: 'audio/webm;codecs=opus',
  quality: 'medium',
  autoStop: true,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  rawAudio: false,
  source: 'microphone' as const,
};

// Re-export everything from testing library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';