# Frontend Service - Modern React User Interface

**Technology Stack**: React 18 + TypeScript + Material-UI + Vite

## Service Overview

The Frontend Service is a dedicated React application that provides the modern user interface for the LiveTranslate system. It focuses on audio capture testing, real-time dashboards, and comprehensive bot management capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Service                             │
│                  [REACT + TYPESCRIPT]                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Audio       │  │ Bot         │  │ Dashboard   │  │ Settings│ │
│  │ Testing     │  │ Management  │  │ Overview    │  │ Config  │ │
│  │ • Recording │  │ • Spawning  │  │ • Metrics   │  │ • Audio │ │
│  │ • Playback  │  │ • Analytics │  │ • Health    │  │ • Theme │ │
│  │ • Analysis  │  │ • Sessions  │  │ • Real-time │  │ • Prefs │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                           ↓                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Redux Store │  │ WebSocket   │  │ API Client  │  │ Testing │ │
│  │ • Audio     │  │ • Real-time │  │ • RTK Query │  │ • Vitest │ │
│  │ • Bot State │  │ • Events    │  │ • Endpoints │  │ • E2E    │ │
│  │ • UI State  │  │ • Heartbeat │  │ • Caching   │  │ • Mocks  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Component Architecture                                      │  │
│  │ • Material-UI Design System                                │  │
│  │ • Responsive Layout with Mobile Support                    │  │
│  │ • Reusable UI Components                                   │  │
│  │ • Accessibility (WCAG 2.1 AA)                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Service Communication                         │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Port 5173) → API Proxy → Orchestration (Port 3000)  │
│  • REST API calls proxied through Vite dev server              │
│  • WebSocket connections for real-time updates                 │
│  • Automatic retry and error handling                          │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

### ✅ COMPLETED - Modern React Architecture
- **React 18 + TypeScript** → ✅ **Complete** with strict type safety
- **Material-UI Design System** → ✅ **Professional** theme and components
- **Redux Toolkit State Management** → ✅ **Complete** with RTK Query
- **Responsive Layout** → ✅ **Mobile-first** design approach
- **Component Architecture** → ✅ **Reusable** and well-structured
- **Testing Framework** → ✅ **Comprehensive** unit, integration, and E2E tests
- **Build Pipeline** → ✅ **Optimized** with Vite and code splitting

### 🎨 UI/UX Features
- **Modern Dashboard** → Real-time metrics and system status
- **Audio Testing Interface** → Comprehensive recording and analysis tools
- **Bot Management Panel** → Complete bot lifecycle management
- **Settings Configuration** → User preferences and system configuration
- **Dark/Light Theme** → Professional theme switching
- **Responsive Design** → Works seamlessly on desktop, tablet, and mobile

### 🔧 Technical Features
- **TypeScript** → Full type safety with strict configuration
- **Hot Module Replacement** → Instant development feedback
- **Code Splitting** → Optimized bundle loading
- **Service Workers** → Offline capability and caching
- **Error Boundaries** → Graceful error handling
- **Performance Monitoring** → Web Vitals and runtime metrics

## Service Communication

### API Integration
The frontend communicates with the orchestration service through:

```typescript
// API Configuration
const API_BASE_URL = 'http://localhost:3000';

// RTK Query API Client
const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers) => {
      headers.set('Content-Type', 'application/json');
      return headers;
    },
  }),
  tagTypes: ['Audio', 'Bot', 'System'],
  endpoints: (builder) => ({
    // Audio endpoints
    uploadAudio: builder.mutation({
      query: (audioData) => ({
        url: '/audio/upload',
        method: 'POST',
        body: audioData,
      }),
    }),
    // Bot endpoints
    spawnBot: builder.mutation({
      query: (botConfig) => ({
        url: '/bot/spawn',
        method: 'POST',
        body: botConfig,
      }),
    }),
    // System endpoints
    getSystemHealth: builder.query({
      query: () => '/system/health',
    }),
  }),
});
```

### WebSocket Integration
```typescript
// WebSocket Hook
export const useWebSocket = (endpoint: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:3000${endpoint}`);
    
    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      // Handle real-time updates
    };
    
    setSocket(ws);
    return () => ws.close();
  }, [endpoint]);
  
  return { socket, isConnected };
};
```

## Development

### Quick Start
```bash
# Navigate to frontend service
cd modules/frontend-service

# Install dependencies
pnpm install

# Start development server
pnpm dev

# Access application
# http://localhost:5173
```

### Available Scripts
```bash
# Development
pnpm dev                    # Start dev server with HMR
pnpm build                  # Production build
pnpm preview               # Preview production build

# Testing
pnpm test                  # Run unit tests
pnpm test:watch           # Watch mode
pnpm test:ui              # Visual test interface
pnpm test:coverage        # Coverage report
pnpm e2e                  # End-to-end tests

# Code Quality
pnpm lint                 # ESLint checking
pnpm lint:fix            # Auto-fix linting issues
pnpm format              # Prettier formatting
pnpm type-check          # TypeScript checking

# Analysis
pnpm analyze             # Bundle analysis
pnpm storybook           # Component documentation
```

## File Structure

```
modules/frontend-service/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── layout/         # Layout components
│   │   └── ui/             # Base UI components
│   ├── pages/              # Route components
│   │   ├── AudioTesting/   # Audio capture and testing
│   │   ├── BotManagement/  # Bot lifecycle management
│   │   ├── Dashboard/      # System overview
│   │   ├── Settings/       # Configuration
│   │   └── WebSocketTest/  # WebSocket testing
│   ├── hooks/              # Custom React hooks
│   │   ├── useAudioProcessing.ts
│   │   ├── useBotManager.ts
│   │   ├── useWebSocket.ts
│   │   └── __tests__/      # Hook tests
│   ├── store/              # Redux store
│   │   ├── slices/         # Redux slices
│   │   │   ├── audioSlice.ts
│   │   │   ├── botSlice.ts
│   │   │   ├── systemSlice.ts
│   │   │   └── apiSlice.ts
│   │   └── index.ts        # Store configuration
│   ├── styles/             # Styling and themes
│   │   ├── theme.ts        # Material-UI theme
│   │   └── globals.css     # Global styles
│   ├── types/              # TypeScript definitions
│   │   ├── audio.ts
│   │   ├── bot.ts
│   │   └── websocket.ts
│   └── test/               # Test utilities
│       ├── setup.ts        # Test setup
│       ├── utils.tsx       # Test utilities
│       └── e2e/           # E2E tests
├── public/                 # Static assets
├── package.json           # Dependencies and scripts
├── vite.config.ts         # Vite configuration
├── tsconfig.json          # TypeScript configuration
├── playwright.config.ts   # E2E test configuration
└── README.md              # Service documentation
```

## Component Architecture

### Page Components
```typescript
// Audio Testing Page
export const AudioTestingPage: React.FC = () => {
  const [audioConfig, setAudioConfig] = useState<AudioConfig>();
  const { processAudio } = useAudioProcessing();
  
  return (
    <Container>
      <AudioConfiguration config={audioConfig} onChange={setAudioConfig} />
      <RecordingControls onRecord={processAudio} />
      <AudioVisualizer />
      <ProcessingPresets />
    </Container>
  );
};

// Bot Management Page
export const BotManagementPage: React.FC = () => {
  const { bots, spawnBot, terminateBot } = useBotManager();
  
  return (
    <Container>
      <BotSpawner onSpawn={spawnBot} />
      <ActiveBots bots={bots} onTerminate={terminateBot} />
      <BotAnalytics />
      <SessionDatabase />
    </Container>
  );
};
```

### Custom Hooks
```typescript
// Audio Processing Hook
export const useAudioProcessing = () => {
  const [uploadAudio] = useUploadAudioMutation();
  
  const processAudio = useCallback(async (audioData: Blob) => {
    try {
      const result = await uploadAudio(audioData).unwrap();
      return result;
    } catch (error) {
      console.error('Audio processing failed:', error);
      throw error;
    }
  }, [uploadAudio]);
  
  return { processAudio };
};

// Bot Manager Hook
export const useBotManager = () => {
  const { data: bots } = useGetBotsQuery();
  const [spawnBot] = useSpawnBotMutation();
  const [terminateBot] = useTerminateBotMutation();
  
  return {
    bots: bots || [],
    spawnBot,
    terminateBot,
  };
};
```

## Testing Strategy

### Unit Tests (Vitest + React Testing Library)
```typescript
// Component Test
describe('AudioConfiguration', () => {
  it('should update configuration when settings change', () => {
    const mockOnChange = vi.fn();
    const { getByLabelText } = render(
      <AudioConfiguration config={mockConfig} onChange={mockOnChange} />
    );
    
    fireEvent.change(getByLabelText('Sample Rate'), {
      target: { value: '48000' }
    });
    
    expect(mockOnChange).toHaveBeenCalledWith({
      ...mockConfig,
      sampleRate: 48000
    });
  });
});

// Hook Test
describe('useAudioProcessing', () => {
  it('should process audio successfully', async () => {
    const { result } = renderHook(() => useAudioProcessing());
    const audioBlob = new Blob(['test'], { type: 'audio/wav' });
    
    await act(async () => {
      const result = await result.current.processAudio(audioBlob);
      expect(result).toBeDefined();
    });
  });
});
```

### Integration Tests
```typescript
// API Integration Test
describe('Bot Management Integration', () => {
  it('should spawn and manage bot lifecycle', async () => {
    const { getByText, getByTestId } = render(
      <Provider store={store}>
        <BotManagementPage />
      </Provider>
    );
    
    // Spawn bot
    fireEvent.click(getByText('Spawn Bot'));
    await waitFor(() => {
      expect(getByTestId('bot-list')).toHaveTextContent('Bot Active');
    });
    
    // Terminate bot
    fireEvent.click(getByText('Terminate'));
    await waitFor(() => {
      expect(getByTestId('bot-list')).toHaveTextContent('No Active Bots');
    });
  });
});
```

### E2E Tests (Playwright)
```typescript
// End-to-End Test
test('complete audio processing workflow', async ({ page }) => {
  await page.goto('http://localhost:5173');
  
  // Navigate to audio testing
  await page.click('text=Audio Testing');
  
  // Configure audio settings
  await page.selectOption('[data-testid=sample-rate]', '16000');
  
  // Start recording
  await page.click('[data-testid=record-button]');
  await page.waitForTimeout(2000);
  await page.click('[data-testid=stop-button]');
  
  // Verify processing
  await expect(page.locator('[data-testid=transcription]')).toBeVisible();
});
```

## Performance Optimization

### Bundle Optimization
```typescript
// Vite Configuration
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          redux: ['@reduxjs/toolkit', 'react-redux'],
        },
      },
    },
  },
});
```

### Code Splitting
```typescript
// Lazy Loading
const AudioTestingPage = lazy(() => import('./pages/AudioTesting'));
const BotManagementPage = lazy(() => import('./pages/BotManagement'));

// Route Configuration
<Route 
  path="/audio" 
  element={
    <Suspense fallback={<LoadingScreen />}>
      <AudioTestingPage />
    </Suspense>
  } 
/>
```

## Deployment

### Development
```bash
# Start frontend service
cd modules/frontend-service
pnpm dev

# Access at http://localhost:5173
# API calls proxied to http://localhost:3000
```

### Production Build
```bash
# Build optimized bundle
pnpm build

# Preview production build
pnpm preview

# Static files generated in dist/
```

### Docker Deployment
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install
COPY . .
RUN pnpm build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

## Environment Configuration

### Development Settings
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:3000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
```

### Environment Variables
```bash
# .env.development
VITE_API_BASE_URL=http://localhost:3000
VITE_WS_BASE_URL=ws://localhost:3000
VITE_ENABLE_DEBUG=true

# .env.production
VITE_API_BASE_URL=https://api.livetranslate.com
VITE_WS_BASE_URL=wss://api.livetranslate.com
VITE_ENABLE_DEBUG=false
```

## Key Features

### 🎙️ Audio Testing Interface
- **Multi-format Recording**: Support for WAV, MP3, WebM formats
- **Real-time Visualization**: Waveform and spectrum analysis
- **Processing Pipeline**: Complete audio processing with parameter control
- **Quality Analysis**: SNR, clipping detection, and audio metrics

### 🤖 Bot Management Dashboard
- **Bot Spawning**: Easy bot creation with configuration options
- **Real-time Monitoring**: Live bot status and performance metrics
- **Session Management**: Complete bot session lifecycle tracking
- **Analytics Dashboard**: Performance insights and usage statistics

### 📊 System Dashboard
- **Service Health**: Real-time status of all backend services
- **Performance Metrics**: API response times and success rates
- **Connection Monitoring**: WebSocket connection status and statistics
- **Error Tracking**: Comprehensive error logging and reporting

### ⚙️ Settings & Configuration
- **Theme Management**: Dark/light theme with system preference detection
- **Audio Configuration**: Comprehensive audio processing parameter control
- **User Preferences**: Persistent user settings and preferences
- **System Configuration**: Backend service configuration management

This frontend service provides a modern, professional, and highly functional user interface for the LiveTranslate system, focusing on excellent user experience and robust functionality.