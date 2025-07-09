# React Migration Plan for LiveTranslate Orchestration Service

## Executive Summary

This document outlines a comprehensive migration plan from the current Flask-based frontend to a modern React application with proper UX/UI design principles, component architecture, and state management.

## Current State Analysis

### Problems with Current Implementation
- **Poor UX/UI Design**: Components are poorly spaced and don't follow design principles
- **Monolithic Templates**: Large HTML files with embedded JavaScript
- **Inconsistent Styling**: Mixed CSS approaches and no design system
- **Poor State Management**: Global variables and scattered state
- **No Component Reusability**: Duplicate code across templates
- **Accessibility Issues**: No proper ARIA labels or keyboard navigation
- **Mobile Responsiveness**: Poor mobile experience
- **🆕 Missing Bot Management UI**: No proper interface for Google Meet bot management and monitoring

### Current Architecture
```
Flask Backend + Jinja2 Templates
├── templates/
│   ├── dashboard.html (2,000+ lines)
│   ├── audio-test-consolidated.html (1,500+ lines)
│   ├── websocket-test.html (800+ lines)
│   ├── settings.html (600+ lines)
│   └── 🆕 Missing: Bot management interfaces
├── static/
│   ├── css/styles.css (mixed styles)
│   └── js/ (vanilla JS with global state)
├── 🆕 src/bot/ (New bot system - needs UI)
│   ├── bot_manager.py (lifecycle management)
│   ├── audio_capture.py (real-time audio)
│   ├── virtual_webcam.py (translation display)
│   └── bot_integration.py (complete pipeline)
└── Python Flask routes
```

## Target React Architecture

### Modern React Stack
```
React Frontend + FastAPI Backend
├── src/
│   ├── components/ (reusable UI components)
│   │   ├── common/ (buttons, forms, modals)
│   │   ├── audio/ (audio processing UI)
│   │   ├── 🆕 bot/ (bot management components)
│   │   └── monitoring/ (system monitoring)
│   ├── pages/ (route-based page components)
│   │   ├── Dashboard/ (main orchestration dashboard)
│   │   ├── AudioProcessing/ (audio pipeline control)
│   │   ├── 🆕 BotManagement/ (Google Meet bot control)
│   │   └── Settings/ (system configuration)
│   ├── hooks/ (custom React hooks)
│   │   ├── useWebSocket/ (WebSocket management)
│   │   ├── 🆕 useBotLifecycle/ (bot state management)
│   │   └── useAudioProcessing/ (audio pipeline hooks)
│   ├── services/ (API and WebSocket services)
│   │   ├── orchestrationApi.ts (main API)
│   │   ├── 🆕 botApi.ts (bot management API)
│   │   └── websocketService.ts (real-time communication)
│   ├── store/ (Redux Toolkit state management)
│   │   ├── slices/ (feature-based state slices)
│   │   │   ├── audioSlice.ts (audio processing state)
│   │   │   ├── 🆕 botSlice.ts (bot management state)
│   │   │   └── systemSlice.ts (system monitoring state)
│   │   └── index.ts (store configuration)
│   ├── utils/ (utility functions)
│   └── styles/ (styled-components + design system)
├── public/
└── Python FastAPI backend (enhanced with bot endpoints)
```

## Phase 1: Project Setup & Foundation (Week 1-2)

### 1.1 Technology Stack Selection

**Frontend Stack:**
- **React 18**: Latest with concurrent features
- **TypeScript**: For type safety and better DX
- **Vite**: Fast build tool and dev server
- **Redux Toolkit**: State management with RTK Query
- **React Router 6**: Client-side routing
- **Material-UI (MUI)**: Component library with excellent UX
- **Styled Components**: CSS-in-JS for theming
- **Framer Motion**: Animations and transitions
- **React Hook Form**: Form management
- **Recharts**: Data visualization
- **Socket.io Client**: WebSocket management

**Backend Transition:**
- **FastAPI**: Replace Flask for better async support
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM (if needed)
- **Redis**: Session management and caching
- **WebSocket**: Real-time communication

**Development Tools:**
- **ESLint + Prettier**: Code formatting
- **Jest + React Testing Library**: Testing
- **Storybook**: Component development
- **MSW**: API mocking for development

### 1.2 Project Structure

```
modules/orchestration-service/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── audio.py
│   │   │   │   ├── websocket.py
│   │   │   │   └── health.py
│   │   │   └── deps.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── websocket_manager.py
│   │   ├── models/
│   │   └── services/
│   ├── requirements.txt
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/ (basic UI components)
│   │   │   ├── audio/ (audio-specific components)
│   │   │   ├── layout/ (layout components)
│   │   │   └── charts/ (visualization components)
│   │   ├── pages/
│   │   │   ├── Dashboard/
│   │   │   ├── AudioTesting/
│   │   │   ├── WebSocketTest/
│   │   │   └── Settings/
│   │   ├── hooks/
│   │   │   ├── useAudio.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useAudioProcessing.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── audio.ts
│   │   ├── store/
│   │   │   ├── slices/
│   │   │   └── index.ts
│   │   ├── types/
│   │   ├── utils/
│   │   └── styles/
│   │       ├── theme.ts
│   │       └── globals.ts
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
└── Dockerfile
```

### 1.3 Component Architecture Design

**Component Hierarchy:**
```
App
├── Layout
│   ├── Header
│   ├── Sidebar
│   └── Main
├── Pages
│   ├── Dashboard
│   │   ├── TranscriptionPanel
│   │   ├── TranslationPanel
│   │   ├── ServiceHealthPanel
│   │   ├── 🆕 BotOverviewPanel
│   │   └── ConnectionPanel
│   ├── AudioTesting
│   │   ├── RecordingConfiguration
│   │   ├── AudioRecorder
│   │   ├── AudioVisualizer
│   │   ├── ProcessingPresets
│   │   ├── PipelineProcessor
│   │   └── ResultsExporter
│   ├── 🆕 BotManagement
│   │   ├── BotDashboard
│   │   ├── BotSpawner
│   │   ├── ActiveBotsPanel
│   │   ├── BotSessionViewer
│   │   ├── VirtualWebcamViewer
│   │   ├── AudioCaptureMonitor
│   │   ├── SpeakerTimelineViewer
│   │   ├── TranslationResultViewer
│   │   └── BotPerformanceMetrics
│   ├── WebSocketTest
│   │   ├── ConnectionManager
│   │   ├── MessageTester
│   │   └── ConnectionLogs
│   └── Settings
│       ├── ServiceSettings
│       ├── AudioSettings
│       ├── 🆕 BotSettings
│       └── UISettings
└── Common Components
    ├── Button
    ├── Card
    ├── Modal
    ├── Slider
    ├── Toggle
    ├── ProgressBar
    ├── Waveform
    └── LogViewer
```

## Phase 2: UI/UX Design System (Week 2-3)

### 2.1 Design System Foundation

**Color Palette:**
```typescript
export const colors = {
  primary: {
    50: '#E3F2FD',
    100: '#BBDEFB',
    500: '#2196F3',
    700: '#1976D2',
    900: '#0D47A1'
  },
  secondary: {
    50: '#F3E5F5',
    100: '#E1BEE7',
    500: '#9C27B0',
    700: '#7B1FA2',
    900: '#4A148C'
  },
  success: {
    50: '#E8F5E8',
    500: '#4CAF50',
    700: '#388E3C'
  },
  warning: {
    50: '#FFF3E0',
    500: '#FF9800',
    700: '#F57C00'
  },
  error: {
    50: '#FFEBEE',
    500: '#F44336',
    700: '#D32F2F'
  },
  background: {
    default: '#FAFAFA',
    paper: '#FFFFFF',
    dark: '#121212'
  }
};
```

**Typography Scale:**
```typescript
export const typography = {
  h1: { fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2 },
  h2: { fontSize: '2rem', fontWeight: 600, lineHeight: 1.3 },
  h3: { fontSize: '1.5rem', fontWeight: 600, lineHeight: 1.4 },
  h4: { fontSize: '1.25rem', fontWeight: 600, lineHeight: 1.4 },
  body1: { fontSize: '1rem', fontWeight: 400, lineHeight: 1.5 },
  body2: { fontSize: '0.875rem', fontWeight: 400, lineHeight: 1.43 },
  caption: { fontSize: '0.75rem', fontWeight: 400, lineHeight: 1.66 }
};
```

**Spacing System:**
```typescript
export const spacing = {
  xs: '0.25rem',   // 4px
  sm: '0.5rem',    // 8px
  md: '1rem',      // 16px
  lg: '1.5rem',    // 24px
  xl: '2rem',      // 32px
  xxl: '3rem'      // 48px
};
```

### 2.2 Component Design Specifications

**Button Component:**
```typescript
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'outline' | 'ghost';
  size: 'small' | 'medium' | 'large';
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
  onClick?: () => void;
}
```

**Card Component:**
```typescript
interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  padding?: 'small' | 'medium' | 'large';
  elevation?: number;
  children: React.ReactNode;
}
```

**Audio Visualizer Component:**
```typescript
interface AudioVisualizerProps {
  audioData: Float32Array;
  type: 'waveform' | 'frequency' | 'level';
  height?: number;
  color?: string;
  animated?: boolean;
}
```

## Phase 3: State Management Architecture (Week 3-4)

### 3.1 Redux Toolkit Setup

**Store Structure:**
```typescript
// store/index.ts
export const store = configureStore({
  reducer: {
    audio: audioSlice.reducer,
    🆕 bot: botSlice.reducer,
    websocket: websocketSlice.reducer,
    settings: settingsSlice.reducer,
    ui: uiSlice.reducer,
    api: apiSlice.reducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiSlice.middleware)
});
```

**Audio State Slice:**
```typescript
interface AudioState {
  recording: {
    isRecording: boolean;
    duration: number;
    maxDuration: number;
    autoStop: boolean;
    format: string;
    sampleRate: number;
    blob: Blob | null;
    status: 'idle' | 'recording' | 'processing' | 'completed' | 'error';
  };
  playback: {
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    volume: number;
  };
  processing: {
    currentStage: number;
    isProcessing: boolean;
    progress: number;
    results: Record<string, any>;
    preset: string;
  };
  visualization: {
    audioLevel: number;
    frequencyData: Float32Array;
    timeData: Float32Array;
  };
  devices: AudioDevice[];
  error: string | null;
}
```

**🆕 Bot State Slice:**
```typescript
interface BotState {
  bots: {
    [botId: string]: {
      botId: string;
      status: 'spawning' | 'active' | 'error' | 'terminated';
      meetingInfo: {
        meetingId: string;
        meetingTitle: string;
        organizerEmail: string;
        participantCount: number;
      };
      audioCapture: {
        isCapturing: boolean;
        totalChunksCaptured: number;
        averageQualityScore: number;
        lastCaptureTimestamp: number;
        deviceInfo: string;
      };
      captionProcessor: {
        totalCaptionsProcessed: number;
        totalSpeakers: number;
        speakerTimeline: SpeakerTimelineEvent[];
        lastCaptionTimestamp: number;
      };
      virtualWebcam: {
        isStreaming: boolean;
        framesGenerated: number;
        currentTranslations: Translation[];
        webcamConfig: WebcamConfig;
      };
      timeCorrelation: {
        totalCorrelations: number;
        successRate: number;
        averageTimingOffset: number;
        lastCorrelationTimestamp: number;
      };
      performance: {
        sessionDuration: number;
        totalProcessingTime: number;
        averageLatency: number;
        errorCount: number;
      };
      createdAt: number;
      lastActiveAt: number;
    };
  };
  activeBotIds: string[];
  spawnerConfig: {
    maxConcurrentBots: number;
    defaultTargetLanguages: string[];
    autoTranslationEnabled: boolean;
    virtualWebcamEnabled: boolean;
  };
  systemStats: {
    totalBotsSpawned: number;
    activeBots: number;
    completedSessions: number;
    errorRate: number;
    averageSessionDuration: number;
  };
  meetingRequests: {
    [requestId: string]: {
      meetingId: string;
      meetingTitle: string;
      organizerEmail: string;
      targetLanguages: string[];
      autoTranslation: boolean;
      priority: 'low' | 'medium' | 'high';
      status: 'pending' | 'processing' | 'completed' | 'failed';
      createdAt: number;
    };
  };
  error: string | null;
  loading: boolean;
}
```

**Bot Management Actions:**
```typescript
// Bot lifecycle actions
const botSlice = createSlice({
  name: 'bot',
  initialState,
  reducers: {
    spawnBot: (state, action: PayloadAction<MeetingRequest>) => {
      state.loading = true;
      const requestId = generateId();
      state.meetingRequests[requestId] = {
        ...action.payload,
        status: 'pending',
        createdAt: Date.now()
      };
    },
    spawnBotSuccess: (state, action: PayloadAction<{requestId: string, botId: string}>) => {
      const { requestId, botId } = action.payload;
      const request = state.meetingRequests[requestId];
      if (request) {
        request.status = 'completed';
        state.bots[botId] = {
          botId,
          status: 'spawning',
          meetingInfo: {
            meetingId: request.meetingId,
            meetingTitle: request.meetingTitle,
            organizerEmail: request.organizerEmail,
            participantCount: 0
          },
          // ... initialize other bot state
        };
        state.activeBotIds.push(botId);
        state.systemStats.totalBotsSpawned++;
        state.systemStats.activeBots++;
      }
      state.loading = false;
    },
    updateBotStatus: (state, action: PayloadAction<{botId: string, status: BotStatus, data?: any}>) => {
      const { botId, status, data } = action.payload;
      if (state.bots[botId]) {
        state.bots[botId].status = status;
        state.bots[botId].lastActiveAt = Date.now();
        if (data) {
          Object.assign(state.bots[botId], data);
        }
      }
    },
    updateAudioCapture: (state, action: PayloadAction<{botId: string, metrics: AudioCaptureMetrics}>) => {
      const { botId, metrics } = action.payload;
      if (state.bots[botId]) {
        Object.assign(state.bots[botId].audioCapture, metrics);
      }
    },
    addTranslation: (state, action: PayloadAction<{botId: string, translation: Translation}>) => {
      const { botId, translation } = action.payload;
      if (state.bots[botId]) {
        state.bots[botId].virtualWebcam.currentTranslations.push(translation);
        // Keep only last 3 translations for display
        if (state.bots[botId].virtualWebcam.currentTranslations.length > 3) {
          state.bots[botId].virtualWebcam.currentTranslations.shift();
        }
      }
    },
    terminateBot: (state, action: PayloadAction<string>) => {
      const botId = action.payload;
      if (state.bots[botId]) {
        state.bots[botId].status = 'terminated';
        state.activeBotIds = state.activeBotIds.filter(id => id !== botId);
        state.systemStats.activeBots = Math.max(0, state.systemStats.activeBots - 1);
        state.systemStats.completedSessions++;
      }
    },
    removeBotFromState: (state, action: PayloadAction<string>) => {
      const botId = action.payload;
      delete state.bots[botId];
      state.activeBotIds = state.activeBotIds.filter(id => id !== botId);
    },
    updateSystemStats: (state, action: PayloadAction<Partial<SystemStats>>) => {
      Object.assign(state.systemStats, action.payload);
    },
    setBotError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
      state.loading = false;
    },
    clearBotError: (state) => {
      state.error = null;
    }
  }
});
```

### 3.2 Custom Hooks

**🆕 Bot Management Hooks:**
```typescript
export const useBotManager = () => {
  const dispatch = useAppDispatch();
  const botState = useAppSelector(state => state.bot);
  
  const spawnBot = useCallback(async (meetingRequest: MeetingRequest) => {
    try {
      dispatch(spawnBot(meetingRequest));
      const response = await fetch('/api/bot/spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(meetingRequest)
      });
      const { botId, requestId } = await response.json();
      dispatch(spawnBotSuccess({ requestId, botId }));
      return botId;
    } catch (error) {
      dispatch(setBotError(error.message));
      return null;
    }
  }, [dispatch]);
  
  const terminateBot = useCallback(async (botId: string) => {
    try {
      await fetch(`/api/bot/${botId}/terminate`, { method: 'POST' });
      dispatch(terminateBot(botId));
      return true;
    } catch (error) {
      dispatch(setBotError(error.message));
      return false;
    }
  }, [dispatch]);
  
  const getActiveBots = useCallback(() => {
    return botState.activeBotIds.map(id => botState.bots[id]).filter(Boolean);
  }, [botState]);
  
  const getBotById = useCallback((botId: string) => {
    return botState.bots[botId] || null;
  }, [botState]);
  
  return {
    ...botState,
    spawnBot,
    terminateBot,
    getActiveBots,
    getBotById,
    activeBots: getActiveBots(),
    systemStats: botState.systemStats
  };
};

export const useBotLifecycle = (botId: string) => {
  const dispatch = useAppDispatch();
  const bot = useAppSelector(state => state.bot.bots[botId]);
  
  const updateStatus = useCallback((status: BotStatus, data?: any) => {
    dispatch(updateBotStatus({ botId, status, data }));
  }, [dispatch, botId]);
  
  const updateAudioMetrics = useCallback((metrics: AudioCaptureMetrics) => {
    dispatch(updateAudioCapture({ botId, metrics }));
  }, [dispatch, botId]);
  
  const addTranslation = useCallback((translation: Translation) => {
    dispatch(addTranslation({ botId, translation }));
  }, [dispatch, botId]);
  
  return {
    bot,
    updateStatus,
    updateAudioMetrics,
    addTranslation,
    isActive: bot?.status === 'active',
    isCapturingAudio: bot?.audioCapture.isCapturing || false,
    isStreamingWebcam: bot?.virtualWebcam.isStreaming || false
  };
};

export const useVirtualWebcam = (botId: string) => {
  const dispatch = useAppDispatch();
  const webcamState = useAppSelector(state => state.bot.bots[botId]?.virtualWebcam);
  
  const getCurrentFrame = useCallback(async () => {
    try {
      const response = await fetch(`/api/bot/${botId}/webcam/frame`);
      const { frameBase64 } = await response.json();
      return frameBase64;
    } catch (error) {
      console.error('Failed to get webcam frame:', error);
      return null;
    }
  }, [botId]);
  
  const updateWebcamConfig = useCallback(async (config: Partial<WebcamConfig>) => {
    try {
      await fetch(`/api/bot/${botId}/webcam/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      // Update local state would be handled by WebSocket updates
    } catch (error) {
      console.error('Failed to update webcam config:', error);
    }
  }, [botId]);
  
  return {
    webcamState,
    getCurrentFrame,
    updateWebcamConfig,
    isStreaming: webcamState?.isStreaming || false,
    currentTranslations: webcamState?.currentTranslations || [],
    framesGenerated: webcamState?.framesGenerated || 0
  };
};
```

**useAudioRecording Hook:**
```typescript
export const useAudioRecording = () => {
  const dispatch = useAppDispatch();
  const audioState = useAppSelector(state => state.audio);
  
  const startRecording = useCallback(async (options: RecordingOptions) => {
    // Implementation
  }, []);
  
  const stopRecording = useCallback(() => {
    // Implementation
  }, []);
  
  const clearRecording = useCallback(() => {
    // Implementation
  }, []);
  
  return {
    ...audioState.recording,
    startRecording,
    stopRecording,
    clearRecording
  };
};
```

**useAudioProcessing Hook:**
```typescript
export const useAudioProcessing = () => {
  const dispatch = useAppDispatch();
  const processingState = useAppSelector(state => state.audio.processing);
  
  const runPipeline = useCallback(async (audioBlob: Blob) => {
    // Implementation
  }, []);
  
  const runStage = useCallback(async (stageId: string) => {
    // Implementation
  }, []);
  
  return {
    ...processingState,
    runPipeline,
    runStage
  };
};
```

## Phase 4: Component Development (Week 4-6)

### 4.1 Core UI Components

**Button Component:**
```typescript
// components/ui/Button.tsx
import styled from 'styled-components';

const StyledButton = styled.button<ButtonProps>`
  padding: ${props => props.size === 'small' ? '8px 16px' : '12px 24px'};
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.2s ease;
  
  ${props => props.variant === 'primary' && `
    background: ${props.theme.colors.primary[500]};
    color: white;
    border: none;
    
    &:hover {
      background: ${props.theme.colors.primary[700]};
    }
  `}
  
  ${props => props.loading && `
    opacity: 0.7;
    cursor: not-allowed;
  `}
`;

export const Button: React.FC<ButtonProps> = ({
  children,
  loading,
  icon,
  ...props
}) => (
  <StyledButton {...props}>
    {loading ? <Spinner /> : icon}
    {children}
  </StyledButton>
);
```

**Card Component:**
```typescript
// components/ui/Card.tsx
export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  actions,
  children,
  padding = 'medium',
  elevation = 1
}) => (
  <StyledCard elevation={elevation}>
    {(title || subtitle || actions) && (
      <CardHeader>
        <CardTitle>
          {title && <Typography variant="h4">{title}</Typography>}
          {subtitle && <Typography variant="body2" color="textSecondary">{subtitle}</Typography>}
        </CardTitle>
        {actions && <CardActions>{actions}</CardActions>}
      </CardHeader>
    )}
    <CardContent padding={padding}>
      {children}
    </CardContent>
  </StyledCard>
);
```

### 4.2 Audio-Specific Components

**AudioRecorder Component:**
```typescript
// components/audio/AudioRecorder.tsx
export const AudioRecorder: React.FC = () => {
  const {
    isRecording,
    duration,
    maxDuration,
    status,
    startRecording,
    stopRecording,
    clearRecording
  } = useAudioRecording();
  
  const { audioLevel, frequencyData } = useAudioVisualization();
  
  return (
    <Card title="Audio Recording">
      <RecorderControls>
        <RecorderSettings>
          <DurationSlider
            min={5}
            max={300}
            value={maxDuration}
            onChange={(value) => setMaxDuration(value)}
          />
          <DeviceSelector />
          <FormatSelector />
        </RecorderSettings>
        
        <RecorderActions>
          <Button
            variant="primary"
            onClick={startRecording}
            disabled={isRecording}
            icon={<MicIcon />}
          >
            {isRecording ? 'Recording...' : 'Start Recording'}
          </Button>
          
          <Button
            variant="secondary"
            onClick={stopRecording}
            disabled={!isRecording}
            icon={<StopIcon />}
          >
            Stop
          </Button>
          
          <Button
            variant="outline"
            onClick={clearRecording}
            icon={<ClearIcon />}
          >
            Clear
          </Button>
        </RecorderActions>
      </RecorderControls>
      
      <AudioVisualizer
        audioLevel={audioLevel}
        frequencyData={frequencyData}
        isRecording={isRecording}
      />
      
      <RecordingStatus status={status} duration={duration} />
    </Card>
  );
};
```

**PipelineProcessor Component:**
```typescript
// components/audio/PipelineProcessor.tsx
export const PipelineProcessor: React.FC = () => {
  const {
    currentStage,
    isProcessing,
    progress,
    results,
    runPipeline,
    runStage
  } = useAudioProcessing();
  
  const stages = useMemo(() => PIPELINE_STAGES, []);
  
  return (
    <Card title="Audio Processing Pipeline">
      <ProcessingControls>
        <Button
          variant="primary"
          onClick={() => runPipeline()}
          disabled={isProcessing}
          loading={isProcessing}
          icon={<PlayIcon />}
        >
          Run Full Pipeline
        </Button>
        
        <Button
          variant="secondary"
          onClick={() => runStage(stages[currentStage]?.id)}
          disabled={isProcessing}
          icon={<NextIcon />}
        >
          Run Next Stage
        </Button>
        
        <ProgressBar value={progress} />
      </ProcessingControls>
      
      <StageGrid>
        {stages.map((stage, index) => (
          <StageCard
            key={stage.id}
            stage={stage}
            index={index}
            isActive={currentStage === index}
            isCompleted={index < currentStage}
            result={results[stage.id]}
            onRun={() => runStage(stage.id)}
          />
        ))}
      </StageGrid>
    </Card>
  );
};
```

## Phase 5: Backend Migration (Week 6-7)

### 5.1 FastAPI Backend Setup

**Main Application:**
```python
# backend/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import audio, websocket, health
from app.core.websocket_manager import WebSocketManager

app = FastAPI(title="LiveTranslate Orchestration Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
websocket_manager = WebSocketManager()

# Include routers
app.include_router(audio.router, prefix="/api/audio", tags=["audio"])
app.include_router(health.router, prefix="/api/health", tags=["health"])

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)

# Serve React app
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

**Audio API Routes:**
```python
# backend/app/api/routes/audio.py
from fastapi import APIRouter, UploadFile, HTTPException
from app.services.audio_service import AudioService
from app.models.audio import AudioProcessingRequest, AudioProcessingResponse

router = APIRouter()
audio_service = AudioService()

@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(request: AudioProcessingRequest):
    try:
        result = await audio_service.process_audio(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_audio(file: UploadFile):
    try:
        result = await audio_service.save_audio_file(file)
        return {"file_id": result.id, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
async def get_audio_devices():
    try:
        devices = await audio_service.get_available_devices()
        return {"devices": devices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 5.2 WebSocket Management

**WebSocket Manager:**
```python
# backend/app/core/websocket_manager.py
import json
from typing import Dict, List
from fastapi import WebSocket
from app.services.audio_service import AudioService

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_service = AudioService()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        connection_id = str(id(websocket))
        self.active_connections[connection_id] = websocket
        
        # Send initial status
        await self.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id
        }, websocket)
    
    async def disconnect(self, websocket: WebSocket):
        connection_id = str(id(websocket))
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
    
    async def handle_message(self, websocket: WebSocket, data: str):
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if message_type == "audio_data":
                await self.handle_audio_data(websocket, message)
            elif message_type == "ping":
                await self.send_personal_message({"type": "pong"}, websocket)
            
        except json.JSONDecodeError:
            await self.send_personal_message({
                "type": "error",
                "message": "Invalid JSON"
            }, websocket)
    
    async def handle_audio_data(self, websocket: WebSocket, message: dict):
        # Process audio data
        result = await self.audio_service.process_realtime_audio(message["data"])
        
        await self.send_personal_message({
            "type": "audio_result",
            "result": result
        }, websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps(message))
```

## Phase 6: Testing Strategy (Week 7-8)

### 6.1 Unit Testing

**Component Testing:**
```typescript
// components/audio/AudioRecorder.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { AudioRecorder } from './AudioRecorder';
import { createMockStore } from '../../test-utils';

describe('AudioRecorder', () => {
  it('should start recording when button is clicked', async () => {
    const store = createMockStore({
      audio: {
        recording: { isRecording: false, status: 'idle' }
      }
    });
    
    render(
      <Provider store={store}>
        <AudioRecorder />
      </Provider>
    );
    
    const startButton = screen.getByText('Start Recording');
    fireEvent.click(startButton);
    
    expect(store.getActions()).toContainEqual({
      type: 'audio/startRecording'
    });
  });
});
```

**Hook Testing:**
```typescript
// hooks/useAudioRecording.test.ts
import { renderHook, act } from '@testing-library/react';
import { useAudioRecording } from './useAudioRecording';

describe('useAudioRecording', () => {
  it('should handle recording lifecycle', async () => {
    const { result } = renderHook(() => useAudioRecording());
    
    expect(result.current.isRecording).toBe(false);
    
    await act(async () => {
      await result.current.startRecording({ duration: 30 });
    });
    
    expect(result.current.isRecording).toBe(true);
  });
});
```

### 6.2 Integration Testing

**API Integration:**
```typescript
// services/api.test.ts
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { audioAPI } from './api';

const server = setupServer(
  rest.post('/api/audio/process', (req, res, ctx) => {
    return res(ctx.json({
      success: true,
      result: { processed: true }
    }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Audio API', () => {
  it('should process audio successfully', async () => {
    const result = await audioAPI.processAudio({
      stages: ['vad', 'noise_reduction']
    });
    
    expect(result.success).toBe(true);
  });
});
```

## Phase 7: Deployment & DevOps (Week 8-9)

### 7.1 Docker Configuration

**Multi-stage Dockerfile:**
```dockerfile
# Dockerfile
FROM node:18-alpine AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-build /app/dist ./static

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  orchestration:
    build: .
    ports:
      - "3000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - WHISPER_SERVICE_URL=http://whisper:5001
      - TRANSLATION_SERVICE_URL=http://translation:5003
    depends_on:
      - redis
    volumes:
      - ./uploads:/app/uploads
    networks:
      - livetranslate

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - livetranslate

networks:
  livetranslate:
    driver: bridge
```

### 7.2 Development Environment

**Package.json:**
```json
{
  "name": "livetranslate-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "storybook": "storybook dev -p 6006"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "@reduxjs/toolkit": "^1.9.1",
    "react-redux": "^8.0.5",
    "@mui/material": "^5.11.0",
    "@mui/icons-material": "^5.11.0",
    "styled-components": "^5.3.6",
    "framer-motion": "^8.5.0",
    "react-hook-form": "^7.43.0",
    "recharts": "^2.5.0",
    "socket.io-client": "^4.6.1"
  },
  "devDependencies": {
    "@types/react": "^18.0.26",
    "@types/react-dom": "^18.0.9",
    "@typescript-eslint/eslint-plugin": "^5.49.0",
    "@typescript-eslint/parser": "^5.49.0",
    "@vitejs/plugin-react": "^3.1.0",
    "typescript": "^4.9.3",
    "vite": "^4.1.0",
    "jest": "^29.3.1",
    "@testing-library/react": "^13.4.0",
    "@testing-library/jest-dom": "^5.16.5",
    "@storybook/react": "^6.5.15",
    "eslint": "^8.32.0",
    "prettier": "^2.8.3",
    "msw": "^1.0.0"
  }
}
```

**Vite Configuration:**
```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
```

## Phase 8: Performance Optimization (Week 9-10)

### 8.1 Code Splitting

**Route-based Splitting:**
```typescript
// App.tsx
import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { LoadingSpinner } from './components/ui/LoadingSpinner';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const AudioTesting = lazy(() => import('./pages/AudioTesting'));
const WebSocketTest = lazy(() => import('./pages/WebSocketTest'));
const Settings = lazy(() => import('./pages/Settings'));

export const App = () => (
  <BrowserRouter>
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/audio-test" element={<AudioTesting />} />
        <Route path="/websocket-test" element={<WebSocketTest />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  </BrowserRouter>
);
```

### 8.2 Audio Processing Optimization

**Web Workers for Audio Processing:**
```typescript
// workers/audioProcessor.worker.ts
self.addEventListener('message', async (event) => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'PROCESS_AUDIO':
      const result = await processAudioData(data);
      self.postMessage({ type: 'AUDIO_PROCESSED', result });
      break;
    
    case 'ANALYZE_FREQUENCY':
      const analysis = analyzeFrequency(data);
      self.postMessage({ type: 'FREQUENCY_ANALYZED', analysis });
      break;
  }
});
```

**Audio Processing Hook with Worker:**
```typescript
// hooks/useAudioProcessing.ts
export const useAudioProcessing = () => {
  const workerRef = useRef<Worker>();
  
  useEffect(() => {
    workerRef.current = new Worker('/workers/audioProcessor.worker.js');
    
    workerRef.current.onmessage = (event) => {
      const { type, result } = event.data;
      
      if (type === 'AUDIO_PROCESSED') {
        dispatch(audioProcessed(result));
      }
    };
    
    return () => workerRef.current?.terminate();
  }, []);
  
  const processAudio = useCallback((audioData: ArrayBuffer) => {
    workerRef.current?.postMessage({
      type: 'PROCESS_AUDIO',
      data: audioData
    });
  }, []);
  
  return { processAudio };
};
```

## Analysis of Example Implementation and Improvements

### Key Differences from EXAMPLE.md Implementation

The example implementation uses DOM scraping of Google Meet captions, while our system uses real audio capture with virtual webcam generation. Here are the key improvements our implementation provides:

**🚀 Superior Approach Advantages:**
1. **Higher Fidelity Audio**: Direct audio capture vs. caption scraping
2. **Virtual Webcam Output**: Real-time translation display generation
3. **Multi-modal Processing**: Audio + caption correlation for accuracy
4. **Enterprise Integration**: Full service pipeline vs. standalone bot
5. **Production Architecture**: Microservices vs. monolithic approach

**🔧 Improvements Identified from Example Analysis:**

1. **Enhanced Observability** (from Example Section 9):
   ```typescript
   // Add comprehensive bot health monitoring
   interface BotHealthMetrics {
     joined_log: boolean;
     caption_region_detected: boolean;
     segments_count: number;
     memory_usage_mb: number;
     dom_selector_status: 'healthy' | 'degraded' | 'failed';
     last_caption_timestamp: number;
   }
   ```

2. **Smoke Testing Integration** (inspired by Example recommendations):
   ```typescript
   // Add automated bot testing
   export const useBotSmokeTest = () => {
     const runSmokeTest = useCallback(async () => {
       // Test DOM selectors, caption detection, virtual webcam output
       const testResults = await fetch('/api/bot/smoke-test');
       return testResults.json();
     }, []);
     
     return { runSmokeTest };
   };
   ```

3. **DOM Selector Monitoring** (Example fragility concerns):
   ```typescript
   // Add DOM health monitoring for caption fallback
   interface DOMHealthState {
     caption_selectors_working: boolean;
     backup_selectors_available: string[];
     last_dom_change_detected: number;
     selector_success_rate: number;
   }
   ```

4. **Memory Management Tracking** (Example unbounded growth issues):
   ```typescript
   // Enhanced memory monitoring from Example learnings
   const useBotMemoryMonitoring = (botId: string) => {
     const [memoryMetrics, setMemoryMetrics] = useState({
       segments_count: 0,
       memory_usage_mb: 0,
       segment_buffer_size: 0,
       max_segment_limit: 10000 // Prevent runaway growth
     });
     
     useEffect(() => {
       const interval = setInterval(async () => {
         const metrics = await fetch(`/api/bot/${botId}/memory-stats`);
         setMemoryMetrics(await metrics.json());
       }, 5000);
       return () => clearInterval(interval);
     }, [botId]);
     
     return memoryMetrics;
   };
   ```

**🛡️ Production Hardening (from Example "Next Steps"):**
- Add circuit breaker for Google Meet DOM changes
- Implement speaker clustering with BERT embeddings
- Add real-time NLP processing capabilities  
- Implement graceful degradation when caption scraping fails
- Add comprehensive retry and backoff logic
- Implement session collision avoidance

## Migration Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1** | 2 weeks | Project setup, tech stack, architecture, 🆕 bot management UI |
| **Phase 2** | 1 week | Design system, component specs, 🆕 bot dashboard components |
| **Phase 3** | 1 week | State management, Redux setup, 🆕 bot state architecture |
| **Phase 4** | 2 weeks | Core component development, 🆕 bot lifecycle components |
| **Phase 5** | 1 week | Backend migration to FastAPI, 🆕 bot API endpoints |
| **Phase 6** | 1 week | Testing implementation, 🆕 bot integration testing |
| **Phase 7** | 1 week | Deployment setup, 🆕 bot containerization |
| **Phase 8** | 1 week | Performance optimization, 🆕 virtual webcam optimization |
| **Phase 9** | 1 week | Documentation, training, 🆕 bot management guides |

**Total Duration: 10 weeks**

## Success Metrics

### Technical Metrics
- **Performance**: Page load time < 2s, audio processing latency < 100ms
- **Bundle Size**: Initial bundle < 500KB, lazy-loaded chunks < 200KB
- **Test Coverage**: >90% component coverage, >80% integration coverage
- **Lighthouse Score**: >95 for performance, accessibility, best practices
- **🆕 Bot Performance**: Bot spawn time < 10s, audio capture latency < 200ms
- **🆕 Virtual Webcam**: Frame generation rate >25fps, translation display latency < 500ms

### User Experience Metrics
- **Accessibility**: WCAG 2.1 AA compliance
- **Mobile Responsiveness**: Full functionality on mobile devices
- **Error Handling**: Graceful error recovery with user-friendly messages
- **Loading States**: Smooth transitions and loading indicators
- **🆕 Bot Management UX**: Intuitive bot spawning workflow, real-time status updates
- **🆕 Translation Quality**: >90% accuracy correlation between audio and caption processing

### Developer Experience Metrics
- **Build Time**: Development build < 3s, production build < 30s
- **Hot Reload**: Component updates < 1s
- **Type Safety**: 100% TypeScript coverage
- **Code Quality**: ESLint/Prettier compliance
- **🆕 Bot Development**: Clear bot component architecture, comprehensive test suite

### Bot System Metrics
- **Reliability**: >99% bot spawn success rate, <1% session failure rate
- **Scalability**: Support 50+ concurrent bots, <5s average response time
- **Audio Quality**: >80% average quality score, <10% clipping detection rate
- **Memory Efficiency**: <500MB per bot instance, no memory leaks detected
- **Translation Accuracy**: >85% confidence score, <2s processing latency

## Risk Mitigation

### Technical Risks
- **Audio API Compatibility**: Implement fallbacks for different browsers
- **WebSocket Reliability**: Implement reconnection logic and offline support
- **Performance Issues**: Use profiling tools and implement optimization strategies
- **Browser Compatibility**: Test on major browsers and provide polyfills

### Project Risks
- **Timeline Delays**: Implement agile methodology with weekly sprints
- **Scope Creep**: Maintain clear requirements and change management
- **Team Coordination**: Regular standups and code reviews
- **Quality Assurance**: Automated testing and CI/CD pipeline

## Conclusion

This React migration plan provides a comprehensive roadmap for modernizing the LiveTranslate orchestration service frontend with integrated Google Meet bot management. The new architecture will deliver:

- **Superior UX/UI**: Professional design with Material-UI components
- **Better Performance**: Optimized bundle sizes and lazy loading
- **Maintainable Code**: TypeScript, proper component architecture
- **Scalable Foundation**: Modern React patterns and state management
- **Developer Experience**: Hot reloading, testing, and tooling
- **🆕 Enterprise Bot Management**: Complete Google Meet bot lifecycle control
- **🆕 Real-time Audio Processing**: Advanced pipeline control with diagnostics
- **🆕 Virtual Webcam Integration**: Live translation display management
- **🆕 Multi-modal Correlation**: Audio capture + caption timeline coordination

### Key Advantages Over Example Implementation

Our approach provides significant improvements over the DOM scraping method shown in EXAMPLE.md:

1. **Higher Fidelity**: Direct audio capture vs. fragile DOM scraping
2. **Real-time Output**: Virtual webcam generation for live translation display
3. **Production Architecture**: Enterprise microservices vs. standalone script
4. **Multi-modal Processing**: Correlation of audio and caption data for accuracy
5. **Comprehensive Monitoring**: Advanced observability and health tracking
6. **Memory Management**: Proper resource handling vs. unbounded growth
7. **Robust Error Handling**: Circuit breakers and graceful degradation

The migration will transform the current monolithic Flask templates into a modern, component-based React application that provides an exceptional user experience for audio testing, processing workflows, and comprehensive Google Meet bot management with real-time translation capabilities.