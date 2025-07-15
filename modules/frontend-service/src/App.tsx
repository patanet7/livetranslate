import React, { useEffect, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { SnackbarProvider } from 'notistack';

// Store
import { store } from '@/store';
import { useAppDispatch, useAppSelector } from '@/store';

// Theme
import { lightTheme, darkTheme } from '@/styles/theme';

// Hooks
import { useBreakpoint } from '@/hooks/useBreakpoint';
import { useWebSocket } from '@/hooks/useWebSocket';

// Layout components
import { AppLayout } from '@/components/layout/AppLayout';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';

// Actions
import { initializeUI, setBreakpoint } from '@/store/slices/uiSlice';

// Lazy-loaded pages for code splitting
const Dashboard = React.lazy(() => import('@/pages/Dashboard'));
const AudioTesting = React.lazy(() => import('@/pages/AudioTesting'));
const TranscriptionTesting = React.lazy(() => import('@/pages/TranscriptionTesting'));
const TranslationTesting = React.lazy(() => import('@/pages/TranslationTesting'));
const MeetingTest = React.lazy(() => import('@/pages/MeetingTest'));
const BotManagement = React.lazy(() => import('@/pages/BotManagement'));
const Analytics = React.lazy(() => import('@/pages/Analytics'));
const WebSocketTest = React.lazy(() => import('@/pages/WebSocketTest'));
const Settings = React.lazy(() => import('@/pages/Settings'));

// App initialization component
const AppInitializer: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const dispatch = useAppDispatch();
  const { theme } = useAppSelector(state => state.ui);
  const breakpoint = useBreakpoint();

  // Initialize WebSocket connection
  useWebSocket();

  // Initialize UI from localStorage and handle responsive breakpoints
  useEffect(() => {
    dispatch(initializeUI());
  }, [dispatch]);

  useEffect(() => {
    dispatch(setBreakpoint(breakpoint));
  }, [dispatch, breakpoint]);

  // Select theme based on UI state
  const selectedTheme = theme === 'light' ? lightTheme : darkTheme;

  return (
    <ThemeProvider theme={selectedTheme}>
      <CssBaseline />
      <SnackbarProvider 
        maxSnack={3}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        dense
      >
        {children}
      </SnackbarProvider>
    </ThemeProvider>
  );
};

// Main App component
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <Provider store={store}>
        <AppInitializer>
          <Router>
            <AppLayout>
              <Suspense fallback={<LoadingScreen />}>
                <Routes>
                  {/* Main Dashboard */}
                  <Route path="/" element={<Dashboard />} />
                  
                  {/* Audio Testing */}
                  <Route path="/audio-test" element={<AudioTesting />} />
                  <Route path="/audio-testing" element={<AudioTesting />} />
                  
                  {/* Transcription Testing */}
                  <Route path="/transcription-testing" element={<TranscriptionTesting />} />
                  <Route path="/transcription-test" element={<TranscriptionTesting />} />
                  <Route path="/transcription" element={<TranscriptionTesting />} />
                  
                  {/* Translation Testing */}
                  <Route path="/translation-testing" element={<TranslationTesting />} />
                  <Route path="/translation-test" element={<TranslationTesting />} />
                  <Route path="/translation" element={<TranslationTesting />} />
                  
                  {/* Meeting Test Dashboard */}
                  <Route path="/meeting-test" element={<MeetingTest />} />
                  <Route path="/meeting" element={<MeetingTest />} />
                  
                  {/* Bot Management */}
                  <Route path="/bot-management" element={<BotManagement />} />
                  <Route path="/bots" element={<BotManagement />} />
                  
                  {/* Analytics Dashboard */}
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/metrics" element={<Analytics />} />
                  
                  {/* WebSocket Testing */}
                  <Route path="/websocket-test" element={<WebSocketTest />} />
                  <Route path="/websocket" element={<WebSocketTest />} />
                  
                  {/* Settings */}
                  <Route path="/settings" element={<Settings />} />
                  
                  {/* Redirect any unknown routes to dashboard */}
                  <Route path="*" element={<Dashboard />} />
                </Routes>
              </Suspense>
            </AppLayout>
          </Router>
        </AppInitializer>
      </Provider>
    </ErrorBoundary>
  );
};

export default App;