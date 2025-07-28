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
const AudioProcessingHub = React.lazy(() => import('@/pages/AudioProcessingHub'));
const BotManagement = React.lazy(() => import('@/pages/BotManagement'));
const Analytics = React.lazy(() => import('@/pages/Analytics'));
const SystemAnalytics = React.lazy(() => import('@/pages/SystemAnalytics'));
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
                  
                  {/* Audio Processing Hub - Unified Audio Processing Interface */}
                  <Route path="/audio-hub" element={<AudioProcessingHub />} />
                  <Route path="/audio-processing" element={<AudioProcessingHub />} />
                  <Route path="/audio-hub/:tab" element={<AudioProcessingHub />} />
                  
                  {/* Legacy Audio Routes - Redirect to AudioProcessingHub */}
                  <Route path="/audio-test" element={<AudioProcessingHub />} />
                  <Route path="/audio-testing" element={<AudioProcessingHub />} />
                  <Route path="/pipeline-studio" element={<AudioProcessingHub />} />
                  <Route path="/pipeline" element={<AudioProcessingHub />} />
                  <Route path="/transcription-testing" element={<AudioProcessingHub />} />
                  <Route path="/transcription-test" element={<AudioProcessingHub />} />
                  <Route path="/transcription" element={<AudioProcessingHub />} />
                  <Route path="/translation-testing" element={<AudioProcessingHub />} />
                  <Route path="/translation-test" element={<AudioProcessingHub />} />
                  <Route path="/translation" element={<AudioProcessingHub />} />
                  <Route path="/meeting-test" element={<AudioProcessingHub />} />
                  <Route path="/meeting" element={<AudioProcessingHub />} />
                  <Route path="/streaming-processor" element={<AudioProcessingHub />} />
                  <Route path="/streaming" element={<AudioProcessingHub />} />
                  <Route path="/stream" element={<AudioProcessingHub />} />
                  
                  {/* Bot Management */}
                  <Route path="/bot-management" element={<BotManagement />} />
                  <Route path="/bots" element={<BotManagement />} />
                  
                  {/* Analytics Dashboard */}
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/metrics" element={<Analytics />} />
                  
                  {/* System Analytics Dashboard */}
                  <Route path="/system-analytics" element={<SystemAnalytics />} />
                  <Route path="/system-metrics" element={<SystemAnalytics />} />
                  <Route path="/monitoring" element={<SystemAnalytics />} />
                  
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