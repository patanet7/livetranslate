/**
 * AudioProcessingHub - Unified Audio Processing Interface
 * 
 * Professional audio processing hub with 6 specialized tabs:
 * 1. Live Analytics - Real-time system monitoring and metrics
 * 2. Pipeline Studio - Visual audio pipeline editor (existing)
 * 3. Quality Analysis - Audio quality analysis and visualization
 * 4. Streaming Processor - Real-time streaming audio processing
 * 5. Transcription Lab - Advanced transcription testing and analysis
 * 6. Translation Lab - Multi-language translation testing
 */

import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Tab,
  Tabs,
  Typography,
  Paper,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  AccountTree as PipelineIcon,
  Assessment as QualityIcon,
  Stream as StreamingIcon,
  RecordVoiceOver as TranscriptionIcon,
  Translate as TranslationIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

// Import existing components
import PipelineStudio from '../PipelineStudio';
import MeetingTest from '../MeetingTest';
import StreamingProcessor from '../StreamingProcessor';
import TranscriptionTesting from '../TranscriptionTesting';
import TranslationTesting from '../TranslationTesting';

// Import new hub components
import LiveAnalytics from './components/LiveAnalytics';
import QualityAnalysis from './components/QualityAnalysis';

// Import unified audio manager
import { useUnifiedAudio } from '@/hooks/useUnifiedAudio';
import { useAppDispatch, useAppSelector } from '@/store';
import { addNotification } from '@/store/slices/uiSlice';

// Tab configuration
interface AudioProcessingTab {
  id: string;
  label: string;
  icon: React.ReactElement;
  description: string;
  component: React.ComponentType;
  badge?: string;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
}

const AUDIO_PROCESSING_TABS: AudioProcessingTab[] = [
  {
    id: 'live-analytics',
    label: 'Live Analytics',
    icon: <AnalyticsIcon />,
    description: 'Real-time system monitoring, performance metrics, and service health',
    component: LiveAnalytics,
    badge: 'Live',
    color: 'success',
  },
  {
    id: 'pipeline-studio',
    label: 'Pipeline Studio',
    icon: <PipelineIcon />,
    description: 'Visual audio pipeline editor with drag-and-drop processing stages',
    component: PipelineStudio,
    badge: 'Visual',
    color: 'primary',
  },
  {
    id: 'quality-analysis',
    label: 'Quality Analysis',
    icon: <QualityIcon />,
    description: 'Audio quality analysis, FFT visualization, and LUFS metering',
    component: QualityAnalysis,
    badge: 'Pro',
    color: 'secondary',
  },
  {
    id: 'streaming-processor',
    label: 'Streaming Processor',
    icon: <StreamingIcon />,
    description: 'Real-time streaming audio processing and meeting integration',
    component: StreamingProcessor,
    badge: 'RT',
    color: 'warning',
  },
  {
    id: 'transcription-lab',
    label: 'Transcription Lab',
    icon: <TranscriptionIcon />,
    description: 'Advanced transcription testing with multiple models and languages',
    component: TranscriptionTesting,
    badge: 'AI',
    color: 'primary',
  },
  {
    id: 'translation-lab',
    label: 'Translation Lab',
    icon: <TranslationIcon />,
    description: 'Multi-language translation testing with quality analysis',
    component: TranslationTesting,
    badge: 'Multi',
    color: 'success',
  },
];

interface AudioProcessingHubProps {
  initialTab?: string;
}

const AudioProcessingHub: React.FC<AudioProcessingHubProps> = ({ 
  initialTab = 'live-analytics' 
}) => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const { tab } = useParams<{ tab?: string }>();
  const [currentTab, setCurrentTab] = useState(tab || initialTab);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Get unified audio manager
  const audioManager = useUnifiedAudio();

  // Get system state
  const { connection } = useAppSelector(state => state.websocket);
  const isConnected = connection.isConnected;
  const { notifications } = useAppSelector(state => state.ui);

  // Load system health on mount
  useEffect(() => {
    loadSystemHealth();
  }, []);

  const loadSystemHealth = async () => {
    setIsRefreshing(true);
    try {
      const health = await audioManager.getServiceStatus();
      setSystemHealth(health);
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'System Health Check Failed',
        message: 'Unable to retrieve system status',
        autoHide: true
      }));
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setCurrentTab(newValue);
  };

  const getCurrentTabConfig = () => {
    return AUDIO_PROCESSING_TABS.find(tab => tab.id === currentTab) || AUDIO_PROCESSING_TABS[0];
  };

  const CurrentComponent = getCurrentTabConfig().component;

  return (
    <Box sx={{ 
      minHeight: '100vh',
      width: '100%',
      background: theme.palette.mode === 'dark' 
        ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.8)} 0%, ${alpha(theme.palette.primary.dark, 0.1)} 100%)`
        : `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.background.default, 0.8)} 100%)`
    }}>
      {/* Header */}
      <Box sx={{ mb: 3, px: 1.5 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          mb: 2 
        }}>
          <Box>
            <Typography 
              variant="h3" 
              component="h1" 
              sx={{ 
                fontWeight: 600,
                background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1
              }}
            >
              Audio Processing Hub
            </Typography>
            <Typography 
              variant="h6" 
              color="textSecondary"
              sx={{ fontWeight: 400 }}
            >
              Professional audio processing suite with real-time analytics
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* System Status */}
            <Chip
              icon={<Box 
                sx={{ 
                  width: 8, 
                  height: 8, 
                  borderRadius: '50%',
                  backgroundColor: isConnected ? 'success.main' : 'error.main'
                }}
              />}
              label={isConnected ? 'Connected' : 'Disconnected'}
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />

            {/* Refresh Button */}
            <Tooltip title="Refresh System Status">
              <IconButton 
                onClick={loadSystemHealth}
                disabled={isRefreshing}
                sx={{
                  animation: isRefreshing ? 'spin 1s linear infinite' : 'none',
                  '@keyframes spin': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                  },
                }}
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>

            {/* Settings Button */}
            <Tooltip title="Hub Settings">
              <IconButton>
                <SettingsIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* System Alerts */}
        {!isConnected && (
          <Alert 
            severity="warning" 
            sx={{ mb: 2 }}
            action={
              <IconButton 
                size="small" 
                onClick={loadSystemHealth}
                color="inherit"
              >
                <RefreshIcon />
              </IconButton>
            }
          >
            WebSocket connection lost. Some real-time features may be unavailable.
          </Alert>
        )}

        {/* Tab Description */}
        <Paper 
          elevation={1}
          sx={{ 
            p: 2, 
            mb: 2,
            backgroundColor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {getCurrentTabConfig().icon}
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 500 }}>
                {getCurrentTabConfig().label}
                {getCurrentTabConfig().badge && (
                  <Chip
                    label={getCurrentTabConfig().badge}
                    size="small"
                    color={getCurrentTabConfig().color}
                    sx={{ ml: 1, fontSize: '0.7rem', height: 20 }}
                  />
                )}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {getCurrentTabConfig().description}
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Box>

      {/* Navigation Tabs */}
      <Paper 
        elevation={2}
        sx={{ 
          mb: 3,
          mx: 1.5,
          backgroundColor: alpha(theme.palette.background.paper, 0.9),
          backdropFilter: 'blur(20px)',
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        }}
      >
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              minHeight: 72,
              textTransform: 'none',
              fontWeight: 500,
              fontSize: '0.9rem',
              '&.Mui-selected': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
              },
            },
            '& .MuiTabs-indicator': {
              height: 3,
              borderRadius: '3px 3px 0 0',
            },
          }}
        >
          {AUDIO_PROCESSING_TABS.map((tab) => (
            <Tab
              key={tab.id}
              value={tab.id}
              icon={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {tab.icon}
                  {tab.badge && (
                    <Chip
                      label={tab.badge}
                      size="small"
                      color={tab.color}
                      sx={{ fontSize: '0.6rem', height: 16 }}
                    />
                  )}
                </Box>
              }
              label={tab.label}
              iconPosition="start"
              sx={{
                minWidth: 160,
                '& .MuiTab-iconWrapper': {
                  marginBottom: 0,
                  marginRight: 1,
                },
              }}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Paper 
        elevation={3}
        sx={{ 
          minHeight: 600,
          mx: 1.5,
          backgroundColor: alpha(theme.palette.background.paper, 0.95),
          backdropFilter: 'blur(20px)',
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <Box sx={{ p: 0, height: '100%' }}>
          <CurrentComponent />
        </Box>
      </Paper>

      {/* Footer */}
      <Box sx={{ 
        mt: 4, 
        pt: 2, 
        px: 1.5,
        borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="body2" color="textSecondary">
          LiveTranslate Audio Processing Hub v2.0
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Typography variant="body2" color="textSecondary">
            Services: {systemHealth ? Object.keys(systemHealth).length : 0} active
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Notifications: {notifications.length}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default AudioProcessingHub;