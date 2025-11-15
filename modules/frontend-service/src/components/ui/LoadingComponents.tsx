/**
 * Comprehensive Loading Components
 * 
 * Provides various loading states with skeleton components,
 * progress indicators, and user-friendly loading messages
 */

import React from 'react';
import {
  Box,
  CircularProgress,
  LinearProgress,
  Typography,
  Paper,
  Skeleton,
  Stack,
  Fade,
  Backdrop,
  Alert,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Psychology as ProcessingIcon,
  Translate as TranslateIcon,
  AudioFile as AudioIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

// ============================================================================
// Loading Screen Component
// ============================================================================

interface LoadingScreenProps {
  message?: string;
  progress?: number;
  type?: 'circular' | 'linear' | 'skeleton';
  size?: 'small' | 'medium' | 'large';
  fullScreen?: boolean;
  showProgress?: boolean;
  icon?: React.ReactNode;
  timeout?: number;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading...',
  progress,
  type = 'circular',
  size = 'medium',
  fullScreen = false,
  showProgress = false,
  icon,
  timeout,
}) => {
  const [showTimeout, setShowTimeout] = React.useState(false);

  React.useEffect(() => {
    if (timeout) {
      const timer = setTimeout(() => {
        setShowTimeout(true);
      }, timeout);
      return () => clearTimeout(timer);
    }
  }, [timeout]);

  const getSize = () => {
    switch (size) {
      case 'small': return 24;
      case 'large': return 64;
      default: return 40;
    }
  };

  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        minHeight: fullScreen ? '100vh' : '200px',
        gap: 2,
      }}
    >
      {/* Icon */}
      {icon && (
        <Box sx={{ mb: 1, color: 'primary.main' }}>
          {icon}
        </Box>
      )}

      {/* Loading Indicator */}
      {type === 'circular' && (
        <CircularProgress
          size={getSize()}
          variant={progress !== undefined ? 'determinate' : 'indeterminate'}
          value={progress}
        />
      )}

      {type === 'linear' && (
        <Box sx={{ width: '100%', maxWidth: 300 }}>
          <LinearProgress
            variant={progress !== undefined ? 'determinate' : 'indeterminate'}
            value={progress}
          />
        </Box>
      )}

      {type === 'skeleton' && (
        <Box sx={{ width: '100%', maxWidth: 300 }}>
          <Skeleton variant="text" width="80%" />
          <Skeleton variant="text" width="60%" />
          <Skeleton variant="rectangular" width="100%" height={60} />
        </Box>
      )}

      {/* Message */}
      <Typography
        variant={size === 'large' ? 'h6' : 'body1'}
        color="text.secondary"
        textAlign="center"
      >
        {message}
      </Typography>

      {/* Progress Text */}
      {showProgress && progress !== undefined && (
        <Typography variant="body2" color="text.secondary">
          {Math.round(progress)}%
        </Typography>
      )}

      {/* Timeout Warning */}
      {showTimeout && (
        <Fade in={showTimeout}>
          <Alert severity="warning" sx={{ mt: 2, maxWidth: 400 }}>
            <Typography variant="body2">
              This is taking longer than expected. Please check your connection.
            </Typography>
          </Alert>
        </Fade>
      )}
    </Box>
  );

  if (fullScreen) {
    return (
      <Backdrop open sx={{ color: '#fff', zIndex: 9999 }}>
        <Paper elevation={3} sx={{ borderRadius: 2 }}>
          {content}
        </Paper>
      </Backdrop>
    );
  }

  return content;
};

// ============================================================================
// Skeleton Components
// ============================================================================

export const AudioProcessingSkeleton: React.FC = () => (
  <Paper elevation={1} sx={{ p: 3 }}>
    <Stack spacing={2}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Skeleton variant="circular" width={40} height={40} />
        <Box sx={{ flex: 1 }}>
          <Skeleton variant="text" width="40%" />
          <Skeleton variant="text" width="80%" />
        </Box>
      </Box>
      <Skeleton variant="rectangular" width="100%" height={60} />
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Skeleton variant="rectangular" width={80} height={36} />
        <Skeleton variant="rectangular" width={80} height={36} />
        <Skeleton variant="rectangular" width={80} height={36} />
      </Box>
    </Stack>
  </Paper>
);

export const AnalyticsSkeleton: React.FC = () => (
  <Stack spacing={2}>
    {/* Metrics Cards */}
    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
      {[1, 2, 3, 4].map((i) => (
        <Paper key={i} elevation={1} sx={{ p: 2 }}>
          <Stack spacing={1}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Skeleton variant="circular" width={24} height={24} />
              <Skeleton variant="text" width="60%" />
            </Box>
            <Skeleton variant="text" width="40%" height={32} />
            <Skeleton variant="text" width="80%" height={20} />
          </Stack>
        </Paper>
      ))}
    </Box>
    
    {/* Chart Area */}
    <Paper elevation={1} sx={{ p: 3 }}>
      <Skeleton variant="text" width="30%" height={24} sx={{ mb: 2 }} />
      <Skeleton variant="rectangular" width="100%" height={300} />
    </Paper>
  </Stack>
);

export const BotManagementSkeleton: React.FC = () => (
  <Stack spacing={3}>
    {/* Header */}
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Skeleton variant="text" width="200px" height={32} />
      <Skeleton variant="rectangular" width={120} height={36} />
    </Box>
    
    {/* Bot Cards */}
    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 2 }}>
      {[1, 2, 3].map((i) => (
        <Paper key={i} elevation={1} sx={{ p: 3 }}>
          <Stack spacing={2}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Skeleton variant="circular" width={48} height={48} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="70%" />
                <Skeleton variant="text" width="50%" />
              </Box>
            </Box>
            <Skeleton variant="rectangular" width="100%" height={80} />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Skeleton variant="rectangular" width={80} height={32} />
              <Skeleton variant="rectangular" width={80} height={32} />
            </Box>
          </Stack>
        </Paper>
      ))}
    </Box>
  </Stack>
);

export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({ 
  rows = 5, 
  columns = 4 
}) => (
  <Paper elevation={1}>
    <Box sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'grid', gridTemplateColumns: `repeat(${columns}, 1fr)`, gap: 2, mb: 2 }}>
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton key={i} variant="text" width="80%" />
        ))}
      </Box>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <Box 
          key={rowIndex} 
          sx={{ 
            display: 'grid', 
            gridTemplateColumns: `repeat(${columns}, 1fr)`, 
            gap: 2, 
            py: 1,
            borderBottom: rowIndex < rows - 1 ? '1px solid' : 'none',
            borderColor: 'divider'
          }}
        >
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton key={colIndex} variant="text" width="90%" />
          ))}
        </Box>
      ))}
    </Box>
  </Paper>
);

// ============================================================================
// Specialized Loading Components
// ============================================================================

export const AudioUploadLoading: React.FC<{ progress?: number }> = ({ progress }) => (
  <LoadingScreen
    message="Uploading audio file..."
    progress={progress}
    type="linear"
    icon={<UploadIcon sx={{ fontSize: 48 }} />}
    showProgress={true}
    timeout={10000}
  />
);

export const AudioProcessingLoading: React.FC<{ stage?: string }> = ({ stage }) => (
  <LoadingScreen
    message={stage ? `Processing: ${stage}` : 'Processing audio...'}
    type="circular"
    icon={<ProcessingIcon sx={{ fontSize: 48 }} />}
    timeout={15000}
  />
);

export const TranscriptionLoading: React.FC = () => (
  <LoadingScreen
    message="Transcribing audio..."
    type="circular"
    icon={<AudioIcon sx={{ fontSize: 48 }} />}
    timeout={20000}
  />
);

export const TranslationLoading: React.FC<{ languages?: string[] }> = ({ languages }) => (
  <LoadingScreen
    message={languages ? `Translating to ${languages.join(', ')}...` : 'Translating text...'}
    type="circular"
    icon={<TranslateIcon sx={{ fontSize: 48 }} />}
    timeout={10000}
  />
);

export const AnalyticsLoading: React.FC = () => (
  <LoadingScreen
    message="Loading analytics data..."
    type="circular"
    icon={<AnalyticsIcon sx={{ fontSize: 48 }} />}
    timeout={5000}
  />
);

// ============================================================================
// Loading Overlay Component
// ============================================================================

interface LoadingOverlayProps {
  loading: boolean;
  children: React.ReactNode;
  message?: string;
  blur?: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  children,
  message = 'Loading...',
  blur = false,
}) => (
  <Box sx={{ position: 'relative' }}>
    {children}
    {loading && (
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: blur ? 'blur(2px)' : 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}
      >
        <Box
          sx={{
            backgroundColor: 'background.paper',
            borderRadius: 2,
            p: 3,
            boxShadow: 3,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <CircularProgress />
          <Typography variant="body2" color="text.secondary">
            {message}
          </Typography>
        </Box>
      </Box>
    )}
  </Box>
);

export default LoadingScreen;