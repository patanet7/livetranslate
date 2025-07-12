import React from 'react';
import {
  Box,
  Chip,
  Button,
  Tooltip,
  Typography,
  Alert,
  AlertTitle,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  Api as ApiIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAppSelector } from '@/store';

export const ConnectionStatus: React.FC = () => {
  const {
    isConnected,
    reconnectAttempts,
    retryConnection,
    useApiMode,
    wsFailureCount,
    canRetryWebSocket,
  } = useWebSocket();
  
  const { connection, errors } = useAppSelector(state => state.websocket);
  const [showDetails, setShowDetails] = React.useState(false);

  const getConnectionStatus = () => {
    if (useApiMode) {
      return {
        label: 'API Mode',
        color: 'warning' as const,
        icon: <ApiIcon />,
        description: 'Using REST API fallback for stability'
      };
    } else if (isConnected) {
      return {
        label: 'Connected',
        color: 'success' as const,
        icon: <WifiIcon />,
        description: 'Real-time WebSocket connection active'
      };
    } else if (reconnectAttempts > 0) {
      return {
        label: `Reconnecting (${reconnectAttempts}/3)`,
        color: 'warning' as const,
        icon: <RefreshIcon />,
        description: 'Attempting to restore WebSocket connection'
      };
    } else {
      return {
        label: 'Disconnected',
        color: 'error' as const,
        icon: <WifiOffIcon />,
        description: 'WebSocket connection not available'
      };
    }
  };

  const status = getConnectionStatus();
  const recentErrors = errors.slice(-3);

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Tooltip title={status.description}>
        <Chip
          icon={status.icon}
          label={status.label}
          color={status.color}
          variant={isConnected ? 'filled' : 'outlined'}
          size="small"
        />
      </Tooltip>

      {/* Connection details toggle */}
      <Tooltip title="Connection Details">
        <IconButton
          size="small"
          onClick={() => setShowDetails(!showDetails)}
          sx={{ ml: 1 }}
        >
          {showDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Tooltip>

      {/* Retry button */}
      {!isConnected && !useApiMode && canRetryWebSocket && (
        <Tooltip title="Retry WebSocket connection">
          <Button
            size="small"
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={retryConnection}
            sx={{ ml: 1 }}
          >
            Retry
          </Button>
        </Tooltip>
      )}

      {/* Connection details panel */}
      <Collapse in={showDetails} sx={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 1000 }}>
        <Box sx={{ mt: 1, p: 2, bgcolor: 'background.paper', border: 1, borderColor: 'divider', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Connection Details
          </Typography>
          
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Status
              </Typography>
              <Typography variant="body2">
                {isConnected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary">
                Mode
              </Typography>
              <Typography variant="body2">
                {useApiMode ? 'REST API' : 'WebSocket'}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary">
                Reconnect Attempts
              </Typography>
              <Typography variant="body2">
                {reconnectAttempts}/3
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary">
                Failure Count
              </Typography>
              <Typography variant="body2">
                {wsFailureCount}
              </Typography>
            </Box>
            
            {connection.connectionId && (
              <Box sx={{ gridColumn: '1 / -1' }}>
                <Typography variant="caption" color="text.secondary">
                  Connection ID
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {connection.connectionId}
                </Typography>
              </Box>
            )}
          </Box>

          {/* API Mode explanation */}
          {useApiMode && (
            <Alert severity="info" sx={{ mb: 2 }}>
              <AlertTitle>API Mode Active</AlertTitle>
              WebSocket failed {wsFailureCount} times. Using REST API for better stability.
              {canRetryWebSocket && (
                <>
                  {' '}
                  <Button size="small" onClick={retryConnection} sx={{ ml: 1 }}>
                    Try WebSocket Again
                  </Button>
                </>
              )}
            </Alert>
          )}

          {/* Recent errors */}
          {recentErrors.length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <WarningIcon fontSize="small" />
                Recent Errors
              </Typography>
              {recentErrors.map((error, index) => (
                <Typography 
                  key={index} 
                  variant="caption" 
                  display="block" 
                  sx={{ 
                    fontFamily: 'monospace', 
                    fontSize: '0.7rem',
                    color: 'error.main',
                    mt: 0.5 
                  }}
                >
                  {new Date(error.timestamp).toLocaleTimeString()}: {error.error}
                </Typography>
              ))}
            </Box>
          )}

          {/* Manual controls */}
          <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
            {!isConnected && canRetryWebSocket && (
              <Button size="small" variant="outlined" onClick={retryConnection}>
                Retry Connection
              </Button>
            )}
            
            {useApiMode && (
              <Button size="small" variant="outlined" color="warning">
                Continue with API
              </Button>
            )}
          </Box>
        </Box>
      </Collapse>
    </Box>
  );
};

export default ConnectionStatus;