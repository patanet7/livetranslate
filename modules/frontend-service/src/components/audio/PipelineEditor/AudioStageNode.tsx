import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Slider,
  Chip,
  LinearProgress,
  Tooltip,
  Badge,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  Handle,
  Position,
  NodeProps,
} from 'reactflow';
import {
  Settings,
  VolumeUp,
  VolumeDown,
  PlayArrow,
  Stop,
  TrendingUp,
  TrendingDown,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Sync,
  SyncDisabled,
} from '@mui/icons-material';
import { usePipelineCallbacks } from './PipelineCallbacksContext';

interface AudioStageNodeData {
  label: string;
  description: string;
  stageType: 'input' | 'processing' | 'output';
  icon: React.ComponentType;
  enabled: boolean;
  
  // Gain Controls
  gainIn: number;      // -20 to +20 dB
  gainOut: number;     // -20 to +20 dB
  
  // Stage Configuration
  stageConfig: Record<string, any>;
  
  // Real-time Metrics
  metrics?: {
    processingTimeMs: number;
    targetLatencyMs: number;
    qualityImprovement: number;
    inputLevel: number;
    outputLevel: number;
    cpuUsage: number;
  };
  
  // Processing Status
  isProcessing: boolean;
  status: 'idle' | 'processing' | 'completed' | 'error';
  
  // Stage-specific parameters
  parameters: StageParameter[];
}

interface StageParameter {
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  description: string;
}

interface AudioStageNodeProps extends NodeProps {
  data: AudioStageNodeData;
}

const AudioStageNode: React.FC<AudioStageNodeProps> = ({
  id,
  data,
  selected,
}) => {
  // Get callbacks from context instead of props
  const {
    onNodeSettingsOpen: onSettingsOpen,
    onGainChange,
    onParameterChange,
    onToggleEnabled,
    websocket,
    isRealtimeActive,
  } = usePipelineCallbacks();
  const [showParameters, setShowParameters] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<number>(0);

  // Debounce timers for parameter changes
  const debounceTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Send parameter update to backend via WebSocket
  const sendParameterUpdate = useCallback((paramName: string, value: number) => {
    if (!websocket || !isRealtimeActive || websocket.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      setIsSyncing(true);
      websocket.send(JSON.stringify({
        type: 'update_stage',
        stage_id: id,
        parameters: {
          [paramName]: value
        }
      }));

      setLastSyncTime(Date.now());

      // Reset syncing indicator after a short delay
      setTimeout(() => setIsSyncing(false), 500);
    } catch (error) {
      console.error('Failed to send parameter update:', error);
      setIsSyncing(false);
    }
  }, [websocket, isRealtimeActive, id]);

  // Debounced parameter change handler
  const handleParameterChangeDebounced = useCallback((paramName: string, value: number) => {
    // Clear existing timer for this parameter
    const existingTimer = debounceTimers.current.get(paramName);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Update local state immediately
    onParameterChange(id, paramName, value);

    // Set new timer to send to backend after 300ms
    const timer = setTimeout(() => {
      sendParameterUpdate(paramName, value);
      debounceTimers.current.delete(paramName);
    }, 300);

    debounceTimers.current.set(paramName, timer);
  }, [id, onParameterChange, sendParameterUpdate]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      debounceTimers.current.forEach(timer => clearTimeout(timer));
      debounceTimers.current.clear();
    };
  }, []);

  const getStageColor = () => {
    switch (data.stageType) {
      case 'input':
        return {
          primary: '#2196f3',
          secondary: '#1976d2',
          background: 'rgba(33, 150, 243, 0.1)',
          border: '#2196f3'
        };
      case 'processing':
        return {
          primary: '#4caf50',
          secondary: '#388e3c',
          background: 'rgba(76, 175, 80, 0.1)',
          border: '#4caf50'
        };
      case 'output':
        return {
          primary: '#ff9800',
          secondary: '#f57c00',
          background: 'rgba(255, 152, 0, 0.1)',
          border: '#ff9800'
        };
      default:
        return {
          primary: '#9e9e9e',
          secondary: '#616161',
          background: 'rgba(158, 158, 158, 0.1)',
          border: '#9e9e9e'
        };
    }
  };

  const getStatusIcon = () => {
    switch (data.status) {
      case 'processing':
        return <PlayArrow sx={{ fontSize: 16, color: '#4caf50' }} />;
      case 'completed':
        return <CheckCircle sx={{ fontSize: 16, color: '#4caf50' }} />;
      case 'error':
        return <Error sx={{ fontSize: 16, color: '#f44336' }} />;
      default:
        return <Stop sx={{ fontSize: 16, color: '#9e9e9e' }} />;
    }
  };

  const getLatencyStatus = (): 'good' | 'warning' | 'poor' => {
    if (!data.metrics) return 'good';
    const { processingTimeMs, targetLatencyMs } = data.metrics;
    
    if (processingTimeMs <= targetLatencyMs) return 'good';
    if (processingTimeMs <= targetLatencyMs * 1.5) return 'warning';
    return 'poor';
  };

  const formatGain = (gain: number): string => {
    return `${gain >= 0 ? '+' : ''}${gain.toFixed(1)}dB`;
  };

  const formatLatency = (ms: number): string => {
    return `${ms.toFixed(1)}ms`;
  };

  const colors = getStageColor();
  const latencyStatus = getLatencyStatus();

  // Get border color based on state
  const getBorderColor = () => {
    if (selected) return colors.primary;
    if (isRealtimeActive && data.enabled) return '#4caf50'; // Green when processing
    if (hovering) return colors.secondary;
    return 'transparent';
  };

  const getBorderAnimation = () => {
    if (isRealtimeActive && data.enabled && data.isProcessing) {
      return {
        animation: 'pulse 2s ease-in-out infinite',
        '@keyframes pulse': {
          '0%, 100%': { borderColor: '#4caf50' },
          '50%': { borderColor: '#81c784' },
        },
      };
    }
    return {};
  };

  return (
    <Card
      sx={{
        minWidth: 280,
        maxWidth: 320,
        border: `2px solid ${getBorderColor()}`,
        borderRadius: 2,
        backgroundColor: colors.background,
        cursor: 'pointer',
        transition: 'all 0.2s ease-in-out',
        transform: selected ? 'scale(1.02)' : 'scale(1)',
        boxShadow: selected ? 4 : hovering ? 2 : 1,
        opacity: data.enabled ? 1 : 0.6,
        ...getBorderAnimation(),
      }}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      {/* Input Handle */}
      {data.stageType !== 'input' && (
        <Handle
          type="target"
          position={Position.Left}
          style={{
            width: 12,
            height: 12,
            backgroundColor: colors.primary,
            border: `2px solid ${colors.secondary}`,
            left: -8,
          }}
        />
      )}

      <CardContent sx={{ p: 2 }}>
        {/* Header */}
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Box display="flex" alignItems="center" gap={1}>
            <Box
              sx={{
                p: 0.5,
                borderRadius: 1,
                backgroundColor: colors.primary,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              {data.icon && typeof data.icon === 'function' ? (
                React.createElement(data.icon, { sx: { fontSize: 16 } })
              ) : (
                <Settings sx={{ fontSize: 16 }} />
              )}
            </Box>
            <Typography variant="subtitle2" fontWeight="bold">
              {data.label}
            </Typography>
          </Box>
          
          <Box display="flex" alignItems="center" gap={0.5}>
            {getStatusIcon()}
            {/* Sync Status Indicator */}
            {isRealtimeActive && (
              <Tooltip title={isSyncing ? 'Syncing parameters...' : 'Connected to backend'}>
                {isSyncing ? (
                  <CircularProgress size={14} thickness={5} />
                ) : (
                  <Sync sx={{ fontSize: 14, color: '#4caf50' }} />
                )}
              </Tooltip>
            )}
            {!isRealtimeActive && websocket && (
              <Tooltip title="Not connected to backend">
                <SyncDisabled sx={{ fontSize: 14, color: '#9e9e9e' }} />
              </Tooltip>
            )}
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onSettingsOpen(id);
              }}
              sx={{ p: 0.5 }}
            >
              <Settings sx={{ fontSize: 16 }} />
            </IconButton>
          </Box>
        </Box>

        {/* Description */}
        <Typography variant="caption" color="text.secondary" display="block" mb={1.5}>
          {data.description}
        </Typography>

        {/* I/O Gain Controls */}
        <Box mb={1.5}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="caption" fontWeight="bold">
              I/O Gain Controls
            </Typography>
            <Chip
              size="small"
              label={data.enabled ? 'ENABLED' : 'DISABLED'}
              color={data.enabled ? 'success' : 'default'}
              onClick={(e) => {
                e.stopPropagation();
                onToggleEnabled(id, !data.enabled);
              }}
              sx={{ cursor: 'pointer' }}
            />
          </Box>
          
          <Box
            className="nodrag nowheel"
            display="flex"
            gap={1}
            onMouseDown={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
            onTouchStart={(e) => e.stopPropagation()}
            sx={{ cursor: 'default' }}
          >
            {/* Input Gain */}
            <Box flex={1}>
              <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                <VolumeDown sx={{ fontSize: 12, color: colors.primary }} />
                <Typography variant="caption">IN</Typography>
                <Typography variant="caption" fontWeight="bold">
                  {formatGain(data.gainIn)}
                </Typography>
              </Box>
              <Slider
                value={data.gainIn}
                min={-20}
                max={20}
                step={0.5}
                size="small"
                onChange={(_, value) => onGainChange(id, 'in', value as number)}
                sx={{
                  color: colors.primary,
                  '& .MuiSlider-thumb': {
                    width: 12,
                    height: 12,
                  },
                  '& .MuiSlider-track': {
                    height: 3,
                  },
                }}
              />
            </Box>

            {/* Output Gain */}
            <Box flex={1}>
              <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                <VolumeUp sx={{ fontSize: 12, color: colors.primary }} />
                <Typography variant="caption">OUT</Typography>
                <Typography variant="caption" fontWeight="bold">
                  {formatGain(data.gainOut)}
                </Typography>
              </Box>
              <Slider
                value={data.gainOut}
                min={-20}
                max={20}
                step={0.5}
                size="small"
                onChange={(_, value) => onGainChange(id, 'out', value as number)}
                sx={{
                  color: colors.primary,
                  '& .MuiSlider-thumb': {
                    width: 12,
                    height: 12,
                  },
                  '& .MuiSlider-track': {
                    height: 3,
                  },
                }}
              />
            </Box>
          </Box>
        </Box>

        {/* Real-time Metrics */}
        {data.metrics && (
          <Box mb={1.5}>
            <Typography variant="caption" fontWeight="bold" display="block" mb={1}>
              Performance Metrics
            </Typography>
            
            <Box display="flex" justifyContent="space-between" mb={0.5}>
              <Typography variant="caption">Latency:</Typography>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Typography
                  variant="caption"
                  fontWeight="bold"
                  color={
                    latencyStatus === 'good' ? 'success.main' :
                    latencyStatus === 'warning' ? 'warning.main' : 'error.main'
                  }
                >
                  {formatLatency(data.metrics.processingTimeMs)}
                </Typography>
                {latencyStatus === 'poor' && <Warning sx={{ fontSize: 12, color: 'error.main' }} />}
              </Box>
            </Box>

            <Box display="flex" justifyContent="space-between" mb={0.5}>
              <Typography variant="caption">Quality:</Typography>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Typography
                  variant="caption"
                  fontWeight="bold"
                  color={data.metrics.qualityImprovement >= 0 ? 'success.main' : 'error.main'}
                >
                  {data.metrics.qualityImprovement >= 0 ? '+' : ''}
                  {data.metrics.qualityImprovement.toFixed(1)}dB
                </Typography>
                {data.metrics.qualityImprovement >= 0 ? 
                  <TrendingUp sx={{ fontSize: 12, color: 'success.main' }} /> :
                  <TrendingDown sx={{ fontSize: 12, color: 'error.main' }} />
                }
              </Box>
            </Box>

            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="caption">CPU:</Typography>
              <Typography variant="caption" fontWeight="bold">
                {data.metrics.cpuUsage.toFixed(0)}%
              </Typography>
            </Box>

            {/* Processing Time Progress Bar */}
            <LinearProgress
              variant="determinate"
              value={Math.min((data.metrics.processingTimeMs / data.metrics.targetLatencyMs) * 100, 100)}
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: 'rgba(0, 0, 0, 0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: 
                    latencyStatus === 'good' ? '#4caf50' :
                    latencyStatus === 'warning' ? '#ff9800' : '#f44336',
                },
              }}
            />
          </Box>
        )}

        {/* Stage Parameters (Collapsible) */}
        {data.parameters.length > 0 && (
          <Box>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              sx={{ cursor: 'pointer' }}
              onClick={() => setShowParameters(!showParameters)}
            >
              <Typography variant="caption" fontWeight="bold">
                Parameters ({data.parameters.length})
              </Typography>
              <Speed 
                sx={{ 
                  fontSize: 12, 
                  color: colors.primary,
                  transform: showParameters ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s ease-in-out',
                }} 
              />
            </Box>
            
            {showParameters && (
              <Box
                className="nodrag nowheel"
                mt={1}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => e.stopPropagation()}
                onDoubleClick={(e) => e.stopPropagation()}
                onTouchStart={(e) => e.stopPropagation()}
                sx={{ cursor: 'default' }}
              >
                <Divider sx={{ mb: 1 }} />
                {data.parameters.slice(0, 3).map((param) => {
                  // Only render numeric parameters as sliders
                  if (typeof param.value !== 'number') return null;

                  return (
                    <Box key={param.name} mb={1}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                        <Tooltip title={param.description} arrow>
                          <Typography variant="caption">{param.name}:</Typography>
                        </Tooltip>
                        <Typography variant="caption" fontWeight="bold">
                          {param.value.toFixed(param.step < 1 ? 1 : 0)}{param.unit}
                        </Typography>
                      </Box>
                      <Slider
                        value={param.value}
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        size="small"
                        onChange={(_, value) => handleParameterChangeDebounced(param.name, value as number)}
                        sx={{
                          color: colors.primary,
                          '& .MuiSlider-thumb': {
                            width: 10,
                            height: 10,
                          },
                          '& .MuiSlider-track': {
                            height: 2,
                          },
                        }}
                      />
                    </Box>
                  );
                })}
                {data.parameters.length > 3 && (
                  <Typography variant="caption" color="text.secondary">
                    +{data.parameters.length - 3} more in settings
                  </Typography>
                )}
              </Box>
            )}
          </Box>
        )}

        {/* Processing Indicator */}
        {data.isProcessing && (
          <Box mt={1}>
            <LinearProgress
              sx={{
                height: 2,
                borderRadius: 1,
                backgroundColor: 'rgba(0, 0, 0, 0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: colors.primary,
                },
              }}
            />
          </Box>
        )}
      </CardContent>

      {/* Output Handle */}
      {data.stageType !== 'output' && (
        <Handle
          type="source"
          position={Position.Right}
          style={{
            width: 12,
            height: 12,
            backgroundColor: colors.primary,
            border: `2px solid ${colors.secondary}`,
            right: -8,
          }}
        />
      )}
    </Card>
  );
};

export default AudioStageNode;