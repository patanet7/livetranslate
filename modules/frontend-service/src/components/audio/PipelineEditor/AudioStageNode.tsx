import React, { useState, useCallback } from 'react';
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
} from '@mui/icons-material';

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
  onSettingsOpen: (nodeId: string) => void;
  onGainChange: (nodeId: string, type: 'in' | 'out', value: number) => void;
  onParameterChange: (nodeId: string, paramName: string, value: number) => void;
  onToggleEnabled: (nodeId: string, enabled: boolean) => void;
}

const AudioStageNode: React.FC<AudioStageNodeProps> = ({ 
  id, 
  data, 
  selected, 
  onSettingsOpen,
  onGainChange,
  onParameterChange,
  onToggleEnabled,
}) => {
  const [showParameters, setShowParameters] = useState(false);
  const [hovering, setHovering] = useState(false);

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

  return (
    <Card
      sx={{
        minWidth: 280,
        maxWidth: 320,
        border: `2px solid ${selected ? colors.primary : hovering ? colors.secondary : 'transparent'}`,
        borderRadius: 2,
        backgroundColor: colors.background,
        cursor: 'pointer',
        transition: 'all 0.2s ease-in-out',
        transform: selected ? 'scale(1.02)' : 'scale(1)',
        boxShadow: selected ? 4 : hovering ? 2 : 1,
        opacity: data.enabled ? 1 : 0.6,
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
              <data.icon sx={{ fontSize: 16 }} />
            </Box>
            <Typography variant="subtitle2" fontWeight="bold">
              {data.label}
            </Typography>
          </Box>
          
          <Box display="flex" alignItems="center" gap={0.5}>
            {getStatusIcon()}
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
          
          <Box display="flex" gap={1}>
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
              <Box mt={1}>
                <Divider sx={{ mb: 1 }} />
                {data.parameters.slice(0, 3).map((param) => (
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
                      onChange={(_, value) => onParameterChange(id, param.name, value as number)}
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
                ))}
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