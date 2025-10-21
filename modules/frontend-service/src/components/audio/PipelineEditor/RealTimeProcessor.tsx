/**
 * Real-Time Audio Processor Component
 * 
 * Provides real-time audio processing controls for the Pipeline Studio
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  LinearProgress,
  Chip,
  Grid,
  Divider,
  Alert,
  Tooltip,
  Switch,
  FormControlLabel,
  Slider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Mic,
  MicOff,
  PlayArrow,
  Stop,
  Settings,
  Analytics,
  VolumeUp,
  VolumeDown,
  GraphicEq,
  Timeline,
  Speed,
  Memory,
  NetworkCheck,
  Warning,
  CheckCircle,
  Upload,
  Download,
  Refresh,
} from '@mui/icons-material';
import { usePipelineProcessing } from '@/hooks/usePipelineProcessing';
import { PipelineData } from './PipelineCanvas';

interface RealTimeProcessorProps {
  currentPipeline: PipelineData | null;
  onMetricsUpdate?: (metrics: any) => void;
  onProcessedAudio?: (audio: string) => void;
  onWebSocketChange?: (websocket: WebSocket | null, isActive: boolean) => void;
}

interface AudioVisualizerProps {
  audioData?: number[];
  isActive: boolean;
  label: string;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ audioData, isActive, label }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current || !audioData) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (!isActive) {
      ctx.fillStyle = '#424242';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No Signal', width / 2, height / 2);
      return;
    }
    
    // Draw waveform
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const sliceWidth = width / audioData.length;
    let x = 0;
    
    for (let i = 0; i < audioData.length; i++) {
      const v = audioData[i] * height / 2;
      const y = height / 2 + v;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      
      x += sliceWidth;
    }
    
    ctx.stroke();
    
    // Draw label
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(5, 5, 60, 20);
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(label, 8, 18);
  }, [audioData, isActive, label]);
  
  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={80}
      style={{
        width: '100%',
        height: '80px',
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
      }}
    />
  );
};

const RealTimeProcessor: React.FC<RealTimeProcessorProps> = ({
  currentPipeline,
  onMetricsUpdate,
  onProcessedAudio,
  onWebSocketChange,
}) => {
  const {
    isProcessing,
    metrics,
    audioAnalysis,
    error,
    realtimeSession,
    isRealtimeActive,
    websocket,
    startRealtimeProcessing,
    startMicrophoneCapture,
    stopRealtimeProcessing,
    updateRealtimeConfig,
    processPipeline,
    analyzeFFT,
    analyzeLUFS,
    clearError,
  } = usePipelineProcessing();

  // Local state
  const [showSettings, setShowSettings] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [outputLevel, setOutputLevel] = useState(0);
  const [realtimeSettings, setRealtimeSettings] = useState({
    chunkSize: 1024,
    bufferSize: 4096,
    latencyTarget: 100,
    qualityMode: 'balanced' as 'low_latency' | 'balanced' | 'high_quality',
  });
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

  // Audio level monitoring
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number>();

  // Monitor audio levels during real-time processing
  useEffect(() => {
    if (isRealtimeActive) {
      const updateLevels = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // Calculate RMS level
          const rms = Math.sqrt(dataArray.reduce((sum, value) => sum + value * value, 0) / dataArray.length);
          setAudioLevel(rms / 128); // Normalize to 0-1
        }
        
        animationRef.current = requestAnimationFrame(updateLevels);
      };
      
      updateLevels();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      setAudioLevel(0);
      setOutputLevel(0);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRealtimeActive]);

  // Update parent with metrics
  useEffect(() => {
    if (metrics && onMetricsUpdate) {
      onMetricsUpdate(metrics);
    }
  }, [metrics, onMetricsUpdate]);

  // Notify parent of WebSocket changes
  useEffect(() => {
    if (onWebSocketChange) {
      onWebSocketChange(websocket, isRealtimeActive);
    }
  }, [websocket, isRealtimeActive, onWebSocketChange]);

  const handleStartRealtime = async () => {
    if (!currentPipeline) {
      alert('Please create a pipeline first');
      return;
    }

    // Find microphone input node to get capture settings
    const micNode = currentPipeline.nodes.find(
      node => node.data.stageType === 'input' &&
              (node.data.label.toLowerCase().includes('microphone') ||
               node.id.includes('microphone_input'))
    );

    // Extract mic settings from the node parameters
    const micSettings = {
      sampleRate: 16000, // default
      channels: 1, // default
    };

    if (micNode?.data?.parameters) {
      const sampleRateParam = micNode.data.parameters.find((p: any) => p.name === 'sampleRate');
      const channelsParam = micNode.data.parameters.find((p: any) => p.name === 'channels');

      if (sampleRateParam) micSettings.sampleRate = sampleRateParam.value;
      if (channelsParam) micSettings.channels = channelsParam.value;
    }

    // Convert pipeline to processing format (backend expects dict for stages, not list)
    // Use componentId (like "microphone_input", "vad_stage") as key, not the React Flow node ID

    // Build mapping from React Flow node ID to component ID
    const nodeIdToComponentId = new Map();
    const stagesDict = currentPipeline.nodes.reduce((dict: any, node: any) => {
      const stageKey = node.data.componentId || node.id; // Fallback to node.id if componentId not set
      nodeIdToComponentId.set(node.id, stageKey); // Store mapping for connections

      dict[stageKey] = {
        enabled: node.data.enabled !== undefined ? node.data.enabled : true,
        gain_in: node.data.gainIn || 0,
        gain_out: node.data.gainOut || 0,
        parameters: node.data.stageConfig || {},
      };
      return dict;
    }, {});

    const pipelineConfig = {
      pipeline_id: currentPipeline.id,
      name: currentPipeline.name,
      stages: stagesDict,
      connections: currentPipeline.edges.map((edge: any) => ({
        id: edge.id,
        source_stage_id: nodeIdToComponentId.get(edge.source) || edge.source,
        target_stage_id: nodeIdToComponentId.get(edge.target) || edge.target,
      })),
    };

    await startRealtimeProcessing(pipelineConfig);
    await startMicrophoneCapture(micSettings);
  };

  const handleStopRealtime = () => {
    stopRealtimeProcessing();
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !currentPipeline) return;

    const pipelineConfig = {
      id: currentPipeline.id,
      name: currentPipeline.name,
      stages: currentPipeline.nodes.map(node => ({
        id: node.id,
        type: node.data.label.toLowerCase().replace(/\s+/g, '_'),
        enabled: node.data.enabled,
        gainIn: node.data.gainIn || 0,
        gainOut: node.data.gainOut || 0,
        parameters: node.data.stageConfig || {},
        position: node.position,
      })),
      connections: currentPipeline.edges.map(edge => ({
        id: edge.id,
        sourceStageId: edge.source,
        targetStageId: edge.target,
      })),
    };

    await processPipeline({
      pipelineConfig,
      audioData: file,
      processingMode: 'batch',
      outputFormat: 'wav',
    });

    // Analyze the uploaded file
    await analyzeFFT(file);
    await analyzeLUFS(file);
  };

  const formatLatency = (ms: number): string => {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    return `${ms.toFixed(1)}ms`;
  };

  const formatLevel = (db: number): string => {
    return `${db >= 0 ? '+' : ''}${db.toFixed(1)}dB`;
  };

  const getLatencyColor = (latency: number): string => {
    if (latency <= 50) return '#4caf50';
    if (latency <= 100) return '#ff9800';
    return '#f44336';
  };

  const getQualityColor = (score: number): string => {
    if (score >= 80) return '#4caf50';
    if (score >= 60) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box>
      {/* Error Display */}
      {error && (
        <Alert 
          severity="error" 
          onClose={clearError}
          sx={{ mb: 2 }}
        >
          {error}
        </Alert>
      )}

      {/* Real-time Controls */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">
              Real-time Audio Processing
            </Typography>
            <Box display="flex" gap={1}>
              <Tooltip title="Upload Audio File">
                <IconButton onClick={() => setUploadDialogOpen(true)}>
                  <Upload />
                </IconButton>
              </Tooltip>
              <Tooltip title="Processing Settings">
                <IconButton onClick={() => setShowSettings(true)}>
                  <Settings />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6}>
              <Box display="flex" gap={2}>
                {!isRealtimeActive ? (
                  <Button
                    variant="contained"
                    startIcon={<Mic />}
                    onClick={handleStartRealtime}
                    disabled={!currentPipeline || isProcessing}
                    color="success"
                  >
                    Start Live Processing
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    startIcon={<MicOff />}
                    onClick={handleStopRealtime}
                    color="error"
                  >
                    Stop Processing
                  </Button>
                )}
              </Box>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Box display="flex" alignItems="center" gap={2}>
                <Typography variant="body2">Status:</Typography>
                <Chip
                  label={
                    isRealtimeActive ? 'LIVE' :
                    isProcessing ? 'PROCESSING' : 'IDLE'
                  }
                  color={
                    isRealtimeActive ? 'success' :
                    isProcessing ? 'warning' : 'default'
                  }
                  size="small"
                />
                {realtimeSession?.sessionId && (
                  <Chip
                    label={`Session: ${realtimeSession.sessionId.slice(0, 8)}`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            </Grid>
          </Grid>

          {/* Audio Level Meters */}
          {isRealtimeActive && (
            <Box mt={2}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" display="block" gutterBottom>
                    Input Level
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <VolumeDown sx={{ fontSize: 16 }} />
                    <LinearProgress
                      variant="determinate"
                      value={audioLevel * 100}
                      sx={{
                        flexGrow: 1,
                        height: 8,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: audioLevel > 0.8 ? '#f44336' : '#4caf50',
                        },
                      }}
                    />
                    <VolumeUp sx={{ fontSize: 16 }} />
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" display="block" gutterBottom>
                    Output Level
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <VolumeDown sx={{ fontSize: 16 }} />
                    <LinearProgress
                      variant="determinate"
                      value={outputLevel * 100}
                      sx={{
                        flexGrow: 1,
                        height: 8,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: outputLevel > 0.8 ? '#f44336' : '#2196f3',
                        },
                      }}
                    />
                    <VolumeUp sx={{ fontSize: 16 }} />
                  </Box>
                </Grid>
              </Grid>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      {metrics && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color={getLatencyColor(metrics.totalLatency)}>
                    {formatLatency(metrics.totalLatency)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Total Latency
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color={getQualityColor(metrics.qualityScore || 0)}>
                    {(metrics.qualityScore || 0).toFixed(0)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Quality Score
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4">
                    {metrics.cpuUsage.toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    CPU Usage
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4">
                    {metrics.chunksProcessed || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Chunks Processed
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Quality Metrics */}
            {metrics.qualityMetrics && (
              <Box mt={2}>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Audio Quality Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2">
                      SNR: <strong>{metrics.qualityMetrics.snr?.toFixed(1) || 'N/A'}dB</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2">
                      THD: <strong>{((metrics.qualityMetrics.thd || 0) * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2">
                      LUFS: <strong>{formatLevel(metrics.qualityMetrics.lufs || 0)}</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2">
                      RMS: <strong>{formatLevel(metrics.qualityMetrics.rms || 0)}</strong>
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Audio Analysis */}
      {(audioAnalysis.fft || audioAnalysis.lufs) && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Audio Analysis
            </Typography>
            
            {audioAnalysis.fft && (
              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Spectral Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2">
                      Fundamental Frequency: <strong>{audioAnalysis.fft.voiceCharacteristics.fundamentalFreq?.toFixed(1) || 'N/A'}Hz</strong>
                    </Typography>
                    <Typography variant="body2">
                      Voice Confidence: <strong>{((audioAnalysis.fft.voiceCharacteristics.voiceConfidence || 0) * 100).toFixed(1)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2">
                      Spectral Centroid: <strong>{audioAnalysis.fft.spectralFeatures.centroid?.toFixed(0) || 'N/A'}Hz</strong>
                    </Typography>
                    <Typography variant="body2">
                      Spectral Rolloff: <strong>{audioAnalysis.fft.spectralFeatures.rolloff?.toFixed(0) || 'N/A'}Hz</strong>
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {audioAnalysis.lufs && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Loudness Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2">
                      Integrated Loudness: <strong>{formatLevel(audioAnalysis.lufs.integratedLoudness)}</strong>
                    </Typography>
                    <Typography variant="body2">
                      Loudness Range: <strong>{audioAnalysis.lufs.loudnessRange.toFixed(1)} LU</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2">
                      True Peak: <strong>{formatLevel(audioAnalysis.lufs.truePeak)}</strong>
                    </Typography>
                    <Typography variant="body2">
                      Compliant: <strong style={{ color: audioAnalysis.lufs.complianceCheck.compliant ? '#4caf50' : '#f44336' }}>
                        {audioAnalysis.lufs.complianceCheck.compliant ? 'Yes' : 'No'}
                      </strong>
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* File Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
        <DialogTitle>Upload Audio File</DialogTitle>
        <DialogContent>
          <Box py={2}>
            <Typography variant="body2" paragraph>
              Upload an audio file to process through your current pipeline configuration.
            </Typography>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              style={{ width: '100%' }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>
            Cancel
          </Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Real-time Processing Settings</DialogTitle>
        <DialogContent>
          <Box py={2}>
            <Typography variant="subtitle2" gutterBottom>
              Buffer Configuration
            </Typography>
            
            <Box mb={2}>
              <Typography gutterBottom>Chunk Size: {realtimeSettings.chunkSize}</Typography>
              <Slider
                value={realtimeSettings.chunkSize}
                min={256}
                max={2048}
                step={256}
                onChange={(_, value) => setRealtimeSettings(prev => ({ 
                  ...prev, 
                  chunkSize: value as number 
                }))}
              />
            </Box>
            
            <Box mb={2}>
              <Typography gutterBottom>Latency Target: {realtimeSettings.latencyTarget}ms</Typography>
              <Slider
                value={realtimeSettings.latencyTarget}
                min={50}
                max={500}
                step={10}
                onChange={(_, value) => setRealtimeSettings(prev => ({ 
                  ...prev, 
                  latencyTarget: value as number 
                }))}
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettings(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RealTimeProcessor;