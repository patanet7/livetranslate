/**
 * Professional Streaming Processor
 * 
 * Advanced real-time audio streaming and processing interface providing:
 * - Professional-grade audio capture and streaming
 * - Real-time transcription with speaker diarization
 * - Multi-language translation capabilities
 * - Advanced audio processing pipeline
 * - Comprehensive monitoring and analytics
 * - Export capabilities and session management
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Slider,
  Tooltip,
  Divider,
  LinearProgress,
  CircularProgress,
  Tab,
  Tabs,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Mic,
  MicOff,
  PlayArrow,
  Stop,
  Pause,
  Settings,
  Translate,
  RecordVoiceOver,
  GraphicEq,
  Timeline,
  Assessment,
  Download,
  Refresh,
  Fullscreen,
  VolumeUp,
  Speed,
  Language,
  Computer,
  Headphones,
  RadioButtonChecked,
  Waves,
} from '@mui/icons-material';

// Import components
import { AudioVisualizer } from '../AudioTesting/components/AudioVisualizer';
import { useAppSelector, useAppDispatch } from '@/store';
import { useAvailableModels } from '@/hooks/useAvailableModels';
import {
  setAudioDevices,
  setVisualizationData,
  addProcessingLog,
  setAudioQualityMetrics,
  updateConfig,
} from '@/store/slices/audioSlice';

// Types
interface StreamingSession {
  id: string;
  startTime: Date;
  duration: number;
  chunksProcessed: number;
  totalTranscriptions: number;
  totalTranslations: number;
  averageLatency: number;
  errorCount: number;
}

interface ProcessingMetrics {
  chunkLatency: number[];
  transcriptionLatency: number[];
  translationLatency: number[];
  audioQuality: number[];
  confidenceScores: number[];
}

interface StreamingResult {
  id: string;
  type: 'transcription' | 'translation';
  timestamp: number;
  sourceLanguage: string;
  targetLanguage?: string;
  text: string;
  confidence: number;
  processingTime: number;
  speakers?: Array<{
    id: string;
    name: string;
    start: number;
    end: number;
  }>;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`streaming-tabpanel-${index}`}
      aria-labelledby={`streaming-tab-${index}`}
      {...other}
    >
      {value === index && children}
    </div>
  );
}

const StreamingProcessor: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const { devices, visualization, config } = useAppSelector(state => state.audio);
  
  // Load available models and device information
  const { 
    models: availableModels, 
    loading: modelsLoading, 
    error: modelsError, 
    status: modelsStatus,
    serviceMessage,
    deviceInfo,
    refetch: refetchModels 
  } = useAvailableModels();

  // UI State
  const [activeTab, setActiveTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Streaming State
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentSession, setCurrentSession] = useState<StreamingSession | null>(null);
  const [streamingResults, setStreamingResults] = useState<StreamingResult[]>([]);
  const [processingMetrics, setProcessingMetrics] = useState<ProcessingMetrics>({
    chunkLatency: [],
    transcriptionLatency: [],
    translationLatency: [],
    audioQuality: [],
    confidenceScores: [],
  });

  // Configuration State
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [chunkDuration, setChunkDuration] = useState(3);
  const [targetLanguages, setTargetLanguages] = useState<string[]>(['es', 'fr', 'de']);
  const [processingConfig, setProcessingConfig] = useState({
    enableTranscription: true,
    enableTranslation: true,
    enableDiarization: true,
    enableVAD: true,
    whisperModel: 'whisper-base',
    translationQuality: 'balanced',
    audioProcessing: true,
    noiseReduction: false,
    speechEnhancement: true,
  });

  // Audio References
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize audio devices
  useEffect(() => {
    const initializeDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices
          .filter(device => device.kind === 'audioinput')
          .map(device => ({
            deviceId: device.deviceId,
            label: device.label || `Microphone ${device.deviceId.substr(0, 8)}`,
            kind: device.kind as 'audioinput',
            groupId: device.groupId || ''
          }));
        
        dispatch(setAudioDevices(audioDevices));
        if (audioDevices.length > 0 && !selectedDevice) {
          setSelectedDevice(audioDevices[0].deviceId);
        }
      } catch (error) {
        console.error('Failed to load audio devices:', error);
      }
    };

    initializeDevices();
  }, [dispatch, selectedDevice]);

  // Session Management
  const startSession = useCallback(async () => {
    try {
      const newSession: StreamingSession = {
        id: `session-${Date.now()}`,
        startTime: new Date(),
        duration: 0,
        chunksProcessed: 0,
        totalTranscriptions: 0,
        totalTranslations: 0,
        averageLatency: 0,
        errorCount: 0,
      };
      
      setCurrentSession(newSession);
      setIsStreaming(true);
      setIsPaused(false);
      setStreamingResults([]);
      
      // Initialize audio streaming
      const constraints = {
        audio: {
          deviceId: selectedDevice,
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      audioStreamRef.current = stream;
      
      // Initialize audio context for visualization
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      microphoneRef.current.connect(analyserRef.current);

      startVisualization();
      startAudioProcessing();
      
    } catch (error) {
      console.error('Failed to start streaming session:', error);
      setIsStreaming(false);
    }
  }, [selectedDevice]);

  const stopSession = useCallback(() => {
    setIsStreaming(false);
    setIsPaused(false);
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    // Update session with final stats
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        duration: Date.now() - currentSession.startTime.getTime(),
      };
      setCurrentSession(updatedSession);
    }
  }, [currentSession]);

  const pauseSession = useCallback(() => {
    setIsPaused(!isPaused);
    if (mediaRecorderRef.current) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
      } else {
        mediaRecorderRef.current.pause();
      }
    }
  }, [isPaused]);

  const startVisualization = useCallback(() => {
    if (!analyserRef.current) return;

    const updateVisualization = () => {
      if (!analyserRef.current || !isStreaming) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(dataArray);

      // Update visualization data
      dispatch(setVisualizationData({
        frequencyData: Array.from(dataArray),
        timestamp: Date.now(),
      }));

      animationFrameRef.current = requestAnimationFrame(updateVisualization);
    };

    updateVisualization();
  }, [dispatch, isStreaming]);

  const startAudioProcessing = useCallback(() => {
    if (!audioStreamRef.current) return;

    const mediaRecorder = new MediaRecorder(audioStreamRef.current, {
      mimeType: 'audio/webm;codecs=opus'
    });

    const chunks: Blob[] = [];
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      if (chunks.length > 0) {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        await processAudioChunk(audioBlob);
        chunks.length = 0;
      }
    };

    // Record in chunks
    mediaRecorder.start();
    const intervalId = setInterval(() => {
      if (mediaRecorder.state === 'recording' && !isPaused) {
        mediaRecorder.stop();
        mediaRecorder.start();
      }
    }, chunkDuration * 1000);

    mediaRecorderRef.current = mediaRecorder;
    
    return () => {
      clearInterval(intervalId);
      if (mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    };
  }, [chunkDuration, isPaused]);

  const processAudioChunk = useCallback(async (audioBlob: Blob) => {
    if (!currentSession) return;

    const startTime = Date.now();
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.webm');
      formData.append('model', processingConfig.whisperModel);
      formData.append('enable_diarization', processingConfig.enableDiarization.toString());
      formData.append('enable_vad', processingConfig.enableVAD.toString());
      formData.append('target_languages', JSON.stringify(targetLanguages));

      const response = await fetch('/api/audio/stream-process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      const processingTime = Date.now() - startTime;

      // Add transcription result
      if (result.transcription && result.transcription.text) {
        const transcriptionResult: StreamingResult = {
          id: `trans-${Date.now()}`,
          type: 'transcription',
          timestamp: Date.now(),
          sourceLanguage: result.transcription.language || 'unknown',
          text: result.transcription.text,
          confidence: result.transcription.confidence || 0,
          processingTime,
          speakers: result.transcription.speakers || [],
        };
        
        setStreamingResults(prev => [...prev, transcriptionResult]);
      }

      // Add translation results
      if (result.translations && result.translations.length > 0) {
        const translationResults: StreamingResult[] = result.translations.map((trans: any) => ({
          id: `trans-${Date.now()}-${trans.target_language}`,
          type: 'translation',
          timestamp: Date.now(),
          sourceLanguage: result.transcription?.language || 'unknown',
          targetLanguage: trans.target_language,
          text: trans.text,
          confidence: trans.confidence || 0,
          processingTime,
        }));
        
        setStreamingResults(prev => [...prev, ...translationResults]);
      }

      // Update session stats
      setCurrentSession(prev => prev ? {
        ...prev,
        chunksProcessed: prev.chunksProcessed + 1,
        totalTranscriptions: prev.totalTranscriptions + (result.transcription ? 1 : 0),
        totalTranslations: prev.totalTranslations + (result.translations?.length || 0),
        averageLatency: (prev.averageLatency * prev.chunksProcessed + processingTime) / (prev.chunksProcessed + 1),
      } : null);

      // Update metrics
      setProcessingMetrics(prev => ({
        chunkLatency: [...prev.chunkLatency.slice(-49), processingTime],
        transcriptionLatency: result.transcription ? [...prev.transcriptionLatency.slice(-49), processingTime] : prev.transcriptionLatency,
        translationLatency: result.translations?.length > 0 ? [...prev.translationLatency.slice(-49), processingTime] : prev.translationLatency,
        audioQuality: [...prev.audioQuality.slice(-49), result.audio_quality || 0.8],
        confidenceScores: [...prev.confidenceScores.slice(-49), result.transcription?.confidence || 0],
      }));

    } catch (error) {
      console.error('Failed to process audio chunk:', error);
      setCurrentSession(prev => prev ? {
        ...prev,
        errorCount: prev.errorCount + 1,
      } : null);
    }
  }, [currentSession, processingConfig, targetLanguages]);

  const exportSession = useCallback(() => {
    if (!currentSession) return;

    const exportData = {
      session: currentSession,
      results: streamingResults,
      metrics: processingMetrics,
      config: processingConfig,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `streaming-session-${currentSession.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentSession, streamingResults, processingMetrics, processingConfig]);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Calculate current metrics
  const currentLatency = processingMetrics.chunkLatency.length > 0 
    ? processingMetrics.chunkLatency[processingMetrics.chunkLatency.length - 1] 
    : 0;
  
  const averageConfidence = processingMetrics.confidenceScores.length > 0
    ? processingMetrics.confidenceScores.reduce((a, b) => a + b, 0) / processingMetrics.confidenceScores.length
    : 0;

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: theme.palette.mode === 'dark' 
        ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.95)} 0%, ${alpha(theme.palette.primary.dark, 0.1)} 100%)`
        : `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.background.default, 0.95)} 100%)`,
      p: 3,
    }}>
      {/* Header */}
      <Card sx={{ 
        mb: 3,
        bgcolor: alpha(theme.palette.background.paper, 0.9),
        backdropFilter: 'blur(20px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Waves sx={{ fontSize: 40, color: 'primary.main' }} />
              <Box>
                <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
                  Professional Streaming Processor
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Real-time audio streaming with advanced transcription and translation
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              {!isStreaming ? (
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<PlayArrow />}
                  onClick={startSession}
                  disabled={!selectedDevice || modelsLoading}
                  sx={{ minWidth: 120 }}
                >
                  Start Stream
                </Button>
              ) : (
                <>
                  <Button
                    variant="outlined"
                    startIcon={isPaused ? <PlayArrow /> : <Pause />}
                    onClick={pauseSession}
                  >
                    {isPaused ? 'Resume' : 'Pause'}
                  </Button>
                  <Button
                    variant="contained"
                    color="error"
                    startIcon={<Stop />}
                    onClick={stopSession}
                  >
                    Stop
                  </Button>
                </>
              )}
              
              <Tooltip title="Export Session">
                <IconButton 
                  onClick={exportSession} 
                  disabled={!currentSession}
                  color="secondary"
                >
                  <Download />
                </IconButton>
              </Tooltip>
              
              <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}>
                <IconButton onClick={toggleFullscreen} color="info">
                  <Fullscreen />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Status and Quick Stats */}
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5,
                p: 2,
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
              }}>
                <RadioButtonChecked sx={{ color: isStreaming ? 'success.main' : 'text.secondary' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1 }}>
                    {isStreaming ? (isPaused ? 'Paused' : 'Live') : 'Stopped'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Stream Status
                  </Typography>
                </Box>
              </Box>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5,
                p: 2,
                bgcolor: alpha(theme.palette.info.main, 0.1),
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              }}>
                <Speed sx={{ color: 'info.main' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1 }}>
                    {currentLatency.toFixed(0)}ms
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Current Latency
                  </Typography>
                </Box>
              </Box>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5,
                p: 2,
                bgcolor: alpha(theme.palette.success.main, 0.1),
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
              }}>
                <Assessment sx={{ color: 'success.main' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1 }}>
                    {(averageConfidence * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Avg Confidence
                  </Typography>
                </Box>
              </Box>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5,
                p: 2,
                bgcolor: alpha(theme.palette.warning.main, 0.1),
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
              }}>
                <Language sx={{ color: 'warning.main' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1 }}>
                    {currentSession?.totalTranslations || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Translations
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Main Content with Tabs */}
      <Card sx={{ 
        bgcolor: alpha(theme.palette.background.paper, 0.98),
        backdropFilter: 'blur(20px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        minHeight: 600,
      }}>
        {/* Tab Navigation */}
        <Tabs 
          value={activeTab} 
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            '& .MuiTab-root': {
              minHeight: 64,
              textTransform: 'none',
              fontWeight: 500,
              fontSize: '0.95rem',
            },
          }}
        >
          <Tab icon={<GraphicEq />} label="Live Stream" iconPosition="start" />
          <Tab icon={<RecordVoiceOver />} label="Transcriptions" iconPosition="start" />
          <Tab icon={<Translate />} label="Translations" iconPosition="start" />
          <Tab icon={<Timeline />} label="Analytics" iconPosition="start" />
          <Tab icon={<Settings />} label="Configuration" iconPosition="start" />
        </Tabs>

        {/* Tab Content */}
        <Box sx={{ p: 3 }}>
          {/* Live Stream Tab */}
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card sx={{ height: 400 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Live Audio Visualization
                    </Typography>
                    <AudioVisualizer 
                      height={320}
                      showControls={false}
                      isLive={isStreaming}
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card sx={{ height: 400 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Session Info
                    </Typography>
                    {currentSession ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box>
                          <Typography variant="body2" color="text.secondary">Session ID:</Typography>
                          <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                            {currentSession.id}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="body2" color="text.secondary">Started:</Typography>
                          <Typography variant="body1">
                            {currentSession.startTime.toLocaleTimeString()}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="body2" color="text.secondary">Chunks Processed:</Typography>
                          <Typography variant="h4" color="primary">
                            {currentSession.chunksProcessed}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="body2" color="text.secondary">Error Rate:</Typography>
                          <Typography variant="body1" color={currentSession.errorCount > 0 ? 'error' : 'success'}>
                            {((currentSession.errorCount / Math.max(currentSession.chunksProcessed, 1)) * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>
                    ) : (
                      <Alert severity="info">
                        No active session. Click "Start Stream" to begin.
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Transcriptions Tab */}
          <TabPanel value={activeTab} index={1}>
            <Box sx={{ maxHeight: 500, overflow: 'auto' }}>
              {streamingResults.filter(r => r.type === 'transcription').length > 0 ? (
                streamingResults
                  .filter(r => r.type === 'transcription')
                  .sort((a, b) => b.timestamp - a.timestamp)
                  .map((result) => (
                    <Card key={result.id} sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Typography variant="body1" sx={{ flex: 1 }}>
                            {result.text}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                            <Chip 
                              label={result.sourceLanguage} 
                              size="small" 
                              color="primary" 
                            />
                            <Chip 
                              label={`${(result.confidence * 100).toFixed(0)}%`} 
                              size="small" 
                              color={result.confidence > 0.8 ? 'success' : 'warning'} 
                            />
                          </Box>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(result.timestamp).toLocaleTimeString()} • 
                          Processing: {result.processingTime}ms
                        </Typography>
                      </CardContent>
                    </Card>
                  ))
              ) : (
                <Alert severity="info">
                  No transcriptions yet. Start streaming to see results.
                </Alert>
              )}
            </Box>
          </TabPanel>

          {/* Translations Tab */}
          <TabPanel value={activeTab} index={2}>
            <Box sx={{ maxHeight: 500, overflow: 'auto' }}>
              {streamingResults.filter(r => r.type === 'translation').length > 0 ? (
                streamingResults
                  .filter(r => r.type === 'translation')
                  .sort((a, b) => b.timestamp - a.timestamp)
                  .map((result) => (
                    <Card key={result.id} sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Typography variant="body1" sx={{ flex: 1 }}>
                            {result.text}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                            <Chip 
                              label={`${result.sourceLanguage} → ${result.targetLanguage}`} 
                              size="small" 
                              color="secondary" 
                            />
                            <Chip 
                              label={`${(result.confidence * 100).toFixed(0)}%`} 
                              size="small" 
                              color={result.confidence > 0.8 ? 'success' : 'warning'} 
                            />
                          </Box>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(result.timestamp).toLocaleTimeString()} • 
                          Processing: {result.processingTime}ms
                        </Typography>
                      </CardContent>
                    </Card>
                  ))
              ) : (
                <Alert severity="info">
                  No translations yet. Enable translation and start streaming.
                </Alert>
              )}
            </Box>
          </TabPanel>

          {/* Analytics Tab */}
          <TabPanel value={activeTab} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Processing Latency Trend
                    </Typography>
                    <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {processingMetrics.chunkLatency.length > 0 ? (
                        <Alert severity="info">
                          Latency chart visualization would be displayed here showing the trend of processing times.
                          Current average: {(processingMetrics.chunkLatency.reduce((a, b) => a + b, 0) / processingMetrics.chunkLatency.length).toFixed(0)}ms
                        </Alert>
                      ) : (
                        <Alert severity="info">
                          No latency data available. Start streaming to see analytics.
                        </Alert>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Confidence Score Distribution
                    </Typography>
                    <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {processingMetrics.confidenceScores.length > 0 ? (
                        <Alert severity="info">
                          Confidence score histogram would be displayed here.
                          Current average: {(averageConfidence * 100).toFixed(1)}%
                        </Alert>
                      ) : (
                        <Alert severity="info">
                          No confidence data available. Start streaming to see analytics.
                        </Alert>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Configuration Tab */}
          <TabPanel value={activeTab} index={4}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Audio Configuration
                    </Typography>
                    
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Audio Input Device</InputLabel>
                      <Select
                        value={selectedDevice}
                        onChange={(e) => setSelectedDevice(e.target.value)}
                        disabled={isStreaming}
                      >
                        {devices.map((device) => (
                          <MenuItem key={device.deviceId} value={device.deviceId}>
                            {device.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Typography gutterBottom>
                      Chunk Duration: {chunkDuration}s
                    </Typography>
                    <Slider
                      value={chunkDuration}
                      onChange={(_, value) => setChunkDuration(value as number)}
                      min={1}
                      max={10}
                      step={1}
                      marks
                      disabled={isStreaming}
                      sx={{ mb: 2 }}
                    />

                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Whisper Model</InputLabel>
                      <Select
                        value={processingConfig.whisperModel}
                        onChange={(e) => setProcessingConfig(prev => ({
                          ...prev,
                          whisperModel: e.target.value
                        }))}
                        disabled={isStreaming}
                      >
                        {availableModels.map((model) => (
                          <MenuItem key={model} value={model}>
                            {model}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Processing Options
                    </Typography>
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={processingConfig.enableTranscription}
                          onChange={(e) => setProcessingConfig(prev => ({
                            ...prev,
                            enableTranscription: e.target.checked
                          }))}
                          disabled={isStreaming}
                        />
                      }
                      label="Enable Transcription"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={processingConfig.enableTranslation}
                          onChange={(e) => setProcessingConfig(prev => ({
                            ...prev,
                            enableTranslation: e.target.checked
                          }))}
                          disabled={isStreaming}
                        />
                      }
                      label="Enable Translation"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={processingConfig.enableDiarization}
                          onChange={(e) => setProcessingConfig(prev => ({
                            ...prev,
                            enableDiarization: e.target.checked
                          }))}
                          disabled={isStreaming}
                        />
                      }
                      label="Speaker Diarization"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={processingConfig.enableVAD}
                          onChange={(e) => setProcessingConfig(prev => ({
                            ...prev,
                            enableVAD: e.target.checked
                          }))}
                          disabled={isStreaming}
                        />
                      }
                      label="Voice Activity Detection"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={processingConfig.speechEnhancement}
                          onChange={(e) => setProcessingConfig(prev => ({
                            ...prev,
                            speechEnhancement: e.target.checked
                          }))}
                          disabled={isStreaming}
                        />
                      }
                      label="Speech Enhancement"
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Box>
      </Card>
    </Box>
  );
};

export default StreamingProcessor;