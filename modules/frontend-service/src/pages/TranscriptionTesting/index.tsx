import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  Chip,
  FormGroup,
  FormControlLabel,
  Switch,
  Stack,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Clear as ClearIcon,
  Mic as MicIcon,
} from '@mui/icons-material';
import { useAppSelector } from '@/store';
import { useNotifications } from '@/hooks/useNotifications';

interface TranscriptionResult {
  id: string;
  audioData?: Blob;
  transcriptText: string;
  language: string;
  confidence: number;
  processingTime: number;
  timestamp: number;
  modelUsed: string;
  chunkId?: string;
  sessionId?: string;
  speakerInfo?: {
    speakerId: string;
    confidence: number;
  }[];
  segments?: {
    start: number;
    end: number;
    text: string;
    confidence: number;
  }[];
}

interface AudioSettings {
  sampleRate: number;
  channels: number;
  bitDepth: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

interface TranscriptionSettings {
  whisperModel: string;
  enableDiarization: boolean;
  enableVAD: boolean;
  language: string;
  enableTimestamps: boolean;
  chunkDuration: number;
  enableRealTime: boolean;
}

const TranscriptionTesting: React.FC = () => {
  const { notifySuccess, notifyError } = useNotifications();
  const { isConnected: webSocketConnected } = useAppSelector(state => state.websocket.connection);

  // Audio Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPreviewingAudio, setIsPreviewingAudio] = useState(false);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  
  // Media References
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const chunkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Settings State
  const [audioSettings, setAudioSettings] = useState<AudioSettings>({
    sampleRate: 16000,
    channels: 1,
    bitDepth: 16,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  });
  
  const [transcriptionSettings, setTranscriptionSettings] = useState<TranscriptionSettings>({
    whisperModel: 'whisper-tiny', // Default to whisper-tiny
    enableDiarization: true,
    enableVAD: true,
    language: 'auto',
    enableTimestamps: true,
    chunkDuration: 5,
    enableRealTime: true,
  });
  
  // Results State
  const [transcriptionResults, setTranscriptionResults] = useState<TranscriptionResult[]>([]);
  const [currentTranscription, setCurrentTranscription] = useState<TranscriptionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamingResults, setStreamingResults] = useState<string>('');
  
  // Session Management
  const [sessionId] = useState(`transcription-${Date.now()}`);
  const [chunkCount, setChunkCount] = useState(0);
  
  // Available Models (this would come from API in real implementation)
  const availableModels = [
    'whisper-tiny',
    'whisper-tiny.en',
    'whisper-base',
    'whisper-base.en',
    'whisper-small',
    'whisper-small.en',
    'whisper-medium',
    'whisper-medium.en',
    'whisper-large-v3',
  ];
  
  const supportedLanguages = [
    { code: 'auto', name: 'Auto Detect' },
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ru', name: 'Russian' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'zh', name: 'Chinese' },
  ];

  // Initialize audio devices
  useEffect(() => {
    const getAudioDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        setAudioDevices(audioInputs);
        if (audioInputs.length > 0 && !selectedDevice) {
          setSelectedDevice(audioInputs[0].deviceId);
        }
        
        // Start audio preview for the default device
        if (audioInputs.length > 0) {
          startAudioPreview(audioInputs[0].deviceId);
        }
      } catch (error) {
        console.error('Failed to enumerate audio devices:', error);
        notifyError('Audio Device Error', 'Failed to access audio devices');
      }
    };

    getAudioDevices();
  }, [notifyError]);

  // Recording timer
  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      setRecordingTime(0);
    }

    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, [isRecording]);

  // Audio level monitoring
  const monitorAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    const average = dataArray.reduce((acc, val) => acc + val, 0) / dataArray.length;
    setAudioLevel(average / 255 * 100);
    
    if (isRecording || isStreaming || isPreviewingAudio) {
      requestAnimationFrame(monitorAudioLevel);
    }
  }, [isRecording, isStreaming, isPreviewingAudio]);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const constraints = {
        audio: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          sampleRate: audioSettings.sampleRate,
          channelCount: audioSettings.channels,
          echoCancellation: audioSettings.echoCancellation,
          noiseSuppression: audioSettings.noiseSuppression,
          autoGainControl: audioSettings.autoGainControl,
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Setup audio analysis
      audioContextRef.current = new AudioContext({ sampleRate: audioSettings.sampleRate });
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      monitorAudioLevel();

      // Setup media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;

      const recordedChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);

      notifySuccess('Recording Started', 'Audio recording has begun');

    } catch (error) {
      console.error('Failed to start recording:', error);
      notifyError('Recording Error', 'Failed to start audio recording');
    }
  }, [selectedDevice, audioSettings, monitorAudioLevel, notifySuccess, notifyError]);

  // Start audio preview when device changes
  useEffect(() => {
    if (selectedDevice && !isRecording && !isStreaming) {
      startAudioPreview(selectedDevice);
    }
  }, [selectedDevice, isRecording, isStreaming]);

  // Audio preview function
  const startAudioPreview = useCallback(async (deviceId: string) => {
    try {
      // Stop existing preview
      if (streamRef.current && isPreviewingAudio) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      const constraints = {
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          sampleRate: audioSettings.sampleRate,
          channelCount: audioSettings.channels,
          echoCancellation: audioSettings.echoCancellation,
          noiseSuppression: audioSettings.noiseSuppression,
          autoGainControl: audioSettings.autoGainControl,
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Setup audio analysis
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      audioContextRef.current = new AudioContext({ sampleRate: audioSettings.sampleRate });
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      setIsPreviewingAudio(true);
      monitorAudioLevel();
      
    } catch (error) {
      console.error('Failed to start audio preview:', error);
      setAudioLevel(0);
    }
  }, [audioSettings, monitorAudioLevel, isPreviewingAudio]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setAudioLevel(0);
    setIsPreviewingAudio(false);
    
    // Restart preview after recording stops
    if (selectedDevice) {
      setTimeout(() => startAudioPreview(selectedDevice), 100);
    }
  }, [isRecording, selectedDevice, startAudioPreview]);

  // Start streaming
  const startStreaming = useCallback(async () => {
    try {
      const constraints = {
        audio: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          sampleRate: audioSettings.sampleRate,
          channelCount: audioSettings.channels,
          echoCancellation: audioSettings.echoCancellation,
          noiseSuppression: audioSettings.noiseSuppression,
          autoGainControl: audioSettings.autoGainControl,
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Setup audio analysis
      audioContextRef.current = new AudioContext({ sampleRate: audioSettings.sampleRate });
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      monitorAudioLevel();

      // Setup streaming media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;

      let chunkCounter = 0;
      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          chunkCounter++;
          setChunkCount(chunkCounter);
          const chunkId = `${sessionId}-chunk-${chunkCounter}`;
          await processAudioChunk(event.data, chunkId);
        }
      };

      // Start recording and set up chunk intervals
      mediaRecorder.start();
      chunkIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start();
        }
      }, transcriptionSettings.chunkDuration * 1000);

      setIsStreaming(true);
      setStreamingResults('');

      notifySuccess('Streaming Started', `Started streaming with ${transcriptionSettings.chunkDuration}s chunks`);

    } catch (error) {
      console.error('Failed to start streaming:', error);
      notifyError('Streaming Error', 'Failed to start audio streaming');
    }
  }, [selectedDevice, audioSettings, transcriptionSettings, sessionId, monitorAudioLevel, notifySuccess, notifyError]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && isStreaming) {
      mediaRecorderRef.current.stop();
      setIsStreaming(false);
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setAudioLevel(0);
    setChunkCount(0);
    setIsPreviewingAudio(false);
    
    // Restart preview after streaming stops
    if (selectedDevice) {
      setTimeout(() => startAudioPreview(selectedDevice), 100);
    }
  }, [isStreaming, selectedDevice, startAudioPreview]);

  // Process single audio file
  const processAudio = useCallback(async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('session_id', sessionId);
      formData.append('enable_transcription', 'true');
      formData.append('enable_diarization', transcriptionSettings.enableDiarization.toString());
      formData.append('whisper_model', transcriptionSettings.whisperModel);
      formData.append('enable_vad', transcriptionSettings.enableVAD.toString());
      
      if (transcriptionSettings.language !== 'auto') {
        formData.append('language', transcriptionSettings.language);
      }

      const response = await fetch('/api/audio/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.processing_result) {
        const transcriptionResult: TranscriptionResult = {
          id: `single-${Date.now()}`,
          audioData: audioBlob,
          transcriptText: result.processing_result.text || '',
          language: result.processing_result.language || 'unknown',
          confidence: result.processing_result.confidence || 0,
          processingTime: result.processing_result.processing_time || 0,
          timestamp: Date.now(),
          modelUsed: transcriptionSettings.whisperModel,
          sessionId,
          segments: result.processing_result.segments,
          speakerInfo: result.processing_result.speaker_info,
        };

        setCurrentTranscription(transcriptionResult);
        setTranscriptionResults(prev => [transcriptionResult, ...prev]);

        notifySuccess('Transcription Complete', 'Audio has been transcribed successfully');
      }

    } catch (error) {
      console.error('Transcription failed:', error);
      notifyError('Transcription Failed', `Failed to transcribe audio: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  }, [sessionId, transcriptionSettings, notifySuccess, notifyError]);

  // Process audio chunk for streaming
  const processAudioChunk = useCallback(async (audioBlob: Blob, chunkId: string) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, `${chunkId}.webm`);
      formData.append('session_id', sessionId);
      formData.append('chunk_id', chunkId);
      formData.append('enable_transcription', 'true');
      formData.append('enable_diarization', transcriptionSettings.enableDiarization.toString());
      formData.append('whisper_model', transcriptionSettings.whisperModel);
      formData.append('enable_vad', transcriptionSettings.enableVAD.toString());
      
      if (transcriptionSettings.language !== 'auto') {
        formData.append('language', transcriptionSettings.language);
      }

      const response = await fetch('/api/audio/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.processing_result && result.processing_result.text) {
        const chunkText = result.processing_result.text;
        setStreamingResults(prev => prev + ' ' + chunkText);

        const transcriptionResult: TranscriptionResult = {
          id: chunkId,
          audioData: audioBlob,
          transcriptText: chunkText,
          language: result.processing_result.language || 'unknown',
          confidence: result.processing_result.confidence || 0,
          processingTime: result.processing_result.processing_time || 0,
          timestamp: Date.now(),
          modelUsed: transcriptionSettings.whisperModel,
          sessionId,
          chunkId,
          segments: result.processing_result.segments,
          speakerInfo: result.processing_result.speaker_info,
        };

        setTranscriptionResults(prev => [transcriptionResult, ...prev]);
      }

    } catch (error) {
      console.error('Chunk transcription failed:', error);
    }
  }, [sessionId, transcriptionSettings]);

  // Clear results
  const clearResults = useCallback(() => {
    setTranscriptionResults([]);
    setCurrentTranscription(null);
    setStreamingResults('');
    setChunkCount(0);
  }, []);

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🎙️ Advanced Transcription Testing
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Comprehensive audio transcription testing with real-time streaming and detailed analysis
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>Orchestration Status:</strong> {webSocketConnected ? 'Connected via WebSocket' : 'Connected via REST API'}
      </Alert>

      <Grid container spacing={3}>
        {/* Audio Controls */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎤 Audio Recording Controls
              </Typography>
              
              <Stack spacing={2}>
                <FormControl fullWidth>
                  <InputLabel>Audio Device</InputLabel>
                  <Select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    label="Audio Device"
                  >
                    {audioDevices.map((device) => (
                      <MenuItem key={device.deviceId} value={device.deviceId}>
                        {device.label || `Device ${device.deviceId.slice(0, 8)}`}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Box>
                  <Typography variant="body2" gutterBottom>
                    Audio Level: {audioLevel.toFixed(1)}%
                    {audioLevel < 5 && ' ⚠️ Signal too weak - check audio settings'}
                    {audioLevel > 5 && audioLevel < 20 && ' ✓ Good signal level'}
                    {audioLevel >= 20 && ' 📢 Strong signal'}
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(audioLevel, 100)} 
                    sx={{ height: 8, borderRadius: 1 }}
                    color={audioLevel < 5 ? 'error' : audioLevel > 80 ? 'warning' : 'success'}
                  />
                  {audioLevel < 5 && (
                    <Typography variant="caption" color="error" sx={{ mt: 0.5, display: 'block' }}>
                      Try disabling audio processing options below or speaking louder
                    </Typography>
                  )}
                </Box>

                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={isRecording ? <StopIcon /> : <MicIcon />}
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isStreaming || isProcessing}
                    color={isRecording ? 'error' : 'primary'}
                  >
                    {isRecording ? `Stop Recording (${formatTime(recordingTime)})` : 'Start Recording'}
                  </Button>

                  <Button
                    variant="outlined"
                    startIcon={isStreaming ? <StopIcon /> : <PlayIcon />}
                    onClick={isStreaming ? stopStreaming : startStreaming}
                    disabled={isRecording || isProcessing}
                    color={isStreaming ? 'error' : 'primary'}
                  >
                    {isStreaming ? `Stop Streaming (${chunkCount} chunks)` : 'Start Streaming'}
                  </Button>
                </Box>

                {isProcessing && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CircularProgress size={20} />
                    <Typography variant="body2">Processing audio...</Typography>
                  </Box>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Transcription Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ⚙️ Transcription Settings
              </Typography>
              
              <Stack spacing={2}>
                <FormControl fullWidth>
                  <InputLabel>Whisper Model</InputLabel>
                  <Select
                    value={transcriptionSettings.whisperModel}
                    onChange={(e) => setTranscriptionSettings(prev => ({ ...prev, whisperModel: e.target.value }))}
                    label="Whisper Model"
                  >
                    {availableModels.map((model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Language</InputLabel>
                  <Select
                    value={transcriptionSettings.language}
                    onChange={(e) => setTranscriptionSettings(prev => ({ ...prev, language: e.target.value }))}
                    label="Language"
                  >
                    {supportedLanguages.map((lang) => (
                      <MenuItem key={lang.code} value={lang.code}>
                        {lang.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <TextField
                  label="Chunk Duration (seconds)"
                  type="number"
                  value={transcriptionSettings.chunkDuration}
                  onChange={(e) => setTranscriptionSettings(prev => ({ 
                    ...prev, 
                    chunkDuration: parseInt(e.target.value) || 5 
                  }))}
                  inputProps={{ min: 1, max: 30 }}
                />

                <FormGroup>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={transcriptionSettings.enableDiarization}
                        onChange={(e) => setTranscriptionSettings(prev => ({ 
                          ...prev, 
                          enableDiarization: e.target.checked 
                        }))}
                      />
                    }
                    label="Enable Speaker Diarization"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={transcriptionSettings.enableVAD}
                        onChange={(e) => setTranscriptionSettings(prev => ({ 
                          ...prev, 
                          enableVAD: e.target.checked 
                        }))}
                      />
                    }
                    label="Enable Voice Activity Detection"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={transcriptionSettings.enableTimestamps}
                        onChange={(e) => setTranscriptionSettings(prev => ({ 
                          ...prev, 
                          enableTimestamps: e.target.checked 
                        }))}
                      />
                    }
                    label="Include Timestamps"
                  />
                </FormGroup>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Audio Processing Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🔧 Audio Processing Settings
              </Typography>
              
              <Stack spacing={2}>
                <TextField
                  label="Sample Rate (Hz)"
                  type="number"
                  value={audioSettings.sampleRate}
                  onChange={(e) => setAudioSettings(prev => ({ 
                    ...prev, 
                    sampleRate: parseInt(e.target.value) || 16000 
                  }))}
                  inputProps={{ min: 8000, max: 48000, step: 1000 }}
                />

                <FormGroup>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={audioSettings.echoCancellation}
                        onChange={(e) => setAudioSettings(prev => ({ 
                          ...prev, 
                          echoCancellation: e.target.checked 
                        }))}
                      />
                    }
                    label="Echo Cancellation (disable for better signal capture)"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={audioSettings.noiseSuppression}
                        onChange={(e) => setAudioSettings(prev => ({ 
                          ...prev, 
                          noiseSuppression: e.target.checked 
                        }))}
                      />
                    }
                    label="Noise Suppression (disable for better signal capture)"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={audioSettings.autoGainControl}
                        onChange={(e) => setAudioSettings(prev => ({ 
                          ...prev, 
                          autoGainControl: e.target.checked 
                        }))}
                      />
                    }
                    label="Auto Gain Control (disable for consistent levels)"
                  />
                </FormGroup>

                <Alert severity="info">
                  <strong>Audio Capture Tips:</strong>
                  <br />• For microphone input: Enable echo cancellation and noise suppression
                  <br />• For system/loopback audio: Disable all processing for better signal capture
                  <br />• If audio levels are very low, try disabling all processing options
                </Alert>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Real-time Streaming Results */}
        {isStreaming && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📝 Real-time Transcription Stream
                </Typography>
                <Paper sx={{ p: 2, minHeight: 100, backgroundColor: 'grey.50' }}>
                  <Typography variant="body1">
                    {streamingResults || 'Listening for speech...'}
                  </Typography>
                </Paper>
                <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
                  <Chip label={`Session: ${sessionId}`} size="small" />
                  <Chip label={`Chunks: ${chunkCount}`} size="small" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Current Transcription Result */}
        {currentTranscription && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📊 Latest Transcription Result
                </Typography>
                <Paper sx={{ p: 2, mb: 2, backgroundColor: 'primary.50' }}>
                  <Typography variant="body1" sx={{ mb: 1 }}>
                    {currentTranscription.transcriptText}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    <Chip 
                      label={`Language: ${currentTranscription.language.toUpperCase()}`}
                      size="small"
                      color="primary"
                    />
                    <Chip 
                      label={`Confidence: ${(currentTranscription.confidence * 100).toFixed(1)}%`}
                      size="small"
                      color={currentTranscription.confidence > 0.8 ? "success" : "warning"}
                    />
                    <Chip 
                      label={`${currentTranscription.processingTime}ms`}
                      size="small"
                    />
                    <Chip 
                      label={`Model: ${currentTranscription.modelUsed}`}
                      size="small"
                    />
                  </Box>
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Transcription History */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  📚 Transcription History ({transcriptionResults.length})
                </Typography>
                <Button
                  startIcon={<ClearIcon />}
                  onClick={clearResults}
                  disabled={transcriptionResults.length === 0}
                >
                  Clear History
                </Button>
              </Box>

              {transcriptionResults.length > 0 ? (
                <Stack spacing={2}>
                  {transcriptionResults.map((result) => (
                    <Accordion key={result.id}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                          <Typography variant="body1" sx={{ flexGrow: 1 }}>
                            {result.transcriptText.slice(0, 100)}...
                          </Typography>
                          <Chip 
                            label={result.language.toUpperCase()} 
                            size="small" 
                            color="primary" 
                          />
                          <Chip 
                            label={`${(result.confidence * 100).toFixed(1)}%`} 
                            size="small" 
                            color={result.confidence > 0.8 ? "success" : "warning"}
                          />
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Stack spacing={2}>
                          <Typography variant="body1">
                            {result.transcriptText}
                          </Typography>
                          
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            <Chip label={`Model: ${result.modelUsed}`} size="small" />
                            <Chip label={`Processing: ${result.processingTime}ms`} size="small" />
                            <Chip label={`Session: ${result.sessionId}`} size="small" />
                            {result.chunkId && (
                              <Chip label={`Chunk: ${result.chunkId}`} size="small" />
                            )}
                          </Box>

                          {result.segments && result.segments.length > 0 && (
                            <Box>
                              <Typography variant="subtitle2" gutterBottom>
                                Timestamped Segments:
                              </Typography>
                              <Stack spacing={1}>
                                {result.segments.map((segment, index) => (
                                  <Paper key={index} sx={{ p: 1, backgroundColor: 'grey.50' }}>
                                    <Typography variant="body2">
                                      <strong>[{segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s]</strong> {segment.text}
                                      <Chip 
                                        label={`${(segment.confidence * 100).toFixed(1)}%`} 
                                        size="small" 
                                        sx={{ ml: 1 }}
                                      />
                                    </Typography>
                                  </Paper>
                                ))}
                              </Stack>
                            </Box>
                          )}

                          {result.speakerInfo && result.speakerInfo.length > 0 && (
                            <Box>
                              <Typography variant="subtitle2" gutterBottom>
                                Speaker Information:
                              </Typography>
                              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                {result.speakerInfo.map((speaker, index) => (
                                  <Chip 
                                    key={index}
                                    label={`Speaker ${speaker.speakerId}: ${(speaker.confidence * 100).toFixed(1)}%`}
                                    size="small"
                                    variant="outlined"
                                  />
                                ))}
                              </Box>
                            </Box>
                          )}
                        </Stack>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Stack>
              ) : (
                <Alert severity="info">
                  Start recording or streaming to see transcription results here.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TranscriptionTesting;