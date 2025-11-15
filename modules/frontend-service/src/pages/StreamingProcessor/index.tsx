import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  Alert,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Stack,
} from '@mui/material';
import {
  Mic,
  MicOff,
  PlayArrow,
  Stop,
  Settings,
  Translate,
  RecordVoiceOver,
  ExpandMore,
  Tune,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store';
import {
  setAudioDevices,
  setVisualizationData,
  addProcessingLog,
  setAudioQualityMetrics,
  updateConfig,
} from '@/store/slices/audioSlice';
import { AudioVisualizer } from '../AudioTesting/components/AudioVisualizer';
import {
  calculateMeetingAudioLevel,
  getMeetingAudioQuality,
  getDisplayLevel,
} from '@/utils/audioLevelCalculation';
import { useAvailableModels } from '@/hooks/useAvailableModels';
import { useAudioDevices } from '@/hooks/useAudioDevices';
import type { StreamingChunk, TranscriptionResult, TranslationResult, StreamingStats } from '@/types/streaming';
import { DEFAULT_TARGET_LANGUAGES, DEFAULT_STREAMING_STATS, DEFAULT_PROCESSING_CONFIG } from '@/constants/defaultConfig';
import { SUPPORTED_LANGUAGES } from '@/constants/languages';

const StreamingProcessor: React.FC = () => {
  const dispatch = useAppDispatch();
  const { devices, visualization, config } = useAppSelector(state => state.audio);
  
  // Load available models and device information dynamically
  const { 
    models: availableModels, 
    loading: modelsLoading, 
    error: modelsError, 
    status: modelsStatus,
    serviceMessage,
    deviceInfo,
    refetch: refetchModels 
  } = useAvailableModels();
  
  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStats, setStreamingStats] = useState<StreamingStats>(DEFAULT_STREAMING_STATS);
  
  // Audio streaming - CHANGE 1: Add English as first option
  const [chunkDuration, setChunkDuration] = useState(3); // 3 seconds default
  const [targetLanguages, setTargetLanguages] = useState<string[]>(['en', ...DEFAULT_TARGET_LANGUAGES]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');

  // Processing configuration - CHANGE 2: Disable audio pipeline features by default
  const [processingConfig, setProcessingConfig] = useState({
    ...DEFAULT_PROCESSING_CONFIG,
    enableDiarization: false,
    enableVAD: false,
    audioProcessing: false,
  });
  
  // Results
  const [transcriptionResults, setTranscriptionResults] = useState<TranscriptionResult[]>([]);
  const [translationResults, setTranslationResults] = useState<TranslationResult[]>([]);
  const [activeChunks, setActiveChunks] = useState<Set<string>>(new Set());
  
  // Audio refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const chunkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);

  // Initialize audio devices with auto-selection
  useAudioDevices({
    autoSelect: true,
    selectedDevice,
    onDeviceSelected: setSelectedDevice
  });

  // Initialize audio visualization
  useEffect(() => {
    if (!selectedDevice) return;

    const initializeAudioVisualization = async () => {
      try {
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
        
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
        
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 2048;
        microphoneRef.current.connect(analyserRef.current);

        startVisualization();
        
        dispatch(addProcessingLog({
          level: 'SUCCESS',
          message: `Audio visualization initialized with device: ${selectedDevice}`,
          timestamp: Date.now()
        }));
      } catch (error) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: `Failed to initialize audio visualization: ${error}`,
          timestamp: Date.now()
        }));
      }
    };

    initializeAudioVisualization();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [dispatch, selectedDevice]);

  const startVisualization = useCallback(() => {
    if (!analyserRef.current) return;

    const updateVisualization = () => {
      if (!analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const frequencyData = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(frequencyData);

      const timeData = new Uint8Array(analyserRef.current.fftSize);
      analyserRef.current.getByteTimeDomainData(timeData);

      // Calculate professional meeting-optimized audio metrics
      const audioMetrics = calculateMeetingAudioLevel(timeData, frequencyData, 16000);
      const qualityAssessment = getMeetingAudioQuality(audioMetrics);
      const displayLevel = getDisplayLevel(audioMetrics);

      // Update Redux store with enhanced visualization data
      dispatch(setVisualizationData({
        frequencyData: Array.from(frequencyData),
        timeData: Array.from(timeData),
        audioLevel: displayLevel
      }));

      // Update comprehensive audio quality metrics
      dispatch(setAudioQualityMetrics({
        rmsLevel: audioMetrics.rmsDb,
        peakLevel: audioMetrics.peakDb,
        signalToNoise: audioMetrics.signalToNoise,
        frequency: 16000,
        clipping: audioMetrics.clipping * 100,
        voiceActivity: audioMetrics.voiceActivity,
        spectralCentroid: audioMetrics.spectralCentroid,
        dynamicRange: audioMetrics.dynamicRange,
        speechClarity: audioMetrics.speechClarity,
        backgroundNoise: audioMetrics.backgroundNoise,
        qualityAssessment: qualityAssessment.quality,
        qualityScore: qualityAssessment.score,
        recommendations: qualityAssessment.recommendations,
        issues: qualityAssessment.issues
      }));

      animationFrameRef.current = requestAnimationFrame(updateVisualization);
    };

    updateVisualization();
  }, [dispatch]);

  // Simple session management for testing
  const [sessionId] = useState(() => `streaming_test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  
  // Handle response from streaming endpoint directly (no WebSocket needed for simple testing)
  const handleStreamingResponse = useCallback((response: any, chunkId: string) => {
    try {
      // Look for transcription in processing_result (backend format) or transcription_result (fallback)
      const transcriptionData = response.processing_result || response.transcription_result;
      
      if (transcriptionData && processingConfig.enableTranscription) {
        const result: TranscriptionResult = {
          id: transcriptionData.id || `transcription-${Date.now()}-${chunkId}`,
          chunkId: chunkId,
          text: transcriptionData.text || transcriptionData.transcription || '',
          confidence: transcriptionData.confidence || transcriptionData.confidence_score || 0.9,
          language: transcriptionData.language || transcriptionData.detected_language || 'en',
          speakers: transcriptionData.speakers || transcriptionData.segments?.speakers,
          timestamp: Date.now(),
          processing_time: transcriptionData.processing_time || response.processing_time || 0
        };
        
        setTranscriptionResults(prev => [...prev, result]);
        
        dispatch(addProcessingLog({
          level: 'SUCCESS',
          message: `Transcription: "${result.text.substring(0, 50)}..." (confidence: ${(result.confidence * 100).toFixed(1)}%)`,
          timestamp: Date.now()
        }));
      }
      
      if (response.translations && processingConfig.enableTranslation) {
        Object.entries(response.translations).forEach(([lang, translation]: [string, any]) => {
          const transcriptionText = transcriptionData?.text || transcriptionData?.transcription || '';
          
          const result: TranslationResult = {
            id: `translation-${Date.now()}-${chunkId}-${lang}`,
            transcriptionId: transcriptionData?.id || '',
            sourceText: translation.source_text || transcriptionText,
            translatedText: translation.translated_text || '',
            sourceLanguage: translation.source_language || 'en',
            targetLanguage: lang,
            confidence: translation.confidence || 0.9,
            timestamp: Date.now(),
            processing_time: translation.processing_time || 0
          };
          
          setTranslationResults(prev => [...prev, result]);
          
          dispatch(addProcessingLog({
            level: 'SUCCESS',
            message: `Translation (${lang}): "${result.translatedText.substring(0, 50)}..."`,
            timestamp: Date.now()
          }));
        });
      }
      
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Failed to process streaming response: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [processingConfig, dispatch]);

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!selectedDevice || !audioStreamRef.current) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: 'No audio device selected or audio stream not available',
        timestamp: Date.now()
      }));
      return;
    }

    try {
      setIsStreaming(true);
      
      // Initialize MediaRecorder for chunk recording
      recordingChunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(audioStreamRef.current, {
        mimeType: 'audio/webm; codecs=opus',
        audioBitsPerSecond: 128000
      });
      
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (recordingChunksRef.current.length > 0) {
          const audioBlob = new Blob(recordingChunksRef.current, { type: 'audio/webm' });
          const chunkId = `chunk_${Date.now()}`;
          
          setActiveChunks(prev => new Set([...prev, chunkId]));
          
          // Send audio chunk to orchestration service
          await sendAudioChunk(chunkId, audioBlob);
          
          setStreamingStats(prev => ({
            ...prev,
            chunksStreamed: prev.chunksStreamed + 1,
            totalDuration: prev.totalDuration + chunkDuration
          }));
          
          recordingChunksRef.current = [];
        }
      };

      // Start recording and set up interval for chunks
      mediaRecorder.start();
      
      chunkIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start();
        }
      }, chunkDuration * 1000);
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: `Started streaming with ${chunkDuration}s chunks`,
        timestamp: Date.now()
      }));
    } catch (error) {
      setIsStreaming(false);
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Failed to start streaming: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [selectedDevice, chunkDuration, dispatch]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Streaming stopped',
      timestamp: Date.now()
    }));
  }, [dispatch]);

  // Send audio chunk to orchestration service
  const sendAudioChunk = useCallback(async (chunkId: string, audioBlob: Blob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'chunk.webm');
      formData.append('chunk_id', chunkId);
      formData.append('session_id', sessionId);
      console.log('🌐 Frontend: targetLanguages state:', targetLanguages);
      console.log('🌐 Frontend: targetLanguages JSON:', JSON.stringify(targetLanguages));
      formData.append('target_languages', JSON.stringify(targetLanguages));
      formData.append('enable_transcription', processingConfig.enableTranscription.toString());
      formData.append('enable_translation', processingConfig.enableTranslation.toString());
      formData.append('enable_diarization', processingConfig.enableDiarization.toString());
      formData.append('whisper_model', processingConfig.whisperModel);
      formData.append('translation_quality', processingConfig.translationQuality);
      formData.append('enable_vad', processingConfig.enableVAD.toString());
      formData.append('audio_processing', processingConfig.audioProcessing.toString());
      formData.append('noise_reduction', processingConfig.noiseReduction.toString());
      formData.append('speech_enhancement', processingConfig.speechEnhancement.toString());
      
      const response = await fetch('/api/audio/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Handle the response directly
      handleStreamingResponse(result, chunkId);
      
      // Remove from active chunks
      setActiveChunks(prev => {
        const newSet = new Set(prev);
        newSet.delete(chunkId);
        return newSet;
      });
      
      dispatch(addProcessingLog({
        level: 'INFO',
        message: `Audio chunk ${chunkId} processed successfully`,
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Failed to process audio chunk ${chunkId}: ${error}`,
        timestamp: Date.now()
      }));
      
      setActiveChunks(prev => {
        const newSet = new Set(prev);
        newSet.delete(chunkId);
        return newSet;
      });
      
      setStreamingStats(prev => ({
        ...prev,
        errorCount: prev.errorCount + 1
      }));
    }
  }, [targetLanguages, processingConfig, sessionId, handleStreamingResponse, dispatch]);

  const handleLanguageToggle = useCallback((language: string) => {
    console.log('🌐 Frontend: Toggling language:', language);
    setTargetLanguages(prev => {
      const newLanguages = prev.includes(language) 
        ? prev.filter(lang => lang !== language)
        : [...prev, language];
      console.log('🌐 Frontend: Languages updated from', prev, 'to', newLanguages);
      return newLanguages;
    });
  }, []);

  // Debug: Track targetLanguages changes
  useEffect(() => {
    console.log('🌐 Frontend: targetLanguages state changed to:', targetLanguages);
  }, [targetLanguages]);

  const clearResults = useCallback(() => {
    setTranscriptionResults([]);
    setTranslationResults([]);
    setActiveChunks(new Set());
    setStreamingStats(DEFAULT_STREAMING_STATS);
  }, []);

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🎤 Real-time Streaming Processor
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Test real-time audio streaming with live transcription and translation. Audio is streamed in {chunkDuration}-second chunks to the orchestration service.
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          This interface streams audio in real-time to the orchestration service, which coordinates with the whisper service for transcription and translation service for multi-language output.
          Each audio chunk is processed separately, showing transcription and translation results as they arrive.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Left Panel - Controls and Configuration */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* Audio Device Selection */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <Settings /> Audio Configuration
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Audio Input Device</InputLabel>
                  <Select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    label="Audio Input Device"
                  >
                    {devices.map((device) => (
                      <MenuItem key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Chunk Duration</InputLabel>
                  <Select
                    value={chunkDuration}
                    onChange={(e) => setChunkDuration(Number(e.target.value))}
                    label="Chunk Duration"
                  >
                    <MenuItem value={2}>2 seconds</MenuItem>
                    <MenuItem value={3}>3 seconds (recommended)</MenuItem>
                    <MenuItem value={4}>4 seconds</MenuItem>
                    <MenuItem value={5}>5 seconds</MenuItem>
                  </Select>
                </FormControl>

                {/* CHANGE 3: Move audio visualization here by the device selection */}
                <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                  📊 Live Audio Visualization
                </Typography>
                <Card sx={{ p: 1, bgcolor: 'background.default' }}>
                  <AudioVisualizer
                    frequencyData={visualization.frequencyData}
                    timeData={visualization.timeData}
                    audioLevel={visualization.audioLevel}
                    isRecording={isStreaming}
                    height={120}
                  />
                </Card>
              </CardContent>
            </Card>

            {/* Processing Configuration */}
            <Card>
              <CardContent>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">
                      <Tune /> Processing Configuration
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={2}>
                      {/* Core Processing Options */}
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Core Processing
                        </Typography>
                        <FormGroup>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={processingConfig.enableTranscription}
                                onChange={(e) => setProcessingConfig(prev => ({
                                  ...prev,
                                  enableTranscription: e.target.checked
                                }))}
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
                                disabled={!processingConfig.enableTranscription}
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
                                disabled={true}
                              />
                            }
                            label="Speaker Diarization (Coming Soon)"
                          />
                          <FormControlLabel
                            control={
                              <Switch
                                checked={processingConfig.enableVAD}
                                onChange={(e) => setProcessingConfig(prev => ({
                                  ...prev,
                                  enableVAD: e.target.checked
                                }))}
                                disabled={true}
                              />
                            }
                            label="Voice Activity Detection (Coming Soon)"
                          />
                        </FormGroup>
                      </Box>

                      <Divider />

                      {/* Model Selection */}
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Model Configuration
                        </Typography>
                        <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                          <InputLabel>Whisper Model</InputLabel>
                          <Select
                            value={processingConfig.whisperModel}
                            onChange={(e) => setProcessingConfig(prev => ({
                              ...prev,
                              whisperModel: e.target.value
                            }))}
                            label="Whisper Model"
                            disabled={modelsLoading}
                          >
                            {modelsLoading ? (
                              <MenuItem value="whisper-base">Loading models...</MenuItem>
                            ) : modelsError ? (
                              <MenuItem value="whisper-base">Error loading models</MenuItem>
                            ) : availableModels.length > 0 ? (
                              availableModels.map((model) => (
                                <MenuItem key={model.name} value={model.name}>
                                  {model.displayName}
                                </MenuItem>
                              ))
                            ) : (
                              <MenuItem value="whisper-base">No models available</MenuItem>
                            )}
                          </Select>
                        </FormControl>
                        
                        {/* Device Information Display */}
                        {deviceInfo && (
                          <Box sx={{ mt: 1, p: 1, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
                            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                              Service Devices:
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                              <Chip 
                                label={`Audio: ${deviceInfo.audio_service.device.toUpperCase()}`}
                                size="small"
                                color={deviceInfo.audio_service.status === 'healthy' ? 'success' : 'warning'}
                                sx={{ fontSize: '0.7rem' }}
                              />
                              <Chip 
                                label={`Translation: ${deviceInfo.translation_service.device.toUpperCase()}`}
                                size="small"
                                color={deviceInfo.translation_service.status === 'healthy' ? 'success' : 'warning'}
                                sx={{ fontSize: '0.7rem' }}
                              />
                            </Box>
                          </Box>
                        )}
                        
                        {/* Service Status Message */}
                        {(modelsStatus === 'fallback' || serviceMessage) && (
                          <Alert severity="warning" sx={{ mt: 1, fontSize: '0.75rem' }}>
                            {serviceMessage || 'Using fallback models'}
                          </Alert>
                        )}
                        
                        <FormControl fullWidth size="small">
                          <InputLabel>Translation Quality</InputLabel>
                          <Select
                            value={processingConfig.translationQuality}
                            onChange={(e) => setProcessingConfig(prev => ({
                              ...prev,
                              translationQuality: e.target.value
                            }))}
                            label="Translation Quality"
                            disabled={!processingConfig.enableTranslation}
                          >
                            <MenuItem value="fast">Fast</MenuItem>
                            <MenuItem value="balanced">Balanced (recommended)</MenuItem>
                            <MenuItem value="high_quality">High Quality</MenuItem>
                          </Select>
                        </FormControl>
                      </Box>

                      <Divider />

                      {/* Audio Processing */}
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Audio Enhancement
                        </Typography>
                        <FormGroup>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={processingConfig.audioProcessing}
                                onChange={(e) => setProcessingConfig(prev => ({
                                  ...prev,
                                  audioProcessing: e.target.checked
                                }))}
                                disabled={true}
                              />
                            }
                            label="Audio Processing Pipeline (Coming Soon)"
                          />
                          <FormControlLabel
                            control={
                              <Switch
                                checked={processingConfig.noiseReduction}
                                onChange={(e) => setProcessingConfig(prev => ({
                                  ...prev,
                                  noiseReduction: e.target.checked
                                }))}
                                disabled={true}
                              />
                            }
                            label="Noise Reduction (Coming Soon)"
                          />
                          <FormControlLabel
                            control={
                              <Switch
                                checked={processingConfig.speechEnhancement}
                                onChange={(e) => setProcessingConfig(prev => ({
                                  ...prev,
                                  speechEnhancement: e.target.checked
                                }))}
                                disabled={true}
                              />
                            }
                            label="Speech Enhancement (Coming Soon)"
                          />
                        </FormGroup>
                      </Box>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
              </CardContent>
            </Card>

            {/* Language Selection */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <Translate /> Target Languages
                </Typography>
                
                <FormGroup>
                  {SUPPORTED_LANGUAGES.slice(0, 9).map((lang) => (
                    <FormControlLabel
                      key={lang.code}
                      control={
                        <Checkbox
                          checked={targetLanguages.includes(lang.code)}
                          onChange={() => handleLanguageToggle(lang.code)}
                          size="small"
                          disabled={!processingConfig.enableTranslation}
                        />
                      }
                      label={lang.name}
                    />
                  ))}
                </FormGroup>
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Selected: {targetLanguages.length} languages
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {targetLanguages.map((lang) => (
                      <Chip
                        key={lang}
                        label={lang.toUpperCase()}
                        size="small"
                        color="primary"
                        onDelete={() => handleLanguageToggle(lang)}
                      />
                    ))}
                  </Box>
                </Box>
              </CardContent>
            </Card>

            {/* Controls */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🎮 Streaming Controls
                </Typography>
                
                <Stack spacing={2}>
                  <Button
                    variant="contained"
                    color={isStreaming ? "error" : "primary"}
                    size="large"
                    startIcon={isStreaming ? <MicOff /> : <Mic />}
                    onClick={isStreaming ? stopStreaming : startStreaming}
                    disabled={
                      !selectedDevice || 
                      (!processingConfig.enableTranscription && !processingConfig.enableTranslation) ||
                      (processingConfig.enableTranslation && targetLanguages.length === 0)
                    }
                    fullWidth
                  >
                    {isStreaming ? 'Stop Streaming' : 'Start Streaming'}
                  </Button>
                  
                  <Button
                    variant="outlined"
                    onClick={clearResults}
                    disabled={isStreaming}
                    fullWidth
                  >
                    Clear Results
                  </Button>
                </Stack>
                
                {/* Streaming Stats */}
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Session Statistics:
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Chunks: {streamingStats.chunksStreamed} | 
                    Duration: {streamingStats.totalDuration}s | 
                    Errors: {streamingStats.errorCount}
                  </Typography>
                  
                  {activeChunks.size > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption">
                        Processing {activeChunks.size} chunk(s)...
                      </Typography>
                      <LinearProgress />
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} lg={8}>
          <Grid container spacing={2}>
            {/* Transcription Results */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="h6" gutterBottom>
                    <RecordVoiceOver /> Transcription Results
                  </Typography>
                  
                  <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                    {transcriptionResults.length === 0 ? (
                      <Alert severity="info">
                        No transcriptions yet. Start streaming to see real-time results.
                      </Alert>
                    ) : (
                      <List dense>
                        {transcriptionResults.slice(-10).reverse().map((result) => (
                          <ListItem key={result.id} sx={{ px: 0 }}>
                            <ListItemText
                              primary={result.text}
                              secondary={
                                <Box>
                                  <Typography variant="caption">
                                    {new Date(result.timestamp).toLocaleTimeString()} | 
                                    Confidence: {(result.confidence * 100).toFixed(1)}% | 
                                    Lang: {result.language.toUpperCase()} |
                                    Time: {result.processing_time}ms
                                  </Typography>
                                  {result.speakers && result.speakers.length > 0 && (
                                    <Box sx={{ mt: 0.5 }}>
                                      {result.speakers.map((speaker, idx) => (
                                        <Chip
                                          key={idx}
                                          label={speaker.speaker_name || `Speaker ${speaker.speaker_id}`}
                                          size="small"
                                          sx={{ mr: 0.5, mb: 0.5 }}
                                        />
                                      ))}
                                    </Box>
                                  )}
                                </Box>
                              }
                            />
                            <Divider />
                          </ListItem>
                        ))}
                      </List>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Translation Results */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="h6" gutterBottom>
                    <Translate /> Translation Results
                  </Typography>
                  
                  <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                    {translationResults.length === 0 ? (
                      <Alert severity="info">
                        No translations yet. Start streaming to see real-time translations.
                      </Alert>
                    ) : (
                      <List dense>
                        {translationResults.slice(-15).reverse().map((result) => (
                          <ListItem key={result.id} sx={{ px: 0 }}>
                            <ListItemText
                              primary={
                                <Box>
                                  <Chip 
                                    label={result.targetLanguage.toUpperCase()} 
                                    size="small" 
                                    color="primary"
                                    sx={{ mr: 1 }}
                                  />
                                  {result.translatedText}
                                </Box>
                              }
                              secondary={
                                <Box>
                                  <Typography variant="caption" color="text.secondary">
                                    Original: {result.sourceText}
                                  </Typography>
                                  <br />
                                  <Typography variant="caption">
                                    {new Date(result.timestamp).toLocaleTimeString()} | 
                                    Confidence: {(result.confidence * 100).toFixed(1)}% |
                                    Time: {result.processing_time}ms
                                  </Typography>
                                </Box>
                              }
                            />
                            <Divider />
                          </ListItem>
                        ))}
                      </List>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default StreamingProcessor;