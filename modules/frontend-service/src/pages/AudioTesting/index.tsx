import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tabs,
  Tab,
  Alert,
  Fade,
  Button,
  Card,
  CardContent,
  Chip,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Divider,
  Stack,
} from '@mui/material';
import { useAppSelector, useAppDispatch } from '@/store';
import {
  setAudioDevices,
  setVisualizationData,
  addProcessingLog,
  setAudioQualityMetrics
} from '@/store/slices/audioSlice';
import { DEFAULT_TARGET_LANGUAGES } from '@/config/translation';
import { 
  calculateMeetingAudioLevel, 
  getMeetingAudioQuality, 
  getDisplayLevel
} from '@/utils/audioLevelCalculation';
import { RecordingControls } from './components/RecordingControls';
import { AudioConfiguration } from './components/AudioConfiguration';
import { AudioVisualizer } from './components/AudioVisualizer';
import { PipelineProcessing } from './components/PipelineProcessing';
import { ProcessingPresets } from './components/ProcessingPresets';
import { ActivityLogs } from './components/ActivityLogs';
import { useAudioProcessing } from '@/hooks/useAudioProcessing';

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
      id={`audio-testing-tabpanel-${index}`}
      aria-labelledby={`audio-testing-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Fade in={true} timeout={300}>
          <Box sx={{ py: 3 }}>
            {children}
          </Box>
        </Fade>
      )}
    </div>
  );
}

const AudioTesting: React.FC = () => {
  const dispatch = useAppDispatch();
  const { recording, visualization, stages, config } = useAppSelector(state => state.audio);
  const [tabValue, setTabValue] = useState(0);
  const [targetLanguages, setTargetLanguages] = useState<string[]>([...DEFAULT_TARGET_LANGUAGES]);
  const [translationResults, setTranslationResults] = useState<any>(null);
  const [transcriptionResult, setTranscriptionResult] = useState<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const {
    startRecording,
    stopRecording,
    playRecording,
    clearRecording,
    downloadRecording,
    runPipeline,
    runStepByStep,
    resetPipeline,
    exportResults,
    processAudioForTranscription,
    processTranscriptionForTranslation,
    processAudioWithTranslation,
    isProcessing,
    processingProgress
  } = useAudioProcessing();

  // Initialize audio devices on mount
  useEffect(() => {
    const initializeAudioDevices = async () => {
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
        dispatch(addProcessingLog({
          level: 'INFO',
          message: `Found ${audioDevices.length} audio input devices`,
          timestamp: Date.now()
        }));
      } catch (error) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: `Failed to load audio devices: ${error}`,
          timestamp: Date.now()
        }));
      }
    };

    initializeAudioDevices();
  }, [dispatch]);

  // Initialize audio context and visualization
  useEffect(() => {
    const initializeAudioContext = async () => {
      try {
        // ✅ Use same device constraints as recording
        const constraints = {
          audio: {
            deviceId: config.deviceId || undefined,
            sampleRate: config.sampleRate,
            channelCount: 1,
            echoCancellation: config.rawAudio ? false : config.echoCancellation,
            noiseSuppression: config.rawAudio ? false : config.noiseSuppression,
            autoGainControl: config.rawAudio ? false : config.autoGainControl,
          }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Clean up previous audio context if it exists
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
          message: `Audio visualization initialized with device: ${config.deviceId || 'default'}`,
          timestamp: Date.now()
        }));
      } catch (error) {
        dispatch(addProcessingLog({
          level: 'ERROR',
          message: `Failed to initialize audio context: ${error}`,
          timestamp: Date.now()
        }));
      }
    };

    initializeAudioContext();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [dispatch, config.deviceId, config.sampleRate, config.echoCancellation, config.noiseSuppression, config.autoGainControl, config.rawAudio]);

  const startVisualization = useCallback(() => {
    if (!analyserRef.current) return;

    const updateVisualization = () => {
      if (!analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const frequencyData = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(frequencyData);

      const timeData = new Uint8Array(analyserRef.current.fftSize);
      analyserRef.current.getByteTimeDomainData(timeData);

      // ✅ Calculate professional meeting-optimized audio metrics
      const audioMetrics = calculateMeetingAudioLevel(timeData, frequencyData, config.sampleRate);
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
        frequency: config.sampleRate,
        clipping: audioMetrics.clipping * 100, // Convert to percentage
        // Meeting-specific metrics
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
  }, [dispatch, config.sampleRate]);

  const handleTabChange = useCallback((_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  }, []);

  const handleStartRecording = useCallback(async () => {
    try {
      await startRecording();
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Recording started',
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Failed to start recording: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [startRecording, dispatch]);

  const handleStopRecording = useCallback(async () => {
    try {
      await stopRecording();
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Recording completed',
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Failed to stop recording: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [stopRecording, dispatch]);

  const handleRunPipeline = useCallback(async () => {
    if (!recording.recordedBlobUrl) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: 'No audio to process. Please record or load audio first.',
        timestamp: Date.now()
      }));
      return;
    }

    try {
      await runPipeline();
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Pipeline processing completed',
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Pipeline processing failed: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [recording.recordedBlobUrl, runPipeline, dispatch]);

  const handleProcessForTranscription = useCallback(async () => {
    if (!recording.recordedBlobUrl) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: 'No audio to process. Please record or load audio first.',
        timestamp: Date.now()
      }));
      return;
    }

    try {
      // Fetch the blob from the URL for processing
      const response = await fetch(recording.recordedBlobUrl);
      const blob = await response.blob();
      const result = await processAudioForTranscription(blob);
      setTranscriptionResult(result.transcription_result || null);
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Audio processing and transcription completed',
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Audio processing for transcription failed: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [recording.recordedBlobUrl, processAudioForTranscription, dispatch]);

  const handleProcessTranscriptionForTranslation = useCallback(async () => {
    if (!transcriptionResult?.text && !transcriptionResult?.transcription) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: 'No transcription available. Please run transcription first.',
        timestamp: Date.now()
      }));
      return;
    }

    try {
      const transcriptionText = transcriptionResult.text || transcriptionResult.transcription || '';
      const sourceLanguage = transcriptionResult.language || 'auto';
      const result = await processTranscriptionForTranslation(transcriptionText, sourceLanguage, targetLanguages);
      setTranslationResults(result.translations || null);
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: `Translation completed for ${targetLanguages.length} languages`,
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Translation failed: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [transcriptionResult, processTranscriptionForTranslation, targetLanguages, dispatch]);

  const handleProcessWithTranslation = useCallback(async () => {
    if (!recording.recordedBlobUrl) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: 'No audio to process. Please record or load audio first.',
        timestamp: Date.now()
      }));
      return;
    }

    try {
      // Fetch the blob from the URL for processing
      const response = await fetch(recording.recordedBlobUrl);
      const blob = await response.blob();
      const result = await processAudioWithTranslation(blob, targetLanguages);
      setTranscriptionResult(result.processing_result || null);
      setTranslationResults(result.translations || null);
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: `Complete audio processing pipeline completed for ${targetLanguages.length} languages`,
        timestamp: Date.now()
      }));
    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Complete audio processing pipeline failed: ${error}`,
        timestamp: Date.now()
      }));
    }
  }, [recording.recordedBlobUrl, processAudioWithTranslation, targetLanguages, dispatch]);

  const handleLanguageToggle = useCallback((language: string) => {
    setTargetLanguages(prev => 
      prev.includes(language) 
        ? prev.filter(lang => lang !== language)
        : [...prev, language]
    );
  }, []);

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🔊 Comprehensive Audio Testing & Pipeline Processing
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Professional audio testing suite with real-time processing pipeline control and comprehensive analysis tools
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          This audio testing interface provides complete control over the audio processing pipeline. 
          Configure recording settings, monitor real-time audio, and test the full processing workflow.
        </Typography>
      </Alert>

      <Paper sx={{ width: '100%', mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="audio testing tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Recording & Configuration" id="audio-testing-tab-0" />
            <Tab label="Pipeline Processing" id="audio-testing-tab-1" />
            <Tab label="Translation Testing" id="audio-testing-tab-2" />
            <Tab label="Presets & Settings" id="audio-testing-tab-3" />
            <Tab label="Activity Logs" id="audio-testing-tab-4" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  🎛️ Recording Configuration
                </Typography>
                <AudioConfiguration />
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  🎵 Audio Recording & Playback
                </Typography>
                <RecordingControls
                  onStartRecording={handleStartRecording}
                  onStopRecording={handleStopRecording}
                  onPlayRecording={playRecording}
                  onDownloadRecording={downloadRecording}
                  onClearRecording={clearRecording}
                  isRecording={recording.isRecording}
                  isPlaying={recording.isPlaying}
                  hasRecording={!!recording.recordedBlobUrl}
                  recordingDuration={recording.duration}
                />
              </Box>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  📊 Audio Visualization
                </Typography>
                <AudioVisualizer
                  frequencyData={visualization.frequencyData}
                  timeData={visualization.timeData}
                  audioLevel={visualization.audioLevel}
                  isRecording={recording.isRecording}
                />
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <PipelineProcessing
            onRunPipeline={handleRunPipeline}
            onRunStepByStep={runStepByStep}
            onResetPipeline={resetPipeline}
            onExportResults={exportResults}
            isProcessing={isProcessing}
            progress={processingProgress}
            hasRecording={!!recording.recordedBlobUrl}
            stages={stages}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🌍 Translation Configuration
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Select target languages for audio translation testing
                  </Typography>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Target Languages:
                  </Typography>
                  <FormGroup row>
                    {[
                      { code: 'es', name: 'Spanish' },
                      { code: 'fr', name: 'French' },
                      { code: 'de', name: 'German' },
                      { code: 'it', name: 'Italian' },
                      { code: 'pt', name: 'Portuguese' },
                      { code: 'ja', name: 'Japanese' },
                      { code: 'ko', name: 'Korean' },
                      { code: 'zh', name: 'Chinese' },
                    ].map((lang) => (
                      <FormControlLabel
                        key={lang.code}
                        control={
                          <Checkbox
                            checked={targetLanguages.includes(lang.code)}
                            onChange={() => handleLanguageToggle(lang.code)}
                            size="small"
                          />
                        }
                        label={lang.name}
                      />
                    ))}
                  </FormGroup>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Selected Languages:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {targetLanguages.length > 0 ? (
                        targetLanguages.map((lang) => (
                          <Chip
                            key={lang}
                            label={lang.toUpperCase()}
                            onDelete={() => handleLanguageToggle(lang)}
                            color="primary"
                            size="small"
                          />
                        ))
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No languages selected
                        </Typography>
                      )}
                    </Box>
                  </Box>
                  
                  <Stack spacing={2}>
                    <Button
                      variant="contained"
                      onClick={handleProcessForTranscription}
                      disabled={!recording.recordedBlobUrl || isProcessing}
                      fullWidth
                    >
                      {isProcessing ? 'Processing...' : '🎤 1. Process Audio + Transcribe'}
                    </Button>
                    
                    <Button
                      variant="outlined"
                      onClick={handleProcessTranscriptionForTranslation}
                      disabled={!transcriptionResult || targetLanguages.length === 0 || isProcessing}
                      fullWidth
                    >
                      {isProcessing ? 'Translating...' : '🌍 2. Translate Transcription'}
                    </Button>
                    
                    <Divider sx={{ my: 1 }}>
                      <Chip label="OR" size="small" />
                    </Divider>
                    
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={handleProcessWithTranslation}
                      disabled={!recording.recordedBlobUrl || targetLanguages.length === 0 || isProcessing}
                      fullWidth
                      sx={{ minHeight: 48 }}
                    >
                      {isProcessing ? 'Processing...' : '🚀 Complete Pipeline (Audio → Translation)'}
                    </Button>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📝 Transcription Result
                  </Typography>
                  {transcriptionResult ? (
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Original Text:
                      </Typography>
                      <Paper sx={{ p: 2, mb: 2, backgroundColor: 'grey.50' }}>
                        <Typography variant="body1">
                          {transcriptionResult.text || transcriptionResult.transcription || 'No transcription available'}
                        </Typography>
                      </Paper>
                      {transcriptionResult.language && (
                        <Typography variant="caption" color="text.secondary">
                          Detected Language: {transcriptionResult.language.toUpperCase()}
                        </Typography>
                      )}
                    </Box>
                  ) : (
                    <Alert severity="info">
                      No transcription available. Record audio and process with translation to see results.
                    </Alert>
                  )}
                </CardContent>
              </Card>
              
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔄 Translation Results
                  </Typography>
                  {translationResults && Object.keys(translationResults).length > 0 ? (
                    <Box>
                      {Object.entries(translationResults).map(([lang, translation]: [string, any]) => (
                        <Box key={lang} sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            {lang.toUpperCase()}:
                          </Typography>
                          <Paper sx={{ p: 2, mb: 1, backgroundColor: 'primary.50' }}>
                            <Typography variant="body1">
                              {translation.translated_text || 'Translation failed'}
                            </Typography>
                          </Paper>
                          {translation.confidence && (
                            <Typography variant="caption" color="text.secondary">
                              Confidence: {(translation.confidence * 100).toFixed(1)}% | 
                              Time: {translation.processing_time?.toFixed(2) || 'N/A'}ms
                            </Typography>
                          )}
                        </Box>
                      ))}
                    </Box>
                  ) : (
                    <Alert severity="info">
                      No translations available. Select target languages and process audio to see translation results.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <ProcessingPresets />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <ActivityLogs />
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default AudioTesting;