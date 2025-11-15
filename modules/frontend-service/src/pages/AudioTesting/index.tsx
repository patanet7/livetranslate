import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tabs,
  Tab,
  Alert,
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
import { addProcessingLog } from '@/store/slices/audioSlice';
import { RecordingControls } from './components/RecordingControls';
import { AudioConfiguration } from './components/AudioConfiguration';
import { AudioVisualizer } from './components/AudioVisualizer';
import { PipelineProcessing } from './components/PipelineProcessing';
import { ProcessingPresets } from './components/ProcessingPresets';
import { ActivityLogs } from './components/ActivityLogs';
import { useAudioProcessing } from '@/hooks/useAudioProcessing';
import { useAudioDevices } from '@/hooks/useAudioDevices';
import { useAudioVisualization } from '@/hooks/useAudioVisualization';
import { TabPanel } from '@/components/ui';
import { DEFAULT_TARGET_LANGUAGES } from '@/constants/defaultConfig';
import { SUPPORTED_LANGUAGES } from '@/constants/languages';

const AudioTesting: React.FC = () => {
  const dispatch = useAppDispatch();
  const { recording, visualization, stages, config } = useAppSelector(state => state.audio);
  const [tabValue, setTabValue] = useState(0);
  const [targetLanguages, setTargetLanguages] = useState<string[]>([...DEFAULT_TARGET_LANGUAGES]);
  const [translationResults, setTranslationResults] = useState<any>(null);
  const [transcriptionResult, setTranscriptionResult] = useState<any>(null);

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

  // Initialize audio devices
  useAudioDevices();

  // Compute audio constraints based on config
  const audioConstraints = useMemo(() => ({
    deviceId: config.deviceId || undefined,
    sampleRate: config.sampleRate,
    channelCount: 1,
    echoCancellation: config.rawAudio ? false : config.echoCancellation,
    noiseSuppression: config.rawAudio ? false : config.noiseSuppression,
    autoGainControl: config.rawAudio ? false : config.autoGainControl,
  }), [config.deviceId, config.sampleRate, config.echoCancellation, config.noiseSuppression, config.autoGainControl, config.rawAudio]);

  // Initialize audio visualization with shared hook
  useAudioVisualization({
    sampleRate: config.sampleRate,
    customConstraints: audioConstraints,
    enableLogging: true
  });

  const handleTabChange = useCallback((__event: React.SyntheticEvent, newValue: number) => {
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

        <TabPanel value={tabValue} index={0} idPrefix="audio-testing">
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

        <TabPanel value={tabValue} index={1} idPrefix="audio-testing">
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

        <TabPanel value={tabValue} index={2} idPrefix="audio-testing">
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
                    {SUPPORTED_LANGUAGES.slice(1, 9).map((lang) => (
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

        <TabPanel value={tabValue} index={3} idPrefix="audio-testing">
          <ProcessingPresets />
        </TabPanel>

        <TabPanel value={tabValue} index={4} idPrefix="audio-testing">
          <ActivityLogs />
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default AudioTesting;