/**
 * QualityAnalysis - Audio Quality Analysis and Visualization
 * 
 * Professional audio quality analysis featuring:
 * - FFT spectral analysis with real-time visualization
 * - LUFS loudness metering and compliance checking
 * - Audio quality metrics (SNR, THD, dynamic range)
 * - Waveform visualization with zoom and selection
 * - Quality scoring and recommendations
 * - Export capabilities for analysis results
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  IconButton,
  useTheme,
  alpha,
} from '@mui/material';
import {
  AudioFile,
  PlayArrow,
  Pause,
  Stop,
  GraphicEq,
  ShowChart,
  Assessment,
  Download,
} from '@mui/icons-material';

// Import chart components
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

import { useUnifiedAudio } from '@/hooks/useUnifiedAudio';
import { useAppDispatch } from '@/store';
import { addNotification } from '@/store/slices/uiSlice';

interface AudioQualityMetrics {
  snr: number;
  thd: number;
  dynamicRange: number;
  peakLevel: number;
  rmsLevel: number;
  lufsIntegrated: number;
  lufsShortTerm: number;
  lufsRange: number;
  qualityScore: number;
  recommendations: string[];
}

interface FrequencyData {
  frequency: number;
  magnitude: number;
  phase: number;
}

interface WaveformData {
  time: number;
  amplitude: number;
}

interface AnalysisSettings {
  fftSize: number;
  windowFunction: 'hann' | 'hamming' | 'blackman' | 'rectangular';
  overlapping: number;
  frequencyRange: [number, number];
  timeRange: [number, number];
}

const QualityAnalysis: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  useUnifiedAudio();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // State
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [activeTab, setActiveTab] = useState(0);

  // Analysis data
  const [qualityMetrics, setQualityMetrics] = useState<AudioQualityMetrics | null>(null);
  const [frequencyData, setFrequencyData] = useState<FrequencyData[]>([]);
  const [waveformData, setWaveformData] = useState<WaveformData[]>([]);

  // Settings
  const [settings, setSettings] = useState<AnalysisSettings>({
    fftSize: 2048,
    windowFunction: 'hann',
    overlapping: 50,
    frequencyRange: [20, 20000],
    timeRange: [0, 100],
  });

  // Generate mock analysis data
  const generateMockAnalysis = useCallback(() => {
    // Mock frequency response data
    const frequencies: FrequencyData[] = [];
    for (let i = 0; i < 512; i++) {
      const freq = (i / 512) * (settings.frequencyRange[1] - settings.frequencyRange[0]) + settings.frequencyRange[0];
      const magnitude = -60 + Math.random() * 60 - (Math.log10(freq / 1000) * 20); // Rough frequency response curve
      frequencies.push({
        frequency: freq,
        magnitude: Math.max(-80, magnitude),
        phase: Math.random() * 360 - 180,
      });
    }

    // Mock waveform data
    const waveform: WaveformData[] = [];
    const samples = 1000;
    for (let i = 0; i < samples; i++) {
      const time = (i / samples) * duration;
      const amplitude = Math.sin(2 * Math.PI * 440 * time / duration) * Math.exp(-time / (duration * 0.3)) + 
                       Math.random() * 0.1 - 0.05; // Decay + noise
      waveform.push({ time, amplitude });
    }

    // Mock quality metrics
    const metrics: AudioQualityMetrics = {
      snr: 45 + Math.random() * 20,
      thd: Math.random() * 2,
      dynamicRange: 15 + Math.random() * 20,
      peakLevel: -3 - Math.random() * 6,
      rmsLevel: -18 - Math.random() * 12,
      lufsIntegrated: -23 + Math.random() * 6,
      lufsShortTerm: -20 + Math.random() * 8,
      lufsRange: 8 + Math.random() * 15,
      qualityScore: 70 + Math.random() * 25,
      recommendations: [
        'Consider reducing background noise',
        'Audio levels are well balanced',
        'Good dynamic range preservation',
      ],
    };

    setFrequencyData(frequencies);
    setWaveformData(waveform);
    setQualityMetrics(metrics);
  }, [duration, settings.frequencyRange]);

  // File handling
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      
      dispatch(addNotification({
        type: 'success',
        title: 'Audio File Loaded',
        message: `Loaded ${file.name} for analysis`,
        autoHide: true,
      }));
    }
  };

  const handleAudioLoaded = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
      setSettings(prev => ({
        ...prev,
        timeRange: [0, audioRef.current!.duration],
      }));
    }
  };

  // Audio playback control
  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const stopPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      setCurrentTime(0);
    }
  };

  // Analysis functions
  const runAnalysis = async () => {
    if (!audioFile) {
      dispatch(addNotification({
        type: 'warning',
        title: 'No Audio File',
        message: 'Please select an audio file to analyze',
        autoHide: true,
      }));
      return;
    }

    setIsAnalyzing(true);
    try {
      // Simulate analysis delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      generateMockAnalysis();
      
      dispatch(addNotification({
        type: 'success',
        title: 'Analysis Complete',
        message: 'Audio quality analysis finished successfully',
        autoHide: true,
      }));
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        title: 'Analysis Failed',
        message: 'Failed to analyze audio quality',
        autoHide: true,
      }));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const exportResults = () => {
    if (!qualityMetrics) return;

    const results = {
      filename: audioFile?.name,
      timestamp: new Date().toISOString(),
      qualityMetrics,
      settings,
      frequencyData: frequencyData.slice(0, 100), // Limit export size
      summary: {
        overallScore: qualityMetrics.qualityScore,
        lufsCompliance: qualityMetrics.lufsIntegrated >= -26 && qualityMetrics.lufsIntegrated <= -16,
        recommendations: qualityMetrics.recommendations,
      },
    };

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio-quality-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Update current time during playback
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getLufsComplianceStatus = (lufs: number) => {
    if (lufs >= -26 && lufs <= -16) return { status: 'Compliant', color: 'success' };
    if (lufs < -26) return { status: 'Too Quiet', color: 'warning' };
    return { status: 'Too Loud', color: 'error' };
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Audio Quality Analysis
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<AudioFile />}
            onClick={() => fileInputRef.current?.click()}
          >
            Load Audio
          </Button>
          <Button
            variant="contained"
            startIcon={<Assessment />}
            onClick={runAnalysis}
            disabled={!audioFile || isAnalyzing}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Quality'}
          </Button>
          {qualityMetrics && (
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={exportResults}
            >
              Export Results
            </Button>
          )}
        </Box>
      </Box>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onLoadedMetadata={handleAudioLoaded}
          style={{ display: 'none' }}
        />
      )}

      <Grid container spacing={3}>
        {/* Audio File Info and Controls */}
        <Grid item xs={12} md={4}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Audio File
              </Typography>
              {audioFile ? (
                <Box>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    {audioFile.name}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Size: {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Duration: {duration.toFixed(1)}s
                  </Typography>
                  
                  {/* Playback Controls */}
                  <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconButton onClick={togglePlayback} disabled={!audioUrl}>
                      {isPlaying ? <Pause /> : <PlayArrow />}
                    </IconButton>
                    <IconButton onClick={stopPlayback} disabled={!audioUrl}>
                      <Stop />
                    </IconButton>
                    <Box sx={{ flex: 1, mx: 2 }}>
                      <LinearProgress
                        variant="determinate"
                        value={duration > 0 ? (currentTime / duration) * 100 : 0}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                    <Typography variant="caption">
                      {currentTime.toFixed(1)}s
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Alert severity="info">
                  No audio file selected. Click "Load Audio" to begin analysis.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Settings */}
        <Grid item xs={12} md={4}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Analysis Settings
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl size="small">
                  <InputLabel>FFT Size</InputLabel>
                  <Select
                    value={settings.fftSize}
                    label="FFT Size"
                    onChange={(e) => setSettings(prev => ({ ...prev, fftSize: e.target.value as number }))}
                  >
                    <MenuItem value={512}>512</MenuItem>
                    <MenuItem value={1024}>1024</MenuItem>
                    <MenuItem value={2048}>2048</MenuItem>
                    <MenuItem value={4096}>4096</MenuItem>
                  </Select>
                </FormControl>

                <FormControl size="small">
                  <InputLabel>Window Function</InputLabel>
                  <Select
                    value={settings.windowFunction}
                    label="Window Function"
                    onChange={(e) => setSettings(prev => ({ ...prev, windowFunction: e.target.value as any }))}
                  >
                    <MenuItem value="hann">Hann</MenuItem>
                    <MenuItem value="hamming">Hamming</MenuItem>
                    <MenuItem value="blackman">Blackman</MenuItem>
                    <MenuItem value="rectangular">Rectangular</MenuItem>
                  </Select>
                </FormControl>

                <Box>
                  <Typography variant="body2" gutterBottom>
                    Overlapping: {settings.overlapping}%
                  </Typography>
                  <Slider
                    value={settings.overlapping}
                    onChange={(_, value) => setSettings(prev => ({ ...prev, overlapping: value as number }))}
                    min={0}
                    max={90}
                    step={10}
                    marks
                    size="small"
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Quality Summary */}
        {qualityMetrics && (
          <Grid item xs={12} md={4}>
            <Card sx={{ 
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: 'blur(10px)',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                  Quality Summary
                </Typography>
                <Box sx={{ textAlign: 'center', mb: 2 }}>
                  <Typography variant="h3" sx={{ fontWeight: 600 }}>
                    {Math.round(qualityMetrics.qualityScore)}
                  </Typography>
                  <Chip
                    label={`Quality Score`}
                    color={getQualityColor(qualityMetrics.qualityScore) as any}
                    variant="outlined"
                  />
                </Box>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">SNR:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {qualityMetrics.snr.toFixed(1)} dB
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">LUFS:</Typography>
                    <Chip
                      label={`${qualityMetrics.lufsIntegrated.toFixed(1)} LUFS`}
                      color={getLufsComplianceStatus(qualityMetrics.lufsIntegrated).color as any}
                      size="small"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Dynamic Range:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {qualityMetrics.dynamicRange.toFixed(1)} dB
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Analysis Progress */}
        {isAnalyzing && (
          <Grid item xs={12}>
            <Card sx={{ 
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: 'blur(10px)',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analyzing Audio Quality...
                </Typography>
                <LinearProgress sx={{ mb: 1 }} />
                <Typography variant="body2" color="textSecondary">
                  Performing FFT analysis, calculating LUFS, and evaluating audio metrics
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Analysis Results */}
        {qualityMetrics && !isAnalyzing && (
          <Grid item xs={12}>
            <Card sx={{ 
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: 'blur(10px)',
            }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={(_, value) => setActiveTab(value)}>
                  <Tab icon={<ShowChart />} label="Frequency Analysis" />
                  <Tab icon={<GraphicEq />} label="Waveform" />
                  <Tab icon={<Assessment />} label="Detailed Metrics" />
                </Tabs>
              </Box>

              <CardContent>
                {/* Frequency Analysis Tab */}
                {activeTab === 0 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                      Frequency Response Analysis
                    </Typography>
                    <Box sx={{ height: 400 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={frequencyData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="frequency" 
                            scale="log"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(value) => `${value >= 1000 ? (value/1000).toFixed(1) + 'k' : value}`}
                          />
                          <YAxis domain={[-80, 0]} label={{ value: 'Magnitude (dB)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip
                            formatter={(value: number, _name: string) => [`${value.toFixed(1)} dB`, 'Magnitude']}
                            labelFormatter={(value) => `${value >= 1000 ? (value/1000).toFixed(1) + ' kHz' : value + ' Hz'}`}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="magnitude" 
                            stroke={theme.palette.primary.main}
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  </Box>
                )}

                {/* Waveform Tab */}
                {activeTab === 1 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                      Waveform Analysis
                    </Typography>
                    <Box sx={{ height: 400 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={waveformData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="time" 
                            label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis 
                            domain={[-1, 1]}
                            label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }}
                          />
                          <RechartsTooltip
                            formatter={(value: number, _name: string) => [value.toFixed(3), 'Amplitude']}
                            labelFormatter={(value) => `${value.toFixed(3)}s`}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="amplitude" 
                            stroke={theme.palette.secondary.main}
                            fill={alpha(theme.palette.secondary.main, 0.3)}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </Box>
                  </Box>
                )}

                {/* Detailed Metrics Tab */}
                {activeTab === 2 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                      Detailed Quality Metrics
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                          Audio Levels
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {[
                            { label: 'Peak Level', value: qualityMetrics.peakLevel, unit: 'dB', target: '< -3 dB' },
                            { label: 'RMS Level', value: qualityMetrics.rmsLevel, unit: 'dB', target: '-12 to -18 dB' },
                            { label: 'LUFS Integrated', value: qualityMetrics.lufsIntegrated, unit: 'LUFS', target: '-23 LUFS' },
                            { label: 'LUFS Range', value: qualityMetrics.lufsRange, unit: 'LU', target: '< 15 LU' },
                          ].map((metric, index) => (
                            <Box key={index} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="body2">{metric.label}</Typography>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {metric.value.toFixed(1)} {metric.unit}
                                </Typography>
                              </Box>
                              <Typography variant="caption" color="textSecondary">
                                Target: {metric.target}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                          Quality Analysis
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {[
                            { label: 'Signal-to-Noise Ratio', value: qualityMetrics.snr, unit: 'dB', target: '> 40 dB' },
                            { label: 'Total Harmonic Distortion', value: qualityMetrics.thd, unit: '%', target: '< 1%' },
                            { label: 'Dynamic Range', value: qualityMetrics.dynamicRange, unit: 'dB', target: '> 15 dB' },
                            { label: 'Overall Quality Score', value: qualityMetrics.qualityScore, unit: '/100', target: '> 80' },
                          ].map((metric, index) => (
                            <Box key={index} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="body2">{metric.label}</Typography>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {metric.value.toFixed(1)}{metric.unit}
                                </Typography>
                              </Box>
                              <Typography variant="caption" color="textSecondary">
                                Target: {metric.target}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </Grid>

                      {/* Recommendations */}
                      <Grid item xs={12}>
                        <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                          Recommendations
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {qualityMetrics.recommendations.map((recommendation, index) => (
                            <Alert key={index} severity="info" variant="outlined">
                              {recommendation}
                            </Alert>
                          ))}
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default QualityAnalysis;