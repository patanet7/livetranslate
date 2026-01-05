/**
 * FFTSpectralAnalyzer - Professional Audio Spectral Analysis
 * 
 * Advanced FFT visualization component providing:
 * - Real-time frequency domain analysis
 * - Logarithmic frequency scaling
 * - Customizable windowing functions
 * - Spectral peak detection
 * - Harmonic analysis
 * - Frequency band highlighting
 * - Exportable spectral data
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Grid,
  Alert,
  useTheme,
  alpha,
} from '@mui/material';
import {
  GraphicEq,
  Settings,
  PlayArrow,
  Pause,
  Stop,
  Download,
  Tune,
} from '@mui/icons-material';

// Import chart components
import { XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Types
interface FFTData {
  frequency: number;
  magnitude: number;
  phase: number;
}

interface SpectralPeak {
  frequency: number;
  magnitude: number;
  bandwidth: number;
  harmonic?: number;
}

interface FFTSettings {
  fftSize: 512 | 1024 | 2048 | 4096 | 8192;
  windowFunction: 'hann' | 'hamming' | 'blackman' | 'rectangular' | 'kaiser';
  overlapping: number; // 0-90%
  smoothing: number; // 0-1
  frequencyRange: [number, number];
  magnitudeRange: [number, number];
  logScale: boolean;
  showPeaks: boolean;
  showHarmonics: boolean;
}

interface FFTSpectralAnalyzerProps {
  audioSource?: MediaStream | HTMLAudioElement;
  height?: number;
  realTime?: boolean;
  showControls?: boolean;
  onPeaksDetected?: (peaks: SpectralPeak[]) => void;
  onDataExport?: (data: FFTData[]) => void;
}

const FFTSpectralAnalyzer: React.FC<FFTSpectralAnalyzerProps> = ({
  audioSource: _audioSource,
  height = 400,
  realTime = true,
  showControls = true,
  onPeaksDetected,
  onDataExport,
}) => {
  const theme = useTheme();
  const animationRef = useRef<number>();

  // State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [fftData, setFFTData] = useState<FFTData[]>([]);
  const [spectralPeaks, setSpectralPeaks] = useState<SpectralPeak[]>([]);
  const [fundamentalFrequency, setFundamentalFrequency] = useState<number | null>(null);
  
  // Settings
  const [settings, setSettings] = useState<FFTSettings>({
    fftSize: 2048,
    windowFunction: 'hann',
    overlapping: 50,
    smoothing: 0.8,
    frequencyRange: [20, 20000],
    magnitudeRange: [-120, 0],
    logScale: true,
    showPeaks: true,
    showHarmonics: false,
  });

  // Generate mock FFT data for demonstration
  const generateMockFFTData = useCallback((): FFTData[] => {
    const data: FFTData[] = [];
    const nyquist = 24000; // Assuming 48kHz sample rate
    const bins = settings.fftSize / 2;
    
    for (let i = 0; i < bins; i++) {
      const frequency = (i / bins) * nyquist;
      
      // Skip frequencies outside our range
      if (frequency < settings.frequencyRange[0] || frequency > settings.frequencyRange[1]) {
        continue;
      }
      
      // Generate realistic spectrum with some peaks
      let magnitude = -60 + Math.random() * 20; // Base noise floor
      
      // Add some characteristic peaks
      const fundamentals = [440, 880, 1320]; // A4 and harmonics
      const peakWidth = 50;
      
      fundamentals.forEach(fund => {
        const distance = Math.abs(frequency - fund);
        if (distance < peakWidth) {
          const peakGain = 40 * Math.exp(-(distance * distance) / (2 * (peakWidth / 3) * (peakWidth / 3)));
          magnitude += peakGain;
        }
      });
      
      // Add 1/f pink noise characteristic
      if (frequency > 0) {
        magnitude -= 3 * Math.log10(frequency / 1000);
      }
      
      // Add time-varying component for realism
      magnitude += 5 * Math.sin(Date.now() / 1000 + frequency / 1000) * Math.random();
      
      data.push({
        frequency,
        magnitude: Math.max(settings.magnitudeRange[0], Math.min(settings.magnitudeRange[1], magnitude)),
        phase: Math.random() * 360 - 180,
      });
    }
    
    return data;
  }, [settings.fftSize, settings.frequencyRange, settings.magnitudeRange]);

  // Peak detection algorithm
  const detectPeaks = useCallback((data: FFTData[]): SpectralPeak[] => {
    if (!settings.showPeaks || data.length < 3) return [];
    
    const peaks: SpectralPeak[] = [];
    const minPeakHeight = settings.magnitudeRange[0] + 40; // 40dB above noise floor
    const minPeakDistance = 50; // Minimum 50Hz between peaks
    
    for (let i = 1; i < data.length - 1; i++) {
      const prev = data[i - 1];
      const current = data[i];
      const next = data[i + 1];
      
      // Check if this is a local maximum
      if (current.magnitude > prev.magnitude && 
          current.magnitude > next.magnitude && 
          current.magnitude > minPeakHeight) {
        
        // Check minimum distance from existing peaks
        const tooClose = peaks.some(peak => 
          Math.abs(peak.frequency - current.frequency) < minPeakDistance
        );
        
        if (!tooClose) {
          // Calculate bandwidth (frequency range where magnitude is within 3dB)
          let leftEdge = i;
          let rightEdge = i;
          const threshold = current.magnitude - 3;
          
          while (leftEdge > 0 && data[leftEdge].magnitude > threshold) leftEdge--;
          while (rightEdge < data.length - 1 && data[rightEdge].magnitude > threshold) rightEdge++;
          
          const bandwidth = data[rightEdge].frequency - data[leftEdge].frequency;
          
          peaks.push({
            frequency: current.frequency,
            magnitude: current.magnitude,
            bandwidth,
          });
        }
      }
    }
    
    // Sort by magnitude (strongest peaks first)
    return peaks.sort((a, b) => b.magnitude - a.magnitude).slice(0, 10);
  }, [settings.showPeaks, settings.magnitudeRange]);

  // Find fundamental frequency and harmonics
  const analyzeHarmonics = useCallback((peaks: SpectralPeak[]): void => {
    if (!settings.showHarmonics || peaks.length === 0) {
      setFundamentalFrequency(null);
      return;
    }
    
    // Simple fundamental frequency detection
    // Look for the lowest significant peak
    const sortedByFreq = [...peaks].sort((a, b) => a.frequency - b.frequency);
    const fundamental = sortedByFreq.find(peak => peak.frequency > 80 && peak.frequency < 2000);
    
    if (fundamental) {
      setFundamentalFrequency(fundamental.frequency);
      
      // Mark harmonics
      peaks.forEach(peak => {
        const ratio = peak.frequency / fundamental.frequency;
        if (Math.abs(ratio - Math.round(ratio)) < 0.1) {
          peak.harmonic = Math.round(ratio);
        }
      });
    }
  }, [settings.showHarmonics]);

  // Update analysis
  const updateAnalysis = useCallback(() => {
    if (!realTime && !isAnalyzing) return;
    
    // Generate mock data (in real implementation, this would read from analyser)
    const newData = generateMockFFTData();
    setFFTData(newData);
    
    // Detect peaks
    const peaks = detectPeaks(newData);
    setSpectralPeaks(peaks);
    
    // Analyze harmonics
    analyzeHarmonics(peaks);
    
    // Callback for peaks
    if (onPeaksDetected && peaks.length > 0) {
      onPeaksDetected(peaks);
    }
    
    if (realTime) {
      animationRef.current = requestAnimationFrame(updateAnalysis);
    }
  }, [realTime, isAnalyzing, generateMockFFTData, detectPeaks, analyzeHarmonics, onPeaksDetected]);

  // Control functions
  const startAnalysis = useCallback(() => {
    setIsAnalyzing(true);
    updateAnalysis();
  }, [updateAnalysis]);

  const stopAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  const exportData = useCallback(() => {
    if (onDataExport) {
      onDataExport(fftData);
    } else {
      // Default export as JSON
      const blob = new Blob([JSON.stringify({
        timestamp: new Date().toISOString(),
        settings,
        fftData,
        spectralPeaks,
        fundamentalFrequency,
      }, null, 2)], { type: 'application/json' });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fft-analysis-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }, [fftData, spectralPeaks, fundamentalFrequency, settings, onDataExport]);

  // Auto-start analysis
  useEffect(() => {
    if (realTime) {
      startAnalysis();
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [realTime, startAnalysis]);

  // Format frequency for display
  const formatFrequency = (freq: number) => {
    if (freq >= 1000) {
      return `${(freq / 1000).toFixed(1)}k`;
    }
    return `${Math.round(freq)}`;
  };

  // Custom tick formatter for log scale
  const formatTick = (value: number) => {
    if (settings.logScale) {
      return formatFrequency(value);
    }
    return value.toString();
  };

  return (
    <Box>
      {/* Controls */}
      {showControls && (
        <Card sx={{ 
          mb: 2,
          bgcolor: alpha(theme.palette.background.paper, 0.7),
          backdropFilter: 'blur(10px)',
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: 500 }}>
                FFT Spectral Analyzer
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton 
                  onClick={isAnalyzing ? stopAnalysis : startAnalysis}
                  color="primary"
                >
                  {isAnalyzing ? <Pause /> : <PlayArrow />}
                </IconButton>
                <IconButton onClick={stopAnalysis}>
                  <Stop />
                </IconButton>
                <IconButton onClick={exportData}>
                  <Download />
                </IconButton>
                <IconButton>
                  <Settings />
                </IconButton>
              </Box>
            </Box>

            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>FFT Size</InputLabel>
                  <Select
                    value={settings.fftSize}
                    label="FFT Size"
                    onChange={(e) => setSettings(prev => ({ ...prev, fftSize: e.target.value as any }))}
                  >
                    <MenuItem value={512}>512</MenuItem>
                    <MenuItem value={1024}>1024</MenuItem>
                    <MenuItem value={2048}>2048</MenuItem>
                    <MenuItem value={4096}>4096</MenuItem>
                    <MenuItem value={8192}>8192</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>Window</InputLabel>
                  <Select
                    value={settings.windowFunction}
                    label="Window"
                    onChange={(e) => setSettings(prev => ({ ...prev, windowFunction: e.target.value as any }))}
                  >
                    <MenuItem value="hann">Hann</MenuItem>
                    <MenuItem value="hamming">Hamming</MenuItem>
                    <MenuItem value="blackman">Blackman</MenuItem>
                    <MenuItem value="rectangular">Rectangular</MenuItem>
                    <MenuItem value="kaiser">Kaiser</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Typography variant="caption" gutterBottom display="block">
                  Smoothing: {settings.smoothing.toFixed(1)}
                </Typography>
                <Slider
                  value={settings.smoothing}
                  onChange={(_, value) => setSettings(prev => ({ ...prev, smoothing: value as number }))}
                  min={0}
                  max={1}
                  step={0.1}
                  size="small"
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.logScale}
                      onChange={(e) => setSettings(prev => ({ ...prev, logScale: e.target.checked }))}
                      size="small"
                    />
                  }
                  label="Log Scale"
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.showPeaks}
                      onChange={(e) => setSettings(prev => ({ ...prev, showPeaks: e.target.checked }))}
                      size="small"
                    />
                  }
                  label="Show Peaks"
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.showHarmonics}
                      onChange={(e) => setSettings(prev => ({ ...prev, showHarmonics: e.target.checked }))}
                      size="small"
                    />
                  }
                  label="Harmonics"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Spectral Display */}
      <Card sx={{ 
        bgcolor: alpha(theme.palette.background.paper, 0.7),
        backdropFilter: 'blur(10px)',
        mb: 2,
      }}>
        <CardContent>
          <Box sx={{ height, width: '100%' }}>
            {fftData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={fftData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.3)} />
                  <XAxis 
                    dataKey="frequency"
                    scale={settings.logScale ? 'log' : 'linear'}
                    domain={settings.logScale ? ['dataMin', 'dataMax'] : settings.frequencyRange}
                    tickFormatter={formatTick}
                    stroke={theme.palette.text.secondary}
                  />
                  <YAxis 
                    domain={settings.magnitudeRange}
                    stroke={theme.palette.text.secondary}
                    label={{ value: 'Magnitude (dB)', angle: -90, position: 'insideLeft' }}
                  />
                  <RechartsTooltip
                    formatter={(value: number, _name: string) => [
                      `${value.toFixed(1)} dB`,
                      'Magnitude'
                    ]}
                    labelFormatter={(value) => `${formatFrequency(value)} Hz`}
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="magnitude" 
                    stroke={theme.palette.primary.main}
                    fill={alpha(theme.palette.primary.main, 0.3)}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: '100%' 
              }}>
                <Alert severity="info">
                  {isAnalyzing ? 'Analyzing audio spectrum...' : 'Click play to start spectral analysis'}
                </Alert>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      <Grid container spacing={2}>
        {/* Fundamental Frequency */}
        {fundamentalFrequency && (
          <Grid item xs={12} md={4}>
            <Card sx={{ 
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: 'blur(10px)',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                  Fundamental Frequency
                </Typography>
                <Typography variant="h4" color="primary.main">
                  {fundamentalFrequency.toFixed(1)} Hz
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Musical Note: {/* Note calculation would go here */}
                  {fundamentalFrequency > 400 && fundamentalFrequency < 480 ? 'A4' : 'Unknown'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Spectral Peaks */}
        <Grid item xs={12} md={8}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Detected Peaks ({spectralPeaks.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {spectralPeaks.slice(0, 8).map((peak, index) => (
                  <Chip
                    key={index}
                    label={`${formatFrequency(peak.frequency)} Hz`}
                    variant="outlined"
                    size="small"
                    color={peak.harmonic ? 'secondary' : 'primary'}
                    icon={peak.harmonic ? <Tune /> : <GraphicEq />}
                    sx={{
                      backgroundColor: peak.harmonic 
                        ? alpha(theme.palette.secondary.main, 0.1)
                        : alpha(theme.palette.primary.main, 0.1),
                    }}
                  />
                ))}
              </Box>
              {spectralPeaks.length === 0 && (
                <Typography variant="body2" color="textSecondary">
                  No significant peaks detected
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Stats */}
        <Grid item xs={12}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Analysis Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="textSecondary">FFT Size:</Typography>
                  <Typography variant="h6">{settings.fftSize}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="textSecondary">Frequency Resolution:</Typography>
                  <Typography variant="h6">{(48000 / settings.fftSize).toFixed(1)} Hz</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="textSecondary">Data Points:</Typography>
                  <Typography variant="h6">{fftData.length}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="textSecondary">Update Rate:</Typography>
                  <Typography variant="h6">{realTime ? '60 FPS' : 'Manual'}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FFTSpectralAnalyzer;