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
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Alert,
  Slider,
  Grid,
} from '@mui/material';
import {
  Equalizer,
  PlayArrow,
  Pause,
  Download,
  ShowChart,
} from '@mui/icons-material';

export interface FFTVisualizerProps {
  audioData?: {
    fftData: Float32Array;
    sampleRate: number;
    windowSize: number;
    timestamp: number;
  };
  isRealTime?: boolean;
  onAnalysisUpdate?: (analysis: FrequencyAnalysis) => void;
  height?: number;
  showControls?: boolean;
}

interface FrequencyAnalysis {
  fundamental_frequency: number;
  spectral_centroid: number;
  spectral_rolloff: number;
  spectral_bandwidth: number;
  spectral_flatness: number;
  peak_frequencies: number[];
  energy_distribution: {
    low: number;    // 0-250 Hz
    mid: number;    // 250-2000 Hz
    high: number;   // 2000+ Hz
  };
  voice_presence: number;
  noise_floor_db: number;
}

type FFTDisplayMode = 'magnitude' | 'power' | 'log_magnitude' | 'mel_scale';
type WindowFunction = 'hanning' | 'hamming' | 'blackman' | 'rectangular';

export const FFTVisualizer: React.FC<FFTVisualizerProps> = ({
  audioData,
  isRealTime = false,
  onAnalysisUpdate,
  height = 300,
  showControls = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  
  const [displayMode, setDisplayMode] = useState<FFTDisplayMode>('magnitude');
  const [windowFunction, setWindowFunction] = useState<WindowFunction>('hanning');
  const [isPlaying, setIsPlaying] = useState(isRealTime);
  const [smoothing, setSmoothing] = useState(0.8);
  const [gainAdjustment, setGainAdjustment] = useState(0);
  const [frequencyRange] = useState<[number, number]>([20, 8000]);
  const [showPeaks, setShowPeaks] = useState(true);
  const [showVoiceRegion, setShowVoiceRegion] = useState(true);
  const [currentAnalysis, setCurrentAnalysis] = useState<FrequencyAnalysis | null>(null);
  
  // Smoothed FFT data for display
  const [smoothedFFT, setSmoothedFFT] = useState<Float32Array | null>(null);
  const [peakFrequencies, setPeakFrequencies] = useState<number[]>([]);

  const drawFFTSpectrum = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height: canvasHeight } = canvas;
    const { fftData, sampleRate } = audioData;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, canvasHeight);
    
    // Apply smoothing
    if (!smoothedFFT || smoothedFFT.length !== fftData.length) {
      setSmoothedFFT(new Float32Array(fftData));
      return;
    }
    
    for (let i = 0; i < fftData.length; i++) {
      smoothedFFT[i] = smoothedFFT[i] * smoothing + fftData[i] * (1 - smoothing);
    }
    
    // Calculate frequency bins
    const nyquist = sampleRate / 2;
    const binWidth = nyquist / (fftData.length / 2);
    const startBin = Math.floor(frequencyRange[0] / binWidth);
    const endBin = Math.min(Math.floor(frequencyRange[1] / binWidth), fftData.length / 2);
    
    // Process data based on display mode
    const processedData = new Float32Array(endBin - startBin);
    const gainLinear = Math.pow(10, gainAdjustment / 20);
    
    for (let i = startBin; i < endBin; i++) {
      let value = smoothedFFT[i] * gainLinear;
      
      switch (displayMode) {
        case 'magnitude':
          processedData[i - startBin] = value;
          break;
        case 'power':
          processedData[i - startBin] = value * value;
          break;
        case 'log_magnitude':
          processedData[i - startBin] = Math.log10(Math.max(value, 1e-10));
          break;
        case 'mel_scale':
          // Simplified mel scale conversion
          processedData[i - startBin] = value;
          break;
      }
    }
    
    // Find max value for normalization
    const maxValue = Math.max(...processedData);
    const minValue = displayMode === 'log_magnitude' ? Math.min(...processedData) : 0;
    const range = maxValue - minValue;
    
    // Draw spectrum
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    const barWidth = width / processedData.length;
    
    for (let i = 0; i < processedData.length; i++) {
      const x = i * barWidth;
      const normalizedValue = range > 0 ? (processedData[i] - minValue) / range : 0;
      const y = canvasHeight - (normalizedValue * canvasHeight * 0.9);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Fill area under curve
    ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
    ctx.lineTo(width, canvasHeight);
    ctx.lineTo(0, canvasHeight);
    ctx.closePath();
    ctx.fill();
    
    // Draw voice frequency region if enabled
    if (showVoiceRegion) {
      const voiceStart = Math.max(0, (85 - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0]) * width);
      const voiceEnd = Math.min(width, (300 - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0]) * width);
      
      ctx.fillStyle = 'rgba(255, 193, 7, 0.1)';
      ctx.fillRect(voiceStart, 0, voiceEnd - voiceStart, canvasHeight);
      
      ctx.strokeStyle = 'rgba(255, 193, 7, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(voiceStart, 0);
      ctx.lineTo(voiceStart, canvasHeight);
      ctx.moveTo(voiceEnd, 0);
      ctx.lineTo(voiceEnd, canvasHeight);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw peak frequencies if enabled
    if (showPeaks && peakFrequencies.length > 0) {
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([2, 2]);
      
      peakFrequencies.forEach(freq => {
        if (freq >= frequencyRange[0] && freq <= frequencyRange[1]) {
          const x = (freq - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0]) * width;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvasHeight);
          ctx.stroke();
        }
      });
      
      ctx.setLineDash([]);
    }
    
    // Draw frequency labels
    ctx.fillStyle = '#888888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    
    const labelCount = 8;
    for (let i = 0; i <= labelCount; i++) {
      const freq = frequencyRange[0] + (i / labelCount) * (frequencyRange[1] - frequencyRange[0]);
      const x = (i / labelCount) * width;
      
      ctx.fillText(
        freq < 1000 ? `${Math.round(freq)}Hz` : `${(freq / 1000).toFixed(1)}kHz`,
        x,
        canvasHeight - 5
      );
    }
    
    // Draw amplitude labels
    ctx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * canvasHeight;
      const amplitude = displayMode === 'log_magnitude' 
        ? (minValue + (1 - i / 4) * range).toFixed(1)
        : ((1 - i / 4) * 100).toFixed(0) + '%';
      
      ctx.fillText(amplitude, 5, y + 12);
    }
    
  }, [audioData, smoothedFFT, displayMode, smoothing, gainAdjustment, frequencyRange, showPeaks, showVoiceRegion, peakFrequencies]);

  const performFrequencyAnalysis = useCallback(() => {
    if (!audioData || !smoothedFFT) return;

    const { sampleRate } = audioData;
    const fftData = smoothedFFT;
    const nyquist = sampleRate / 2;
    const binWidth = nyquist / (fftData.length / 2);
    
    // Find peaks
    const peaks: number[] = [];
    const threshold = 0.1;
    
    for (let i = 2; i < fftData.length / 2 - 2; i++) {
      if (fftData[i] > threshold &&
          fftData[i] > fftData[i-1] && fftData[i] > fftData[i+1] &&
          fftData[i] > fftData[i-2] && fftData[i] > fftData[i+2]) {
        peaks.push(i * binWidth);
      }
    }
    
    setPeakFrequencies(peaks.slice(0, 10)); // Top 10 peaks
    
    // Calculate spectral features
    let spectralCentroid = 0;
    let totalMagnitude = 0;

    for (let i = 1; i < fftData.length / 2; i++) {
      const frequency = i * binWidth;
      const magnitude = fftData[i];
      
      spectralCentroid += frequency * magnitude;
      totalMagnitude += magnitude;
    }
    
    spectralCentroid = totalMagnitude > 0 ? spectralCentroid / totalMagnitude : 0;
    
    // Calculate spectral rolloff (85% of energy)
    let cumulativeEnergy = 0;
    const totalEnergy = totalMagnitude;
    let rolloffFreq = 0;
    
    for (let i = 1; i < fftData.length / 2; i++) {
      cumulativeEnergy += fftData[i];
      if (cumulativeEnergy >= 0.85 * totalEnergy) {
        rolloffFreq = i * binWidth;
        break;
      }
    }
    
    // Energy distribution
    let lowEnergy = 0, midEnergy = 0, highEnergy = 0;
    
    for (let i = 1; i < fftData.length / 2; i++) {
      const frequency = i * binWidth;
      const magnitude = fftData[i];
      
      if (frequency < 250) lowEnergy += magnitude;
      else if (frequency < 2000) midEnergy += magnitude;
      else highEnergy += magnitude;
    }
    
    const energySum = lowEnergy + midEnergy + highEnergy;
    
    // Voice presence (energy in 85-300 Hz range)
    let voiceEnergy = 0;
    const voiceStartBin = Math.floor(85 / binWidth);
    const voiceEndBin = Math.floor(300 / binWidth);
    
    for (let i = voiceStartBin; i <= voiceEndBin; i++) {
      voiceEnergy += fftData[i];
    }
    
    const analysis: FrequencyAnalysis = {
      fundamental_frequency: peaks[0] || 0,
      spectral_centroid: spectralCentroid,
      spectral_rolloff: rolloffFreq,
      spectral_bandwidth: 0, // Simplified for now
      spectral_flatness: 0,  // Simplified for now
      peak_frequencies: peaks.slice(0, 5),
      energy_distribution: {
        low: energySum > 0 ? lowEnergy / energySum : 0,
        mid: energySum > 0 ? midEnergy / energySum : 0,
        high: energySum > 0 ? highEnergy / energySum : 0,
      },
      voice_presence: voiceEnergy / totalMagnitude,
      noise_floor_db: -60, // Simplified calculation
    };
    
    setCurrentAnalysis(analysis);
    onAnalysisUpdate?.(analysis);
  }, [audioData, smoothedFFT, onAnalysisUpdate]);

  useEffect(() => {
    if (isPlaying && audioData) {
      const animate = () => {
        drawFFTSpectrum();
        performFrequencyAnalysis();
        
        if (isPlaying) {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      };
      
      animate();
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, audioData, drawFFTSpectrum, performFrequencyAnalysis]);

  useEffect(() => {
    setIsPlaying(isRealTime);
  }, [isRealTime]);

  const handlePlayToggle = () => {
    setIsPlaying(!isPlaying);
  };

  const handleDownloadSpectrum = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = `fft_spectrum_${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  if (!audioData) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
            <Typography variant="h6" component="h3">
              FFT Spectrum Analyzer
            </Typography>
            <Chip label="No Data" size="small" color="default" />
          </Box>
          
          <Alert severity="info">
            No audio data available for FFT analysis. Start audio processing to see real-time frequency spectrum.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h3">
            <Equalizer sx={{ verticalAlign: 'middle', mr: 1, fontSize: 24 }} />
            FFT Spectrum Analyzer
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={isRealTime ? 'Real-time' : 'Static'} 
              size="small" 
              color={isRealTime ? 'success' : 'default'}
              variant={isRealTime ? 'filled' : 'outlined'}
            />
            {!isRealTime && (
              <IconButton size="small" onClick={handlePlayToggle}>
                {isPlaying ? <Pause /> : <PlayArrow />}
              </IconButton>
            )}
            <IconButton size="small" onClick={handleDownloadSpectrum}>
              <Download />
            </IconButton>
          </Box>
        </Box>

        {/* Controls */}
        {showControls && (
          <Grid container spacing={2} mb={2}>
            <Grid item xs={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Display Mode</InputLabel>
                <Select
                  value={displayMode}
                  label="Display Mode"
                  onChange={(e) => setDisplayMode(e.target.value as FFTDisplayMode)}
                >
                  <MenuItem value="magnitude">Magnitude</MenuItem>
                  <MenuItem value="power">Power</MenuItem>
                  <MenuItem value="log_magnitude">Log Magnitude</MenuItem>
                  <MenuItem value="mel_scale">Mel Scale</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Window Function</InputLabel>
                <Select
                  value={windowFunction}
                  label="Window Function"
                  onChange={(e) => setWindowFunction(e.target.value as WindowFunction)}
                >
                  <MenuItem value="hanning">Hanning</MenuItem>
                  <MenuItem value="hamming">Hamming</MenuItem>
                  <MenuItem value="blackman">Blackman</MenuItem>
                  <MenuItem value="rectangular">Rectangular</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Typography variant="body2" gutterBottom>Smoothing: {smoothing.toFixed(2)}</Typography>
              <Slider
                value={smoothing}
                min={0}
                max={0.95}
                step={0.05}
                onChange={(_, value) => setSmoothing(value as number)}
                size="small"
              />
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Typography variant="body2" gutterBottom>Gain: {gainAdjustment > 0 ? '+' : ''}{gainAdjustment}dB</Typography>
              <Slider
                value={gainAdjustment}
                min={-20}
                max={20}
                step={1}
                onChange={(_, value) => setGainAdjustment(value as number)}
                size="small"
              />
            </Grid>
          </Grid>
        )}

        {/* Visualization Options */}
        <Box mb={2}>
          <FormControlLabel
            control={
              <Switch
                checked={showPeaks}
                onChange={(e) => setShowPeaks(e.target.checked)}
                size="small"
              />
            }
            label="Show peaks"
          />
          <FormControlLabel
            control={
              <Switch
                checked={showVoiceRegion}
                onChange={(e) => setShowVoiceRegion(e.target.checked)}
                size="small"
              />
            }
            label="Highlight voice region"
          />
        </Box>

        {/* FFT Canvas */}
        <Box 
          position="relative" 
          height={height} 
          bgcolor="#000" 
          borderRadius={1}
          overflow="hidden"
        >
          <canvas
            ref={canvasRef}
            width={800}
            height={height}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
            }}
          />
        </Box>

        {/* Analysis Results */}
        {currentAnalysis && (
          <Box mt={2} p={2} bgcolor="background.default" borderRadius={1}>
            <Typography variant="subtitle2" gutterBottom>
              <ShowChart sx={{ verticalAlign: 'middle', mr: 1, fontSize: 16 }} />
              Spectral Analysis
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Fundamental:</Typography>
                <Typography variant="body2">
                  {currentAnalysis.fundamental_frequency.toFixed(1)} Hz
                </Typography>
              </Grid>
              
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Centroid:</Typography>
                <Typography variant="body2">
                  {currentAnalysis.spectral_centroid.toFixed(1)} Hz
                </Typography>
              </Grid>
              
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Voice Presence:</Typography>
                <Typography variant="body2">
                  {(currentAnalysis.voice_presence * 100).toFixed(1)}%
                </Typography>
              </Grid>
              
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">Rolloff:</Typography>
                <Typography variant="body2">
                  {(currentAnalysis.spectral_rolloff / 1000).toFixed(1)} kHz
                </Typography>
              </Grid>
            </Grid>
            
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">Energy Distribution:</Typography>
              <Typography variant="body2">
                Low: {(currentAnalysis.energy_distribution.low * 100).toFixed(1)}% | 
                Mid: {(currentAnalysis.energy_distribution.mid * 100).toFixed(1)}% | 
                High: {(currentAnalysis.energy_distribution.high * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        )}
        
        {/* Metadata */}
        <Box mt={2} textAlign="center">
          <Typography variant="caption" color="text.secondary">
            Sample Rate: {audioData.sampleRate}Hz | 
            Window: {audioData.windowSize} samples | 
            Resolution: {(audioData.sampleRate / audioData.windowSize).toFixed(1)}Hz/bin
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default FFTVisualizer;