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
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ShowChart,
  Download,
  ExpandMore,
  Analytics,
} from '@mui/icons-material';

export interface SpectralAnalyzerProps {
  audioData?: {
    fft_data: Float32Array;
    frequency_analysis: {
      fundamental_frequency: number;
      spectral_centroid: number;
      spectral_rolloff: number;
      spectral_bandwidth: number;
      spectral_flatness: number;
      peak_frequencies: number[];
      energy_distribution: {
        low: number;
        mid: number;
        high: number;
      };
      voice_presence: number;
      noise_floor_db: number;
    };
    metadata: {
      window_size: number;
      sample_rate: number;
      timestamp: number;
    };
  };
  isRealTime?: boolean;
  onAnalysisUpdate?: (analysis: SpectralAnalysis) => void;
  height?: number;
  showControls?: boolean;
}

interface SpectralAnalysis {
  harmonic_content: {
    fundamental: number;
    harmonics: number[];
    harmonic_distortion: number;
    inharmonicity: number;
  };
  formant_analysis: {
    f1: number;
    f2: number;
    f3: number;
    formant_bandwidth: number[];
  };
  spectral_features: {
    centroid: number;
    rolloff: number;
    bandwidth: number;
    flatness: number;
    flux: number;
    kurtosis: number;
    skewness: number;
  };
  voice_characteristics: {
    voice_probability: number;
    pitch_confidence: number;
    voiced_unvoiced_ratio: number;
    shimmer: number;
    jitter: number;
  };
  noise_analysis: {
    snr_estimate: number;
    noise_floor: number;
    signal_to_noise_ratio: number;
    spectral_noise_level: number;
  };
}

type AnalysisView = 'spectrum' | 'formants' | 'harmonics' | 'voice' | 'noise';
type SpectralDisplay = 'magnitude' | 'power' | 'log_power' | 'mel_scale' | 'bark_scale';

export const SpectralAnalyzer: React.FC<SpectralAnalyzerProps> = ({
  audioData,
  isRealTime = false,
  onAnalysisUpdate,
  height = 400,
  showControls = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  
  const [analysisView, setAnalysisView] = useState<AnalysisView>('spectrum');
  const [spectralDisplay, setSpectralDisplay] = useState<SpectralDisplay>('log_power');
  const [frequencyRange] = useState<[number, number]>([20, 8000]);
  const [smoothingFactor, setSmoothingFactor] = useState(0.8);
  const [showPeaks, setShowPeaks] = useState(true);
  const [showFormants, setShowFormants] = useState(true);
  const [showHarmonics, setShowHarmonics] = useState(true);
  const [currentTab, setCurrentTab] = useState(0);
  
  const [currentAnalysis, setCurrentAnalysis] = useState<SpectralAnalysis | null>(null);
  const [smoothedSpectrum, setSmoothedSpectrum] = useState<Float32Array | null>(null);
  const [spectralHistory, setSpectralHistory] = useState<Float32Array[]>([]);

  const performAdvancedSpectralAnalysis = useCallback(() => {
    if (!audioData || !smoothedSpectrum) return;

    const { frequency_analysis, metadata } = audioData;
    const { sample_rate } = metadata;
    const nyquist = sample_rate / 2;
    const binWidth = nyquist / (smoothedSpectrum.length / 2);
    
    // Harmonic analysis
    const fundamental = frequency_analysis.fundamental_frequency;
    const harmonics: number[] = [];
    let harmonicDistortion = 0;
    
    if (fundamental > 0) {
      for (let h = 2; h <= 10; h++) {
        const harmonicFreq = fundamental * h;
        if (harmonicFreq < nyquist) {
          const harmonicBin = Math.round(harmonicFreq / binWidth);
          if (harmonicBin < smoothedSpectrum.length / 2) {
            harmonics.push(smoothedSpectrum[harmonicBin]);
            harmonicDistortion += smoothedSpectrum[harmonicBin];
          }
        }
      }
      harmonicDistortion = harmonicDistortion / smoothedSpectrum[Math.round(fundamental / binWidth)];
    }
    
    // Formant estimation (simplified)
    const formants = estimateFormants(smoothedSpectrum, binWidth);
    
    // Spectral features calculation
    let spectralCentroid = 0;
    let totalMagnitude = 0;
    let spectralSpread = 0;
    let spectralFlux = 0;
    
    for (let i = 1; i < smoothedSpectrum.length / 2; i++) {
      const frequency = i * binWidth;
      const magnitude = smoothedSpectrum[i];
      
      spectralCentroid += frequency * magnitude;
      totalMagnitude += magnitude;
    }
    
    spectralCentroid = totalMagnitude > 0 ? spectralCentroid / totalMagnitude : 0;
    
    // Spectral spread (bandwidth)
    for (let i = 1; i < smoothedSpectrum.length / 2; i++) {
      const frequency = i * binWidth;
      const magnitude = smoothedSpectrum[i];
      spectralSpread += Math.pow(frequency - spectralCentroid, 2) * magnitude;
    }
    spectralSpread = totalMagnitude > 0 ? Math.sqrt(spectralSpread / totalMagnitude) : 0;
    
    // Spectral flux (change between frames)
    if (spectralHistory.length > 0) {
      const prevSpectrum = spectralHistory[spectralHistory.length - 1];
      for (let i = 0; i < Math.min(smoothedSpectrum.length, prevSpectrum.length); i++) {
        const diff = smoothedSpectrum[i] - prevSpectrum[i];
        spectralFlux += diff > 0 ? diff : 0;
      }
    }
    
    // Voice characteristics
    const voiceProbability = calculateVoiceProbability(smoothedSpectrum, binWidth, fundamental);
    const pitchConfidence = fundamental > 0 ? Math.min(1.0, smoothedSpectrum[Math.round(fundamental / binWidth)] * 2) : 0;
    
    // Noise analysis
    const noiseFloor = frequency_analysis.noise_floor_db;
    const signalPower = calculateSignalPower(smoothedSpectrum);
    const noisePower = Math.pow(10, noiseFloor / 10);
    const snrEstimate = 10 * Math.log10(signalPower / noisePower);
    
    const analysis: SpectralAnalysis = {
      harmonic_content: {
        fundamental,
        harmonics,
        harmonic_distortion: harmonicDistortion,
        inharmonicity: calculateInharmonicity(fundamental, harmonics)
      },
      formant_analysis: {
        f1: formants.f1,
        f2: formants.f2,
        f3: formants.f3,
        formant_bandwidth: [formants.f1_bw, formants.f2_bw, formants.f3_bw]
      },
      spectral_features: {
        centroid: spectralCentroid,
        rolloff: frequency_analysis.spectral_rolloff,
        bandwidth: spectralSpread,
        flatness: frequency_analysis.spectral_flatness,
        flux: spectralFlux,
        kurtosis: calculateSpectralKurtosis(smoothedSpectrum, spectralCentroid, spectralSpread),
        skewness: calculateSpectralSkewness(smoothedSpectrum, spectralCentroid, spectralSpread)
      },
      voice_characteristics: {
        voice_probability: voiceProbability,
        pitch_confidence: pitchConfidence,
        voiced_unvoiced_ratio: frequency_analysis.voice_presence,
        shimmer: 0, // Would require time-domain analysis
        jitter: 0   // Would require time-domain analysis
      },
      noise_analysis: {
        snr_estimate: snrEstimate,
        noise_floor: noiseFloor,
        signal_to_noise_ratio: snrEstimate,
        spectral_noise_level: calculateSpectralNoiseLevel(smoothedSpectrum)
      }
    };
    
    setCurrentAnalysis(analysis);
    onAnalysisUpdate?.(analysis);
  }, [audioData, smoothedSpectrum, spectralHistory, onAnalysisUpdate]);

  const drawSpectralAnalysis = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height: canvasHeight } = canvas;
    const { fft_data, metadata } = audioData;
    const { sample_rate } = metadata;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, canvasHeight);
    
    // Apply smoothing
    if (!smoothedSpectrum || smoothedSpectrum.length !== fft_data.length) {
      setSmoothedSpectrum(new Float32Array(fft_data));
      return;
    }
    
    for (let i = 0; i < fft_data.length; i++) {
      smoothedSpectrum[i] = smoothedSpectrum[i] * smoothingFactor + fft_data[i] * (1 - smoothingFactor);
    }
    
    // Calculate display parameters
    const nyquist = sample_rate / 2;
    const binWidth = nyquist / (fft_data.length / 2);
    const startBin = Math.floor(frequencyRange[0] / binWidth);
    const endBin = Math.min(Math.floor(frequencyRange[1] / binWidth), fft_data.length / 2);
    
    // Process spectrum data based on display mode
    const processedData = new Float32Array(endBin - startBin);
    
    for (let i = startBin; i < endBin; i++) {
      let value = smoothedSpectrum[i];
      
      switch (spectralDisplay) {
        case 'magnitude':
          processedData[i - startBin] = value;
          break;
        case 'power':
          processedData[i - startBin] = value * value;
          break;
        case 'log_power':
          processedData[i - startBin] = Math.log10(Math.max(value * value, 1e-10));
          break;
        case 'mel_scale':
          // Mel scale conversion
          processedData[i - startBin] = value;
          break;
        case 'bark_scale':
          // Bark scale conversion
          processedData[i - startBin] = value;
          break;
      }
    }
    
    // Find normalization values
    const maxValue = Math.max(...processedData);
    const minValue = spectralDisplay === 'log_power' ? Math.min(...processedData) : 0;
    const range = maxValue - minValue;
    
    // Draw spectrum based on current view
    switch (analysisView) {
      case 'spectrum':
        drawSpectrumView(ctx, processedData, width, canvasHeight, minValue, range);
        break;
      case 'formants':
        drawFormantView(ctx, processedData, width, canvasHeight, minValue, range, binWidth);
        break;
      case 'harmonics':
        drawHarmonicView(ctx, processedData, width, canvasHeight, minValue, range, binWidth);
        break;
      case 'voice':
        drawVoiceView(ctx, processedData, width, canvasHeight, minValue, range);
        break;
      case 'noise':
        drawNoiseView(ctx, processedData, width, canvasHeight, minValue, range);
        break;
    }
    
    // Draw frequency labels
    drawFrequencyLabels(ctx, width, canvasHeight, frequencyRange);
    
  }, [audioData, smoothedSpectrum, spectralDisplay, analysisView, frequencyRange, smoothingFactor, showPeaks, showFormants, showHarmonics]);

  const drawSpectrumView = (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number, minValue: number, range: number) => {
    // Draw main spectrum
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const barWidth = width / data.length;
    
    for (let i = 0; i < data.length; i++) {
      const x = i * barWidth;
      const normalizedValue = range > 0 ? (data[i] - minValue) / range : 0;
      const y = height - (normalizedValue * height * 0.9);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Fill area under curve
    ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fill();
  };

  const drawFormantView = (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number, minValue: number, range: number, _binWidth: number) => {
    // Draw spectrum background
    drawSpectrumView(ctx, data, width, height, minValue, range);
    
    if (currentAnalysis && showFormants) {
      const { formant_analysis } = currentAnalysis;
      
      // Draw formant markers
      ctx.strokeStyle = '#ff6b6b';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      [formant_analysis.f1, formant_analysis.f2, formant_analysis.f3].forEach((formant, index) => {
        if (formant > frequencyRange[0] && formant < frequencyRange[1]) {
          const x = ((formant - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0])) * width;
          
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
          
          // Formant label
          ctx.fillStyle = '#ff6b6b';
          ctx.font = '12px monospace';
          ctx.fillText(`F${index + 1}`, x + 2, 15);
        }
      });
      
      ctx.setLineDash([]);
    }
  };

  const drawHarmonicView = (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number, minValue: number, range: number, _binWidth: number) => {
    // Draw spectrum background
    drawSpectrumView(ctx, data, width, height, minValue, range);
    
    if (currentAnalysis && showHarmonics) {
      const { harmonic_content } = currentAnalysis;
      
      if (harmonic_content.fundamental > 0) {
        // Draw fundamental
        const f0 = harmonic_content.fundamental;
        if (f0 > frequencyRange[0] && f0 < frequencyRange[1]) {
          const x = ((f0 - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0])) * width;
          
          ctx.strokeStyle = '#ffff00';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
          
          ctx.fillStyle = '#ffff00';
          ctx.font = '12px monospace';
          ctx.fillText('F0', x + 2, 15);
        }
        
        // Draw harmonics
        ctx.strokeStyle = '#ff9f43';
        ctx.lineWidth = 2;
        
        for (let h = 2; h <= 10; h++) {
          const harmonicFreq = f0 * h;
          if (harmonicFreq > frequencyRange[1]) break;
          
          if (harmonicFreq > frequencyRange[0]) {
            const x = ((harmonicFreq - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0])) * width;
            
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            
            ctx.fillStyle = '#ff9f43';
            ctx.font = '10px monospace';
            ctx.fillText(`H${h}`, x + 1, 25);
          }
        }
      }
    }
  };

  const drawVoiceView = (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number, minValue: number, range: number) => {
    // Draw spectrum with voice regions highlighted
    drawSpectrumView(ctx, data, width, height, minValue, range);
    
    // Voice frequency regions
    const voiceRegions = [
      { min: 85, max: 300, color: 'rgba(255, 193, 7, 0.2)', label: 'Fundamental' },
      { min: 300, max: 3400, color: 'rgba(76, 175, 80, 0.2)', label: 'Voice' },
      { min: 2000, max: 5000, color: 'rgba(156, 39, 176, 0.2)', label: 'Presence' }
    ];
    
    voiceRegions.forEach(region => {
      const startX = Math.max(0, ((region.min - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0])) * width);
      const endX = Math.min(width, ((region.max - frequencyRange[0]) / (frequencyRange[1] - frequencyRange[0])) * width);
      
      if (endX > startX) {
        ctx.fillStyle = region.color;
        ctx.fillRect(startX, 0, endX - startX, height);
        
        // Region label
        ctx.fillStyle = region.color.replace('0.2', '0.8');
        ctx.font = '10px monospace';
        ctx.fillText(region.label, startX + 5, 15);
      }
    });
  };

  const drawNoiseView = (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number, minValue: number, range: number) => {
    // Draw spectrum
    drawSpectrumView(ctx, data, width, height, minValue, range);
    
    if (currentAnalysis) {
      const { noise_analysis } = currentAnalysis;
      
      // Draw noise floor line
      const noiseY = height - ((Math.log10(Math.pow(10, noise_analysis.noise_floor / 10)) - minValue) / range) * height * 0.9;
      
      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, noiseY);
      ctx.lineTo(width, noiseY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Noise floor label
      ctx.fillStyle = '#e74c3c';
      ctx.font = '12px monospace';
      ctx.fillText(`Noise Floor: ${noise_analysis.noise_floor.toFixed(1)} dB`, 10, noiseY - 5);
    }
  };

  const drawFrequencyLabels = (ctx: CanvasRenderingContext2D, width: number, height: number, range: [number, number]) => {
    ctx.fillStyle = '#888888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    
    const labelCount = 8;
    for (let i = 0; i <= labelCount; i++) {
      const freq = range[0] + (i / labelCount) * (range[1] - range[0]);
      const x = (i / labelCount) * width;
      
      const label = freq < 1000 ? `${Math.round(freq)}` : `${(freq / 1000).toFixed(1)}k`;
      ctx.fillText(label, x, height - 5);
    }
  };

  // Helper functions
  const estimateFormants = (spectrum: Float32Array, binWidth: number) => {
    // Simplified formant estimation using peak detection in voice range
    const voiceStart = Math.floor(200 / binWidth);
    const voiceEnd = Math.floor(4000 / binWidth);
    
    const peaks: { freq: number; magnitude: number }[] = [];
    
    for (let i = voiceStart + 2; i < Math.min(voiceEnd - 2, spectrum.length / 2); i++) {
      if (spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] &&
          spectrum[i] > spectrum[i-2] && spectrum[i] > spectrum[i+2]) {
        peaks.push({ freq: i * binWidth, magnitude: spectrum[i] });
      }
    }
    
    peaks.sort((a, b) => b.magnitude - a.magnitude);
    
    return {
      f1: peaks[0]?.freq || 500,
      f2: peaks[1]?.freq || 1500,
      f3: peaks[2]?.freq || 2500,
      f1_bw: 100,
      f2_bw: 150,
      f3_bw: 200
    };
  };

  const calculateVoiceProbability = (spectrum: Float32Array, binWidth: number, fundamental: number): number => {
    const voiceStart = Math.floor(85 / binWidth);
    const voiceEnd = Math.floor(300 / binWidth);
    
    let voiceEnergy = 0;
    let totalEnergy = 0;
    
    for (let i = 1; i < spectrum.length / 2; i++) {
      const energy = spectrum[i] * spectrum[i];
      totalEnergy += energy;
      
      if (i >= voiceStart && i <= voiceEnd) {
        voiceEnergy += energy;
      }
    }
    
    const voiceRatio = totalEnergy > 0 ? voiceEnergy / totalEnergy : 0;
    const pitchStrength = fundamental > 0 ? Math.min(1, spectrum[Math.round(fundamental / binWidth)]) : 0;
    
    return Math.min(1, (voiceRatio * 0.7 + pitchStrength * 0.3));
  };

  const calculateSignalPower = (spectrum: Float32Array): number => {
    let power = 0;
    for (let i = 0; i < spectrum.length; i++) {
      power += spectrum[i] * spectrum[i];
    }
    return power / spectrum.length;
  };

  const calculateInharmonicity = (fundamental: number, harmonics: number[]): number => {
    if (fundamental <= 0 || harmonics.length === 0) return 0;
    
    let inharmonicity = 0;
    harmonics.forEach((harmonic, index) => {
      const expectedFreq = fundamental * (index + 2);
      const actualFreq = harmonic;
      inharmonicity += Math.abs(actualFreq - expectedFreq) / expectedFreq;
    });
    
    return harmonics.length > 0 ? inharmonicity / harmonics.length : 0;
  };

  const calculateSpectralKurtosis = (_spectrum: Float32Array, _centroid: number, _spread: number): number => {
    // Simplified spectral kurtosis calculation
    return 0; // Would require more complex implementation
  };

  const calculateSpectralSkewness = (_spectrum: Float32Array, _centroid: number, _spread: number): number => {
    // Simplified spectral skewness calculation
    return 0; // Would require more complex implementation
  };

  const calculateSpectralNoiseLevel = (spectrum: Float32Array): number => {
    // Estimate noise level from lower percentile of spectrum
    const sortedMagnitudes = Array.from(spectrum).sort((a, b) => a - b);
    const percentile = Math.floor(sortedMagnitudes.length * 0.1); // 10th percentile
    return 20 * Math.log10(sortedMagnitudes[percentile] || 1e-10);
  };

  useEffect(() => {
    if (isRealTime && audioData) {
      const animate = () => {
        drawSpectralAnalysis();
        performAdvancedSpectralAnalysis();
        
        // Update spectral history
        if (smoothedSpectrum) {
          setSpectralHistory(prev => {
            const newHistory = [...prev, new Float32Array(smoothedSpectrum)];
            return newHistory.slice(-10); // Keep last 10 frames
          });
        }
        
        if (isRealTime) {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      };
      
      animate();
    } else if (audioData) {
      drawSpectralAnalysis();
      performAdvancedSpectralAnalysis();
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRealTime, audioData, drawSpectralAnalysis, performAdvancedSpectralAnalysis]);

  const handleDownloadAnalysis = () => {
    if (!currentAnalysis) return;
    
    const analysisReport = {
      timestamp: new Date().toISOString(),
      analysis_type: 'advanced_spectral_analysis',
      settings: {
        view: analysisView,
        display: spectralDisplay,
        frequency_range: frequencyRange,
        smoothing: smoothingFactor
      },
      results: currentAnalysis
    };
    
    const blob = new Blob([JSON.stringify(analysisReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `spectral_analysis_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!audioData) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
            <Typography variant="h6" component="h3">
              Spectral Analyzer
            </Typography>
            <Chip label="No Data" size="small" color="default" />
          </Box>
          
          <Alert severity="info">
            No audio data available for spectral analysis. Start audio processing to see advanced frequency domain analysis.
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
            <ShowChart sx={{ verticalAlign: 'middle', mr: 1, fontSize: 24 }} />
            Advanced Spectral Analyzer
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={isRealTime ? 'Real-time' : 'Static'} 
              size="small" 
              color={isRealTime ? 'success' : 'default'}
              variant={isRealTime ? 'filled' : 'outlined'}
            />
            <IconButton size="small" onClick={handleDownloadAnalysis}>
              <Download />
            </IconButton>
          </Box>
        </Box>

        {/* Analysis View Tabs */}
        <Tabs 
          value={currentTab} 
          onChange={(_, newValue) => setCurrentTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ mb: 2 }}
        >
          <Tab label="Spectrum" onClick={() => setAnalysisView('spectrum')} />
          <Tab label="Formants" onClick={() => setAnalysisView('formants')} />
          <Tab label="Harmonics" onClick={() => setAnalysisView('harmonics')} />
          <Tab label="Voice" onClick={() => setAnalysisView('voice')} />
          <Tab label="Noise" onClick={() => setAnalysisView('noise')} />
        </Tabs>

        {/* Controls */}
        {showControls && (
          <Grid container spacing={2} mb={2}>
            <Grid item xs={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Display</InputLabel>
                <Select
                  value={spectralDisplay}
                  label="Display"
                  onChange={(e) => setSpectralDisplay(e.target.value as SpectralDisplay)}
                >
                  <MenuItem value="magnitude">Magnitude</MenuItem>
                  <MenuItem value="power">Power</MenuItem>
                  <MenuItem value="log_power">Log Power</MenuItem>
                  <MenuItem value="mel_scale">Mel Scale</MenuItem>
                  <MenuItem value="bark_scale">Bark Scale</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="body2" gutterBottom>Smoothing: {smoothingFactor.toFixed(2)}</Typography>
              <Slider
                value={smoothingFactor}
                min={0}
                max={0.95}
                step={0.05}
                onChange={(_, value) => setSmoothingFactor(value as number)}
                size="small"
              />
            </Grid>
            
            <Grid item xs={12} md={8}>
              <Box display="flex" gap={1} flexWrap="wrap">
                <FormControlLabel
                  control={
                    <Switch
                      checked={showPeaks}
                      onChange={(e) => setShowPeaks(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Peaks"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={showFormants}
                      onChange={(e) => setShowFormants(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Formants"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={showHarmonics}
                      onChange={(e) => setShowHarmonics(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Harmonics"
                />
              </Box>
            </Grid>
          </Grid>
        )}

        {/* Spectral Analysis Canvas */}
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
          <Accordion sx={{ mt: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">
                <Analytics sx={{ verticalAlign: 'middle', mr: 1, fontSize: 16 }} />
                Detailed Analysis Results
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom><strong>Spectral Features</strong></Typography>
                  <Typography variant="caption" display="block">
                    Centroid: {currentAnalysis.spectral_features.centroid.toFixed(1)} Hz
                  </Typography>
                  <Typography variant="caption" display="block">
                    Rolloff: {currentAnalysis.spectral_features.rolloff.toFixed(1)} Hz
                  </Typography>
                  <Typography variant="caption" display="block">
                    Bandwidth: {currentAnalysis.spectral_features.bandwidth.toFixed(1)} Hz
                  </Typography>
                  <Typography variant="caption" display="block">
                    Flatness: {currentAnalysis.spectral_features.flatness.toFixed(3)}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom><strong>Voice Analysis</strong></Typography>
                  <Typography variant="caption" display="block">
                    Voice Probability: {(currentAnalysis.voice_characteristics.voice_probability * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" display="block">
                    Pitch Confidence: {(currentAnalysis.voice_characteristics.pitch_confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" display="block">
                    V/UV Ratio: {(currentAnalysis.voice_characteristics.voiced_unvoiced_ratio * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                
                {currentAnalysis.harmonic_content.fundamental > 0 && (
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" gutterBottom><strong>Harmonic Content</strong></Typography>
                    <Typography variant="caption" display="block">
                      Fundamental: {currentAnalysis.harmonic_content.fundamental.toFixed(1)} Hz
                    </Typography>
                    <Typography variant="caption" display="block">
                      Harmonics: {currentAnalysis.harmonic_content.harmonics.length}
                    </Typography>
                    <Typography variant="caption" display="block">
                      THD: {(currentAnalysis.harmonic_content.harmonic_distortion * 100).toFixed(2)}%
                    </Typography>
                  </Grid>
                )}
                
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom><strong>Noise Analysis</strong></Typography>
                  <Typography variant="caption" display="block">
                    SNR: {currentAnalysis.noise_analysis.snr_estimate.toFixed(1)} dB
                  </Typography>
                  <Typography variant="caption" display="block">
                    Noise Floor: {currentAnalysis.noise_analysis.noise_floor.toFixed(1)} dB
                  </Typography>
                  <Typography variant="caption" display="block">
                    Spectral Noise: {currentAnalysis.noise_analysis.spectral_noise_level.toFixed(1)} dB
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}
        
        {/* Technical Info */}
        <Box mt={2} textAlign="center">
          <Typography variant="caption" color="text.secondary">
            View: {analysisView} | Display: {spectralDisplay} | 
            Range: {frequencyRange[0]}-{frequencyRange[1]} Hz | 
            Resolution: {audioData.metadata ? (audioData.metadata.sample_rate / audioData.metadata.window_size).toFixed(1) : 0}Hz/bin
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SpectralAnalyzer;