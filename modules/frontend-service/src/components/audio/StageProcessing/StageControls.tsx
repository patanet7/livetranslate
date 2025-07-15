import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Grid,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  VolumeUp,
  VolumeDown,
  Settings,
  Refresh,
  Save,
  Equalizer,
  Tune,
} from '@mui/icons-material';

interface StageControlsProps {
  stageName: string;
  stageDisplayName: string;
  onConfigChange: (stageName: string, config: any) => void;
  initialConfig?: any;
  isEnabled?: boolean;
  onEnabledChange?: (enabled: boolean) => void;
}

interface StageParameter {
  key: string;
  label: string;
  type: 'slider' | 'select' | 'switch';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: any; label: string }[];
  unit?: string;
  defaultValue: any;
  description?: string;
}

const STAGE_PARAMETERS: Record<string, StageParameter[]> = {
  vad: [
    { key: 'aggressiveness', label: 'Aggressiveness', type: 'slider', min: 0, max: 3, step: 1, defaultValue: 2, description: 'VAD sensitivity level' },
    { key: 'energy_threshold', label: 'Energy Threshold', type: 'slider', min: 0.001, max: 0.1, step: 0.001, defaultValue: 0.01, description: 'Voice detection threshold' },
    { key: 'voice_freq_min', label: 'Min Voice Freq', type: 'slider', min: 50, max: 150, step: 5, defaultValue: 85, unit: 'Hz' },
    { key: 'voice_freq_max', label: 'Max Voice Freq', type: 'slider', min: 200, max: 500, step: 10, defaultValue: 300, unit: 'Hz' },
  ],
  voice_filter: [
    { key: 'fundamental_min', label: 'Fundamental Min', type: 'slider', min: 50, max: 150, step: 5, defaultValue: 85, unit: 'Hz' },
    { key: 'fundamental_max', label: 'Fundamental Max', type: 'slider', min: 200, max: 500, step: 10, defaultValue: 300, unit: 'Hz' },
    { key: 'voice_band_gain', label: 'Voice Band Gain', type: 'slider', min: 0.1, max: 3.0, step: 0.1, defaultValue: 1.1, description: 'Voice frequency amplification' },
    { key: 'preserve_formants', label: 'Preserve Formants', type: 'switch', defaultValue: true, description: 'Maintain F1/F2 formant frequencies' },
  ],
  noise_reduction: [
    { key: 'strength', label: 'Strength', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.7, description: 'Noise reduction intensity' },
    { key: 'voice_protection', label: 'Voice Protection', type: 'switch', defaultValue: true, description: 'Protect speech frequencies' },
    { key: 'adaptation_rate', label: 'Adaptation Rate', type: 'slider', min: 0.01, max: 1.0, step: 0.01, defaultValue: 0.1, description: 'Noise profile adaptation speed' },
    { key: 'noise_floor_db', label: 'Noise Floor', type: 'slider', min: -60, max: -20, step: 1, defaultValue: -40, unit: 'dB' },
  ],
  voice_enhancement: [
    { key: 'clarity_enhancement', label: 'Clarity', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.2, description: 'Speech clarity enhancement' },
    { key: 'presence_boost', label: 'Presence', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.1, description: '2-5kHz presence boost' },
    { key: 'warmth_adjustment', label: 'Warmth', type: 'slider', min: -1.0, max: 1.0, step: 0.1, defaultValue: 0.0, description: 'Tonal warmth adjustment' },
    { key: 'brightness_adjustment', label: 'Brightness', type: 'slider', min: -1.0, max: 1.0, step: 0.1, defaultValue: 0.0, description: 'High frequency brightness' },
  ],
  equalizer: [
    { key: 'preset_name', label: 'Preset', type: 'select', defaultValue: 'flat', options: [
      { value: 'flat', label: 'Flat Response' },
      { value: 'voice_enhance', label: 'Voice Enhance' },
      { value: 'broadcast', label: 'Broadcast' },
      { value: 'bass_boost', label: 'Bass Boost' },
      { value: 'treble_boost', label: 'Treble Boost' },
    ]},
  ],
  spectral_denoising: [
    { key: 'mode', label: 'Algorithm', type: 'select', defaultValue: 'minimal', options: [
      { value: 'minimal', label: 'Minimal Processing' },
      { value: 'spectral_subtraction', label: 'Spectral Subtraction' },
      { value: 'wiener_filter', label: 'Wiener Filter' },
      { value: 'adaptive', label: 'Adaptive' },
    ]},
    { key: 'noise_reduction_factor', label: 'Reduction Factor', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.5 },
    { key: 'spectral_floor', label: 'Spectral Floor', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.1, description: 'Artifact reduction floor' },
  ],
  conventional_denoising: [
    { key: 'mode', label: 'Filter Type', type: 'select', defaultValue: 'median_filter', options: [
      { value: 'median_filter', label: 'Median Filter' },
      { value: 'gaussian_filter', label: 'Gaussian Filter' },
      { value: 'bilateral_filter', label: 'Bilateral Filter' },
      { value: 'wavelet_denoising', label: 'Wavelet Denoising' },
    ]},
    { key: 'filter_strength', label: 'Filter Strength', type: 'slider', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.3 },
    { key: 'preserve_transients', label: 'Preserve Transients', type: 'switch', defaultValue: true, description: 'Maintain speech transients' },
  ],
  lufs_normalization: [
    { key: 'target_lufs', label: 'Target LUFS', type: 'slider', min: -30, max: -10, step: 1, defaultValue: -23, unit: 'LUFS', description: 'EBU R128 standard: -23 LUFS' },
    { key: 'max_adjustment', label: 'Max Adjustment', type: 'slider', min: 3, max: 20, step: 1, defaultValue: 12, unit: 'dB' },
    { key: 'gating_threshold', label: 'Gating Threshold', type: 'slider', min: -80, max: -60, step: 1, defaultValue: -70, unit: 'dB' },
  ],
  agc: [
    { key: 'target_level', label: 'Target Level', type: 'slider', min: -30, max: -6, step: 1, defaultValue: -18, unit: 'dB' },
    { key: 'max_gain', label: 'Max Gain', type: 'slider', min: 6, max: 20, step: 1, defaultValue: 12, unit: 'dB' },
    { key: 'attack_time', label: 'Attack Time', type: 'slider', min: 1, max: 50, step: 1, defaultValue: 10, unit: 'ms' },
    { key: 'release_time', label: 'Release Time', type: 'slider', min: 50, max: 500, step: 10, defaultValue: 100, unit: 'ms' },
  ],
  compression: [
    { key: 'threshold', label: 'Threshold', type: 'slider', min: -40, max: -5, step: 1, defaultValue: -20, unit: 'dB' },
    { key: 'ratio', label: 'Ratio', type: 'slider', min: 1, max: 20, step: 0.5, defaultValue: 3, unit: ':1' },
    { key: 'knee', label: 'Knee Width', type: 'slider', min: 0, max: 10, step: 0.5, defaultValue: 2, unit: 'dB' },
    { key: 'attack_time', label: 'Attack Time', type: 'slider', min: 0.1, max: 20, step: 0.1, defaultValue: 5, unit: 'ms' },
    { key: 'release_time', label: 'Release Time', type: 'slider', min: 10, max: 500, step: 10, defaultValue: 100, unit: 'ms' },
  ],
  limiter: [
    { key: 'threshold', label: 'Threshold', type: 'slider', min: -10, max: 0, step: 0.1, defaultValue: -1, unit: 'dB' },
    { key: 'release_time', label: 'Release Time', type: 'slider', min: 10, max: 200, step: 5, defaultValue: 50, unit: 'ms' },
    { key: 'lookahead', label: 'Lookahead', type: 'slider', min: 1, max: 20, step: 1, defaultValue: 5, unit: 'ms' },
    { key: 'soft_clip', label: 'Soft Clipping', type: 'switch', defaultValue: true, description: 'Harmonic saturation vs hard clipping' },
  ],
};

export const StageControls: React.FC<StageControlsProps> = ({
  stageName,
  stageDisplayName,
  onConfigChange,
  initialConfig,
  isEnabled = true,
  onEnabledChange,
}) => {
  const [config, setConfig] = useState<any>(initialConfig || {});
  const [gainIn, setGainIn] = useState(initialConfig?.gain_in || 0);
  const [gainOut, setGainOut] = useState(initialConfig?.gain_out || 0);

  const stageParameters = STAGE_PARAMETERS[stageName] || [];

  useEffect(() => {
    if (initialConfig) {
      setConfig(initialConfig);
      setGainIn(initialConfig.gain_in || 0);
      setGainOut(initialConfig.gain_out || 0);
    }
  }, [initialConfig]);

  const handleParameterChange = (key: string, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onConfigChange(stageName, newConfig);
  };

  const handleGainChange = (type: 'in' | 'out', value: number) => {
    const newConfig = { 
      ...config, 
      [`gain_${type}`]: value 
    };
    
    if (type === 'in') {
      setGainIn(value);
    } else {
      setGainOut(value);
    }
    
    setConfig(newConfig);
    onConfigChange(stageName, newConfig);
  };

  const resetToDefaults = () => {
    const defaultConfig = {
      gain_in: 0,
      gain_out: 0,
    };
    
    stageParameters.forEach(param => {
      defaultConfig[param.key] = param.defaultValue;
    });
    
    setConfig(defaultConfig);
    setGainIn(0);
    setGainOut(0);
    onConfigChange(stageName, defaultConfig);
  };

  const renderParameter = (param: StageParameter) => {
    const value = config[param.key] ?? param.defaultValue;

    switch (param.type) {
      case 'slider':
        return (
          <Box key={param.key} mb={3}>
            <Typography variant="body2" gutterBottom>
              {param.label} {param.unit && `(${param.unit})`}
            </Typography>
            {param.description && (
              <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                {param.description}
              </Typography>
            )}
            <Slider
              value={value}
              min={param.min}
              max={param.max}
              step={param.step}
              onChange={(_, newValue) => handleParameterChange(param.key, newValue)}
              valueLabelDisplay="auto"
              size="small"
            />
            <Typography variant="caption" color="text.secondary">
              Current: {value}{param.unit}
            </Typography>
          </Box>
        );

      case 'select':
        return (
          <FormControl key={param.key} fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel>{param.label}</InputLabel>
            <Select
              value={value}
              label={param.label}
              onChange={(e) => handleParameterChange(param.key, e.target.value)}
            >
              {param.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );

      case 'switch':
        return (
          <Box key={param.key} mb={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={value}
                  onChange={(e) => handleParameterChange(param.key, e.target.checked)}
                  size="small"
                />
              }
              label={param.label}
            />
            {param.description && (
              <Typography variant="caption" color="text.secondary" display="block">
                {param.description}
              </Typography>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h3">
            {stageDisplayName} Controls
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={isEnabled ? 'Enabled' : 'Disabled'} 
              size="small" 
              color={isEnabled ? 'success' : 'default'}
              variant="outlined"
            />
            <Tooltip title="Reset to defaults">
              <IconButton size="small" onClick={resetToDefaults}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Stage Enable/Disable */}
        {onEnabledChange && (
          <Box mb={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={isEnabled}
                  onChange={(e) => onEnabledChange(e.target.checked)}
                />
              }
              label={`Enable ${stageDisplayName} Stage`}
            />
          </Box>
        )}

        {/* Individual Gain Controls */}
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Individual Gain Controls
          </Typography>
          <Typography variant="body2">
            Professional gain controls (-20dB to +20dB) applied before and after stage processing
          </Typography>
        </Alert>

        <Grid container spacing={3} mb={3}>
          <Grid item xs={6}>
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <VolumeDown fontSize="small" />
                <Typography variant="body2">Input Gain</Typography>
              </Box>
              <Slider
                value={gainIn}
                min={-20}
                max={20}
                step={0.1}
                onChange={(_, value) => handleGainChange('in', value as number)}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value > 0 ? '+' : ''}${value}dB`}
                size="small"
                color="primary"
              />
              <Typography variant="caption" color="text.secondary">
                {gainIn > 0 ? '+' : ''}{gainIn}dB
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <VolumeUp fontSize="small" />
                <Typography variant="body2">Output Gain</Typography>
              </Box>
              <Slider
                value={gainOut}
                min={-20}
                max={20}
                step={0.1}
                onChange={(_, value) => handleGainChange('out', value as number)}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value > 0 ? '+' : ''}${value}dB`}
                size="small"
                color="secondary"
              />
              <Typography variant="caption" color="text.secondary">
                {gainOut > 0 ? '+' : ''}{gainOut}dB
              </Typography>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ mb: 3 }} />

        {/* Stage-Specific Parameters */}
        <Typography variant="subtitle1" gutterBottom>
          <Tune sx={{ verticalAlign: 'middle', mr: 1, fontSize: 20 }} />
          Stage Parameters
        </Typography>

        {stageParameters.length > 0 ? (
          <Box>
            {stageParameters.map(renderParameter)}
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary" style={{ fontStyle: 'italic' }}>
            No configurable parameters for this stage
          </Typography>
        )}

        {/* Configuration Summary */}
        <Box mt={3} p={2} bgcolor="background.default" borderRadius={1}>
          <Typography variant="caption" color="text.secondary" display="block" mb={1}>
            Current Configuration:
          </Typography>
          <Typography variant="caption" component="pre" sx={{ fontSize: '0.7rem', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify({ gain_in: gainIn, gain_out: gainOut, ...config }, null, 2)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};