import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Switch,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RestoreIcon from '@mui/icons-material/Restore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

interface AudioProcessingConfig {
  // VAD Configuration
  vad: {
    enabled: boolean;
    mode: 'basic' | 'webrtc' | 'aggressive' | 'silero';
    aggressiveness: number;
    energy_threshold: number;
    sensitivity: number;
    voice_freq_min: number;
    voice_freq_max: number;
  };
  
  // Voice Filter Configuration
  voice_filter: {
    enabled: boolean;
    fundamental_min: number;
    fundamental_max: number;
    voice_band_gain: number;
    preserve_formants: boolean;
    formant1_min: number;
    formant1_max: number;
  };
  
  // Noise Reduction Configuration
  noise_reduction: {
    enabled: boolean;
    mode: 'light' | 'moderate' | 'aggressive' | 'adaptive';
    strength: number;
    voice_protection: boolean;
    adaptation_rate: number;
  };
  
  // Voice Enhancement Configuration
  voice_enhancement: {
    enabled: boolean;
    clarity_enhancement: number;
    presence_boost: number;
    normalize: boolean;
    sibilance_enhancement: number;
  };
  
  // Compression Configuration
  compression: {
    enabled: boolean;
    mode: 'soft_knee' | 'hard_knee' | 'adaptive';
    threshold: number;
    ratio: number;
    knee: number;
    attack_time: number;
    release_time: number;
  };
  
  // Limiter Configuration
  limiter: {
    enabled: boolean;
    threshold: number;
    release_time: number;
    soft_clip: boolean;
  };
  
  // Pipeline Configuration
  pipeline: {
    enabled_stages: string[];
    pause_after_stage: Record<string, boolean>;
    bypass_on_low_quality: boolean;
    quality_threshold: number;
  };
  
  // General Configuration
  general: {
    preset_name: string;
    sample_rate: number;
    buffer_duration: number;
    processing_timeout: number;
  };
}

interface AudioProcessingSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const defaultConfig: AudioProcessingConfig = {
  vad: {
    enabled: true,
    mode: 'webrtc',
    aggressiveness: 2,
    energy_threshold: 0.01,
    sensitivity: 0.7,
    voice_freq_min: 85,
    voice_freq_max: 300,
  },
  voice_filter: {
    enabled: true,
    fundamental_min: 85,
    fundamental_max: 300,
    voice_band_gain: 1.1,
    preserve_formants: true,
    formant1_min: 200,
    formant1_max: 1000,
  },
  noise_reduction: {
    enabled: true,
    mode: 'moderate',
    strength: 0.7,
    voice_protection: true,
    adaptation_rate: 0.1,
  },
  voice_enhancement: {
    enabled: true,
    clarity_enhancement: 0.3,
    presence_boost: 0.2,
    normalize: false,
    sibilance_enhancement: 0.15,
  },
  compression: {
    enabled: true,
    mode: 'soft_knee',
    threshold: -20,
    ratio: 3.0,
    knee: 2.0,
    attack_time: 5.0,
    release_time: 50.0,
  },
  limiter: {
    enabled: true,
    threshold: -1.0,
    release_time: 50.0,
    soft_clip: true,
  },
  pipeline: {
    enabled_stages: ['vad', 'voice_filter', 'noise_reduction', 'voice_enhancement', 'compression', 'limiter'],
    pause_after_stage: {},
    bypass_on_low_quality: false,
    quality_threshold: 0.3,
  },
  general: {
    preset_name: 'default',
    sample_rate: 16000,
    buffer_duration: 5.0,
    processing_timeout: 30000,
  },
};

const presets = {
  default: 'Default Processing',
  voice_optimized: 'Voice Optimized',
  noisy_environment: 'Noisy Environment',
  music_mode: 'Music Mode',
  minimal_processing: 'Minimal Processing',
  aggressive_cleanup: 'Aggressive Cleanup',
};

const AudioProcessingSettings: React.FC<AudioProcessingSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<AudioProcessingConfig>(defaultConfig);
  const [testingAudio, setTestingAudio] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);

  // Load current configuration from backend
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch('/api/settings/audio');
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultConfig, ...currentConfig });
        }
      } catch (error) {
        console.error('Failed to load audio configuration:', error);
      }
    };
    
    loadConfiguration();
  }, []);

  const handleConfigChange = (section: keyof AudioProcessingConfig, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  const handleStageToggle = (stage: string, enabled: boolean) => {
    setConfig(prev => ({
      ...prev,
      pipeline: {
        ...prev.pipeline,
        enabled_stages: enabled 
          ? [...prev.pipeline.enabled_stages, stage]
          : prev.pipeline.enabled_stages.filter(s => s !== stage),
      },
    }));
  };

  const handlePresetChange = async (presetName: string) => {
    try {
      const response = await fetch(`/api/settings/audio/presets/${presetName}`);
      if (response.ok) {
        const presetConfig = await response.json();
        setConfig({ ...presetConfig, general: { ...config.general, preset_name: presetName } });
        onSave(`Applied ${presets[presetName as keyof typeof presets]} preset`);
      }
    } catch (error) {
      onSave('Failed to load preset', false);
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch('/api/settings/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        onSave('Audio processing settings saved successfully');
      } else {
        onSave('Failed to save audio processing settings', false);
      }
    } catch (error) {
      onSave('Error saving audio processing settings', false);
    }
  };

  const handleTestAudio = async () => {
    setTestingAudio(true);
    try {
      const response = await fetch('/api/settings/audio/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        onSave('Audio processing test completed successfully');
      } else {
        onSave('Audio processing test failed', false);
      }
    } catch (error) {
      onSave('Error testing audio processing', false);
    } finally {
      setTestingAudio(false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultConfig);
    onSave('Configuration reset to defaults');
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Audio Processing Configuration
        </Typography>
        <Box>
          <Tooltip title="Reset to defaults">
            <IconButton onClick={handleResetToDefaults} color="secondary">
              <RestoreIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<PlayArrowIcon />}
            onClick={handleTestAudio}
            disabled={testingAudio}
            sx={{ mr: 2 }}
          >
            {testingAudio ? 'Testing...' : 'Test Configuration'}
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure advanced audio processing parameters for optimal speech recognition and quality.
        Changes are applied in real-time with preview mode.
      </Alert>

      {/* Preset Selection */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Quick Presets" />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Processing Preset</InputLabel>
                <Select
                  value={config.general.preset_name}
                  label="Processing Preset"
                  onChange={(e) => handlePresetChange(e.target.value)}
                >
                  {Object.entries(presets).map(([key, label]) => (
                    <MenuItem key={key} value={key}>{label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={previewMode}
                    onChange={(e) => setPreviewMode(e.target.checked)}
                  />
                }
                label="Real-time Preview"
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Pipeline Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Processing Pipeline" />
        <CardContent>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Configure which processing stages are active and their order
          </Typography>
          <Box sx={{ mb: 2 }}>
            {['vad', 'voice_filter', 'noise_reduction', 'voice_enhancement', 'compression', 'limiter'].map((stage) => (
              <Chip
                key={stage}
                label={stage.replace('_', ' ').toUpperCase()}
                color={config.pipeline.enabled_stages.includes(stage) ? 'primary' : 'default'}
                onClick={() => handleStageToggle(stage, !config.pipeline.enabled_stages.includes(stage))}
                sx={{ mr: 1, mb: 1 }}
              />
            ))}
          </Box>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.pipeline.bypass_on_low_quality}
                    onChange={(e) => handleConfigChange('pipeline', 'bypass_on_low_quality', e.target.checked)}
                  />
                }
                label="Bypass on Low Quality"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Quality Threshold: {config.pipeline.quality_threshold}</Typography>
              <Slider
                value={config.pipeline.quality_threshold}
                onChange={(_, value) => handleConfigChange('pipeline', 'quality_threshold', value)}
                min={0}
                max={1}
                step={0.1}
                marks
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Voice Activity Detection */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Voice Activity Detection (VAD)</Typography>
          <Chip 
            label={config.vad.enabled ? 'Enabled' : 'Disabled'} 
            color={config.vad.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.vad.enabled}
                    onChange={(e) => handleConfigChange('vad', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Voice Activity Detection"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>VAD Mode</InputLabel>
                <Select
                  value={config.vad.mode}
                  label="VAD Mode"
                  onChange={(e) => handleConfigChange('vad', 'mode', e.target.value)}
                  disabled={!config.vad.enabled}
                >
                  <MenuItem value="basic">Basic Energy-based</MenuItem>
                  <MenuItem value="webrtc">WebRTC VAD</MenuItem>
                  <MenuItem value="aggressive">Aggressive</MenuItem>
                  <MenuItem value="silero">Silero VAD</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Aggressiveness: {config.vad.aggressiveness}</Typography>
              <Slider
                value={config.vad.aggressiveness}
                onChange={(_, value) => handleConfigChange('vad', 'aggressiveness', value)}
                min={0}
                max={3}
                step={1}
                marks
                disabled={!config.vad.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Energy Threshold: {config.vad.energy_threshold}</Typography>
              <Slider
                value={config.vad.energy_threshold}
                onChange={(_, value) => handleConfigChange('vad', 'energy_threshold', value)}
                min={0.001}
                max={0.1}
                step={0.001}
                disabled={!config.vad.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Sensitivity: {config.vad.sensitivity}</Typography>
              <Slider
                value={config.vad.sensitivity}
                onChange={(_, value) => handleConfigChange('vad', 'sensitivity', value)}
                min={0}
                max={1}
                step={0.1}
                disabled={!config.vad.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Voice Frequency Filter */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Voice Frequency Filter</Typography>
          <Chip 
            label={config.voice_filter.enabled ? 'Enabled' : 'Disabled'} 
            color={config.voice_filter.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.voice_filter.enabled}
                    onChange={(e) => handleConfigChange('voice_filter', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Voice Frequency Filtering"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Fundamental Min (Hz)"
                type="number"
                value={config.voice_filter.fundamental_min}
                onChange={(e) => handleConfigChange('voice_filter', 'fundamental_min', Number(e.target.value))}
                disabled={!config.voice_filter.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Fundamental Max (Hz)"
                type="number"
                value={config.voice_filter.fundamental_max}
                onChange={(e) => handleConfigChange('voice_filter', 'fundamental_max', Number(e.target.value))}
                disabled={!config.voice_filter.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Voice Band Gain: {config.voice_filter.voice_band_gain}</Typography>
              <Slider
                value={config.voice_filter.voice_band_gain}
                onChange={(_, value) => handleConfigChange('voice_filter', 'voice_band_gain', value)}
                min={0.5}
                max={2.0}
                step={0.1}
                disabled={!config.voice_filter.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.voice_filter.preserve_formants}
                    onChange={(e) => handleConfigChange('voice_filter', 'preserve_formants', e.target.checked)}
                    disabled={!config.voice_filter.enabled}
                  />
                }
                label="Preserve Formants"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Noise Reduction */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Noise Reduction</Typography>
          <Chip 
            label={config.noise_reduction.enabled ? 'Enabled' : 'Disabled'} 
            color={config.noise_reduction.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.noise_reduction.enabled}
                    onChange={(e) => handleConfigChange('noise_reduction', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Noise Reduction"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Noise Reduction Mode</InputLabel>
                <Select
                  value={config.noise_reduction.mode}
                  label="Noise Reduction Mode"
                  onChange={(e) => handleConfigChange('noise_reduction', 'mode', e.target.value)}
                  disabled={!config.noise_reduction.enabled}
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="moderate">Moderate</MenuItem>
                  <MenuItem value="aggressive">Aggressive</MenuItem>
                  <MenuItem value="adaptive">Adaptive</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Strength: {config.noise_reduction.strength}</Typography>
              <Slider
                value={config.noise_reduction.strength}
                onChange={(_, value) => handleConfigChange('noise_reduction', 'strength', value)}
                min={0}
                max={1}
                step={0.1}
                disabled={!config.noise_reduction.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.noise_reduction.voice_protection}
                    onChange={(e) => handleConfigChange('noise_reduction', 'voice_protection', e.target.checked)}
                    disabled={!config.noise_reduction.enabled}
                  />
                }
                label="Voice Protection"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Voice Enhancement */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Voice Enhancement</Typography>
          <Chip 
            label={config.voice_enhancement.enabled ? 'Enabled' : 'Disabled'} 
            color={config.voice_enhancement.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.voice_enhancement.enabled}
                    onChange={(e) => handleConfigChange('voice_enhancement', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Voice Enhancement"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Clarity Enhancement: {config.voice_enhancement.clarity_enhancement}</Typography>
              <Slider
                value={config.voice_enhancement.clarity_enhancement}
                onChange={(_, value) => handleConfigChange('voice_enhancement', 'clarity_enhancement', value)}
                min={0}
                max={1}
                step={0.1}
                disabled={!config.voice_enhancement.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Presence Boost: {config.voice_enhancement.presence_boost}</Typography>
              <Slider
                value={config.voice_enhancement.presence_boost}
                onChange={(_, value) => handleConfigChange('voice_enhancement', 'presence_boost', value)}
                min={0}
                max={1}
                step={0.1}
                disabled={!config.voice_enhancement.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.voice_enhancement.normalize}
                    onChange={(e) => handleConfigChange('voice_enhancement', 'normalize', e.target.checked)}
                    disabled={!config.voice_enhancement.enabled}
                  />
                }
                label="Normalize Output"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Dynamic Compression */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Dynamic Range Compression</Typography>
          <Chip 
            label={config.compression.enabled ? 'Enabled' : 'Disabled'} 
            color={config.compression.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.compression.enabled}
                    onChange={(e) => handleConfigChange('compression', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Dynamic Compression"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Compression Mode</InputLabel>
                <Select
                  value={config.compression.mode}
                  label="Compression Mode"
                  onChange={(e) => handleConfigChange('compression', 'mode', e.target.value)}
                  disabled={!config.compression.enabled}
                >
                  <MenuItem value="soft_knee">Soft Knee</MenuItem>
                  <MenuItem value="hard_knee">Hard Knee</MenuItem>
                  <MenuItem value="adaptive">Adaptive</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Threshold (dB)"
                type="number"
                value={config.compression.threshold}
                onChange={(e) => handleConfigChange('compression', 'threshold', Number(e.target.value))}
                disabled={!config.compression.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Ratio: {config.compression.ratio}</Typography>
              <Slider
                value={config.compression.ratio}
                onChange={(_, value) => handleConfigChange('compression', 'ratio', value)}
                min={1}
                max={10}
                step={0.5}
                disabled={!config.compression.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Knee: {config.compression.knee}</Typography>
              <Slider
                value={config.compression.knee}
                onChange={(_, value) => handleConfigChange('compression', 'knee', value)}
                min={0}
                max={5}
                step={0.5}
                disabled={!config.compression.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Audio Limiter */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Audio Limiter</Typography>
          <Chip 
            label={config.limiter.enabled ? 'Enabled' : 'Disabled'} 
            color={config.limiter.enabled ? 'success' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.limiter.enabled}
                    onChange={(e) => handleConfigChange('limiter', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Audio Limiting"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Threshold (dB)"
                type="number"
                value={config.limiter.threshold}
                onChange={(e) => handleConfigChange('limiter', 'threshold', Number(e.target.value))}
                disabled={!config.limiter.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Release Time (ms)"
                type="number"
                value={config.limiter.release_time}
                onChange={(e) => handleConfigChange('limiter', 'release_time', Number(e.target.value))}
                disabled={!config.limiter.enabled}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.limiter.soft_clip}
                    onChange={(e) => handleConfigChange('limiter', 'soft_clip', e.target.checked)}
                    disabled={!config.limiter.enabled}
                  />
                }
                label="Soft Clipping"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* General Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">General Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Sample Rate</InputLabel>
                <Select
                  value={config.general.sample_rate}
                  label="Sample Rate"
                  onChange={(e) => handleConfigChange('general', 'sample_rate', e.target.value)}
                >
                  <MenuItem value={8000}>8 kHz</MenuItem>
                  <MenuItem value={16000}>16 kHz</MenuItem>
                  <MenuItem value={22050}>22.05 kHz</MenuItem>
                  <MenuItem value={44100}>44.1 kHz</MenuItem>
                  <MenuItem value={48000}>48 kHz</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Buffer Duration (seconds)"
                type="number"
                value={config.general.buffer_duration}
                onChange={(e) => handleConfigChange('general', 'buffer_duration', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Processing Timeout (ms)"
                type="number"
                value={config.general.processing_timeout}
                onChange={(e) => handleConfigChange('general', 'processing_timeout', Number(e.target.value))}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default AudioProcessingSettings;