import React, { useState } from 'react';
import {
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Typography,
  Switch,
  FormControlLabel,
  FormGroup,
  Button,
  Stack,
  Chip,
  Alert,
} from '@mui/material';
import { useAppSelector, useAppDispatch } from '@/store';
import { updateRecordingConfig } from '@/store/slices/audioSlice';

export const AudioConfiguration: React.FC = () => {
  const dispatch = useAppDispatch();
  const { devices, config } = useAppSelector(state => state.audio);
  const [selectedSource, setSelectedSource] = useState<'microphone' | 'file' | 'sample'>('microphone');

  const handleConfigChange = (key: string, value: any) => {
    dispatch(updateRecordingConfig({ [key]: value }));
  };

  const handleSourceChange = (source: 'microphone' | 'file' | 'sample') => {
    setSelectedSource(source);
    handleConfigChange('source', source);
  };

  const sampleRates = [
    { value: 16000, label: '16 kHz (Recommended)' },
    { value: 22050, label: '22 kHz' },
    { value: 44100, label: '44 kHz' },
    { value: 48000, label: '48 kHz' },
  ];

  const audioFormats = [
    { value: 'audio/webm;codecs=opus', label: 'WebM/Opus (Best Quality)', extension: 'webm' },
    { value: 'audio/mp4;codecs=mp4a.40.2', label: 'MP4/AAC (High Compatibility)', extension: 'mp4' },
    { value: 'audio/ogg;codecs=opus', label: 'OGG/Opus (Open Source)', extension: 'ogg' },
    { value: 'audio/wav', label: 'WAV (Uncompressed)', extension: 'wav' },
  ];

  const qualityOptions = [
    { value: 'lossless', label: 'Lossless (WAV) - Meeting Recommended' },
    { value: 'high', label: 'High (256 kbps)' },
    { value: 'medium', label: 'Medium (128 kbps)' },
    { value: 'low', label: 'Low (64 kbps)' },
  ];

  return (
    <Paper sx={{ p: 3 }}>
      <Grid container spacing={3}>
        {/* Recording Duration - Meeting Optimized */}
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>
            Recording Duration: {config.duration}s
          </Typography>
          <Slider
            value={config.duration}
            onChange={(_, value) => handleConfigChange('duration', value)}
            min={1}
            max={30}
            marks={[
              { value: 1, label: '1s' },
              { value: 5, label: '5s' },
              { value: 10, label: '10s' },
              { value: 15, label: '15s' },
              { value: 30, label: '30s' },
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}s`}
          />
          <Typography variant="caption" color="text.secondary">
            Meeting-optimized duration (1-30 seconds for real-time processing)
          </Typography>
        </Grid>

        {/* Audio Device Selection */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Audio Device</InputLabel>
            <Select
              value={config.deviceId}
              label="Audio Device"
              onChange={(e) => handleConfigChange('deviceId', e.target.value)}
            >
              <MenuItem value="">System Default</MenuItem>
              {devices.map((device) => (
                <MenuItem key={device.deviceId} value={device.deviceId}>
                  {device.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary">
            Select microphone input device
          </Typography>
        </Grid>

        {/* Sample Rate */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Sample Rate</InputLabel>
            <Select
              value={config.sampleRate}
              label="Sample Rate"
              onChange={(e) => handleConfigChange('sampleRate', e.target.value)}
            >
              {sampleRates.map((rate) => (
                <MenuItem key={rate.value} value={rate.value}>
                  {rate.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary">
            Audio sample rate for recording
          </Typography>
        </Grid>

        {/* Recording Format */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Recording Format</InputLabel>
            <Select
              value={config.format}
              label="Recording Format"
              onChange={(e) => handleConfigChange('format', e.target.value)}
            >
              {audioFormats.map((format) => (
                <MenuItem key={format.value} value={format.value}>
                  {format.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary">
            Audio format for recording (higher quality codecs)
          </Typography>
        </Grid>

        {/* Audio Quality */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Audio Quality</InputLabel>
            <Select
              value={config.quality}
              label="Audio Quality"
              onChange={(e) => handleConfigChange('quality', e.target.value)}
            >
              {qualityOptions.map((quality) => (
                <MenuItem key={quality.value} value={quality.value}>
                  {quality.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary">
            Lossless quality recommended for meeting audio accuracy
          </Typography>
        </Grid>

        {/* Auto-stop Recording */}
        <Grid item xs={12} md={6}>
          <FormControlLabel
            control={
              <Switch
                checked={config.autoStop}
                onChange={(e) => handleConfigChange('autoStop', e.target.checked)}
              />
            }
            label="Auto-Stop Recording"
          />
          <Typography variant="caption" color="text.secondary" display="block">
            Automatically stop recording after set duration
          </Typography>
        </Grid>

        {/* Audio Processing Controls */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Audio Processing
          </Typography>
          <FormGroup row>
            <FormControlLabel
              control={
                <Switch
                  checked={config.echoCancellation}
                  onChange={(e) => handleConfigChange('echoCancellation', e.target.checked)}
                  disabled={config.rawAudio}
                />
              }
              label="Echo Cancellation"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={config.noiseSuppression}
                  onChange={(e) => handleConfigChange('noiseSuppression', e.target.checked)}
                  disabled={config.rawAudio}
                />
              }
              label="Noise Suppression"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={config.autoGainControl}
                  onChange={(e) => handleConfigChange('autoGainControl', e.target.checked)}
                  disabled={config.rawAudio}
                />
              }
              label="Auto Gain Control"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={config.rawAudio}
                  onChange={(e) => handleConfigChange('rawAudio', e.target.checked)}
                />
              }
              label="Raw Audio (Disable All Processing)"
            />
          </FormGroup>
          <Typography variant="caption" color="text.secondary">
            Control browser audio processing features
          </Typography>
        </Grid>

        {/* Meeting-Specific Presets */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            🏢 Meeting Presets
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 2 }}>
            <Chip
              icon={<span>🏠</span>}
              label="Conference Room"
              clickable
              variant="outlined"
              onClick={() => {
                handleConfigChange('duration', 15);
                handleConfigChange('quality', 'lossless');
                handleConfigChange('sampleRate', 16000);
                handleConfigChange('echoCancellation', true);
                handleConfigChange('noiseSuppression', true);
                handleConfigChange('autoGainControl', false);
                handleConfigChange('rawAudio', false);
              }}
            />
            <Chip
              icon={<span>💻</span>}
              label="Virtual Meeting"
              clickable
              variant="outlined"
              onClick={() => {
                handleConfigChange('duration', 10);
                handleConfigChange('quality', 'high');
                handleConfigChange('sampleRate', 16000);
                handleConfigChange('echoCancellation', false);
                handleConfigChange('noiseSuppression', false);
                handleConfigChange('autoGainControl', false);
                handleConfigChange('rawAudio', true);
              }}
            />
            <Chip
              icon={<span>🔊</span>}
              label="Noisy Environment"
              clickable
              variant="outlined"
              onClick={() => {
                handleConfigChange('duration', 20);
                handleConfigChange('quality', 'lossless');
                handleConfigChange('sampleRate', 22050);
                handleConfigChange('echoCancellation', true);
                handleConfigChange('noiseSuppression', true);
                handleConfigChange('autoGainControl', true);
                handleConfigChange('rawAudio', false);
              }}
            />
            <Chip
              icon={<span>🎙️</span>}
              label="Interview/Presentation"
              clickable
              variant="outlined"
              onClick={() => {
                handleConfigChange('duration', 30);
                handleConfigChange('quality', 'lossless');
                handleConfigChange('sampleRate', 22050);
                handleConfigChange('echoCancellation', false);
                handleConfigChange('noiseSuppression', false);
                handleConfigChange('autoGainControl', false);
                handleConfigChange('rawAudio', true);
              }}
            />
          </Stack>
          <Typography variant="caption" color="text.secondary">
            Click preset to optimize settings for specific meeting environments
          </Typography>
        </Grid>

        {/* Audio Source Selection */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Audio Source
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Chip
              icon={<span>🎤</span>}
              label="Microphone"
              clickable
              color={selectedSource === 'microphone' ? 'primary' : 'default'}
              variant={selectedSource === 'microphone' ? 'filled' : 'outlined'}
              onClick={() => handleSourceChange('microphone')}
            />
            <Chip
              icon={<span>📁</span>}
              label="File Upload"
              clickable
              color={selectedSource === 'file' ? 'primary' : 'default'}
              variant={selectedSource === 'file' ? 'filled' : 'outlined'}
              onClick={() => handleSourceChange('file')}
            />
            <Chip
              icon={<span>🎼</span>}
              label="Test Sample"
              clickable
              color={selectedSource === 'sample' ? 'primary' : 'default'}
              variant={selectedSource === 'sample' ? 'filled' : 'outlined'}
              onClick={() => handleSourceChange('sample')}
            />
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            Select audio input source
          </Typography>
        </Grid>

        {/* File Upload (shown when file source is selected) */}
        {selectedSource === 'file' && (
          <Grid item xs={12}>
            <Alert severity="info" sx={{ mb: 2 }}>
              File upload functionality will be available when you select "File Upload" source.
            </Alert>
            <Button
              variant="outlined"
              component="label"
              fullWidth
              sx={{ height: 60 }}
            >
              Choose Audio File
              <input
                type="file"
                accept="audio/*"
                hidden
                onChange={(e) => {
                  // Handle file upload
                  const file = e.target.files?.[0];
                  if (file) {
                    // TODO: Implement file handling
                    console.log('File selected:', file.name);
                  }
                }}
              />
            </Button>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};