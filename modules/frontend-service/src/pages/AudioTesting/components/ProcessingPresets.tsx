import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
  Slider,
  Paper,
} from '@mui/material';
import {
  Save as SaveIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

interface PresetConfig {
  vad: { enabled: boolean; aggressiveness: number };
  voiceFilter: { enabled: boolean; lowcut?: number; highcut?: number };
  noiseReduction: { enabled: boolean; strength?: number };
  enhancement: { enabled: boolean; compression?: number };
}

interface PresetConfiguration {
  id: string;
  name: string;
  icon: string;
  description: string;
  config: PresetConfig;
}

const presetConfigurations: PresetConfiguration[] = [
  {
    id: 'speech',
    name: 'Speech',
    icon: '🗣️',
    description: 'Optimized for clear speech recording',
    config: {
      vad: { enabled: true, aggressiveness: 2 },
      voiceFilter: { enabled: true, lowcut: 85, highcut: 300 },
      noiseReduction: { enabled: true, strength: 0.3 },
      enhancement: { enabled: true, compression: 2.5 },
    },
  },
  {
    id: 'podcast',
    name: 'Podcast',
    icon: '🎙️',
    description: 'Professional podcast recording quality',
    config: {
      vad: { enabled: true, aggressiveness: 1 },
      voiceFilter: { enabled: true, lowcut: 80, highcut: 350 },
      noiseReduction: { enabled: true, strength: 0.5 },
      enhancement: { enabled: true, compression: 3.0 },
    },
  },
  {
    id: 'noisy',
    name: 'Noisy Environment',
    icon: '🔊',
    description: 'Aggressive noise reduction for noisy environments',
    config: {
      vad: { enabled: true, aggressiveness: 3 },
      voiceFilter: { enabled: true, lowcut: 100, highcut: 280 },
      noiseReduction: { enabled: true, strength: 0.8 },
      enhancement: { enabled: true, compression: 4.0 },
    },
  },
  {
    id: 'clean',
    name: 'Clean Audio',
    icon: '✨',
    description: 'Minimal processing for clean audio sources',
    config: {
      vad: { enabled: true, aggressiveness: 1 },
      voiceFilter: { enabled: false },
      noiseReduction: { enabled: false },
      enhancement: { enabled: true, compression: 1.5 },
    },
  },
  {
    id: 'music',
    name: 'Music Vocal',
    icon: '🎵',
    description: 'Optimized for vocals in music',
    config: {
      vad: { enabled: true, aggressiveness: 1 },
      voiceFilter: { enabled: true, lowcut: 70, highcut: 400 },
      noiseReduction: { enabled: true, strength: 0.2 },
      enhancement: { enabled: true, compression: 2.0 },
    },
  },
  {
    id: 'broadcast',
    name: 'Broadcast',
    icon: '📻',
    description: 'Broadcast-quality audio processing',
    config: {
      vad: { enabled: true, aggressiveness: 2 },
      voiceFilter: { enabled: true, lowcut: 90, highcut: 320 },
      noiseReduction: { enabled: true, strength: 0.4 },
      enhancement: { enabled: true, compression: 3.5 },
    },
  },
];

export const ProcessingPresets: React.FC = () => {
  const [selectedPreset, setSelectedPreset] = useState<string>('speech');
  const [customDialogOpen, setCustomDialogOpen] = useState(false);
  const [customPreset, setCustomPreset] = useState<{
    name: string;
    description: string;
    config: PresetConfig;
  }>({
    name: '',
    description: '',
    config: presetConfigurations[0].config,
  });

  const handlePresetSelect = (presetId: string) => {
    setSelectedPreset(presetId);
    const preset = presetConfigurations.find(p => p.id === presetId);
    if (preset) {
      // Apply preset configuration
      console.log('Applying preset:', preset.name, preset.config);
    }
  };

  const handleSaveCustomPreset = () => {
    // Save custom preset logic
    console.log('Saving custom preset:', customPreset);
    setCustomDialogOpen(false);
  };

  const handleExportPresets = () => {
    const dataStr = JSON.stringify(presetConfigurations, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'audio-processing-presets.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleImportPresets = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target?.result as string);
          console.log('Imported presets:', imported);
          // Handle imported presets
        } catch (error) {
          console.error('Error importing presets:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          ⚙️ Processing Presets
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<SaveIcon />}
            onClick={() => setCustomDialogOpen(true)}
            size="small"
          >
            Create Custom
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportPresets}
            size="small"
          >
            Export
          </Button>
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            component="label"
            size="small"
          >
            Import
            <input
              type="file"
              accept=".json"
              hidden
              onChange={handleImportPresets}
            />
          </Button>
        </Stack>
      </Box>

      <Grid container spacing={2}>
        {presetConfigurations.map((preset) => (
          <Grid item xs={12} sm={6} md={4} key={preset.id}>
            <Card
              sx={{
                cursor: 'pointer',
                border: selectedPreset === preset.id ? 2 : 1,
                borderColor: selectedPreset === preset.id ? 'primary.main' : 'divider',
                transition: 'all 0.3s ease',
                '&:hover': {
                  borderColor: 'primary.main',
                  transform: 'translateY(-2px)',
                  boxShadow: 3,
                },
              }}
              onClick={() => handlePresetSelect(preset.id)}
            >
              <CardContent>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      fontSize: '2rem',
                      width: 48,
                      height: 48,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: selectedPreset === preset.id ? 'primary.main' : 'grey.100',
                      borderRadius: '50%',
                    }}
                  >
                    {preset.icon}
                  </Box>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" component="h3">
                      {preset.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {preset.description}
                    </Typography>
                  </Box>
                </Stack>

                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {Object.entries(preset.config).map(([key, value]) => (
                    <Chip
                      key={key}
                      label={`${key}: ${(value as any).enabled ? 'ON' : 'OFF'}`}
                      size="small"
                      color={(value as any).enabled ? 'primary' : 'default'}
                      variant="outlined"
                    />
                  ))}
                </Stack>
              </CardContent>

              <CardActions>
                <Button
                  size="small"
                  startIcon={<SettingsIcon />}
                  disabled={selectedPreset !== preset.id}
                >
                  Configure
                </Button>
                {selectedPreset === preset.id && (
                  <Chip
                    label="Active"
                    color="primary"
                    size="small"
                    sx={{ ml: 'auto' }}
                  />
                )}
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Custom Preset Dialog */}
      <Dialog
        open={customDialogOpen}
        onClose={() => setCustomDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create Custom Preset</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 1 }}>
            <TextField
              label="Preset Name"
              value={customPreset.name}
              onChange={(e) => setCustomPreset({ ...customPreset, name: e.target.value })}
              fullWidth
            />
            <TextField
              label="Description"
              value={customPreset.description}
              onChange={(e) => setCustomPreset({ ...customPreset, description: e.target.value })}
              fullWidth
              multiline
              rows={2}
            />

            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Voice Activity Detection
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={customPreset.config.vad.enabled}
                    onChange={(e) => setCustomPreset({
                      ...customPreset,
                      config: {
                        ...customPreset.config,
                        vad: { ...customPreset.config.vad, enabled: e.target.checked }
                      }
                    })}
                  />
                }
                label="Enable VAD"
              />
              <Typography gutterBottom>
                Aggressiveness: {customPreset.config.vad.aggressiveness}
              </Typography>
              <Slider
                value={customPreset.config.vad.aggressiveness}
                onChange={(_, value) => setCustomPreset({
                  ...customPreset,
                  config: {
                    ...customPreset.config,
                    vad: { ...customPreset.config.vad, aggressiveness: value as number }
                  }
                })}
                min={0}
                max={3}
                step={1}
                marks
                disabled={!customPreset.config.vad.enabled}
              />
            </Paper>

            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Voice Filter
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={customPreset.config.voiceFilter.enabled}
                    onChange={(e) => setCustomPreset({
                      ...customPreset,
                      config: {
                        ...customPreset.config,
                        voiceFilter: { ...customPreset.config.voiceFilter, enabled: e.target.checked }
                      }
                    })}
                  />
                }
                label="Enable Voice Filter"
              />
              <Typography gutterBottom>
                Low Cut: {customPreset.config.voiceFilter.lowcut}Hz
              </Typography>
              <Slider
                value={customPreset.config.voiceFilter.lowcut}
                onChange={(_, value) => setCustomPreset({
                  ...customPreset,
                  config: {
                    ...customPreset.config,
                    voiceFilter: { ...customPreset.config.voiceFilter, lowcut: value as number }
                  }
                })}
                min={50}
                max={200}
                disabled={!customPreset.config.voiceFilter.enabled}
              />
              <Typography gutterBottom>
                High Cut: {customPreset.config.voiceFilter.highcut}Hz
              </Typography>
              <Slider
                value={customPreset.config.voiceFilter.highcut}
                onChange={(_, value) => setCustomPreset({
                  ...customPreset,
                  config: {
                    ...customPreset.config,
                    voiceFilter: { ...customPreset.config.voiceFilter, highcut: value as number }
                  }
                })}
                min={250}
                max={500}
                disabled={!customPreset.config.voiceFilter.enabled}
              />
            </Paper>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCustomDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSaveCustomPreset} variant="contained">
            Save Preset
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};