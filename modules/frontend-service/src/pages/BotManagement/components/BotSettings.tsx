import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  Switch,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Slider,
  Alert,
  Tabs,
  Tab,
  Stack,
  Paper,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Save as SaveIcon,
  RestoreFromTrash as RestoreIcon,
  ExpandMore as ExpandMoreIcon,
  VolumeUp as AudioIcon,
  Language as LanguageIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';
import { TabPanel } from '@/components/ui';
import { SUPPORTED_LANGUAGES } from '@/constants/languages';

interface BotSettingsProps {
  onSettingsUpdate: (settings: BotConfiguration) => void;
}

interface BotConfiguration {
  // Audio Processing Settings
  audioProcessing: {
    sampleRate: number;
    channels: number;
    bitDepth: number;
    vadEnabled: boolean;
    vadAggressiveness: number;
    noiseReduction: boolean;
    noiseReductionLevel: number;
    echoCancellation: boolean;
    autoGainControl: boolean;
  };
  
  // Translation Settings
  translation: {
    enabledLanguages: string[];
    defaultSourceLanguage: string;
    simultaneousTranslation: boolean;
    translationQuality: 'fast' | 'balanced' | 'accurate';
    confidenceThreshold: number;
    maxTranslationLength: number;
    fallbackLanguage: string;
  };
  
  // Performance Settings
  performance: {
    maxConcurrentBots: number;
    processingTimeout: number;
    retryAttempts: number;
    memoryLimit: number;
    cpuThreshold: number;
    enableProfiling: boolean;
    logLevel: 'debug' | 'info' | 'warning' | 'error';
  };
  
  // Security Settings
  security: {
    enableEncryption: boolean;
    sessionTimeout: number;
    maxSessionDuration: number;
    enableRateLimiting: boolean;
    rateLimit: number;
    allowedDomains: string[];
    enableAuditLog: boolean;
  };
  
  // Storage Settings
  storage: {
    retentionPeriod: number;
    enableBackup: boolean;
    backupFrequency: number;
    compressionEnabled: boolean;
    maxStorageSize: number;
    autoCleanup: boolean;
  };
}

const defaultSettings: BotConfiguration = {
  audioProcessing: {
    sampleRate: 16000,
    channels: 1,
    bitDepth: 16,
    vadEnabled: true,
    vadAggressiveness: 2,
    noiseReduction: true,
    noiseReductionLevel: 0.5,
    echoCancellation: true,
    autoGainControl: true,
  },
  translation: {
    enabledLanguages: ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ru'],
    defaultSourceLanguage: 'en',
    simultaneousTranslation: true,
    translationQuality: 'balanced',
    confidenceThreshold: 0.7,
    maxTranslationLength: 500,
    fallbackLanguage: 'en',
  },
  performance: {
    maxConcurrentBots: 10,
    processingTimeout: 30000,
    retryAttempts: 3,
    memoryLimit: 2048,
    cpuThreshold: 80,
    enableProfiling: false,
    logLevel: 'info',
  },
  security: {
    enableEncryption: true,
    sessionTimeout: 1800,
    maxSessionDuration: 14400,
    enableRateLimiting: true,
    rateLimit: 100,
    allowedDomains: ['meet.google.com', 'teams.microsoft.com'],
    enableAuditLog: true,
  },
  storage: {
    retentionPeriod: 30,
    enableBackup: true,
    backupFrequency: 24,
    compressionEnabled: true,
    maxStorageSize: 10240,
    autoCleanup: true,
  },
};

const availableLanguages = SUPPORTED_LANGUAGES;

export const BotSettings: React.FC<BotSettingsProps> = ({ onSettingsUpdate }) => {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState<BotConfiguration>(defaultSettings);
  const [hasChanges, setHasChanges] = useState(false);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState<{
    show: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({ show: false, message: '', severity: 'info' });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/bot/settings');
      if (response.ok) {
        const data = await response.json();
        setSettings({ ...defaultSettings, ...data });
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const handleSettingChange = (section: keyof BotConfiguration, field: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSaveSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/bot/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        setHasChanges(false);
        setNotification({
          show: true,
          message: 'Settings saved successfully',
          severity: 'success',
        });
        onSettingsUpdate(settings);
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      setNotification({
        show: true,
        message: 'Failed to save settings',
        severity: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleResetSettings = () => {
    setSettings(defaultSettings);
    setHasChanges(true);
  };

  const toggleLanguage = (languageCode: string) => {
    const currentLanguages = settings.translation.enabledLanguages;
    const newLanguages = currentLanguages.includes(languageCode)
      ? currentLanguages.filter(lang => lang !== languageCode)
      : [...currentLanguages, languageCode];
    
    handleSettingChange('translation', 'enabledLanguages', newLanguages);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Bot Configuration Settings
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            onClick={handleResetSettings}
            startIcon={<RestoreIcon />}
          >
            Reset to Defaults
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveSettings}
            disabled={!hasChanges || loading}
            startIcon={<SaveIcon />}
          >
            {loading ? 'Saving...' : 'Save Settings'}
          </Button>
        </Stack>
      </Box>

      {notification.show && (
        <Alert 
          severity={notification.severity} 
          sx={{ mb: 2 }}
          onClose={() => setNotification(prev => ({ ...prev, show: false }))}
        >
          {notification.message}
        </Alert>
      )}

      <Paper sx={{ width: '100%' }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab icon={<AudioIcon />} label="Audio Processing" />
          <Tab icon={<LanguageIcon />} label="Translation" />
          <Tab icon={<SpeedIcon />} label="Performance" />
          <Tab icon={<SecurityIcon />} label="Security" />
          <Tab icon={<StorageIcon />} label="Storage" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Audio Quality</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControl fullWidth>
                      <InputLabel>Sample Rate</InputLabel>
                      <Select
                        value={settings.audioProcessing.sampleRate}
                        label="Sample Rate"
                        onChange={(e) => handleSettingChange('audioProcessing', 'sampleRate', e.target.value)}
                      >
                        <MenuItem value={8000}>8000 Hz</MenuItem>
                        <MenuItem value={16000}>16000 Hz (Recommended)</MenuItem>
                        <MenuItem value={44100}>44100 Hz</MenuItem>
                        <MenuItem value={48000}>48000 Hz</MenuItem>
                      </Select>
                    </FormControl>

                    <FormControl fullWidth>
                      <InputLabel>Channels</InputLabel>
                      <Select
                        value={settings.audioProcessing.channels}
                        label="Channels"
                        onChange={(e) => handleSettingChange('audioProcessing', 'channels', e.target.value)}
                      >
                        <MenuItem value={1}>Mono (Recommended)</MenuItem>
                        <MenuItem value={2}>Stereo</MenuItem>
                      </Select>
                    </FormControl>

                    <FormControl fullWidth>
                      <InputLabel>Bit Depth</InputLabel>
                      <Select
                        value={settings.audioProcessing.bitDepth}
                        label="Bit Depth"
                        onChange={(e) => handleSettingChange('audioProcessing', 'bitDepth', e.target.value)}
                      >
                        <MenuItem value={16}>16-bit (Recommended)</MenuItem>
                        <MenuItem value={24}>24-bit</MenuItem>
                        <MenuItem value={32}>32-bit</MenuItem>
                      </Select>
                    </FormControl>
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>

            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Audio Processing</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.audioProcessing.vadEnabled}
                          onChange={(e) => handleSettingChange('audioProcessing', 'vadEnabled', e.target.checked)}
                        />
                      }
                      label="Voice Activity Detection"
                    />

                    {settings.audioProcessing.vadEnabled && (
                      <Box>
                        <Typography variant="body2" gutterBottom>
                          VAD Aggressiveness: {settings.audioProcessing.vadAggressiveness}
                        </Typography>
                        <Slider
                          value={settings.audioProcessing.vadAggressiveness}
                          onChange={(_, value) => handleSettingChange('audioProcessing', 'vadAggressiveness', value)}
                          min={0}
                          max={3}
                          step={1}
                          marks={[
                            { value: 0, label: 'Least' },
                            { value: 1, label: 'Low' },
                            { value: 2, label: 'Normal' },
                            { value: 3, label: 'Most' },
                          ]}
                        />
                      </Box>
                    )}

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.audioProcessing.noiseReduction}
                          onChange={(e) => handleSettingChange('audioProcessing', 'noiseReduction', e.target.checked)}
                        />
                      }
                      label="Noise Reduction"
                    />

                    {settings.audioProcessing.noiseReduction && (
                      <Box>
                        <Typography variant="body2" gutterBottom>
                          Noise Reduction Level: {Math.round(settings.audioProcessing.noiseReductionLevel * 100)}%
                        </Typography>
                        <Slider
                          value={settings.audioProcessing.noiseReductionLevel}
                          onChange={(_, value) => handleSettingChange('audioProcessing', 'noiseReductionLevel', value)}
                          min={0}
                          max={1}
                          step={0.1}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                        />
                      </Box>
                    )}

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.audioProcessing.echoCancellation}
                          onChange={(e) => handleSettingChange('audioProcessing', 'echoCancellation', e.target.checked)}
                        />
                      }
                      label="Echo Cancellation"
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.audioProcessing.autoGainControl}
                          onChange={(e) => handleSettingChange('audioProcessing', 'autoGainControl', e.target.checked)}
                        />
                      }
                      label="Auto Gain Control"
                    />
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Language Configuration</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControl fullWidth>
                      <InputLabel>Default Source Language</InputLabel>
                      <Select
                        value={settings.translation.defaultSourceLanguage}
                        label="Default Source Language"
                        onChange={(e) => handleSettingChange('translation', 'defaultSourceLanguage', e.target.value)}
                      >
                        {availableLanguages.map((lang) => (
                          <MenuItem key={lang.code} value={lang.code}>
                            {lang.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <FormControl fullWidth>
                      <InputLabel>Fallback Language</InputLabel>
                      <Select
                        value={settings.translation.fallbackLanguage}
                        label="Fallback Language"
                        onChange={(e) => handleSettingChange('translation', 'fallbackLanguage', e.target.value)}
                      >
                        {availableLanguages.map((lang) => (
                          <MenuItem key={lang.code} value={lang.code}>
                            {lang.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Typography variant="body2" gutterBottom>
                      Enabled Languages ({settings.translation.enabledLanguages.length})
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {availableLanguages.map((language) => (
                        <Chip
                          key={language.code}
                          label={language.name}
                          clickable
                          color={settings.translation.enabledLanguages.includes(language.code) ? 'primary' : 'default'}
                          variant={settings.translation.enabledLanguages.includes(language.code) ? 'filled' : 'outlined'}
                          onClick={() => toggleLanguage(language.code)}
                        />
                      ))}
                    </Box>
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>

            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Translation Quality</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControl fullWidth>
                      <InputLabel>Translation Quality</InputLabel>
                      <Select
                        value={settings.translation.translationQuality}
                        label="Translation Quality"
                        onChange={(e) => handleSettingChange('translation', 'translationQuality', e.target.value)}
                      >
                        <MenuItem value="fast">Fast (Lower accuracy)</MenuItem>
                        <MenuItem value="balanced">Balanced (Recommended)</MenuItem>
                        <MenuItem value="accurate">Accurate (Higher latency)</MenuItem>
                      </Select>
                    </FormControl>

                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Confidence Threshold: {Math.round(settings.translation.confidenceThreshold * 100)}%
                      </Typography>
                      <Slider
                        value={settings.translation.confidenceThreshold}
                        onChange={(_, value) => handleSettingChange('translation', 'confidenceThreshold', value)}
                        min={0.1}
                        max={1}
                        step={0.05}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                      />
                    </Box>

                    <TextField
                      fullWidth
                      label="Max Translation Length"
                      type="number"
                      value={settings.translation.maxTranslationLength}
                      onChange={(e) => handleSettingChange('translation', 'maxTranslationLength', parseInt(e.target.value))}
                      helperText="Maximum characters per translation"
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.translation.simultaneousTranslation}
                          onChange={(e) => handleSettingChange('translation', 'simultaneousTranslation', e.target.checked)}
                        />
                      }
                      label="Simultaneous Translation"
                    />
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Resource Limits</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="Max Concurrent Bots"
                      type="number"
                      value={settings.performance.maxConcurrentBots}
                      onChange={(e) => handleSettingChange('performance', 'maxConcurrentBots', parseInt(e.target.value))}
                      helperText="Maximum number of bots running simultaneously"
                    />

                    <TextField
                      fullWidth
                      label="Processing Timeout (ms)"
                      type="number"
                      value={settings.performance.processingTimeout}
                      onChange={(e) => handleSettingChange('performance', 'processingTimeout', parseInt(e.target.value))}
                      helperText="Maximum time for processing operations"
                    />

                    <TextField
                      fullWidth
                      label="Memory Limit (MB)"
                      type="number"
                      value={settings.performance.memoryLimit}
                      onChange={(e) => handleSettingChange('performance', 'memoryLimit', parseInt(e.target.value))}
                      helperText="Maximum memory usage per bot"
                    />

                    <Box>
                      <Typography variant="body2" gutterBottom>
                        CPU Threshold: {settings.performance.cpuThreshold}%
                      </Typography>
                      <Slider
                        value={settings.performance.cpuThreshold}
                        onChange={(_, value) => handleSettingChange('performance', 'cpuThreshold', value)}
                        min={50}
                        max={100}
                        step={5}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => `${value}%`}
                      />
                    </Box>
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>

            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Error Handling</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="Retry Attempts"
                      type="number"
                      value={settings.performance.retryAttempts}
                      onChange={(e) => handleSettingChange('performance', 'retryAttempts', parseInt(e.target.value))}
                      helperText="Number of retry attempts for failed operations"
                    />

                    <FormControl fullWidth>
                      <InputLabel>Log Level</InputLabel>
                      <Select
                        value={settings.performance.logLevel}
                        label="Log Level"
                        onChange={(e) => handleSettingChange('performance', 'logLevel', e.target.value)}
                      >
                        <MenuItem value="debug">Debug (Most verbose)</MenuItem>
                        <MenuItem value="info">Info (Recommended)</MenuItem>
                        <MenuItem value="warning">Warning</MenuItem>
                        <MenuItem value="error">Error (Least verbose)</MenuItem>
                      </Select>
                    </FormControl>

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.performance.enableProfiling}
                          onChange={(e) => handleSettingChange('performance', 'enableProfiling', e.target.checked)}
                        />
                      }
                      label="Enable Performance Profiling"
                    />
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Security Options</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.security.enableEncryption}
                          onChange={(e) => handleSettingChange('security', 'enableEncryption', e.target.checked)}
                        />
                      }
                      label="Enable Encryption"
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.security.enableRateLimiting}
                          onChange={(e) => handleSettingChange('security', 'enableRateLimiting', e.target.checked)}
                        />
                      }
                      label="Enable Rate Limiting"
                    />

                    {settings.security.enableRateLimiting && (
                      <TextField
                        fullWidth
                        label="Rate Limit (requests/minute)"
                        type="number"
                        value={settings.security.rateLimit}
                        onChange={(e) => handleSettingChange('security', 'rateLimit', parseInt(e.target.value))}
                      />
                    )}

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.security.enableAuditLog}
                          onChange={(e) => handleSettingChange('security', 'enableAuditLog', e.target.checked)}
                        />
                      }
                      label="Enable Audit Logging"
                    />
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>

            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Session Management</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="Session Timeout (seconds)"
                      type="number"
                      value={settings.security.sessionTimeout}
                      onChange={(e) => handleSettingChange('security', 'sessionTimeout', parseInt(e.target.value))}
                      helperText="Idle session timeout"
                    />

                    <TextField
                      fullWidth
                      label="Max Session Duration (seconds)"
                      type="number"
                      value={settings.security.maxSessionDuration}
                      onChange={(e) => handleSettingChange('security', 'maxSessionDuration', parseInt(e.target.value))}
                      helperText="Maximum session duration"
                    />

                    <Typography variant="body2" gutterBottom>
                      Allowed Domains
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {settings.security.allowedDomains.map((domain, index) => (
                        <Chip
                          key={index}
                          label={domain}
                          onDelete={() => {
                            const newDomains = settings.security.allowedDomains.filter((_, i) => i !== index);
                            handleSettingChange('security', 'allowedDomains', newDomains);
                          }}
                        />
                      ))}
                    </Box>
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Data Retention</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="Retention Period (days)"
                      type="number"
                      value={settings.storage.retentionPeriod}
                      onChange={(e) => handleSettingChange('storage', 'retentionPeriod', parseInt(e.target.value))}
                      helperText="How long to keep session data"
                    />

                    <TextField
                      fullWidth
                      label="Max Storage Size (MB)"
                      type="number"
                      value={settings.storage.maxStorageSize}
                      onChange={(e) => handleSettingChange('storage', 'maxStorageSize', parseInt(e.target.value))}
                      helperText="Maximum storage space usage"
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.storage.autoCleanup}
                          onChange={(e) => handleSettingChange('storage', 'autoCleanup', e.target.checked)}
                        />
                      }
                      label="Auto Cleanup Old Data"
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.storage.compressionEnabled}
                          onChange={(e) => handleSettingChange('storage', 'compressionEnabled', e.target.checked)}
                        />
                      }
                      label="Enable Data Compression"
                    />
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>

            <Grid item xs={12} md={6}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Backup Settings</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.storage.enableBackup}
                          onChange={(e) => handleSettingChange('storage', 'enableBackup', e.target.checked)}
                        />
                      }
                      label="Enable Automatic Backups"
                    />

                    {settings.storage.enableBackup && (
                      <TextField
                        fullWidth
                        label="Backup Frequency (hours)"
                        type="number"
                        value={settings.storage.backupFrequency}
                        onChange={(e) => handleSettingChange('storage', 'backupFrequency', parseInt(e.target.value))}
                        helperText="How often to create backups"
                      />
                    )}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};