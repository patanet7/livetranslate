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
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RestoreIcon from '@mui/icons-material/Restore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import AddIcon from '@mui/icons-material/Add';

interface BotConfig {
  // Bot Manager Configuration
  manager: {
    enabled: boolean;
    max_concurrent_bots: number;
    bot_spawn_timeout: number;
    health_check_interval: number;
    auto_recovery: boolean;
    max_recovery_attempts: number;
    cleanup_on_shutdown: boolean;
  };

  // Google Meet Integration
  google_meet: {
    enabled: boolean;
    credentials_path: string;
    oauth_scopes: string[];
    api_timeout_ms: number;
    fallback_mode: boolean;
    meeting_detection: boolean;
  };

  // Bot Lifecycle
  lifecycle: {
    startup_delay_ms: number;
    shutdown_timeout_ms: number;
    graceful_shutdown: boolean;
    force_kill_timeout_ms: number;
    restart_on_failure: boolean;
    max_session_duration_hours: number;
  };

  // Audio Capture Configuration
  audio_capture: {
    enabled: boolean;
    capture_method: 'loopback' | 'virtual_cable' | 'screen_audio';
    audio_device: string;
    sample_rate: number;
    channels: number;
    buffer_duration_ms: number;
    quality_threshold: number;
  };

  // Caption Processing
  caption_processing: {
    enabled: boolean;
    extract_google_captions: boolean;
    caption_language: string;
    timing_accuracy_ms: number;
    caption_filtering: boolean;
    merge_duplicate_captions: boolean;
  };

  // Database Integration
  database: {
    enabled: boolean;
    session_persistence: boolean;
    store_audio_files: boolean;
    store_transcripts: boolean;
    store_correlations: boolean;
    cleanup_old_sessions: boolean;
    retention_days: number;
  };

  // Performance and Resources
  performance: {
    cpu_limit_percent: number;
    memory_limit_mb: number;
    disk_space_limit_gb: number;
    priority_level: 'low' | 'normal' | 'high';
    process_isolation: boolean;
  };

  // Error Handling and Monitoring
  monitoring: {
    enabled: boolean;
    log_level: 'debug' | 'info' | 'warning' | 'error';
    metrics_collection: boolean;
    performance_tracking: boolean;
    error_reporting: boolean;
    health_dashboard: boolean;
  };
}

interface BotTemplate {
  id: string;
  name: string;
  description: string;
  config: Partial<BotConfig>;
  is_default: boolean;
}

interface BotSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const defaultBotConfig: BotConfig = {
  manager: {
    enabled: true,
    max_concurrent_bots: 10,
    bot_spawn_timeout: 30000,
    health_check_interval: 5000,
    auto_recovery: true,
    max_recovery_attempts: 3,
    cleanup_on_shutdown: true,
  },
  google_meet: {
    enabled: true,
    credentials_path: '/config/google-meet-credentials.json',
    oauth_scopes: [
      'https://www.googleapis.com/auth/meetings.space.created',
      'https://www.googleapis.com/auth/meetings.space.readonly',
    ],
    api_timeout_ms: 10000,
    fallback_mode: true,
    meeting_detection: true,
  },
  lifecycle: {
    startup_delay_ms: 2000,
    shutdown_timeout_ms: 10000,
    graceful_shutdown: true,
    force_kill_timeout_ms: 5000,
    restart_on_failure: true,
    max_session_duration_hours: 4,
  },
  audio_capture: {
    enabled: true,
    capture_method: 'loopback',
    audio_device: 'default',
    sample_rate: 16000,
    channels: 1,
    buffer_duration_ms: 100,
    quality_threshold: 0.3,
  },
  caption_processing: {
    enabled: true,
    extract_google_captions: true,
    caption_language: 'en',
    timing_accuracy_ms: 500,
    caption_filtering: true,
    merge_duplicate_captions: true,
  },
  database: {
    enabled: true,
    session_persistence: true,
    store_audio_files: true,
    store_transcripts: true,
    store_correlations: true,
    cleanup_old_sessions: true,
    retention_days: 30,
  },
  performance: {
    cpu_limit_percent: 50,
    memory_limit_mb: 1024,
    disk_space_limit_gb: 5,
    priority_level: 'normal',
    process_isolation: true,
  },
  monitoring: {
    enabled: true,
    log_level: 'info',
    metrics_collection: true,
    performance_tracking: true,
    error_reporting: true,
    health_dashboard: true,
  },
};

const BotSettings: React.FC<BotSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<BotConfig>(defaultBotConfig);
  const [botStats, setBotStats] = useState({
    total_bots_spawned: 0,
    currently_active: 0,
    successful_sessions: 0,
    failed_sessions: 0,
    average_session_duration: 0,
  });
  const [templates, setTemplates] = useState<BotTemplate[]>([]);
  const [templateDialog, setTemplateDialog] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<BotTemplate | null>(null);

  // Load current configuration
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch('/api/settings/bot');
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultBotConfig, ...currentConfig });
        }
      } catch (error) {
        console.error('Failed to load bot configuration:', error);
      }
    };

    const loadStats = async () => {
      try {
        const response = await fetch('/api/settings/bot/stats');
        if (response.ok) {
          const stats = await response.json();
          setBotStats(stats);
        }
      } catch (error) {
        console.error('Failed to load bot stats:', error);
      }
    };

    const loadTemplates = async () => {
      try {
        const response = await fetch('/api/settings/bot/templates');
        if (response.ok) {
          const templateList = await response.json();
          setTemplates(templateList);
        }
      } catch (error) {
        console.error('Failed to load bot templates:', error);
      }
    };
    
    loadConfiguration();
    loadStats();
    loadTemplates();
  }, []);

  const handleConfigChange = (section: keyof BotConfig, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  const handleSave = async () => {
    try {
      const response = await fetch('/api/settings/bot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        onSave('Bot management settings saved successfully');
      } else {
        onSave('Failed to save bot management settings', false);
      }
    } catch (error) {
      onSave('Error saving bot management settings', false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultBotConfig);
    onSave('Bot configuration reset to defaults');
  };

  const handleTestBotSpawn = async () => {
    try {
      const response = await fetch('/api/settings/bot/test-spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });
      
      if (response.ok) {
        const result = await response.json();
        onSave(`Bot spawn test successful: ${result.message}`);
      } else {
        onSave('Bot spawn test failed', false);
      }
    } catch (error) {
      onSave('Error testing bot spawn', false);
    }
  };

  const handleSaveTemplate = async () => {
    if (!editingTemplate) return;

    try {
      const response = await fetch('/api/settings/bot/templates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...editingTemplate,
          config: config, // Use current config as template
        }),
      });
      
      if (response.ok) {
        // Reload templates
        const templatesResponse = await fetch('/api/settings/bot/templates');
        if (templatesResponse.ok) {
          const templateList = await templatesResponse.json();
          setTemplates(templateList);
        }
        onSave('Bot template saved successfully');
        setTemplateDialog(false);
      } else {
        onSave('Failed to save bot template', false);
      }
    } catch (error) {
      onSave('Error saving bot template', false);
    }
  };

  const handleLoadTemplate = async (template: BotTemplate) => {
    setConfig({ ...defaultBotConfig, ...template.config });
    onSave(`Loaded template: ${template.name}`);
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Bot Management Configuration
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
            onClick={handleTestBotSpawn}
            disabled={!config.manager.enabled}
            sx={{ mr: 2 }}
          >
            Test Bot Spawn
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure Google Meet bot management including spawning, lifecycle, audio capture, and database integration.
        Supports enterprise-grade bot orchestration with automatic recovery.
      </Alert>

      {/* Bot Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Bot Management Statistics" />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {botStats.total_bots_spawned}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Spawned
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="success.main">
                  {botStats.currently_active}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Currently Active
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="info.main">
                  {botStats.successful_sessions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Successful Sessions
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="error.main">
                  {botStats.failed_sessions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Failed Sessions
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="secondary.main">
                  {(botStats.average_session_duration / 60).toFixed(1)}m
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Duration
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Bot Templates */}
      <Card sx={{ mb: 3 }}>
        <CardHeader 
          title="Bot Configuration Templates" 
          action={
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={() => {
                setEditingTemplate({
                  id: '',
                  name: '',
                  description: '',
                  config: {},
                  is_default: false,
                });
                setTemplateDialog(true);
              }}
            >
              Save Template
            </Button>
          }
        />
        <CardContent>
          <Grid container spacing={2}>
            {templates.map((template) => (
              <Grid item xs={12} md={6} lg={4} key={template.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {template.name}
                      {template.is_default && (
                        <Chip label="Default" color="primary" size="small" sx={{ ml: 1 }} />
                      )}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {template.description}
                    </Typography>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => handleLoadTemplate(template)}
                      sx={{ mt: 1 }}
                    >
                      Load Template
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Bot Manager Configuration */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Bot Manager Settings</Typography>
          <Chip 
            label={config.manager.enabled ? 'Enabled' : 'Disabled'} 
            color={config.manager.enabled ? 'success' : 'default'}
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
                    checked={config.manager.enabled}
                    onChange={(e) => handleConfigChange('manager', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Bot Management"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Max Concurrent Bots: {config.manager.max_concurrent_bots}</Typography>
              <Slider
                value={config.manager.max_concurrent_bots}
                onChange={(_, value) => handleConfigChange('manager', 'max_concurrent_bots', value)}
                min={1}
                max={50}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 10, label: '10' },
                  { value: 25, label: '25' },
                  { value: 50, label: '50' },
                ]}
                disabled={!config.manager.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Bot Spawn Timeout (ms)"
                type="number"
                value={config.manager.bot_spawn_timeout}
                onChange={(e) => handleConfigChange('manager', 'bot_spawn_timeout', Number(e.target.value))}
                disabled={!config.manager.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Health Check Interval (ms)"
                type="number"
                value={config.manager.health_check_interval}
                onChange={(e) => handleConfigChange('manager', 'health_check_interval', Number(e.target.value))}
                disabled={!config.manager.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Recovery Attempts"
                type="number"
                value={config.manager.max_recovery_attempts}
                onChange={(e) => handleConfigChange('manager', 'max_recovery_attempts', Number(e.target.value))}
                disabled={!config.manager.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.manager.auto_recovery}
                    onChange={(e) => handleConfigChange('manager', 'auto_recovery', e.target.checked)}
                    disabled={!config.manager.enabled}
                  />
                }
                label="Auto Recovery"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.manager.cleanup_on_shutdown}
                    onChange={(e) => handleConfigChange('manager', 'cleanup_on_shutdown', e.target.checked)}
                    disabled={!config.manager.enabled}
                  />
                }
                label="Cleanup on Shutdown"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Google Meet Integration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Google Meet Integration</Typography>
          <Chip 
            label={config.google_meet.enabled ? 'Enabled' : 'Disabled'} 
            color={config.google_meet.enabled ? 'success' : 'default'}
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
                    checked={config.google_meet.enabled}
                    onChange={(e) => handleConfigChange('google_meet', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Google Meet Integration"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Credentials Path"
                value={config.google_meet.credentials_path}
                onChange={(e) => handleConfigChange('google_meet', 'credentials_path', e.target.value)}
                disabled={!config.google_meet.enabled}
                helperText="Path to Google Meet API credentials JSON file"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="API Timeout (ms)"
                type="number"
                value={config.google_meet.api_timeout_ms}
                onChange={(e) => handleConfigChange('google_meet', 'api_timeout_ms', Number(e.target.value))}
                disabled={!config.google_meet.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.fallback_mode}
                    onChange={(e) => handleConfigChange('google_meet', 'fallback_mode', e.target.checked)}
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Fallback Mode (without API)"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.meeting_detection}
                    onChange={(e) => handleConfigChange('google_meet', 'meeting_detection', e.target.checked)}
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Automatic Meeting Detection"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Audio Capture Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Audio Capture Configuration</Typography>
          <Chip 
            label={config.audio_capture.capture_method} 
            color="primary"
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
                    checked={config.audio_capture.enabled}
                    onChange={(e) => handleConfigChange('audio_capture', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Audio Capture"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Capture Method</InputLabel>
                <Select
                  value={config.audio_capture.capture_method}
                  label="Capture Method"
                  onChange={(e) => handleConfigChange('audio_capture', 'capture_method', e.target.value)}
                  disabled={!config.audio_capture.enabled}
                >
                  <MenuItem value="loopback">Audio Loopback</MenuItem>
                  <MenuItem value="virtual_cable">Virtual Audio Cable</MenuItem>
                  <MenuItem value="screen_audio">Screen Audio Capture</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Audio Device"
                value={config.audio_capture.audio_device}
                onChange={(e) => handleConfigChange('audio_capture', 'audio_device', e.target.value)}
                disabled={!config.audio_capture.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Sample Rate</InputLabel>
                <Select
                  value={config.audio_capture.sample_rate}
                  label="Sample Rate"
                  onChange={(e) => handleConfigChange('audio_capture', 'sample_rate', e.target.value)}
                  disabled={!config.audio_capture.enabled}
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
              <FormControl fullWidth>
                <InputLabel>Channels</InputLabel>
                <Select
                  value={config.audio_capture.channels}
                  label="Channels"
                  onChange={(e) => handleConfigChange('audio_capture', 'channels', e.target.value)}
                  disabled={!config.audio_capture.enabled}
                >
                  <MenuItem value={1}>Mono</MenuItem>
                  <MenuItem value={2}>Stereo</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Buffer Duration (ms)"
                type="number"
                value={config.audio_capture.buffer_duration_ms}
                onChange={(e) => handleConfigChange('audio_capture', 'buffer_duration_ms', Number(e.target.value))}
                disabled={!config.audio_capture.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Quality Threshold: {config.audio_capture.quality_threshold}</Typography>
              <Slider
                value={config.audio_capture.quality_threshold}
                onChange={(_, value) => handleConfigChange('audio_capture', 'quality_threshold', value)}
                min={0}
                max={1}
                step={0.1}
                disabled={!config.audio_capture.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Performance and Resources */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Performance and Resources</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>CPU Limit: {config.performance.cpu_limit_percent}%</Typography>
              <Slider
                value={config.performance.cpu_limit_percent}
                onChange={(_, value) => handleConfigChange('performance', 'cpu_limit_percent', value)}
                min={10}
                max={100}
                step={10}
                marks={[
                  { value: 25, label: '25%' },
                  { value: 50, label: '50%' },
                  { value: 75, label: '75%' },
                  { value: 100, label: '100%' },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Memory Limit (MB)"
                type="number"
                value={config.performance.memory_limit_mb}
                onChange={(e) => handleConfigChange('performance', 'memory_limit_mb', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Disk Space Limit (GB)"
                type="number"
                value={config.performance.disk_space_limit_gb}
                onChange={(e) => handleConfigChange('performance', 'disk_space_limit_gb', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Priority Level</InputLabel>
                <Select
                  value={config.performance.priority_level}
                  label="Priority Level"
                  onChange={(e) => handleConfigChange('performance', 'priority_level', e.target.value)}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="normal">Normal</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.process_isolation}
                    onChange={(e) => handleConfigChange('performance', 'process_isolation', e.target.checked)}
                  />
                }
                label="Process Isolation"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Monitoring Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Monitoring and Logging</Typography>
          <Chip 
            label={config.monitoring.log_level} 
            color="info"
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
                    checked={config.monitoring.enabled}
                    onChange={(e) => handleConfigChange('monitoring', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Monitoring"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Log Level</InputLabel>
                <Select
                  value={config.monitoring.log_level}
                  label="Log Level"
                  onChange={(e) => handleConfigChange('monitoring', 'log_level', e.target.value)}
                  disabled={!config.monitoring.enabled}
                >
                  <MenuItem value="debug">Debug</MenuItem>
                  <MenuItem value="info">Info</MenuItem>
                  <MenuItem value="warning">Warning</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.metrics_collection}
                    onChange={(e) => handleConfigChange('monitoring', 'metrics_collection', e.target.checked)}
                    disabled={!config.monitoring.enabled}
                  />
                }
                label="Metrics Collection"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.performance_tracking}
                    onChange={(e) => handleConfigChange('monitoring', 'performance_tracking', e.target.checked)}
                    disabled={!config.monitoring.enabled}
                  />
                }
                label="Performance Tracking"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.error_reporting}
                    onChange={(e) => handleConfigChange('monitoring', 'error_reporting', e.target.checked)}
                    disabled={!config.monitoring.enabled}
                  />
                }
                label="Error Reporting"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.health_dashboard}
                    onChange={(e) => handleConfigChange('monitoring', 'health_dashboard', e.target.checked)}
                    disabled={!config.monitoring.enabled}
                  />
                }
                label="Health Dashboard"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Template Dialog */}
      <Dialog open={templateDialog} onClose={() => setTemplateDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Save Current Configuration as Template</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Template Name"
                value={editingTemplate?.name || ''}
                onChange={(e) => setEditingTemplate(prev => prev ? { ...prev, name: e.target.value } : null)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={3}
                value={editingTemplate?.description || ''}
                onChange={(e) => setEditingTemplate(prev => prev ? { ...prev, description: e.target.value } : null)}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editingTemplate?.is_default || false}
                    onChange={(e) => setEditingTemplate(prev => prev ? { ...prev, is_default: e.target.checked } : null)}
                  />
                }
                label="Set as Default Template"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveTemplate} variant="contained">Save Template</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BotSettings;