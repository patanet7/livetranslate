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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RestoreIcon from '@mui/icons-material/Restore';
import RefreshIcon from '@mui/icons-material/Refresh';
import SecurityIcon from '@mui/icons-material/Security';
import StorageIcon from '@mui/icons-material/Storage';
import NetworkCheckIcon from '@mui/icons-material/NetworkCheck';
import BugReportIcon from '@mui/icons-material/BugReport';
import PaletteIcon from '@mui/icons-material/Palette';
import NotificationsIcon from '@mui/icons-material/Notifications';

interface SystemConfig {
  // General System Settings
  general: {
    system_name: string;
    environment: 'development' | 'staging' | 'production';
    debug_mode: boolean;
    log_level: 'debug' | 'info' | 'warning' | 'error';
    timezone: string;
    language: string;
  };

  // Security Settings
  security: {
    enable_authentication: boolean;
    session_timeout_minutes: number;
    rate_limiting: boolean;
    rate_limit_requests_per_minute: number;
    cors_enabled: boolean;
    allowed_origins: string[];
    api_key_required: boolean;
  };

  // Performance Settings
  performance: {
    max_concurrent_sessions: number;
    request_timeout_ms: number;
    connection_pool_size: number;
    cache_enabled: boolean;
    cache_ttl_minutes: number;
    compression_enabled: boolean;
  };

  // Storage Settings
  storage: {
    data_retention_days: number;
    auto_cleanup: boolean;
    backup_enabled: boolean;
    backup_interval_hours: number;
    max_storage_gb: number;
    storage_monitoring: boolean;
  };

  // Monitoring and Alerts
  monitoring: {
    health_checks: boolean;
    metrics_collection: boolean;
    error_tracking: boolean;
    performance_monitoring: boolean;
    alert_email_enabled: boolean;
    alert_email_addresses: string[];
    alert_thresholds: {
      cpu_usage_percent: number;
      memory_usage_percent: number;
      disk_usage_percent: number;
      error_rate_percent: number;
    };
  };

  // User Interface
  ui: {
    theme: 'light' | 'dark' | 'auto';
    compact_mode: boolean;
    animations_enabled: boolean;
    auto_refresh_interval_seconds: number;
    show_advanced_features: boolean;
    notification_sounds: boolean;
  };

  // API Configuration
  api: {
    enable_swagger_docs: boolean;
    enable_api_versioning: boolean;
    default_response_format: 'json' | 'xml';
    enable_webhooks: boolean;
    webhook_timeout_ms: number;
    enable_batch_requests: boolean;
  };

  // Database Settings
  database: {
    connection_string: string;
    max_connections: number;
    connection_timeout_ms: number;
    query_timeout_ms: number;
    enable_query_logging: boolean;
    backup_retention_days: number;
  };
}

interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_connections: number;
  uptime_seconds: number;
  last_backup: string;
  service_status: {
    orchestration: 'healthy' | 'warning' | 'error';
    whisper: 'healthy' | 'warning' | 'error';
    translation: 'healthy' | 'warning' | 'error';
    database: 'healthy' | 'warning' | 'error';
  };
}

interface SystemSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const defaultSystemConfig: SystemConfig = {
  general: {
    system_name: 'LiveTranslate System',
    environment: 'development',
    debug_mode: true,
    log_level: 'info',
    timezone: 'UTC',
    language: 'en',
  },
  security: {
    enable_authentication: false,
    session_timeout_minutes: 60,
    rate_limiting: true,
    rate_limit_requests_per_minute: 100,
    cors_enabled: true,
    allowed_origins: ['http://localhost:5173', 'http://localhost:3000'],
    api_key_required: false,
  },
  performance: {
    max_concurrent_sessions: 100,
    request_timeout_ms: 30000,
    connection_pool_size: 20,
    cache_enabled: true,
    cache_ttl_minutes: 15,
    compression_enabled: true,
  },
  storage: {
    data_retention_days: 30,
    auto_cleanup: true,
    backup_enabled: true,
    backup_interval_hours: 24,
    max_storage_gb: 100,
    storage_monitoring: true,
  },
  monitoring: {
    health_checks: true,
    metrics_collection: true,
    error_tracking: true,
    performance_monitoring: true,
    alert_email_enabled: false,
    alert_email_addresses: [],
    alert_thresholds: {
      cpu_usage_percent: 80,
      memory_usage_percent: 85,
      disk_usage_percent: 90,
      error_rate_percent: 5,
    },
  },
  ui: {
    theme: 'auto',
    compact_mode: false,
    animations_enabled: true,
    auto_refresh_interval_seconds: 30,
    show_advanced_features: false,
    notification_sounds: true,
  },
  api: {
    enable_swagger_docs: true,
    enable_api_versioning: true,
    default_response_format: 'json',
    enable_webhooks: false,
    webhook_timeout_ms: 5000,
    enable_batch_requests: true,
  },
  database: {
    connection_string: 'postgresql://localhost:5432/livetranslate',
    max_connections: 20,
    connection_timeout_ms: 10000,
    query_timeout_ms: 30000,
    enable_query_logging: false,
    backup_retention_days: 7,
  },
};

const timezones = [
  'UTC', 'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
  'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Asia/Tokyo', 'Asia/Shanghai',
  'Australia/Sydney', 'Pacific/Auckland'
];

const languages = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'ja', name: 'Japanese' },
  { code: 'zh', name: 'Chinese' },
];

const SystemSettings: React.FC<SystemSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<SystemConfig>(defaultSystemConfig);
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    active_connections: 0,
    uptime_seconds: 0,
    last_backup: '',
    service_status: {
      orchestration: 'healthy',
      whisper: 'healthy',
      translation: 'healthy',
      database: 'healthy',
    },
  });

  // Load current configuration and health status
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch('/api/settings/system');
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultSystemConfig, ...currentConfig });
        }
      } catch (error) {
        console.error('Failed to load system configuration:', error);
      }
    };

    const loadSystemHealth = async () => {
      try {
        const response = await fetch('/api/system/health');
        if (response.ok) {
          const health = await response.json();
          setSystemHealth(health);
        }
      } catch (error) {
        console.error('Failed to load system health:', error);
      }
    };
    
    loadConfiguration();
    loadSystemHealth();

    // Set up periodic health updates
    const healthInterval = setInterval(loadSystemHealth, 30000); // Every 30 seconds

    return () => clearInterval(healthInterval);
  }, []);

  const handleConfigChange = (section: keyof SystemConfig, key: string, value: any) => {
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
      const response = await fetch('/api/settings/system', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        onSave('System settings saved successfully');
      } else {
        onSave('Failed to save system settings', false);
      }
    } catch (error) {
      onSave('Error saving system settings', false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultSystemConfig);
    onSave('System configuration reset to defaults');
  };

  const handleRestartServices = async () => {
    try {
      const response = await fetch('/api/system/restart', {
        method: 'POST',
      });
      
      if (response.ok) {
        onSave('System restart initiated successfully');
      } else {
        onSave('Failed to restart system services', false);
      }
    } catch (error) {
      onSave('Error restarting system services', false);
    }
  };

  const handleTestConnections = async () => {
    try {
      const response = await fetch('/api/system/test-connections', {
        method: 'POST',
      });
      
      if (response.ok) {
        const results = await response.json();
        onSave(`Connection test completed: ${results.summary}`);
      } else {
        onSave('Connection test failed', false);
      }
    } catch (error) {
      onSave('Error testing connections', false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          System Configuration
        </Typography>
        <Box>
          <Tooltip title="Reset to defaults">
            <IconButton onClick={handleResetToDefaults} color="secondary">
              <RestoreIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleTestConnections}
            sx={{ mr: 2 }}
          >
            Test Connections
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure system-wide settings including security, performance, monitoring, and user interface preferences.
        Changes may require a system restart to take effect.
      </Alert>

      {/* System Health Overview */}
      <Card sx={{ mb: 3 }}>
        <CardHeader 
          title="System Health Overview" 
          action={
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleRestartServices}
              color="warning"
            >
              Restart Services
            </Button>
          }
        />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {systemHealth.cpu_usage.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  CPU Usage
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {systemHealth.memory_usage.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Memory Usage
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {systemHealth.disk_usage.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Disk Usage
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {formatUptime(systemHealth.uptime_seconds)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Uptime
                </Typography>
              </Paper>
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="h6" gutterBottom>Service Status</Typography>
          <Grid container spacing={1}>
            {Object.entries(systemHealth.service_status).map(([service, status]) => (
              <Grid item xs={6} md={3} key={service}>
                <Chip
                  label={`${service}: ${status}`}
                  color={getStatusColor(status) as any}
                  variant={status === 'healthy' ? 'filled' : 'outlined'}
                  size="small"
                  sx={{ width: '100%' }}
                />
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* General Settings */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">General System Settings</Typography>
          <Chip 
            label={config.general.environment} 
            color={config.general.environment === 'production' ? 'error' : 'default'}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="System Name"
                value={config.general.system_name}
                onChange={(e) => handleConfigChange('general', 'system_name', e.target.value)}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Environment</InputLabel>
                <Select
                  value={config.general.environment}
                  label="Environment"
                  onChange={(e) => handleConfigChange('general', 'environment', e.target.value)}
                >
                  <MenuItem value="development">Development</MenuItem>
                  <MenuItem value="staging">Staging</MenuItem>
                  <MenuItem value="production">Production</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Log Level</InputLabel>
                <Select
                  value={config.general.log_level}
                  label="Log Level"
                  onChange={(e) => handleConfigChange('general', 'log_level', e.target.value)}
                >
                  <MenuItem value="debug">Debug</MenuItem>
                  <MenuItem value="info">Info</MenuItem>
                  <MenuItem value="warning">Warning</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Timezone</InputLabel>
                <Select
                  value={config.general.timezone}
                  label="Timezone"
                  onChange={(e) => handleConfigChange('general', 'timezone', e.target.value)}
                >
                  {timezones.map((tz) => (
                    <MenuItem key={tz} value={tz}>{tz}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={config.general.language}
                  label="Language"
                  onChange={(e) => handleConfigChange('general', 'language', e.target.value)}
                >
                  {languages.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>{lang.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.general.debug_mode}
                    onChange={(e) => handleConfigChange('general', 'debug_mode', e.target.checked)}
                  />
                }
                label="Debug Mode"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Security Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SecurityIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Security Settings</Typography>
          <Chip 
            label={config.security.enable_authentication ? 'Auth Enabled' : 'Auth Disabled'} 
            color={config.security.enable_authentication ? 'success' : 'warning'}
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
                    checked={config.security.enable_authentication}
                    onChange={(e) => handleConfigChange('security', 'enable_authentication', e.target.checked)}
                  />
                }
                label="Enable Authentication"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Session Timeout (minutes)"
                type="number"
                value={config.security.session_timeout_minutes}
                onChange={(e) => handleConfigChange('security', 'session_timeout_minutes', Number(e.target.value))}
                disabled={!config.security.enable_authentication}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.security.rate_limiting}
                    onChange={(e) => handleConfigChange('security', 'rate_limiting', e.target.checked)}
                  />
                }
                label="Rate Limiting"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Rate Limit (requests/minute)"
                type="number"
                value={config.security.rate_limit_requests_per_minute}
                onChange={(e) => handleConfigChange('security', 'rate_limit_requests_per_minute', Number(e.target.value))}
                disabled={!config.security.rate_limiting}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.security.cors_enabled}
                    onChange={(e) => handleConfigChange('security', 'cors_enabled', e.target.checked)}
                  />
                }
                label="CORS Enabled"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Allowed Origins (comma-separated)"
                value={config.security.allowed_origins.join(', ')}
                onChange={(e) => handleConfigChange('security', 'allowed_origins', e.target.value.split(', ').filter(Boolean))}
                disabled={!config.security.cors_enabled}
                helperText="List of allowed CORS origins"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Performance Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <NetworkCheckIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Performance Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Max Concurrent Sessions: {config.performance.max_concurrent_sessions}</Typography>
              <Slider
                value={config.performance.max_concurrent_sessions}
                onChange={(_, value) => handleConfigChange('performance', 'max_concurrent_sessions', value)}
                min={10}
                max={1000}
                step={10}
                marks={[
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                  { value: 500, label: '500' },
                  { value: 1000, label: '1000' },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Request Timeout (ms)"
                type="number"
                value={config.performance.request_timeout_ms}
                onChange={(e) => handleConfigChange('performance', 'request_timeout_ms', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Connection Pool Size"
                type="number"
                value={config.performance.connection_pool_size}
                onChange={(e) => handleConfigChange('performance', 'connection_pool_size', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Cache TTL (minutes)"
                type="number"
                value={config.performance.cache_ttl_minutes}
                onChange={(e) => handleConfigChange('performance', 'cache_ttl_minutes', Number(e.target.value))}
                disabled={!config.performance.cache_enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.cache_enabled}
                    onChange={(e) => handleConfigChange('performance', 'cache_enabled', e.target.checked)}
                  />
                }
                label="Enable Caching"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.compression_enabled}
                    onChange={(e) => handleConfigChange('performance', 'compression_enabled', e.target.checked)}
                  />
                }
                label="Enable Compression"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Storage Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <StorageIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Storage and Backup</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Data Retention (days)"
                type="number"
                value={config.storage.data_retention_days}
                onChange={(e) => handleConfigChange('storage', 'data_retention_days', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Storage (GB)"
                type="number"
                value={config.storage.max_storage_gb}
                onChange={(e) => handleConfigChange('storage', 'max_storage_gb', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Backup Interval (hours)"
                type="number"
                value={config.storage.backup_interval_hours}
                onChange={(e) => handleConfigChange('storage', 'backup_interval_hours', Number(e.target.value))}
                disabled={!config.storage.backup_enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.storage.auto_cleanup}
                    onChange={(e) => handleConfigChange('storage', 'auto_cleanup', e.target.checked)}
                  />
                }
                label="Auto Cleanup"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.storage.backup_enabled}
                    onChange={(e) => handleConfigChange('storage', 'backup_enabled', e.target.checked)}
                  />
                }
                label="Enable Backups"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.storage.storage_monitoring}
                    onChange={(e) => handleConfigChange('storage', 'storage_monitoring', e.target.checked)}
                  />
                }
                label="Storage Monitoring"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* User Interface */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <PaletteIcon sx={{ mr: 1 }} />
          <Typography variant="h6">User Interface</Typography>
          <Chip 
            label={config.ui.theme} 
            color="primary"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Theme</InputLabel>
                <Select
                  value={config.ui.theme}
                  label="Theme"
                  onChange={(e) => handleConfigChange('ui', 'theme', e.target.value)}
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="auto">Auto (System)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Auto Refresh Interval (seconds)"
                type="number"
                value={config.ui.auto_refresh_interval_seconds}
                onChange={(e) => handleConfigChange('ui', 'auto_refresh_interval_seconds', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.ui.compact_mode}
                    onChange={(e) => handleConfigChange('ui', 'compact_mode', e.target.checked)}
                  />
                }
                label="Compact Mode"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.ui.animations_enabled}
                    onChange={(e) => handleConfigChange('ui', 'animations_enabled', e.target.checked)}
                  />
                }
                label="Animations"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.ui.show_advanced_features}
                    onChange={(e) => handleConfigChange('ui', 'show_advanced_features', e.target.checked)}
                  />
                }
                label="Advanced Features"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.ui.notification_sounds}
                    onChange={(e) => handleConfigChange('ui', 'notification_sounds', e.target.checked)}
                  />
                }
                label="Notification Sounds"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Monitoring and Alerts */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <BugReportIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Monitoring and Alerts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.health_checks}
                    onChange={(e) => handleConfigChange('monitoring', 'health_checks', e.target.checked)}
                  />
                }
                label="Health Checks"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.metrics_collection}
                    onChange={(e) => handleConfigChange('monitoring', 'metrics_collection', e.target.checked)}
                  />
                }
                label="Metrics Collection"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.error_tracking}
                    onChange={(e) => handleConfigChange('monitoring', 'error_tracking', e.target.checked)}
                  />
                }
                label="Error Tracking"
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Alert Thresholds</Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>CPU Usage Alert: {config.monitoring.alert_thresholds.cpu_usage_percent}%</Typography>
              <Slider
                value={config.monitoring.alert_thresholds.cpu_usage_percent}
                onChange={(_, value) => handleConfigChange('monitoring', 'alert_thresholds', {
                  ...config.monitoring.alert_thresholds,
                  cpu_usage_percent: value
                })}
                min={50}
                max={100}
                step={5}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Memory Usage Alert: {config.monitoring.alert_thresholds.memory_usage_percent}%</Typography>
              <Slider
                value={config.monitoring.alert_thresholds.memory_usage_percent}
                onChange={(_, value) => handleConfigChange('monitoring', 'alert_thresholds', {
                  ...config.monitoring.alert_thresholds,
                  memory_usage_percent: value
                })}
                min={50}
                max={100}
                step={5}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.monitoring.alert_email_enabled}
                    onChange={(e) => handleConfigChange('monitoring', 'alert_email_enabled', e.target.checked)}
                  />
                }
                label="Email Alerts"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Alert Email Addresses (comma-separated)"
                value={config.monitoring.alert_email_addresses.join(', ')}
                onChange={(e) => handleConfigChange('monitoring', 'alert_email_addresses', e.target.value.split(', ').filter(Boolean))}
                disabled={!config.monitoring.alert_email_enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default SystemSettings;