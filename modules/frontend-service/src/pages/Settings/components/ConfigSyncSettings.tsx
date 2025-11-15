import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore,
  Sync,
  CheckCircle,
  Warning,
  Error,
  Settings,
  Refresh,
  CloudSync,
  Info,
} from '@mui/icons-material';

interface ConfigSyncSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

interface ServiceConfiguration {
  whisper_service: {
    sample_rate: number;
    buffer_duration: number;
    inference_interval: number;
    overlap_duration: number;
    enable_vad: boolean;
    max_concurrent_requests: number;
  };
  orchestration_service: {
    chunking_config: {
      chunk_duration: number;
      overlap_duration: number;
      processing_interval: number;
      buffer_duration: number;
      max_concurrent_chunks: number;
      speaker_correlation_enabled: boolean;
    };
    service_mode: string;
  };
  frontend_compatible: {
    whisper_service_settings: any;
    orchestration_chunking_settings: any;
    migration_status: any;
  };
  sync_info: {
    last_sync: string | null;
    sync_source: string;
    configuration_version: string;
  };
  presets: Record<string, any>;
}

interface CompatibilityStatus {
  compatible: boolean;
  issues: string[];
  warnings: string[];
  sync_required: boolean;
}

interface SyncResult {
  success: boolean;
  sync_time: string;
  services_synced: string[];
  errors: string[];
  compatibility_status?: CompatibilityStatus;
}

const ConfigSyncSettings: React.FC<ConfigSyncSettingsProps> = ({ onSave }) => {
  const [configuration, setConfiguration] = useState<ServiceConfiguration | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [compatibilityStatus, setCompatibilityStatus] = useState<CompatibilityStatus | null>(null);
  const [autoSyncEnabled, setAutoSyncEnabled] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState('exact_whisper_match');
  const [lastSyncResult, setLastSyncResult] = useState<SyncResult | null>(null);

  // Load unified configuration
  const loadConfiguration = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/settings/sync/unified');
      if (response.ok) {
        const config = await response.json();
        setConfiguration(config);
        
        // Check compatibility status
        const compatResponse = await fetch('/api/settings/sync/compatibility');
        if (compatResponse.ok) {
          const compatStatus = await compatResponse.json();
          setCompatibilityStatus(compatStatus);
        }
      } else {
        onSave('Failed to load configuration', false);
      }
    } catch (error) {
      console.error('Configuration load error:', error);
      onSave('Failed to load configuration', false);
    } finally {
      setLoading(false);
    }
  };

  // Force sync all configurations
  const handleForceSync = async () => {
    setSyncing(true);
    try {
      const response = await fetch('/api/settings/sync/force', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (response.ok) {
        const result = await response.json();
        setLastSyncResult(result);
        
        if (result.success) {
          onSave('Configuration synchronization completed successfully');
          await loadConfiguration(); // Reload to get updated config
        } else {
          onSave(`Synchronization failed: ${result.errors.join(', ')}`, false);
        }
      } else {
        onSave('Failed to trigger synchronization', false);
      }
    } catch (error) {
      console.error('Sync error:', error);
      onSave('Synchronization error occurred', false);
    } finally {
      setSyncing(false);
    }
  };

  // Apply configuration preset
  const handleApplyPreset = async () => {
    if (!selectedPreset) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/settings/sync/preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_name: selectedPreset }),
      });
      
      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          onSave(`Applied preset: ${selectedPreset}`);
          await loadConfiguration();
        } else {
          onSave(`Failed to apply preset: ${result.errors?.join(', ') || 'Unknown error'}`, false);
        }
      } else {
        onSave('Failed to apply preset', false);
      }
    } catch (error) {
      console.error('Preset application error:', error);
      onSave('Failed to apply preset', false);
    } finally {
      setLoading(false);
    }
  };


  useEffect(() => {
    loadConfiguration();
  }, []);

  const getCompatibilityStatusIcon = () => {
    if (!compatibilityStatus) return <Info color="action" />;
    
    if (compatibilityStatus.compatible && (compatibilityStatus.issues?.length || 0) === 0) {
      return <CheckCircle color="success" />;
    } else if ((compatibilityStatus.issues?.length || 0) > 0) {
      return <Error color="error" />;
    } else {
      return <Warning color="warning" />;
    }
  };

  const getCompatibilityStatusText = () => {
    if (!compatibilityStatus) return "Unknown";
    
    if (compatibilityStatus.compatible && (compatibilityStatus.issues?.length || 0) === 0) {
      return "Fully Compatible";
    } else if ((compatibilityStatus.issues?.length || 0) > 0) {
      return "Compatibility Issues";
    } else {
      return "Warnings Present";
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Configuration Synchronization
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Manage configuration synchronization between whisper service, orchestration coordinator, and frontend settings.
        Ensures all components stay in sync with consistent parameters.
      </Typography>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Sync Status Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Synchronization Status</Typography>
            <Box display="flex" gap={1}>
              <Tooltip title="Force sync all configurations">
                <IconButton 
                  onClick={handleForceSync} 
                  disabled={syncing}
                  color="primary"
                >
                  {syncing ? <CloudSync className="spinning" /> : <Sync />}
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh configuration">
                <IconButton onClick={loadConfiguration} disabled={loading}>
                  <Refresh />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center" gap={1}>
                {getCompatibilityStatusIcon()}
                <Typography variant="body1">
                  {getCompatibilityStatusText()}
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography variant="body2" color="text.secondary">
                Last Sync: {configuration?.sync_info?.last_sync 
                  ? new Date(configuration.sync_info.last_sync).toLocaleString()
                  : 'Never'
                }
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoSyncEnabled}
                    onChange={(e) => setAutoSyncEnabled(e.target.checked)}
                  />
                }
                label="Auto-sync enabled"
              />
            </Grid>
          </Grid>

          {compatibilityStatus && (
            <Box mt={2}>
              {(compatibilityStatus?.issues?.length || 0) > 0 && (
                <Alert severity="error" sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Configuration Issues:</Typography>
                  <List dense>
                    {compatibilityStatus.issues.map((issue, index) => (
                      <ListItem key={index} disablePadding>
                        <ListItemText primary={`• ${issue}`} />
                      </ListItem>
                    ))}
                  </List>
                </Alert>
              )}

              {(compatibilityStatus?.warnings?.length || 0) > 0 && (
                <Alert severity="warning" sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Configuration Warnings:</Typography>
                  <List dense>
                    {compatibilityStatus.warnings.map((warning, index) => (
                      <ListItem key={index} disablePadding>
                        <ListItemText primary={`• ${warning}`} />
                      </ListItem>
                    ))}
                  </List>
                </Alert>
              )}

              {compatibilityStatus.sync_required && (
                <Alert severity="info">
                  Configuration synchronization is recommended to resolve timing mismatches.
                </Alert>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Configuration Presets */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configuration Presets
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Apply predefined configuration templates for common use cases.
          </Typography>

          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Configuration Preset</InputLabel>
                <Select
                  value={selectedPreset}
                  onChange={(e) => setSelectedPreset(e.target.value)}
                  label="Configuration Preset"
                >
                  {configuration?.presets && Object.entries(configuration.presets).map(([key, _preset]) => (
                    <MenuItem key={key} value={key}>
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Button
                variant="contained"
                onClick={handleApplyPreset}
                disabled={!selectedPreset || loading}
                startIcon={<Settings />}
              >
                Apply Preset
              </Button>
            </Grid>
          </Grid>

          {configuration?.presets && selectedPreset && configuration.presets[selectedPreset] && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="subtitle2">
                {configuration.presets[selectedPreset].description}
              </Typography>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Service Configurations */}
      {configuration && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6">Service Configurations</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {/* Whisper Service Configuration */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Box display="flex" alignItems="center" gap={1} mb={2}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        Whisper Service
                      </Typography>
                      <Chip 
                        size="small" 
                        label={configuration.orchestration_service.service_mode}
                        color={configuration.orchestration_service.service_mode === 'orchestration' ? 'success' : 'default'}
                      />
                    </Box>
                    
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Sample Rate" 
                          secondary={`${configuration.whisper_service.sample_rate} Hz`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Buffer Duration" 
                          secondary={`${configuration.whisper_service.buffer_duration}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Inference Interval" 
                          secondary={`${configuration.whisper_service.inference_interval}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Overlap Duration" 
                          secondary={`${configuration.whisper_service.overlap_duration}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="VAD Enabled" 
                          secondary={configuration.whisper_service.enable_vad ? 'Yes' : 'No'} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Max Concurrent Requests" 
                          secondary={configuration.whisper_service.max_concurrent_requests} 
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Orchestration Service Configuration */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Orchestration Service
                    </Typography>
                    
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Chunk Duration" 
                          secondary={`${configuration.orchestration_service.chunking_config.chunk_duration}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Overlap Duration" 
                          secondary={`${configuration.orchestration_service.chunking_config.overlap_duration}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Processing Interval" 
                          secondary={`${configuration.orchestration_service.chunking_config.processing_interval}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Buffer Duration" 
                          secondary={`${configuration.orchestration_service.chunking_config.buffer_duration}s`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Max Concurrent Chunks" 
                          secondary={configuration.orchestration_service.chunking_config.max_concurrent_chunks} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Speaker Correlation" 
                          secondary={configuration.orchestration_service.chunking_config.speaker_correlation_enabled ? 'Enabled' : 'Disabled'} 
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Sync Results */}
      {lastSyncResult && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Last Sync Results
            </Typography>
            
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              {lastSyncResult.success ? (
                <CheckCircle color="success" />
              ) : (
                <Error color="error" />
              )}
              <Typography variant="body1">
                {lastSyncResult.success ? 'Synchronization Successful' : 'Synchronization Failed'}
              </Typography>
              <Chip 
                size="small" 
                label={new Date(lastSyncResult.sync_time).toLocaleString()}
              />
            </Box>

            {(lastSyncResult?.services_synced?.length || 0) > 0 && (
              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Services Synchronized:
                </Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {lastSyncResult.services_synced.map((service, index) => (
                    <Chip 
                      key={index} 
                      size="small" 
                      label={service.replace(/_/g, ' ')}
                      color="success"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            )}

            {(lastSyncResult?.errors?.length || 0) > 0 && (
              <Alert severity="error">
                <Typography variant="subtitle2">Sync Errors:</Typography>
                <List dense>
                  {lastSyncResult.errors.map((error, index) => (
                    <ListItem key={index} disablePadding>
                      <ListItemText primary={`• ${error}`} />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          onClick={handleForceSync}
          disabled={syncing || loading}
          startIcon={syncing ? <CloudSync className="spinning" /> : <Sync />}
        >
          {syncing ? 'Synchronizing...' : 'Force Sync All'}
        </Button>
        
        <Button
          variant="outlined"
          onClick={loadConfiguration}
          disabled={loading}
          startIcon={<Refresh />}
        >
          Refresh Configuration
        </Button>
      </Box>
    </Box>
  );
};

export default ConfigSyncSettings;