import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Slider,
  Switch,
  FormControlLabel,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Divider,
  Alert,
  IconButton,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Paper,
  Grid,
} from '@mui/material';
import {
  Close,
  RestoreOutlined,
  Save,
  Info,
  VolumeUp,
  VolumeDown,
  Settings,
  Speed,
  Memory,
  BatteryFull,
  Warning,
  CheckCircle,
} from '@mui/icons-material';

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  nodeId: string;
  nodeData: any;
  onParameterChange: (nodeId: string, paramName: string, value: any) => void;
  onGainChange: (nodeId: string, type: 'in' | 'out', value: number) => void;
  onToggleEnabled: (nodeId: string, enabled: boolean) => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const SettingsPanel: React.FC<SettingsPanelProps> = ({
  open,
  onClose,
  nodeId,
  nodeData,
  onParameterChange,
  onGainChange,
  onToggleEnabled,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!nodeData) return null;

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleResetToDefaults = () => {
    // Reset all parameters to their default values
    nodeData.parameters.forEach((param: any) => {
      onParameterChange(nodeId, param.name, param.defaultValue || param.min);
    });
    onGainChange(nodeId, 'in', 0);
    onGainChange(nodeId, 'out', 0);
    setUnsavedChanges(false);
  };

  const handleSaveAsPreset = () => {
    // Save current settings as a preset (would integrate with PresetManager)
    console.log('Saving settings as preset...');
  };

  const formatGain = (value: number): string => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}dB`;
  };

  const getQualityImpact = (param: any): 'positive' | 'negative' | 'neutral' => {
    // Simple heuristic for quality impact
    if (param.name.includes('quality') || param.name.includes('enhancement')) return 'positive';
    if (param.name.includes('reduction') || param.name.includes('compression')) return 'negative';
    return 'neutral';
  };

  const getPerformanceImpact = (param: any): 'low' | 'medium' | 'high' => {
    // Simple heuristic for performance impact
    if (param.name.includes('quality') || param.name.includes('advanced')) return 'high';
    if (param.name.includes('basic') || param.name.includes('simple')) return 'low';
    return 'medium';
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { height: '80vh', display: 'flex', flexDirection: 'column' }
      }}
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={2}>
            <nodeData.icon sx={{ fontSize: 24, color: 'primary.main' }} />
            <Box>
              <Typography variant="h6">{nodeData.label} Settings</Typography>
              <Typography variant="caption" color="text.secondary">
                {nodeData.description}
              </Typography>
            </Box>
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ flexGrow: 1, p: 0 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="settings tabs">
            <Tab label="Basic Settings" icon={<Settings sx={{ fontSize: 16 }} />} iconPosition="start" />
            <Tab label="Audio I/O" icon={<VolumeUp sx={{ fontSize: 16 }} />} iconPosition="start" />
            <Tab label="Performance" icon={<Speed sx={{ fontSize: 16 }} />} iconPosition="start" />
            <Tab label="Advanced" icon={<Memory sx={{ fontSize: 16 }} />} iconPosition="start" />
          </Tabs>
        </Box>

        {/* Basic Settings Tab */}
        <TabPanel value={activeTab} index={0}>
          <Box display="flex" flexDirection="column" gap={3}>
            {/* Enable/Disable Toggle */}
            <Paper variant="outlined" sx={{ p: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={nodeData.enabled}
                    onChange={(e) => {
                      onToggleEnabled(nodeId, e.target.checked);
                      setUnsavedChanges(true);
                    }}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1">
                      {nodeData.enabled ? 'Component Enabled' : 'Component Disabled'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {nodeData.enabled ? 
                        'Audio will be processed through this component' : 
                        'Audio will bypass this component'}
                    </Typography>
                  </Box>
                }
              />
            </Paper>

            {/* Main Parameters */}
            <Box>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Main Parameters
              </Typography>
              {nodeData.parameters.slice(0, 5).map((param: any) => (
                <Box key={param.name} mb={3}>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2">{param.displayName || param.name}</Typography>
                      <Tooltip title={param.description} arrow>
                        <Info sx={{ fontSize: 16, color: 'text.secondary' }} />
                      </Tooltip>
                    </Box>
                    <Typography variant="body2" fontWeight="bold">
                      {param.value.toFixed(param.step < 1 ? 1 : 0)}{param.unit || ''}
                    </Typography>
                  </Box>
                  
                  {param.type === 'slider' && (
                    <Slider
                      value={param.value}
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      marks
                      onChange={(_, value) => {
                        onParameterChange(nodeId, param.name, value as number);
                        setUnsavedChanges(true);
                      }}
                      sx={{ mt: 1 }}
                    />
                  )}
                  
                  {param.type === 'select' && (
                    <FormControl fullWidth size="small">
                      <Select
                        value={param.value}
                        onChange={(e) => {
                          onParameterChange(nodeId, param.name, e.target.value);
                          setUnsavedChanges(true);
                        }}
                      >
                        {param.options?.map((option: any) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                  
                  {param.type === 'toggle' && (
                    <Switch
                      checked={param.value}
                      onChange={(e) => {
                        onParameterChange(nodeId, param.name, e.target.checked);
                        setUnsavedChanges(true);
                      }}
                    />
                  )}
                </Box>
              ))}
            </Box>

            {/* Quality Presets (for certain components) */}
            {nodeData.label.includes('Enhancement') && (
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Quick Presets
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Button 
                      fullWidth 
                      variant="outlined" 
                      size="small"
                      onClick={() => {
                        // Apply light enhancement preset
                        onParameterChange(nodeId, 'clarityEnhancement', 0.3);
                        onParameterChange(nodeId, 'presenceBoost', 0.2);
                        setUnsavedChanges(true);
                      }}
                    >
                      Light
                    </Button>
                  </Grid>
                  <Grid item xs={4}>
                    <Button 
                      fullWidth 
                      variant="outlined" 
                      size="small"
                      onClick={() => {
                        // Apply moderate enhancement preset
                        onParameterChange(nodeId, 'clarityEnhancement', 0.6);
                        onParameterChange(nodeId, 'presenceBoost', 0.5);
                        setUnsavedChanges(true);
                      }}
                    >
                      Moderate
                    </Button>
                  </Grid>
                  <Grid item xs={4}>
                    <Button 
                      fullWidth 
                      variant="outlined" 
                      size="small"
                      onClick={() => {
                        // Apply heavy enhancement preset
                        onParameterChange(nodeId, 'clarityEnhancement', 0.9);
                        onParameterChange(nodeId, 'presenceBoost', 0.8);
                        setUnsavedChanges(true);
                      }}
                    >
                      Heavy
                    </Button>
                  </Grid>
                </Grid>
              </Paper>
            )}
          </Box>
        </TabPanel>

        {/* Audio I/O Tab */}
        <TabPanel value={activeTab} index={1}>
          <Box display="flex" flexDirection="column" gap={3}>
            {/* Input Gain */}
            <Box>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Input Gain Control
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <VolumeDown />
                    <Typography variant="body2">Input Gain</Typography>
                  </Box>
                  <Typography variant="body1" fontWeight="bold">
                    {formatGain(nodeData.gainIn)}
                  </Typography>
                </Box>
                <Slider
                  value={nodeData.gainIn}
                  min={-20}
                  max={20}
                  step={0.5}
                  marks={[
                    { value: -20, label: '-20dB' },
                    { value: 0, label: '0dB' },
                    { value: 20, label: '+20dB' },
                  ]}
                  onChange={(_, value) => {
                    onGainChange(nodeId, 'in', value as number);
                    setUnsavedChanges(true);
                  }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  Adjust the input level before processing. Be careful not to cause clipping.
                </Typography>
              </Paper>
            </Box>

            {/* Output Gain */}
            <Box>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Output Gain Control
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <VolumeUp />
                    <Typography variant="body2">Output Gain</Typography>
                  </Box>
                  <Typography variant="body1" fontWeight="bold">
                    {formatGain(nodeData.gainOut)}
                  </Typography>
                </Box>
                <Slider
                  value={nodeData.gainOut}
                  min={-20}
                  max={20}
                  step={0.5}
                  marks={[
                    { value: -20, label: '-20dB' },
                    { value: 0, label: '0dB' },
                    { value: 20, label: '+20dB' },
                  ]}
                  onChange={(_, value) => {
                    onGainChange(nodeId, 'out', value as number);
                    setUnsavedChanges(true);
                  }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  Adjust the output level after processing. Match levels with other components.
                </Typography>
              </Paper>
            </Box>

            {/* Level Meters (placeholder) */}
            <Alert severity="info" icon={<Info />}>
              Real-time level meters will appear here during audio processing
            </Alert>
          </Box>
        </TabPanel>

        {/* Performance Tab */}
        <TabPanel value={activeTab} index={2}>
          <Box display="flex" flexDirection="column" gap={3}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              Performance Metrics
            </Typography>

            {/* Current Performance */}
            {nodeData.metrics && (
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Speed sx={{ fontSize: 20 }} />
                      <Typography variant="body2">Processing Latency</Typography>
                    </Box>
                    <Typography variant="h5" fontWeight="bold">
                      {nodeData.metrics.processingTimeMs.toFixed(1)}ms
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Target: {nodeData.metrics.targetLatencyMs}ms
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Memory sx={{ fontSize: 20 }} />
                      <Typography variant="body2">CPU Usage</Typography>
                    </Box>
                    <Typography variant="h5" fontWeight="bold">
                      {nodeData.metrics.cpuUsage.toFixed(0)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {nodeData.metrics.cpuUsage < 20 ? 'Low' : 
                       nodeData.metrics.cpuUsage < 50 ? 'Moderate' : 'High'} usage
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            )}

            {/* Performance Optimization Tips */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Optimization Tips
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary="Reduce quality settings for lower latency"
                    secondary="Consider using 'Fast' mode if available"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Info sx={{ fontSize: 16, color: 'info.main' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary="Disable unused features"
                    secondary="Turn off advanced processing when not needed"
                  />
                </ListItem>
                {nodeData.metrics && nodeData.metrics.processingTimeMs > nodeData.metrics.targetLatencyMs && (
                  <ListItem>
                    <ListItemIcon>
                      <Warning sx={{ fontSize: 16, color: 'warning.main' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary="Processing time exceeds target"
                      secondary="Consider simplifying settings or upgrading hardware"
                    />
                  </ListItem>
                )}
              </List>
            </Box>

            {/* Quality vs Performance Trade-off */}
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Quality vs Performance
              </Typography>
              <FormControl fullWidth size="small">
                <InputLabel>Processing Mode</InputLabel>
                <Select
                  value={nodeData.stageConfig.mode || 'balanced'}
                  label="Processing Mode"
                  onChange={(e) => {
                    onParameterChange(nodeId, 'mode', e.target.value);
                    setUnsavedChanges(true);
                  }}
                >
                  <MenuItem value="fast">
                    <Box>
                      <Typography variant="body2">Fast (Low Latency)</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Prioritize speed over quality
                      </Typography>
                    </Box>
                  </MenuItem>
                  <MenuItem value="balanced">
                    <Box>
                      <Typography variant="body2">Balanced</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Good quality with reasonable latency
                      </Typography>
                    </Box>
                  </MenuItem>
                  <MenuItem value="quality">
                    <Box>
                      <Typography variant="body2">High Quality</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Best quality, higher latency
                      </Typography>
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Box>
        </TabPanel>

        {/* Advanced Tab */}
        <TabPanel value={activeTab} index={3}>
          <Box display="flex" flexDirection="column" gap={3}>
            <Alert severity="warning" icon={<Warning />}>
              Advanced settings can significantly impact audio quality and performance. 
              Modify with caution.
            </Alert>

            {/* Additional Parameters */}
            {nodeData.parameters.length > 5 && (
              <Box>
                <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                  Advanced Parameters
                </Typography>
                {nodeData.parameters.slice(5).map((param: any) => (
                  <Box key={param.name} mb={3}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body2">{param.displayName || param.name}</Typography>
                        <Tooltip title={param.description} arrow>
                          <Info sx={{ fontSize: 16, color: 'text.secondary' }} />
                        </Tooltip>
                        <Chip
                          label={getPerformanceImpact(param)}
                          size="small"
                          color={
                            getPerformanceImpact(param) === 'high' ? 'error' :
                            getPerformanceImpact(param) === 'medium' ? 'warning' : 'success'
                          }
                          sx={{ height: 18, fontSize: '0.7rem' }}
                        />
                      </Box>
                      <Typography variant="body2" fontWeight="bold">
                        {param.value.toFixed(param.step < 1 ? 1 : 0)}{param.unit || ''}
                      </Typography>
                    </Box>
                    
                    {param.type === 'slider' && (
                      <Slider
                        value={param.value}
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        marks
                        onChange={(_, value) => {
                          onParameterChange(nodeId, param.name, value as number);
                          setUnsavedChanges(true);
                        }}
                        sx={{ mt: 1 }}
                      />
                    )}
                  </Box>
                ))}
              </Box>
            )}

            {/* Debug Information */}
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Debug Information
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Component ID"
                    secondary={nodeId}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Component Type"
                    secondary={nodeData.stageType}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Processing Status"
                    secondary={nodeData.status}
                  />
                </ListItem>
              </List>
            </Paper>
          </Box>
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ p: 2 }}>
        <Box display="flex" justifyContent="space-between" width="100%">
          <Box>
            <Button
              startIcon={<RestoreOutlined />}
              onClick={handleResetToDefaults}
            >
              Reset to Defaults
            </Button>
            <Button
              startIcon={<Save />}
              onClick={handleSaveAsPreset}
            >
              Save as Preset
            </Button>
          </Box>
          <Box>
            {unsavedChanges && (
              <Chip
                label="Unsaved Changes"
                color="warning"
                size="small"
                sx={{ mr: 2 }}
              />
            )}
            <Button onClick={onClose}>
              Close
            </Button>
          </Box>
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default SettingsPanel;