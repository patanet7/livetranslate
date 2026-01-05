import React, { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  Drawer,
  Toolbar,
  AppBar,
  Chip,
  Alert,
  Snackbar,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Menu,
  Info,
  Share,
  FileUpload,
  AutoFixHigh,
  Speaker,
} from '@mui/icons-material';
import { ReactFlowProvider } from 'reactflow';

import {
  PipelineCanvas,
  ComponentLibrary,
  PresetManager,
  PipelineValidation,
  RealTimeProcessor,
  type PipelineData,
  type PipelinePreset,
  type AudioComponent,
} from '@/components/audio/PipelineEditor';

const DRAWER_WIDTH = 320;

interface PipelineStudioState {
  currentPipeline: PipelineData | null;
  isProcessing: boolean;
  validationResult: any;
  selectedPreset: PipelinePreset | null;
}

const PipelineStudio: React.FC = () => {
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(true);
  const [rightDrawerOpen, setRightDrawerOpen] = useState(false);
  const [activeLeftTab, setActiveLeftTab] = useState<'components' | 'presets'>('components');
  const [activeRightTab, setActiveRightTab] = useState<'validation' | 'processor' | 'settings'>('validation');
  
  const [pipelineState, setPipelineState] = useState<PipelineStudioState>({
    currentPipeline: null,
    isProcessing: false,
    validationResult: null,
    selectedPreset: null,
  });

  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });

  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [shareFormat, setShareFormat] = useState<'json' | 'url' | 'embed'>('json');

  // Pipeline handlers
  const handlePipelineChange = useCallback((pipeline: PipelineData) => {
    setPipelineState(prev => ({
      ...prev,
      currentPipeline: pipeline,
    }));
  }, []);

  const handleProcessingStart = useCallback(() => {
    if (!pipelineState.currentPipeline) return;
    
    setPipelineState(prev => ({ ...prev, isProcessing: true }));
    setSnackbar({
      open: true,
      message: 'Starting audio processing pipeline...',
      severity: 'info',
    });

    // Simulate processing - in real implementation, this would call the backend
    setTimeout(() => {
      setPipelineState(prev => ({ ...prev, isProcessing: false }));
      setSnackbar({
        open: true,
        message: 'Pipeline processing completed successfully!',
        severity: 'success',
      });
    }, 3000);
  }, [pipelineState.currentPipeline]);

  const handleProcessingStop = useCallback(() => {
    setPipelineState(prev => ({ ...prev, isProcessing: false }));
    setSnackbar({
      open: true,
      message: 'Pipeline processing stopped',
      severity: 'warning',
    });
  }, []);

  // Preset handlers
  const handleLoadPreset = useCallback((preset: PipelinePreset) => {
    // Create some sample nodes for the preset
    const sampleNodes = [
      {
        id: 'input_1',
        type: 'audioStage',
        position: { x: 100, y: 100 },
        data: {
          label: 'File Input',
          description: 'Upload and process audio files',
          stageType: 'input',
          icon: FileUpload,
          enabled: true,
          gainIn: 0,
          gainOut: 0,
          stageConfig: {},
          parameters: [],
          isProcessing: false,
          status: 'idle',
        },
      },
      {
        id: 'process_1',
        type: 'audioStage',
        position: { x: 400, y: 100 },
        data: {
          label: 'Voice Enhancement',
          description: 'Professional voice clarity enhancement',
          stageType: 'processing',
          icon: AutoFixHigh,
          enabled: true,
          gainIn: 0,
          gainOut: 0,
          stageConfig: {},
          parameters: [],
          isProcessing: false,
          status: 'idle',
        },
      },
      {
        id: 'output_1',
        type: 'audioStage',
        position: { x: 700, y: 100 },
        data: {
          label: 'Speaker Output',
          description: 'Real-time audio playback',
          stageType: 'output',
          icon: Speaker,
          enabled: true,
          gainIn: 0,
          gainOut: 0,
          stageConfig: {},
          parameters: [],
          isProcessing: false,
          status: 'idle',
        },
      },
    ];

    const sampleEdges = [
      {
        id: 'edge_1',
        source: 'input_1',
        target: 'process_1',
        type: 'audioConnection',
        animated: true,
      },
      {
        id: 'edge_2',
        source: 'process_1',
        target: 'output_1',
        type: 'audioConnection',
        animated: true,
      },
    ];

    const pipelineData: PipelineData = {
      id: `pipeline_${Date.now()}`,
      name: `${preset.name} (Loaded)`,
      description: preset.description,
      nodes: sampleNodes,
      edges: sampleEdges,
      created: new Date(),
      modified: new Date(),
      metadata: {
        totalLatency: 45.5,
        complexity: 'moderate',
        validated: true,
        errors: [],
        warnings: [],
      },
    };

    setPipelineState(prev => ({
      ...prev,
      currentPipeline: pipelineData,
      selectedPreset: preset,
    }));

    setSnackbar({
      open: true,
      message: `Loaded preset: ${preset.name}`,
      severity: 'success',
    });
  }, []);

  const handleSavePreset = useCallback((preset: PipelinePreset) => {
    setSnackbar({
      open: true,
      message: `Saved preset: ${preset.name}`,
      severity: 'success',
    });
  }, []);

  const handleDeletePreset = useCallback((_presetId: string) => {
    setSnackbar({
      open: true,
      message: 'Preset deleted',
      severity: 'info',
    });
  }, []);

  const handleExportPreset = useCallback((preset: PipelinePreset) => {
    const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${preset.name.replace(/\s+/g, '_')}_preset.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setSnackbar({
      open: true,
      message: `Exported preset: ${preset.name}`,
      severity: 'success',
    });
  }, []);

  const handleImportPreset = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedPreset = JSON.parse(e.target?.result as string);
        setSnackbar({
          open: true,
          message: `Imported preset: ${importedPreset.name}`,
          severity: 'success',
        });
      } catch (error) {
        setSnackbar({
          open: true,
          message: 'Failed to import preset: Invalid file format',
          severity: 'error',
        });
      }
    };
    reader.readAsText(file);
  }, []);

  // Component library handlers
  const handleComponentSelect = useCallback((component: AudioComponent) => {
    setSnackbar({
      open: true,
      message: `Selected component: ${component.label}`,
      severity: 'info',
    });
  }, []);

  const handleDragStart = useCallback((event: React.DragEvent, component: AudioComponent) => {
    event.dataTransfer.setData('application/audioComponent', JSON.stringify(component));
    event.dataTransfer.effectAllowed = 'move';
  }, []);

  // Share functionality
  const handleShare = useCallback(() => {
    if (!pipelineState.currentPipeline) return;

    let shareData: string;
    switch (shareFormat) {
      case 'json':
        shareData = JSON.stringify(pipelineState.currentPipeline, null, 2);
        break;
      case 'url':
        const encodedPipeline = btoa(JSON.stringify(pipelineState.currentPipeline));
        shareData = `${window.location.origin}/pipeline-studio?pipeline=${encodedPipeline}`;
        break;
      case 'embed':
        const embedCode = `<iframe src="${window.location.origin}/pipeline-studio?embed=true&pipeline=${btoa(JSON.stringify(pipelineState.currentPipeline))}" width="800" height="600"></iframe>`;
        shareData = embedCode;
        break;
      default:
        shareData = '';
    }

    navigator.clipboard.writeText(shareData).then(() => {
      setSnackbar({
        open: true,
        message: `Pipeline ${shareFormat.toUpperCase()} copied to clipboard`,
        severity: 'success',
      });
      setShareDialogOpen(false);
    });
  }, [pipelineState.currentPipeline, shareFormat]);

  // Create a blank pipeline
  const handleCreateBlankPipeline = useCallback(() => {
    const blankPipeline: PipelineData = {
      id: `pipeline_${Date.now()}`,
      name: 'New Pipeline',
      description: 'Custom audio processing pipeline',
      nodes: [],
      edges: [],
      created: new Date(),
      modified: new Date(),
      metadata: {
        totalLatency: 0,
        complexity: 'simple',
        validated: false,
        errors: ['Pipeline must have at least one input component', 'Pipeline must have at least one output component'],
        warnings: [],
      },
    };

    setPipelineState(prev => ({
      ...prev,
      currentPipeline: blankPipeline,
      selectedPreset: null,
    }));

    setSnackbar({
      open: true,
      message: 'Created new blank pipeline',
      severity: 'success',
    });
  }, []);

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Main App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            edge="start"
            onClick={() => setLeftDrawerOpen(!leftDrawerOpen)}
            sx={{ mr: 1 }}
          >
            <Menu />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Pipeline Studio - Visual Audio Processing Editor
          </Typography>

          {pipelineState.currentPipeline && (
            <Box display="flex" alignItems="center" gap={1} mr={2}>
              <Chip
                label={pipelineState.currentPipeline.name}
                size="small"
                variant="outlined"
              />
              {pipelineState.selectedPreset && (
                <Chip
                  label={`Based on: ${pipelineState.selectedPreset.name}`}
                  size="small"
                  color="primary"
                />
              )}
            </Box>
          )}

          <Box display="flex" alignItems="center" gap={1}>
            <IconButton onClick={() => setShareDialogOpen(true)} disabled={!pipelineState.currentPipeline}>
              <Share />
            </IconButton>
            <IconButton onClick={() => setRightDrawerOpen(!rightDrawerOpen)}>
              <Info />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Left Drawer - Components & Presets */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={leftDrawerOpen}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Tab Selection */}
          <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant={activeLeftTab === 'components' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setActiveLeftTab('components')}
                >
                  Components
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant={activeLeftTab === 'presets' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setActiveLeftTab('presets')}
                >
                  Presets
                </Button>
              </Grid>
            </Grid>
          </Box>

          {/* Tab Content */}
          <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
            {activeLeftTab === 'components' && (
              <ComponentLibrary
                onComponentSelect={handleComponentSelect}
                onDragStart={handleDragStart}
              />
            )}
            {activeLeftTab === 'presets' && (
              <PresetManager
                currentPipeline={pipelineState.selectedPreset || undefined}
                onLoadPreset={handleLoadPreset}
                onSavePreset={handleSavePreset}
                onDeletePreset={handleDeletePreset}
                onExportPreset={handleExportPreset}
                onImportPreset={handleImportPreset}
              />
            )}
          </Box>
        </Box>
      </Drawer>

      {/* Main Content - Pipeline Canvas */}
      <Box
        component="main"
        sx={{
          position: 'fixed',
          top: 64, // Account for AppBar
          left: leftDrawerOpen ? DRAWER_WIDTH : 0,
          right: rightDrawerOpen ? DRAWER_WIDTH : 0,
          bottom: 0,
          transition: 'left 0.3s ease-in-out, right 0.3s ease-in-out',
          overflow: 'hidden',
        }}
      >
        {pipelineState.currentPipeline === null ? (
          // Welcome Screen
          <Box
            sx={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'linear-gradient(45deg, #f5f5f5 30%, #e0e0e0 90%)',
            }}
          >
            <Card sx={{ maxWidth: 600, textAlign: 'center', p: 4 }}>
              <CardContent>
                <Typography variant="h4" gutterBottom>
                  Welcome to Pipeline Studio
                </Typography>
                <Typography variant="h6" color="text.secondary" paragraph>
                  Build professional audio processing pipelines with drag-and-drop simplicity
                </Typography>
                <Typography variant="body1" paragraph>
                  • Drag components from the library to create your pipeline<br />
                  • Use built-in presets for common audio processing scenarios<br />
                  • Real-time validation ensures your pipeline is always valid<br />
                  • Process audio through your custom pipeline in real-time
                </Typography>
                <Box sx={{ mt: 3 }}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => {
                      setActiveLeftTab('presets');
                      setLeftDrawerOpen(true);
                    }}
                    sx={{ mr: 2 }}
                  >
                    Load a Preset
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => {
                      handleCreateBlankPipeline();
                      setActiveLeftTab('components');
                      setLeftDrawerOpen(true);
                    }}
                  >
                    Start Building
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Box>
        ) : (
          // Pipeline Canvas - No extra wrappers
          <ReactFlowProvider>
            <PipelineCanvas
              initialPipeline={pipelineState.currentPipeline}
              onPipelineChange={handlePipelineChange}
              onProcessingStart={handleProcessingStart}
              onProcessingStop={handleProcessingStop}
              isProcessing={pipelineState.isProcessing}
              showMinimap={true}
              showGrid={true}
              height={window.innerHeight - 64}
            />
          </ReactFlowProvider>
        )}
      </Box>

      {/* Right Drawer - Validation & Settings */}
      <Drawer
        variant="persistent"
        anchor="right"
        open={rightDrawerOpen}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Tab Selection */}
          <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Button
                  fullWidth
                  variant={activeRightTab === 'validation' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setActiveRightTab('validation')}
                >
                  Validation
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  fullWidth
                  variant={activeRightTab === 'processor' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setActiveRightTab('processor')}
                >
                  Live
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  fullWidth
                  variant={activeRightTab === 'settings' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setActiveRightTab('settings')}
                >
                  Info
                </Button>
              </Grid>
            </Grid>
          </Box>

          {/* Tab Content */}
          <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
            {activeRightTab === 'validation' && pipelineState.validationResult && (
              <PipelineValidation validationResult={pipelineState.validationResult} />
            )}
            {activeRightTab === 'processor' && (
              <RealTimeProcessor
                currentPipeline={pipelineState.currentPipeline}
                onMetricsUpdate={(metrics) => {
                  // Update pipeline metrics in real-time
                  console.log('Real-time metrics:', metrics);
                }}
              />
            )}
            {activeRightTab === 'settings' && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Pipeline Information
                </Typography>
                {pipelineState.currentPipeline ? (
                  <Box>
                    <Typography variant="body2" paragraph>
                      <strong>Name:</strong> {pipelineState.currentPipeline.name}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Components:</strong> {pipelineState.currentPipeline.nodes.length}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Connections:</strong> {pipelineState.currentPipeline.edges.length}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Complexity:</strong> {pipelineState.currentPipeline.metadata.complexity}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Total Latency:</strong> {pipelineState.currentPipeline.metadata.totalLatency.toFixed(1)}ms
                    </Typography>
                  </Box>
                ) : (
                  <Alert severity="info">
                    No pipeline loaded. Create a new pipeline or load a preset to see information here.
                  </Alert>
                )}
              </Box>
            )}
          </Box>
        </Box>
      </Drawer>

      {/* Share Dialog */}
      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Share Pipeline</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 1 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Share Format</InputLabel>
              <Select
                value={shareFormat}
                label="Share Format"
                onChange={(e) => setShareFormat(e.target.value as any)}
              >
                <MenuItem value="json">JSON File</MenuItem>
                <MenuItem value="url">Shareable URL</MenuItem>
                <MenuItem value="embed">Embed Code</MenuItem>
              </Select>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              {shareFormat === 'json' && 'Download pipeline as JSON file for import into other instances'}
              {shareFormat === 'url' && 'Generate a shareable URL that loads this pipeline'}
              {shareFormat === 'embed' && 'Generate HTML embed code for websites'}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleShare} variant="contained">Copy to Clipboard</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default PipelineStudio;