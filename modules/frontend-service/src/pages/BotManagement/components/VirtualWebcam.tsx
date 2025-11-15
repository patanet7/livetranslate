import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Paper,
  Chip,
  Stack,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Videocam as VideocamIcon,
  VideocamOff as VideocamOffIcon,
  Settings as SettingsIcon,
  Fullscreen as FullscreenIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Language as LanguageIcon,
} from '@mui/icons-material';
import { BotInstance, WebcamConfig } from '@/types';
import { TabPanel } from '@/components/ui';

interface VirtualWebcamProps {
  bots: Record<string, BotInstance>;
  activeBotIds: string[];
  onWebcamUpdate: (botId: string, config: WebcamConfig) => void;
}

export const VirtualWebcam: React.FC<VirtualWebcamProps> = ({
  bots,
  activeBotIds,
  onWebcamUpdate,
}) => {
  const [selectedBotId, setSelectedBotId] = useState<string>('');
  const [webcamConfig, setWebcamConfig] = useState<WebcamConfig>({
    width: 1920,
    height: 1080,
    fps: 30,
    displayMode: 'overlay',
    theme: 'dark',
    maxTranslationsDisplayed: 5,
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [currentFrame, setCurrentFrame] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (activeBotIds.length > 0 && !selectedBotId) {
      setSelectedBotId(activeBotIds[0]);
    }
  }, [activeBotIds, selectedBotId]);

  useEffect(() => {
    if (selectedBotId && bots[selectedBotId]) {
      setWebcamConfig(bots[selectedBotId].virtualWebcam.webcamConfig);
      setIsStreaming(bots[selectedBotId].virtualWebcam.isStreaming);
    }
  }, [selectedBotId, bots]);

  useEffect(() => {
    // Start frame polling when bot is selected and streaming
    if (selectedBotId && isStreaming) {
      intervalRef.current = setInterval(fetchLatestFrame, 1000 / webcamConfig.fps);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [selectedBotId, isStreaming, webcamConfig.fps]);

  const fetchLatestFrame = async () => {
    if (!selectedBotId) return;
    
    try {
      const response = await fetch(`/api/bot/${selectedBotId}/webcam/frame`);
      if (response.ok) {
        const frameData = await response.text();
        setCurrentFrame(frameData);
      }
    } catch (error) {
      console.error('Failed to fetch webcam frame:', error);
    }
  };

  const handleConfigChange = (field: keyof WebcamConfig, value: any) => {
    const newConfig = { ...webcamConfig, [field]: value };
    setWebcamConfig(newConfig);
    
    if (selectedBotId) {
      onWebcamUpdate(selectedBotId, newConfig);
    }
  };

  const handleStartStreaming = async () => {
    if (!selectedBotId) return;
    
    try {
      const response = await fetch(`/api/bot/${selectedBotId}/webcam/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: webcamConfig }),
      });
      
      if (response.ok) {
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Failed to start webcam streaming:', error);
    }
  };

  const handleStopStreaming = async () => {
    if (!selectedBotId) return;
    
    try {
      const response = await fetch(`/api/bot/${selectedBotId}/webcam/stop`, {
        method: 'POST',
      });
      
      if (response.ok) {
        setIsStreaming(false);
        setCurrentFrame('');
      }
    } catch (error) {
      console.error('Failed to stop webcam streaming:', error);
    }
  };

  const handleDownloadFrame = () => {
    if (!currentFrame) return;
    
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${currentFrame}`;
    a.download = `webcam_frame_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const activeBots = activeBotIds.map(id => bots[id]).filter(Boolean);
  const selectedBot = selectedBotId ? bots[selectedBotId] : null;
  const recentTranslations = selectedBot?.virtualWebcam.currentTranslations || [];

  if (activeBots.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <VideocamOffIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Active Bots
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Spawn a bot to enable virtual webcam functionality
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          <VideocamIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Virtual Webcam Manager
        </Typography>
        <Stack direction="row" spacing={2}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Select Bot</InputLabel>
            <Select
              value={selectedBotId}
              label="Select Bot"
              onChange={(e) => setSelectedBotId(e.target.value)}
            >
              {activeBots.map((bot) => (
                <MenuItem key={bot.botId} value={bot.botId}>
                  {bot.meetingInfo.meetingTitle || bot.meetingInfo.meetingId}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </Button>
        </Stack>
      </Box>

      {selectedBot && (
        <Grid container spacing={3}>
          {/* Webcam Preview */}
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Live Webcam Preview
                  </Typography>
                  <Stack direction="row" spacing={1}>
                    <Chip
                      label={isStreaming ? 'Streaming' : 'Stopped'}
                      color={isStreaming ? 'success' : 'error'}
                      icon={isStreaming ? <VideocamIcon /> : <VideocamOffIcon />}
                    />
                    <Chip
                      label={`${webcamConfig.width}x${webcamConfig.height} @ ${webcamConfig.fps}fps`}
                      variant="outlined"
                    />
                  </Stack>
                </Box>

                <Box sx={{ position: 'relative', width: '100%', bgcolor: 'black', borderRadius: 1 }}>
                  {currentFrame ? (
                    <img
                      src={`data:image/png;base64,${currentFrame}`}
                      alt="Virtual Webcam"
                      style={{
                        width: '100%',
                        height: 'auto',
                        borderRadius: 4,
                      }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: '100%',
                        height: 400,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                      }}
                    >
                      <Typography variant="h6">
                        {isStreaming ? 'Loading preview...' : 'Webcam not active'}
                      </Typography>
                    </Box>
                  )}
                </Box>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Stack direction="row" spacing={1}>
                    {isStreaming ? (
                      <Button
                        variant="contained"
                        color="error"
                        onClick={handleStopStreaming}
                        startIcon={<VideocamOffIcon />}
                      >
                        Stop Streaming
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        color="success"
                        onClick={handleStartStreaming}
                        startIcon={<VideocamIcon />}
                      >
                        Start Streaming
                      </Button>
                    )}
                    <Button
                      variant="outlined"
                      onClick={fetchLatestFrame}
                      startIcon={<RefreshIcon />}
                    >
                      Refresh
                    </Button>
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    <Button
                      variant="outlined"
                      onClick={handleDownloadFrame}
                      disabled={!currentFrame}
                      startIcon={<DownloadIcon />}
                    >
                      Download Frame
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<FullscreenIcon />}
                      onClick={() => {
                        if (currentFrame) {
                          const newWindow = window.open('', '_blank');
                          if (newWindow) {
                            newWindow.document.write(`
                              <html>
                                <head><title>Virtual Webcam - Fullscreen</title></head>
                                <body style="margin:0; background:black; display:flex; justify-content:center; align-items:center; height:100vh;">
                                  <img src="data:image/png;base64,${currentFrame}" style="max-width:100%; max-height:100%;" />
                                </body>
                              </html>
                            `);
                          }
                        }
                      }}
                      disabled={!currentFrame}
                    >
                      Fullscreen
                    </Button>
                  </Stack>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Translation Display */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <LanguageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Live Translations
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Frames Generated: {selectedBot.virtualWebcam.framesGenerated.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Translations: {recentTranslations.length}
                  </Typography>
                </Box>

                <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                  {recentTranslations.length === 0 ? (
                    <Alert severity="info">
                      No recent translations to display
                    </Alert>
                  ) : (
                    <Stack spacing={1}>
                      {recentTranslations.map((translation) => (
                        <Paper key={translation.translationId} sx={{ p: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Chip
                              label={translation.speakerName}
                              size="small"
                              variant="outlined"
                            />
                            <Typography variant="caption" color="text.secondary">
                              {new Date(translation.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Original ({translation.sourceLanguage.toUpperCase()}):</strong> {translation.translatedText}
                          </Typography>
                          <Typography variant="body2" color="primary">
                            <strong>Translation ({translation.targetLanguage.toUpperCase()}):</strong> {translation.translatedText}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Confidence: {Math.round(translation.translationConfidence * 100)}%
                            </Typography>
                            <Chip
                              label={`${translation.sourceLanguage} → ${translation.targetLanguage}`}
                              size="small"
                              color="primary"
                            />
                          </Box>
                        </Paper>
                      ))}
                    </Stack>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Virtual Webcam Settings</DialogTitle>
        <DialogContent>
          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
            <Tab label="Video Settings" />
            <Tab label="Display Settings" />
            <Tab label="Translation Settings" />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Resolution
                </Typography>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Resolution Preset</InputLabel>
                  <Select
                    value={`${webcamConfig.width}x${webcamConfig.height}`}
                    label="Resolution Preset"
                    onChange={(e) => {
                      const [width, height] = e.target.value.split('x').map(Number);
                      handleConfigChange('width', width);
                      handleConfigChange('height', height);
                    }}
                  >
                    <MenuItem value="1920x1080">1920x1080 (Full HD)</MenuItem>
                    <MenuItem value="1280x720">1280x720 (HD)</MenuItem>
                    <MenuItem value="854x480">854x480 (SD)</MenuItem>
                    <MenuItem value="640x360">640x360 (Low)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Frame Rate: {webcamConfig.fps} FPS
                </Typography>
                <Slider
                  value={webcamConfig.fps}
                  onChange={(_, value) => handleConfigChange('fps', value)}
                  min={15}
                  max={60}
                  step={5}
                  marks={[
                    { value: 15, label: '15' },
                    { value: 30, label: '30' },
                    { value: 60, label: '60' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Display Mode</InputLabel>
                  <Select
                    value={webcamConfig.displayMode}
                    label="Display Mode"
                    onChange={(e) => handleConfigChange('displayMode', e.target.value)}
                  >
                    <MenuItem value="overlay">Overlay on Video</MenuItem>
                    <MenuItem value="sidebar">Sidebar Panel</MenuItem>
                    <MenuItem value="fullscreen">Fullscreen Translations</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Theme</InputLabel>
                  <Select
                    value={webcamConfig.theme}
                    label="Theme"
                    onChange={(e) => handleConfigChange('theme', e.target.value)}
                  >
                    <MenuItem value="light">Light</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                    <MenuItem value="auto">Auto</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Max Translations Displayed: {webcamConfig.maxTranslationsDisplayed}
                </Typography>
                <Slider
                  value={webcamConfig.maxTranslationsDisplayed}
                  onChange={(_, value) => handleConfigChange('maxTranslationsDisplayed', value)}
                  min={1}
                  max={10}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 5, label: '5' },
                    { value: 10, label: '10' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>
            </Grid>
          </TabPanel>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button
            onClick={() => {
              if (selectedBotId) {
                onWebcamUpdate(selectedBotId, webcamConfig);
              }
              setSettingsOpen(false);
            }}
            variant="contained"
          >
            Apply Settings
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};