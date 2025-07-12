import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Button,
  Chip,
  Avatar,
  IconButton,
  Menu,
  MenuItem,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Divider,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  ExpandMore as ExpandMoreIcon,
  SmartToy as BotIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon,
  Videocam as VideocamIcon,
  VideocamOff as VideocamOffIcon,
  Language as LanguageIcon,
  Timeline as TimelineIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import { BotInstance, BotStatus } from '@/types';
import { formatDistanceToNow } from 'date-fns';

interface ActiveBotsProps {
  bots: Record<string, BotInstance>;
  activeBotIds: string[];
  onTerminateBot: (botId: string) => void;
  onBotError: (botId: string, error: string) => void;
}

export const ActiveBots: React.FC<ActiveBotsProps> = ({
  bots,
  activeBotIds,
  onTerminateBot,
  onBotError,
}) => {
  const [anchorEls, setAnchorEls] = useState<Record<string, HTMLElement | null>>({});
  const [expandedBot, setExpandedBot] = useState<string | null>(null);

  const handleMenuClick = (botId: string, event: React.MouseEvent<HTMLElement>) => {
    setAnchorEls(prev => ({ ...prev, [botId]: event.currentTarget }));
  };

  const handleMenuClose = (botId: string) => {
    setAnchorEls(prev => ({ ...prev, [botId]: null }));
  };

  const getStatusColor = (status: BotStatus) => {
    switch (status) {
      case 'active': return 'success';
      case 'spawning': return 'warning';
      case 'error': return 'error';
      case 'terminated': return 'default';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: BotStatus) => {
    switch (status) {
      case 'active': return <CheckCircleIcon />;
      case 'spawning': return <ScheduleIcon />;
      case 'error': return <ErrorIcon />;
      case 'terminated': return <StopIcon />;
      default: return <BotIcon />;
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const calculateQualityScore = (bot: BotInstance) => {
    const { audioCapture, performance } = bot;
    const audioQuality = audioCapture.averageQualityScore || 0;
    const latencyScore = Math.max(0, 1 - (performance.averageLatency / 1000));
    const errorScore = Math.max(0, 1 - (performance.errorCount / 100));
    
    return Math.round(((audioQuality + latencyScore + errorScore) / 3) * 100);
  };

  const activeBots = activeBotIds.map(id => bots[id]).filter(Boolean);

  if (activeBots.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <BotIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Active Bots
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Spawn a new bot to start monitoring Google Meet sessions
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Active Bots ({activeBots.length})
      </Typography>
      
      <Grid container spacing={2}>
        {activeBots.map((bot) => (
          <Grid item xs={12} key={bot.botId}>
            <Card>
              <CardContent>
                {/* Bot Header */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ mr: 2, bgcolor: `${getStatusColor(bot.status)}.main` }}>
                    {getStatusIcon(bot.status)}
                  </Avatar>
                  
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" component="div">
                      {bot.meetingInfo.meetingTitle || 'Untitled Meeting'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Bot ID: {bot.botId} • Meeting: {bot.meetingInfo.meetingId}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={bot.status}
                      color={getStatusColor(bot.status)}
                      size="small"
                      variant="filled"
                    />
                    <Chip
                      label={`${calculateQualityScore(bot)}% Quality`}
                      color={calculateQualityScore(bot) > 80 ? 'success' : calculateQualityScore(bot) > 60 ? 'warning' : 'error'}
                      size="small"
                      variant="outlined"
                    />
                    <IconButton
                      onClick={(e) => handleMenuClick(bot.botId, e)}
                      size="small"
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>
                </Box>

                {/* Status Indicators */}
                <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                  <Chip
                    icon={bot.audioCapture.isCapturing ? <MicIcon /> : <MicOffIcon />}
                    label={bot.audioCapture.isCapturing ? 'Audio Active' : 'Audio Inactive'}
                    color={bot.audioCapture.isCapturing ? 'success' : 'error'}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={bot.virtualWebcam.isStreaming ? <VideocamIcon /> : <VideocamOffIcon />}
                    label={bot.virtualWebcam.isStreaming ? 'Webcam Active' : 'Webcam Inactive'}
                    color={bot.virtualWebcam.isStreaming ? 'success' : 'error'}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<LanguageIcon />}
                    label={`${bot.virtualWebcam.currentTranslations.length} Translations`}
                    color="info"
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<TimelineIcon />}
                    label={`${bot.captionProcessor.totalSpeakers} Speakers`}
                    color="secondary"
                    size="small"
                    variant="outlined"
                  />
                </Box>

                {/* Performance Metrics */}
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Session Duration
                    </Typography>
                    <Typography variant="h6">
                      {formatDuration(bot.performance.sessionDuration)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Audio Chunks
                    </Typography>
                    <Typography variant="h6">
                      {bot.audioCapture.totalChunksCaptured.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Captions Processed
                    </Typography>
                    <Typography variant="h6">
                      {bot.captionProcessor.totalCaptionsProcessed.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Avg Latency
                    </Typography>
                    <Typography variant="h6">
                      {bot.performance.averageLatency}ms
                    </Typography>
                  </Grid>
                </Grid>

                {/* Progress Indicators */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Audio Quality: {bot.audioCapture.averageQualityScore?.toFixed(2) || 'N/A'}
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(bot.audioCapture.averageQualityScore || 0) * 100} 
                    color={bot.audioCapture.averageQualityScore > 0.8 ? 'success' : 'warning'}
                  />
                </Box>

                {/* Expandable Details */}
                <Accordion 
                  expanded={expandedBot === bot.botId} 
                  onChange={() => setExpandedBot(expandedBot === bot.botId ? null : bot.botId)}
                  sx={{ mt: 2 }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="body2">
                      Detailed Metrics & Timeline
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle2" gutterBottom>
                          Time Correlation
                        </Typography>
                        <Stack spacing={1}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">Total Correlations:</Typography>
                            <Typography variant="body2">{bot.timeCorrelation.totalCorrelations}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">Success Rate:</Typography>
                            <Typography variant="body2">{(bot.timeCorrelation.successRate * 100).toFixed(1)}%</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">Avg Timing Offset:</Typography>
                            <Typography variant="body2">{bot.timeCorrelation.averageTimingOffset}ms</Typography>
                          </Box>
                        </Stack>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle2" gutterBottom>
                          Virtual Webcam
                        </Typography>
                        <Stack spacing={1}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">Frames Generated:</Typography>
                            <Typography variant="body2">{bot.virtualWebcam.framesGenerated.toLocaleString()}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">Resolution:</Typography>
                            <Typography variant="body2">{bot.virtualWebcam.webcamConfig.width}x{bot.virtualWebcam.webcamConfig.height}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">FPS:</Typography>
                            <Typography variant="body2">{bot.virtualWebcam.webcamConfig.fps}</Typography>
                          </Box>
                        </Stack>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="subtitle2" gutterBottom>
                          Recent Speaker Activity
                        </Typography>
                        <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                          {bot.captionProcessor.speakerTimeline.slice(-5).map((event, index) => (
                            <Box key={event.eventId} sx={{ mb: 1, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
                              <Typography variant="body2">
                                <strong>{event.speakerName}</strong> • {event.eventType} • {formatDistanceToNow(new Date(event.timestamp))} ago
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Confidence: {(event.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>

                {/* Bot Actions Menu */}
                <Menu
                  anchorEl={anchorEls[bot.botId]}
                  open={Boolean(anchorEls[bot.botId])}
                  onClose={() => handleMenuClose(bot.botId)}
                >
                  <MenuItem onClick={() => {
                    // Refresh bot data
                    handleMenuClose(bot.botId);
                  }}>
                    <RefreshIcon sx={{ mr: 1 }} />
                    Refresh
                  </MenuItem>
                  <MenuItem onClick={() => {
                    onTerminateBot(bot.botId);
                    handleMenuClose(bot.botId);
                  }}>
                    <StopIcon sx={{ mr: 1 }} />
                    Terminate Bot
                  </MenuItem>
                </Menu>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};