import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  Button,
  CardActions,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  SmartToy,
  AudioFile,
  Cable,
  Warning,
  CheckCircle,
  Error,
  Analytics,
  Translate,
  Settings,
  ArrowForward,
  Assessment,
} from '@mui/icons-material';
import { useAppSelector } from '@/store';
import { useGetSystemHealthQuery, useGetBotsQuery } from '@/store/slices/apiSlice';
import { LoadingScreen } from '@/components/ui/LoadingScreen';

// Dashboard widgets
const SystemHealthWidget: React.FC = () => {
  const { data: healthData, isLoading } = useGetSystemHealthQuery();
  const { serviceHealth, performance } = useAppSelector(state => state.system);

  if (isLoading) return <LoadingScreen variant="minimal" size="small" />;

  // Use API data directly if available, otherwise fall back to Redux state
  const apiData = healthData;
  const overallStatus = apiData?.status || 'unknown';
  const services = apiData?.services || {};
  const performanceData = apiData?.performance || performance;
  
  const statusColor = overallStatus === 'healthy' ? 'success' : 
                     overallStatus === 'degraded' ? 'warning' : 'error';

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h2">
            System Health
          </Typography>
          <Chip 
            label={overallStatus.toUpperCase()} 
            color={statusColor}
            size="small"
          />
        </Box>

        <Stack spacing={2}>
          {/* Service statuses */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Services
            </Typography>
            <Grid container spacing={1}>
              {Object.entries(services).map(([service, health]: [string, any]) => (
                <Grid item xs={6} key={service}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {health?.status === 'healthy' ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : health?.status === 'degraded' ? (
                      <Warning color="warning" fontSize="small" />
                    ) : (
                      <Error color="error" fontSize="small" />
                    )}
                    <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                      {service}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>

          {/* Performance metrics */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Performance
            </Typography>
            <Stack spacing={1}>
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption">CPU Usage</Typography>
                  <Typography variant="caption">{performanceData.cpu?.usage || 0}%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={performanceData.cpu?.usage || 0} 
                  color={(performanceData.cpu?.usage || 0) > 80 ? 'error' : (performanceData.cpu?.usage || 0) > 60 ? 'warning' : 'primary'}
                />
              </Box>
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption">Memory Usage</Typography>
                  <Typography variant="caption">{performanceData.memory?.percentage || 0}%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={performanceData.memory?.percentage || 0} 
                  color={(performanceData.memory?.percentage || 0) > 80 ? 'error' : (performanceData.memory?.percentage || 0) > 60 ? 'warning' : 'primary'}
                />
              </Box>
            </Stack>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

const BotManagementWidget: React.FC = () => {
  const { data: botsData, isLoading } = useGetBotsQuery();
  const { systemStats } = useAppSelector(state => state.bot);

  if (isLoading) return <LoadingScreen variant="minimal" size="small" />;

  const bots = botsData?.data || [];
  const activeBotsCount = bots.filter(bot => bot.status === 'active').length;
  const spawningBotsCount = bots.filter(bot => bot.status === 'spawning').length;
  const errorBotsCount = bots.filter(bot => bot.status === 'error').length;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h2">
            Bot Management
          </Typography>
          <SmartToy color="primary" />
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {activeBotsCount}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Active Bots
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="primary.main">
                {systemStats.totalBotsSpawned}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Total Spawned
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {spawningBotsCount}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Spawning
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="error.main">
                {errorBotsCount}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Errors
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {systemStats.errorRate > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Error Rate: {(systemStats.errorRate * 100).toFixed(1)}%
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={systemStats.errorRate * 100} 
              color="error"
              sx={{ mt: 0.5 }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

const AudioSystemWidget: React.FC = () => {
  const { devices, stats, currentQualityMetrics } = useAppSelector(state => state.audio);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h2">
            Audio System
          </Typography>
          <AudioFile color="primary" />
        </Box>

        <Stack spacing={2}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h5" color="primary.main">
                  {devices.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Audio Devices
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h5" color="success.main">
                  {stats.totalRecordings}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Recordings
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {currentQualityMetrics && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Current Quality
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption">Quality Score</Typography>
                <Typography variant="caption">
                  {((currentQualityMetrics.qualityScore || 0) * 100).toFixed(0)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={(currentQualityMetrics.qualityScore || 0) * 100} 
                color={(currentQualityMetrics.qualityScore || 0) > 0.8 ? 'success' : 
                       (currentQualityMetrics.qualityScore || 0) > 0.6 ? 'warning' : 'error'}
              />
            </Box>
          )}

          <Box>
            <Typography variant="caption" color="text.secondary">
              Average Quality: {(stats.averageQualityScore * 100).toFixed(1)}%
            </Typography>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

const ConnectionWidget: React.FC = () => {
  const { connection, stats } = useAppSelector(state => state.websocket);

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h2">
            WebSocket Connection
          </Typography>
          <Cable color={connection.isConnected ? 'success' : 'error'} />
        </Box>

        <Stack spacing={2}>
          <Box>
            <Chip 
              label={connection.isConnected ? 'Connected' : 'Disconnected'}
              color={connection.isConnected ? 'success' : 'error'}
              size="small"
            />
          </Box>

          {connection.isConnected && (
            <>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" color="primary.main">
                      {stats.messagesSent}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Messages Sent
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" color="success.main">
                      {stats.messagesReceived}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Messages Received
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              <Box>
                <Typography variant="caption" color="text.secondary">
                  Uptime: {stats.connectionDuration > 0 
                    ? formatDuration(Date.now() - stats.connectionDuration)
                    : '0s'
                  }
                </Typography>
              </Box>

              {stats.averageLatency > 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Average Latency: {Math.round(stats.averageLatency)}ms
                  </Typography>
                </Box>
              )}
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

const QuickActionsWidget: React.FC = () => {
  const navigate = useNavigate();

  const quickActions = [
    {
      title: 'Translation Testing',
      description: 'Test translation capabilities and prompt performance',
      icon: <Translate />,
      path: '/translation-testing',
      color: 'primary' as const,
    },
    {
      title: 'Analytics Dashboard',
      description: 'View comprehensive performance metrics',
      icon: <Analytics />,
      path: '/analytics',
      color: 'secondary' as const,
    },
    {
      title: 'Bot Management',
      description: 'Spawn and manage translation bots',
      icon: <SmartToy />,
      path: '/bot-management',
      color: 'success' as const,
    },
    {
      title: 'Audio Testing',
      description: 'Test audio capture and processing',
      icon: <AudioFile />,
      path: '/audio-testing',
      color: 'warning' as const,
    },
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" component="h2" gutterBottom>
          Quick Actions
        </Typography>
        <Grid container spacing={2}>
          {quickActions.map((action, index) => (
            <Grid item xs={12} sm={6} key={index}>
              <Paper 
                variant="outlined" 
                sx={{ 
                  p: 2, 
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': {
                    elevation: 2,
                    transform: 'translateY(-2px)',
                    borderColor: `${action.color}.main`,
                  }
                }}
                onClick={() => navigate(action.path)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 1,
                      backgroundColor: `${action.color}.main`,
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {action.icon}
                  </Box>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle2" fontWeight="medium">
                      {action.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {action.description}
                    </Typography>
                  </Box>
                  <ArrowForward fontSize="small" color="action" />
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

const TranslationStatsWidget: React.FC = () => {
  // Mock data - in real implementation, this would come from the analytics API
  const translationStats = {
    totalTranslations: 12847,
    successRate: 98.2,
    avgQualityScore: 0.89,
    avgLatency: 240,
    dailyGrowth: 12.5,
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h2">
            Translation Performance
          </Typography>
          <Assessment color="primary" />
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary.main">
                {translationStats.totalTranslations.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Total Translations
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 0.5 }}>
                <Typography variant="caption" color="success.main">
                  +{translationStats.dailyGrowth}% today
                </Typography>
              </Box>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {translationStats.successRate}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Success Rate
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary.main">
                {translationStats.avgQualityScore.toFixed(2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Avg Quality Score
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {translationStats.avgLatency}ms
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Avg Latency
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
      <CardActions>
        <Button
          size="small"
          startIcon={<Analytics />}
          onClick={() => window.location.href = '/analytics'}
          fullWidth
        >
          View Full Analytics
        </Button>
      </CardActions>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Page header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time monitoring and control for LiveTranslate orchestration services
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="View Analytics">
            <IconButton onClick={() => navigate('/analytics')} color="primary">
              <Analytics />
            </IconButton>
          </Tooltip>
          <Tooltip title="System Settings">
            <IconButton onClick={() => navigate('/settings')} color="inherit">
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Dashboard widgets */}
      <Grid container spacing={3}>
        {/* System Health */}
        <Grid item xs={12} md={6} lg={3}>
          <SystemHealthWidget />
        </Grid>

        {/* Bot Management */}
        <Grid item xs={12} md={6} lg={3}>
          <BotManagementWidget />
        </Grid>

        {/* Audio System */}
        <Grid item xs={12} md={6} lg={3}>
          <AudioSystemWidget />
        </Grid>

        {/* WebSocket Connection */}
        <Grid item xs={12} md={6} lg={3}>
          <ConnectionWidget />
        </Grid>

        {/* Translation Performance */}
        <Grid item xs={12} md={6}>
          <TranslationStatsWidget />
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <QuickActionsWidget />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;