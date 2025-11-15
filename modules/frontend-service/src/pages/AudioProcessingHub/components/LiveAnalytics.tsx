/**
 * LiveAnalytics - Real-time System Monitoring and Metrics
 * 
 * Provides comprehensive real-time monitoring of:
 * - System health and service status
 * - Performance metrics and trends
 * - Audio processing statistics
 * - Resource utilization
 * - Error tracking and alerts
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Alert,
  Button,
  useTheme,
  alpha,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Speed,
  Memory,
  NetworkCheck,
  Refresh,
  Notifications,
  Timeline,
} from '@mui/icons-material';

// Import chart components (we'll use simple implementations for now)
import { XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

import { useUnifiedAudio } from '@/hooks/useUnifiedAudio';
import { useAnalytics } from '@/hooks/useAnalytics';
import { useAppDispatch } from '@/store';
import { addNotification } from '@/store/slices/uiSlice';

// Component-specific service health interface
interface ServiceHealthDisplay {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  responseTime: number;
  lastCheck: Date;
  uptime: number;
  version?: string;
}

interface SystemMetrics {
  timestamp: Date;
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  activeConnections: number;
  requestsPerSecond: number;
  errorRate: number;
}

interface ProcessingStats {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageProcessingTime: number;
  peakProcessingTime: number;
  activeStreams: number;
}

const LiveAnalytics: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  useUnifiedAudio();

  // State
  const [services, setServices] = useState<ServiceHealthDisplay[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics[]>([]);
  const [processingStats, setProcessingStats] = useState<ProcessingStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [hasError, setHasError] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  // Use analytics hook
  const analyticsHook = useAnalytics();
  const {
    systemMetrics: systemMetricsData,
    serviceHealth: serviceHealthData,
    analyticsOverview: analyticsOverviewData,
    connectionStatus: connectionStatusData,
    refreshData: refreshAnalyticsData,
  } = analyticsHook;

  // Transform analytics hook data to component format
  const transformData = useCallback(() => {
    const systemMetrics: SystemMetrics = {
      timestamp: new Date(),
      cpu: systemMetricsData?.cpu?.percentage || 0,
      memory: systemMetricsData?.memory?.percentage || 0,
      disk: systemMetricsData?.disk?.percentage || 0,
      network: systemMetricsData?.network?.utilization || 0,
      activeConnections: systemMetricsData?.performance?.activeConnections || 0,
      requestsPerSecond: systemMetricsData?.performance?.requestsPerSecond || 0,
      errorRate: systemMetricsData?.performance?.errorRate || 0,
    };

    const serviceHealth: ServiceHealthDisplay[] = serviceHealthData.map(service => ({
      name: service.name,
      status: service.status === 'healthy' ? 'healthy' : 
             service.status === 'degraded' ? 'degraded' : 'down',
      responseTime: service.responseTime || 0,
      lastCheck: new Date(),
      uptime: service.uptime || 0,
      version: service.version || 'unknown',
    }));

    const processingStats: ProcessingStats = {
      totalRequests: analyticsOverviewData?.totalRequests || 0,
      successfulRequests: analyticsOverviewData?.successfulRequests || 0,
      failedRequests: analyticsOverviewData?.failedRequests || 0,
      averageProcessingTime: analyticsOverviewData?.averageProcessingTime || 0,
      peakProcessingTime: analyticsOverviewData?.peakProcessingTime || 0,
      activeStreams: analyticsOverviewData?.activeStreams || 0,
    };

    return { systemMetrics, serviceHealth, processingStats };
  }, [systemMetricsData, serviceHealthData, analyticsOverviewData]);

  // Load data using unified analytics service
  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Use unified analytics service instead of direct API calls
      const { systemMetrics, serviceHealth, processingStats } = transformData();

      setServices(serviceHealth);
      // Add new metrics to the array, keeping last 20 entries
      setMetrics(prevMetrics => {
        const newMetrics = [...prevMetrics, systemMetrics].slice(-20);
        return newMetrics;
      });
      setProcessingStats(processingStats);
      setLastUpdate(new Date());
      setHasError(false);
      setConnectionStatus(connectionStatusData?.isConnected ? 'connected' : 'disconnected');
    } catch (error: unknown) {
      console.error('Failed to load analytics data:', error);
      setHasError(true);
      setConnectionStatus('disconnected');

      const errorMessage = (error as Error)?.message || 'Unable to load system analytics data';
      dispatch(addNotification({
        type: 'error',
        title: 'Analytics Load Failed',
        message: errorMessage,
        autoHide: true,
      }));
      
      // Keep existing data when API is unavailable (don't clear it)
      // Only clear on first load if no data exists
      if (services.length === 0) {
        setServices([]);
        setMetrics([]);
        setProcessingStats(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, [transformData, connectionStatusData, dispatch]);

  // Auto-refresh effect
  useEffect(() => {
    if (systemMetricsData || serviceHealthData.length > 0) {
      loadData();
    }
  }, [systemMetricsData, serviceHealthData, loadData]);

  // Manual refresh from analytics hook
  const handleRefresh = useCallback(async () => {
    await refreshAnalyticsData();
  }, [refreshAnalyticsData]);

  // Helper functions
  const getStatusIcon = (status: ServiceHealthDisplay['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'degraded':
        return <Warning color="warning" />;
      case 'down':
        return <Error color="error" />;
      default:
        return <Warning color="disabled" />;
    }
  };

  const getStatusColor = (status: ServiceHealthDisplay['status']) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'down':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatUptime = (uptime: number) => `${uptime.toFixed(2)}%`;
  const formatResponseTime = (time: number) => `${Math.round(time)}ms`;

  // Current metrics (latest data point)
  const currentMetrics = metrics[metrics.length - 1];
  const previousMetrics = metrics[metrics.length - 2];

  const getTrend = (current: number, previous: number) => {
    if (!previous) return 'stable';
    return current > previous ? 'up' : current < previous ? 'down' : 'stable';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp color="error" />;
      case 'down':
        return <TrendingDown color="success" />;
      default:
        return <Timeline color="disabled" />;
    }
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Loading Analytics...</Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (hasError && services.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Unable to Connect to Analytics API
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            The analytics dashboard cannot connect to the backend services. Please check:
          </Typography>
          <Box component="ul" sx={{ ml: 2 }}>
            <li>Backend orchestration service is running on port 3000</li>
            <li>All required services are healthy</li>
            <li>Network connectivity to the API endpoints</li>
          </Box>
        </Alert>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button 
            variant="contained" 
            onClick={handleRefresh}
            disabled={isLoading}
            startIcon={<Refresh />}
          >
            Retry Connection
          </Button>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Live System Analytics
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {hasError && (
            <Alert severity="warning" sx={{ py: 0.5, px: 1 }}>
              <Typography variant="caption">
                API Connection Lost - Showing cached data
              </Typography>
            </Alert>
          )}
          <Chip
            icon={<Notifications />}
            label={`Last Update: ${lastUpdate.toLocaleTimeString()}`}
            variant="outlined"
            size="small"
            color={connectionStatus === 'connected' ? 'success' : 'error'}
          />
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} size="small" disabled={isLoading}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Service Health Cards */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
            Service Health
          </Typography>
          <Grid container spacing={2}>
            {services.map((service) => (
              <Grid item xs={12} sm={6} md={3} key={service.name}>
                <Card sx={{ 
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: 'blur(10px)',
                }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                        {service.name}
                      </Typography>
                      {getStatusIcon(service.status)}
                    </Box>
                    <Chip
                      label={service.status.toUpperCase()}
                      color={getStatusColor(service.status) as any}
                      size="small"
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="body2" color="textSecondary">
                      Response: {formatResponseTime(service.responseTime)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Uptime: {formatUptime(service.uptime)}
                    </Typography>
                    {service.version && (
                      <Typography variant="caption" color="textSecondary">
                        v{service.version}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* System Metrics */}
        <Grid item xs={12} md={8}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                System Performance
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <RechartsTooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                      formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="cpu" 
                      stroke={theme.palette.primary.main} 
                      fill={alpha(theme.palette.primary.main, 0.3)}
                      name="CPU"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="memory" 
                      stroke={theme.palette.secondary.main} 
                      fill={alpha(theme.palette.secondary.main, 0.3)}
                      name="Memory"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Metrics */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {currentMetrics && [
              { label: 'CPU Usage', value: currentMetrics.cpu, unit: '%', icon: <Speed />, trend: getTrend(currentMetrics.cpu, previousMetrics?.cpu || 0) },
              { label: 'Memory', value: currentMetrics.memory, unit: '%', icon: <Memory />, trend: getTrend(currentMetrics.memory, previousMetrics?.memory || 0) },
              { label: 'Active Connections', value: currentMetrics.activeConnections, unit: '', icon: <NetworkCheck />, trend: getTrend(currentMetrics.activeConnections, previousMetrics?.activeConnections || 0) },
              { label: 'Requests/sec', value: currentMetrics.requestsPerSecond, unit: '', icon: <TrendingUp />, trend: getTrend(currentMetrics.requestsPerSecond, previousMetrics?.requestsPerSecond || 0) },
            ].map((metric, index) => (
              <Grid item xs={6} key={index}>
                <Card sx={{ 
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: 'blur(10px)',
                }}>
                  <CardContent sx={{ pb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                      {metric.icon}
                      {getTrendIcon(metric.trend)}
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {Math.round(metric.value)}{metric.unit}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {metric.label}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Processing Statistics */}
        {processingStats && (
          <Grid item xs={12} md={6}>
            <Card sx={{ 
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: 'blur(10px)',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                  Audio Processing Statistics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Total Requests</Typography>
                    <Typography variant="h6">{processingStats.totalRequests.toLocaleString()}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Success Rate</Typography>
                    <Typography variant="h6" color="success.main">
                      {((processingStats.successfulRequests / processingStats.totalRequests) * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Avg Processing Time</Typography>
                    <Typography variant="h6">{Math.round(processingStats.averageProcessingTime)}ms</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Active Streams</Typography>
                    <Typography variant="h6" color="primary.main">{processingStats.activeStreams}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Error Log */}
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Recent Alerts
              </Typography>
              <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                {services.filter(s => s.status !== 'healthy').map((service, index) => (
                  <Alert 
                    key={index}
                    severity={service.status === 'down' ? 'error' : 'warning'}
                    sx={{ mb: 1, fontSize: '0.875rem' }}
                  >
                    {service.name}: {service.status === 'down' ? 'Service unavailable' : 'Performance degraded'}
                  </Alert>
                ))}
                {services.every(s => s.status === 'healthy') && (
                  <Alert severity="success">All systems operational</Alert>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveAnalytics;