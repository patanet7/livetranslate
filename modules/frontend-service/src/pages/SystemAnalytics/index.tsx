/**
 * System Analytics Dashboard
 * 
 * Comprehensive monitoring dashboard providing:
 * - Real-time system metrics and performance monitoring
 * - Professional audio analysis and visualization tools
 * - Service health indicators and infrastructure monitoring
 * - Advanced analytics with export capabilities
 * - Professional latency analysis and spectral visualization
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Button,
  IconButton,
  Tooltip,
  Chip,
  Alert,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';

// Import API hooks
import { 
  useGetSystemHealthQuery,
  useGetSystemMetricsQuery,
  useGetServiceHealthQuery
} from '@/store/slices/apiSlice';
import {
  Analytics as AnalyticsIcon,
  Dashboard,
  ShowChart,
  HealthAndSafety,
  GraphicEq,
  VolumeUp,
  Timeline,
  Download,
  Refresh,
  Fullscreen,
  Settings,
  Assessment,
  Speed,
  Memory,
  Computer,
  NetworkCheck,
} from '@mui/icons-material';

// Import our professional components
import {
  RealTimeMetrics,
  PerformanceCharts,
  SystemHealthIndicators
} from '@/components/analytics';

import {
  FFTSpectralAnalyzer,
  LUFSMeter,
  LatencyHeatmap,
} from '@/components/visualizations';

import { TabPanel } from '@/components/ui';

const SystemAnalytics: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  // Load real API data
  const { data: systemHealth, isLoading: healthLoading, error: healthError } = useGetSystemHealthQuery();
  const { data: systemMetrics, isLoading: metricsLoading, error: metricsError } = useGetSystemMetricsQuery();
  const { data: serviceHealth, isLoading: serviceLoading, error: serviceError } = useGetServiceHealthQuery();

  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  const handleRefresh = useCallback(() => {
    setLastRefresh(new Date());
    // Trigger refresh for all components
  }, []);

  const handleExportData = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      dashboard: 'system-analytics',
      activeTab,
      lastRefresh: lastRefresh.toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `system-analytics-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeTab, lastRefresh]);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Tab configuration
  const tabs = [
    {
      label: 'Live Monitoring',
      icon: <Dashboard />,
      description: 'Real-time system metrics and performance monitoring',
      color: 'primary',
    },
    {
      label: 'Performance Charts',
      icon: <ShowChart />,
      description: 'Historical performance analysis and trends',
      color: 'secondary',
    },
    {
      label: 'System Health',
      icon: <HealthAndSafety />,
      description: 'Service health indicators and infrastructure status',
      color: 'success',
    },
    {
      label: 'Audio Analysis',
      icon: <GraphicEq />,
      description: 'Professional audio spectral analysis and FFT visualization',
      color: 'info',
    },
    {
      label: 'LUFS Metering',
      icon: <VolumeUp />,
      description: 'Professional loudness metering and EBU R128 compliance',
      color: 'warning',
    },
    {
      label: 'Latency Analysis',
      icon: <Timeline />,
      description: 'Advanced latency visualization and heatmap analysis',
      color: 'error',
    },
  ];

  // Calculate real quick stats from API data
  const quickStats = React.useMemo(() => {
    const defaultStats = [
      { label: 'Services Online', value: '0/0', color: 'error' as const, icon: <Computer /> },
      { label: 'Avg Latency', value: '0ms', color: 'primary' as const, icon: <Speed /> },
      { label: 'Memory Usage', value: '0%', color: 'success' as const, icon: <Memory /> },
      { label: 'Active Connections', value: '0', color: 'info' as const, icon: <NetworkCheck /> },
    ];
    
    if (healthLoading || metricsLoading || serviceLoading) {
      return defaultStats;
    }
    
    // Calculate services online
    const services = serviceHealth?.data || {};
    const totalServices = Object.keys(services).length;
    const healthyServices = Object.values(services).filter((s: any) => s.status === 'healthy').length;
    const servicesColor = healthyServices === totalServices ? 'success' : healthyServices > 0 ? 'warning' : 'error';
    
    // Get system metrics
    const metrics = systemMetrics?.data || {};
    const avgLatency = metrics.avgLatency || systemHealth?.data?.performance?.avgLatency || 0;
    const memoryUsage = metrics.memoryUsage || systemHealth?.data?.performance?.memory?.percentage || 0;
    const activeConnections = metrics.activeConnections || 0;
    
    // Determine colors based on values
    const latencyColor = avgLatency < 200 ? 'success' : avgLatency < 500 ? 'warning' : 'error';
    const memoryColor = memoryUsage < 70 ? 'success' : memoryUsage < 85 ? 'warning' : 'error';
    
    return [
      { 
        label: 'Services Online', 
        value: `${healthyServices}/${totalServices}`, 
        color: servicesColor, 
        icon: <Computer /> 
      },
      { 
        label: 'Avg Latency', 
        value: `${avgLatency.toFixed(0)}ms`, 
        color: latencyColor, 
        icon: <Speed /> 
      },
      { 
        label: 'Memory Usage', 
        value: `${memoryUsage.toFixed(0)}%`, 
        color: memoryColor, 
        icon: <Memory /> 
      },
      { 
        label: 'Active Connections', 
        value: activeConnections.toString(), 
        color: 'info' as const, 
        icon: <NetworkCheck /> 
      },
    ];
  }, [systemHealth, systemMetrics, serviceHealth, healthLoading, metricsLoading, serviceLoading]);

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: theme.palette.mode === 'dark' 
        ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.95)} 0%, ${alpha(theme.palette.primary.dark, 0.1)} 100%)`
        : `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.background.default, 0.95)} 100%)`,
      p: 3,
    }}>
      {/* Header Section */}
      <Card sx={{ 
        mb: 3,
        bgcolor: alpha(theme.palette.background.paper, 0.9),
        backdropFilter: 'blur(20px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}>
        <CardContent sx={{ pb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <AnalyticsIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              <Box>
                <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
                  System Analytics Dashboard
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Comprehensive monitoring and analysis platform for LiveTranslate system performance
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Tooltip title="Refresh All Data">
                <IconButton onClick={handleRefresh} color="primary">
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Export Analytics Data">
                <IconButton onClick={handleExportData} color="secondary">
                  <Download />
                </IconButton>
              </Tooltip>
              <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}>
                <IconButton onClick={toggleFullscreen} color="info">
                  <Fullscreen />
                </IconButton>
              </Tooltip>
              <Tooltip title="Dashboard Settings">
                <IconButton color="default">
                  <Settings />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Quick Stats */}
          <Grid container spacing={2}>
            {quickStats.map((stat, index) => (
              <Grid item xs={6} sm={3} key={index}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1.5,
                  p: 2,
                  bgcolor: alpha(theme.palette[stat.color as keyof typeof theme.palette].main, 0.1),
                  borderRadius: 2,
                  border: `1px solid ${alpha(theme.palette[stat.color as keyof typeof theme.palette].main, 0.2)}`,
                }}>
                  <Box sx={{ 
                    color: `${stat.color}.main`,
                    display: 'flex',
                    alignItems: 'center',
                  }}>
                    {stat.icon}
                  </Box>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1 }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>

          {/* Status Alert */}
          {(healthError || metricsError || serviceError) ? (
            <Alert 
              severity="error" 
              sx={{ 
                mt: 2,
                bgcolor: alpha(theme.palette.error.main, 0.1),
                border: `1px solid ${alpha(theme.palette.error.main, 0.2)}`,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span>Unable to load system data. Please check your connection and try refreshing.</span>
                <Chip 
                  icon={<Assessment />} 
                  label="Error Loading Data" 
                  size="small" 
                  color="error" 
                  variant="outlined"
                />
              </Box>
            </Alert>
          ) : (
            <Alert 
              severity="success" 
              sx={{ 
                mt: 2,
                bgcolor: alpha(theme.palette.success.main, 0.1),
                border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span>System data loaded successfully • Last updated: {lastRefresh.toLocaleTimeString()}</span>
                <Chip 
                  icon={<Assessment />} 
                  label="Professional Analytics Active" 
                  size="small" 
                  color="success" 
                  variant="outlined"
                />
              </Box>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Navigation Tabs */}
      <Card sx={{ 
        mb: 3,
        bgcolor: alpha(theme.palette.background.paper, 0.95),
        backdropFilter: 'blur(20px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              minHeight: 72,
              textTransform: 'none',
              fontWeight: 500,
              fontSize: '0.95rem',
              '&.Mui-selected': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                borderBottom: `3px solid ${theme.palette.primary.main}`,
              },
            },
            '& .MuiTabs-indicator': {
              display: 'none', // Custom indicator above
            },
          }}
        >
          {tabs.map((tab, index) => (
            <Tab 
              key={index}
              icon={tab.icon} 
              label={
                <Box sx={{ textAlign: 'left' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {tab.label}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                    {tab.description}
                  </Typography>
                </Box>
              }
              iconPosition="start"
              sx={{ 
                minWidth: 220,
                alignItems: 'flex-start',
                gap: 1.5,
                py: 2,
              }}
            />
          ))}
        </Tabs>
      </Card>

      {/* Tab Content */}
      <Card sx={{ 
        bgcolor: alpha(theme.palette.background.paper, 0.98),
        backdropFilter: 'blur(20px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        minHeight: 600,
      }}>
        <CardContent sx={{ p: 3 }}>
          {/* Live Monitoring Tab */}
          <TabPanel value={activeTab} index={0} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Real-Time System Monitoring
              </Typography>
              <RealTimeMetrics
                updateInterval={2000}
                showHistory={true}
                compact={false}
              />
            </Box>
          </TabPanel>

          {/* Performance Charts Tab */}
          <TabPanel value={activeTab} index={1} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Performance Analysis & Trends
              </Typography>
              <PerformanceCharts
                height={500}
                showControls={true}
                autoRefresh={true}
              />
            </Box>
          </TabPanel>

          {/* System Health Tab */}
          <TabPanel value={activeTab} index={2} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                System Health & Infrastructure Status
              </Typography>
              <SystemHealthIndicators
                compact={false}
                showTrends={true}
                autoRefresh={true}
                refreshInterval={30000}
              />
            </Box>
          </TabPanel>

          {/* Audio Analysis Tab */}
          <TabPanel value={activeTab} index={3} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Professional Audio Spectral Analysis
              </Typography>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  Professional FFT spectral analysis with customizable windowing functions,
                  peak detection, and harmonic analysis for audio quality assessment.
                </Typography>
              </Alert>
              <FFTSpectralAnalyzer
                height={450}
                realTime={true}
                showControls={true}
                onPeaksDetected={(peaks) => {
                  console.log('Spectral peaks detected:', peaks);
                }}
              />
            </Box>
          </TabPanel>

          {/* LUFS Metering Tab */}
          <TabPanel value={activeTab} index={4} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Professional LUFS Loudness Metering
              </Typography>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  EBU R128 / ITU-R BS.1770 compliant loudness measurement with integrated,
                  short-term, and momentary loudness analysis for broadcast standards compliance.
                </Typography>
              </Alert>
              <LUFSMeter
                height={400}
                showControls={true}
                standard="ebu"
                realTime={true}
                onComplianceChange={(isCompliant, violations) => {
                  if (!isCompliant) {
                    console.warn('LUFS compliance violations:', violations);
                  }
                }}
              />
            </Box>
          </TabPanel>

          {/* Latency Analysis Tab */}
          <TabPanel value={activeTab} index={5} idPrefix="analytics">
            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Advanced Latency Analysis & Heatmap
              </Typography>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  Professional latency visualization with service dependency tracking, 
                  percentile analysis, and anomaly detection for performance optimization.
                </Typography>
              </Alert>
              <LatencyHeatmap 
                timeRange="1h"
                services={['whisper-service', 'translation-service', 'orchestration-service', 'database']}
                height={450}
                showControls={true}
                showStats={true}
                alertThreshold={1000}
                onAnomalyDetected={(anomalies) => {
                  if (anomalies.length > 0) {
                    console.warn('Latency anomalies detected:', anomalies);
                  }
                }}
              />
            </Box>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Footer */}
      <Box sx={{ 
        mt: 3, 
        pt: 2, 
        borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="caption" color="text.secondary">
          LiveTranslate System Analytics Dashboard • Professional monitoring and analysis platform
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Last refresh: {lastRefresh.toLocaleString()} • Tab: {tabs[activeTab]?.label}
        </Typography>
      </Box>
    </Box>
  );
};

export default SystemAnalytics;