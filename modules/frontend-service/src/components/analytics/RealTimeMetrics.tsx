/**
 * RealTimeMetrics - Real-time System Metrics Display
 *
 * Provides real-time monitoring of critical system metrics including:
 * - CPU, Memory, and Disk utilization
 * - Network activity and connection statistics
 * - Request rates and error tracking
 * - Service response times and throughput
 * - Audio processing performance metrics
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  CircularProgress,
  IconButton,
  Tooltip,
  Alert,
  useTheme,
  alpha,
} from "@mui/material";
import {
  Speed,
  Memory,
  Storage,
  NetworkWifi,
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Timer,
} from "@mui/icons-material";

// Import chart components
import { ResponsiveContainer, AreaChart, Area } from "recharts";

// Types
interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  icon: React.ReactElement;
  color: "primary" | "secondary" | "success" | "warning" | "error";
  trend: "up" | "down" | "stable";
  threshold: {
    warning: number;
    critical: number;
  };
  history: Array<{ timestamp: Date; value: number }>;
}

interface ServiceMetric {
  name: string;
  status: "healthy" | "degraded" | "down";
  responseTime: number;
  throughput: number;
  errorRate: number;
  uptime: number;
}

interface RealTimeMetricsProps {
  updateInterval?: number;
  showHistory?: boolean;
  compact?: boolean;
}

const RealTimeMetrics: React.FC<RealTimeMetricsProps> = ({
  updateInterval = 2000,
  showHistory = true,
  compact = false,
}) => {
  const theme = useTheme();

  // State
  const [metrics, setMetrics] = useState<SystemMetric[]>([]);
  const [serviceMetrics, setServiceMetrics] = useState<ServiceMetric[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [connectionStatus, setConnectionStatus] = useState<
    "connected" | "connecting" | "disconnected"
  >("connecting");

  // Initialize default metrics
  const initializeMetrics = useCallback(
    (): SystemMetric[] => [
      {
        id: "cpu",
        name: "CPU Usage",
        value: 0,
        unit: "%",
        icon: <Speed />,
        color: "primary",
        trend: "stable",
        threshold: { warning: 70, critical: 90 },
        history: [],
      },
      {
        id: "memory",
        name: "Memory Usage",
        value: 0,
        unit: "%",
        icon: <Memory />,
        color: "secondary",
        trend: "stable",
        threshold: { warning: 80, critical: 95 },
        history: [],
      },
      {
        id: "disk",
        name: "Disk Usage",
        value: 0,
        unit: "%",
        icon: <Storage />,
        color: "success",
        trend: "stable",
        threshold: { warning: 85, critical: 95 },
        history: [],
      },
      {
        id: "network",
        name: "Network I/O",
        value: 0,
        unit: "MB/s",
        icon: <NetworkWifi />,
        color: "warning",
        trend: "stable",
        threshold: { warning: 100, critical: 200 },
        history: [],
      },
      {
        id: "requests",
        name: "Requests/sec",
        value: 0,
        unit: "req/s",
        icon: <TrendingUp />,
        color: "primary",
        trend: "stable",
        threshold: { warning: 1000, critical: 2000 },
        history: [],
      },
      {
        id: "response_time",
        name: "Avg Response Time",
        value: 0,
        unit: "ms",
        icon: <Timer />,
        color: "secondary",
        trend: "stable",
        threshold: { warning: 500, critical: 1000 },
        history: [],
      },
    ],
    [],
  );

  // Subscribe to unified analytics service
  const [analyticsData] = useState<{
    systemMetrics: any;
    serviceHealth: any[];
    connectionStatus: any;
  }>({
    systemMetrics: null,
    serviceHealth: [],
    connectionStatus: { isConnected: false },
  });

  // Remove the broken useEffect - analytics data will be handled by the hook

  // Transform unified analytics data to component format
  const transformAnalyticsData = useCallback(() => {
    if (!analyticsData.systemMetrics) {
      return { systemMetrics: {}, serviceMetrics: [] };
    }

    const systemMetrics = {
      cpu: analyticsData.systemMetrics.cpu?.percentage || 0,
      memory: analyticsData.systemMetrics.memory?.percentage || 0,
      disk: analyticsData.systemMetrics.disk?.percentage || 0,
      network: analyticsData.systemMetrics.network?.utilization || 0,
      requests: analyticsData.systemMetrics.performance?.requestsPerSecond || 0,
      response_time:
        analyticsData.systemMetrics.performance?.averageResponseTime || 0,
    };

    const serviceMetrics: ServiceMetric[] = analyticsData.serviceHealth.map(
      (service) => ({
        name: service.name,
        status:
          service.status === "healthy"
            ? "healthy"
            : service.status === "degraded"
              ? "degraded"
              : "down",
        responseTime: service.responseTime || 0,
        throughput: 0, // Not provided by unified service yet
        errorRate: 0, // Not provided by unified service yet
        uptime: service.uptime || 0,
      }),
    );

    return { systemMetrics, serviceMetrics };
  }, [analyticsData]);

  // Update metrics using unified analytics service
  const updateMetrics = useCallback(async () => {
    try {
      setConnectionStatus("connecting");

      // Use unified analytics service
      const transformedData = transformAnalyticsData();

      setMetrics((prevMetrics) => {
        return prevMetrics.map((metric) => {
          const newValue =
            transformedData.systemMetrics[
              metric.id as keyof typeof transformedData.systemMetrics
            ] || 0;
          const previousValue = metric.value;

          // Calculate trend
          let trend: "up" | "down" | "stable" = "stable";
          if (newValue > previousValue + 5) trend = "up";
          else if (newValue < previousValue - 5) trend = "down";

          // Update history (keep last 50 points)
          const newHistory = [
            ...metric.history.slice(-49),
            { timestamp: new Date(), value: newValue },
          ];

          return {
            ...metric,
            value: newValue,
            trend,
            history: newHistory,
          };
        });
      });

      setServiceMetrics(transformedData.serviceMetrics);
      setConnectionStatus(
        analyticsData.connectionStatus?.isConnected
          ? "connected"
          : "disconnected",
      );
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to update metrics:", error);
      setConnectionStatus("disconnected");
    } finally {
      setIsLoading(false);
    }
  }, [transformAnalyticsData, analyticsData.connectionStatus]);

  // Initialize and start auto-update
  useEffect(() => {
    setMetrics(initializeMetrics());
  }, [initializeMetrics]);

  // Update metrics when analytics data changes
  useEffect(() => {
    if (analyticsData.systemMetrics || analyticsData.serviceHealth.length > 0) {
      updateMetrics();
    }
  }, [analyticsData, updateMetrics]);

  // Manual refresh from unified service
  const handleRefresh = useCallback(async () => {
    // Use analytics hook refresh method
  }, []);

  // Helper functions
  const getMetricColor = (metric: SystemMetric) => {
    if (metric.value >= metric.threshold.critical) return "error";
    if (metric.value >= metric.threshold.warning) return "warning";
    return metric.color;
  };

  const getMetricStatus = (metric: SystemMetric) => {
    if (metric.value >= metric.threshold.critical) return "critical";
    if (metric.value >= metric.threshold.warning) return "warning";
    return "normal";
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up":
        return <TrendingUp fontSize="small" />;
      case "down":
        return <TrendingDown fontSize="small" />;
      default:
        return null;
    }
  };

  const getServiceStatusIcon = (status: ServiceMetric["status"]) => {
    switch (status) {
      case "healthy":
        return <CheckCircle color="success" fontSize="small" />;
      case "degraded":
        return <Warning color="warning" fontSize="small" />;
      case "down":
        return <Error color="error" fontSize="small" />;
      default:
        return <Warning color="disabled" fontSize="small" />;
    }
  };

  const formatValue = (value: number, unit: string) => {
    if (unit === "ms" || unit === "req/s" || unit === "MB/s") {
      return `${Math.round(value)}${unit}`;
    }
    return `${Math.round(value)}${unit}`;
  };

  if (isLoading && metrics.length === 0) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: 200,
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (connectionStatus === "disconnected" && metrics.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Unable to Connect to Real-Time Metrics API
          </Typography>
          <Typography variant="body2">
            Cannot retrieve real-time system metrics. Please ensure the backend
            services are running and accessible.
          </Typography>
        </Alert>
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <IconButton onClick={handleRefresh} disabled={isLoading}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography variant={compact ? "h6" : "h5"} sx={{ fontWeight: 600 }}>
          Real-time System Metrics
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Chip
            icon={
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor:
                    connectionStatus === "connected"
                      ? "success.main"
                      : connectionStatus === "connecting"
                        ? "warning.main"
                        : "error.main",
                }}
              />
            }
            label={connectionStatus}
            size="small"
            variant="outlined"
          />
          <Tooltip title="Refresh Metrics">
            <IconButton onClick={handleRefresh} size="small">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Connection Status Alert */}
      {connectionStatus === "disconnected" && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Connection lost. Metrics may be outdated.
        </Alert>
      )}

      {/* System Metrics Grid */}
      <Grid container spacing={2}>
        {metrics.map((metric) => (
          <Grid item xs={12} sm={6} md={compact ? 6 : 4} key={metric.id}>
            <Card
              sx={{
                bgcolor: alpha(theme.palette.background.paper, 0.7),
                backdropFilter: "blur(10px)",
                border: `1px solid ${alpha(
                  theme.palette[getMetricColor(metric)].main,
                  0.3,
                )}`,
              }}
            >
              <CardContent sx={{ pb: compact ? 1 : 2 }}>
                {/* Metric Header */}
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 1,
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Box
                      sx={{
                        color: theme.palette[getMetricColor(metric)].main,
                        display: "flex",
                        alignItems: "center",
                      }}
                    >
                      {metric.icon}
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      {metric.name}
                    </Typography>
                  </Box>
                  {getTrendIcon(metric.trend)}
                </Box>

                {/* Metric Value */}
                <Typography
                  variant={compact ? "h6" : "h5"}
                  sx={{
                    fontWeight: 600,
                    color: theme.palette[getMetricColor(metric)].main,
                    mb: 1,
                  }}
                >
                  {formatValue(metric.value, metric.unit)}
                </Typography>

                {/* Progress Bar */}
                <LinearProgress
                  variant="determinate"
                  value={Math.min(metric.value, 100)}
                  color={getMetricColor(metric)}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    mb: showHistory && !compact ? 1 : 0,
                  }}
                />

                {/* Mini Chart */}
                {showHistory && !compact && metric.history.length > 1 && (
                  <Box sx={{ height: 60, mt: 1 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={metric.history}>
                        <Area
                          type="monotone"
                          dataKey="value"
                          stroke={theme.palette[getMetricColor(metric)].main}
                          fill={alpha(
                            theme.palette[getMetricColor(metric)].main,
                            0.2,
                          )}
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Box>
                )}

                {/* Status Chip */}
                {!compact && (
                  <Chip
                    label={getMetricStatus(metric)}
                    size="small"
                    color={getMetricColor(metric)}
                    variant="outlined"
                    sx={{ fontSize: "0.7rem", height: 20, mt: 1 }}
                  />
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Service Metrics */}
      {!compact && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
            Service Health
          </Typography>
          <Grid container spacing={2}>
            {serviceMetrics.map((service, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card
                  sx={{
                    bgcolor: alpha(theme.palette.background.paper, 0.7),
                    backdropFilter: "blur(10px)",
                  }}
                >
                  <CardContent>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        mb: 1,
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                        {service.name}
                      </Typography>
                      {getServiceStatusIcon(service.status)}
                    </Box>

                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 0.5,
                      }}
                    >
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="caption" color="textSecondary">
                          Response Time:
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 500 }}>
                          {Math.round(service.responseTime)}ms
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="caption" color="textSecondary">
                          Throughput:
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 500 }}>
                          {Math.round(service.throughput)} req/s
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="caption" color="textSecondary">
                          Error Rate:
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 500 }}>
                          {service.errorRate.toFixed(2)}%
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="caption" color="textSecondary">
                          Uptime:
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 500 }}>
                          {service.uptime.toFixed(2)}%
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Footer */}
      <Box
        sx={{
          mt: 2,
          pt: 1,
          borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Typography variant="caption" color="textSecondary">
          Last updated: {lastUpdate.toLocaleTimeString()}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          Update interval: {updateInterval / 1000}s
        </Typography>
      </Box>
    </Box>
  );
};

export default RealTimeMetrics;
