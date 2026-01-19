/**
 * PerformanceCharts - Professional Performance Visualization
 *
 * Provides comprehensive performance charts and visualizations including:
 * - Multi-metric time series charts
 * - Performance trend analysis
 * - Response time distributions
 * - Throughput and latency correlation
 * - Error rate tracking
 * - Comparative performance analysis
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ButtonGroup,
  Button,
  Chip,
  Grid,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
} from "@mui/material";
import { Refresh, Download, Settings } from "@mui/icons-material";

// Import chart components
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from "recharts";

// Types
interface TimeSeriesData {
  timestamp: Date;
  [key: string]: any;
}

interface PerformanceMetric {
  name: string;
  color: string;
  unit: string;
  dataKey: string;
}

interface ChartConfig {
  type: "line" | "area" | "bar" | "pie" | "scatter" | "composed";
  title: string;
  metrics: PerformanceMetric[];
  timeRange: "1h" | "6h" | "24h" | "7d" | "30d";
  refreshInterval: number;
}

interface PerformanceChartsProps {
  height?: number;
  showControls?: boolean;
  autoRefresh?: boolean;
}

const PerformanceCharts: React.FC<PerformanceChartsProps> = ({
  height = 400,
  showControls = true,
  autoRefresh = true,
}) => {
  const theme = useTheme();

  // State
  const [data, setData] = useState<TimeSeriesData[]>([]);
  const [selectedChart, setSelectedChart] = useState<
    "overview" | "latency" | "throughput" | "errors" | "resources"
  >("overview");
  const [timeRange, setTimeRange] = useState<
    "1h" | "6h" | "24h" | "7d" | "30d"
  >("1h");
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Chart configurations
  const chartConfigs: Record<string, ChartConfig> = {
    overview: {
      type: "composed",
      title: "System Overview",
      metrics: [
        {
          name: "CPU Usage",
          color: theme.palette.primary.main,
          unit: "%",
          dataKey: "cpu",
        },
        {
          name: "Memory Usage",
          color: theme.palette.secondary.main,
          unit: "%",
          dataKey: "memory",
        },
        {
          name: "Request Rate",
          color: theme.palette.success.main,
          unit: "req/s",
          dataKey: "requests",
        },
      ],
      timeRange,
      refreshInterval: 5000,
    },
    latency: {
      type: "line",
      title: "Response Time Analysis",
      metrics: [
        {
          name: "Avg Response Time",
          color: theme.palette.primary.main,
          unit: "ms",
          dataKey: "avgResponseTime",
        },
        {
          name: "P95 Response Time",
          color: theme.palette.warning.main,
          unit: "ms",
          dataKey: "p95ResponseTime",
        },
        {
          name: "P99 Response Time",
          color: theme.palette.error.main,
          unit: "ms",
          dataKey: "p99ResponseTime",
        },
      ],
      timeRange,
      refreshInterval: 3000,
    },
    throughput: {
      type: "area",
      title: "Throughput Metrics",
      metrics: [
        {
          name: "Audio Requests",
          color: theme.palette.primary.main,
          unit: "req/s",
          dataKey: "audioRequests",
        },
        {
          name: "Translation Requests",
          color: theme.palette.secondary.main,
          unit: "req/s",
          dataKey: "translationRequests",
        },
        {
          name: "Pipeline Requests",
          color: theme.palette.success.main,
          unit: "req/s",
          dataKey: "pipelineRequests",
        },
      ],
      timeRange,
      refreshInterval: 2000,
    },
    errors: {
      type: "bar",
      title: "Error Rate Analysis",
      metrics: [
        {
          name: "HTTP 4xx",
          color: theme.palette.warning.main,
          unit: "errors/min",
          dataKey: "http4xx",
        },
        {
          name: "HTTP 5xx",
          color: theme.palette.error.main,
          unit: "errors/min",
          dataKey: "http5xx",
        },
        {
          name: "Timeouts",
          color: theme.palette.info.main,
          unit: "errors/min",
          dataKey: "timeouts",
        },
      ],
      timeRange,
      refreshInterval: 10000,
    },
    resources: {
      type: "area",
      title: "Resource Utilization",
      metrics: [
        {
          name: "CPU Cores",
          color: theme.palette.primary.main,
          unit: "cores",
          dataKey: "cpuCores",
        },
        {
          name: "Memory GB",
          color: theme.palette.secondary.main,
          unit: "GB",
          dataKey: "memoryGB",
        },
        {
          name: "Disk I/O",
          color: theme.palette.success.main,
          unit: "MB/s",
          dataKey: "diskIO",
        },
      ],
      timeRange,
      refreshInterval: 5000,
    },
  };

  // Generate mock performance data
  const generateMockData = useCallback(
    (points: number = 50) => {
      const now = new Date();
      const interval =
        timeRange === "1h"
          ? 60000
          : timeRange === "6h"
            ? 360000
            : timeRange === "24h"
              ? 1440000
              : timeRange === "7d"
                ? 10080000
                : 43200000;

      const data: TimeSeriesData[] = [];

      for (let i = points - 1; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * interval);

        // Generate correlated performance data
        const baseLoad = 0.3 + 0.4 * Math.sin(i / 10) + 0.1 * Math.random();
        const spike = Math.random() > 0.95 ? 2 : 1; // Occasional spikes

        data.push({
          timestamp,
          // System metrics
          cpu: Math.max(
            10,
            Math.min(95, baseLoad * 60 * spike + Math.random() * 10),
          ),
          memory: Math.max(
            20,
            Math.min(90, baseLoad * 70 + Math.random() * 15),
          ),
          requests: Math.max(0, baseLoad * 100 * spike + Math.random() * 20),

          // Latency metrics (inversely correlated with load)
          avgResponseTime: Math.max(
            50,
            (1 + baseLoad) * 150 * spike + Math.random() * 50,
          ),
          p95ResponseTime: Math.max(
            100,
            (1 + baseLoad) * 300 * spike + Math.random() * 100,
          ),
          p99ResponseTime: Math.max(
            200,
            (1 + baseLoad) * 600 * spike + Math.random() * 200,
          ),

          // Throughput metrics
          audioRequests: Math.max(0, baseLoad * 50 + Math.random() * 15),
          translationRequests: Math.max(0, baseLoad * 30 + Math.random() * 10),
          pipelineRequests: Math.max(0, baseLoad * 20 + Math.random() * 8),

          // Error metrics (correlated with load)
          http4xx: Math.max(0, baseLoad * spike * 5 + Math.random() * 2),
          http5xx: Math.max(0, baseLoad * spike * 2 + Math.random() * 1),
          timeouts: Math.max(0, baseLoad * spike * 1 + Math.random() * 0.5),

          // Resource metrics
          cpuCores: Math.max(1, baseLoad * 8 + Math.random() * 2),
          memoryGB: Math.max(2, baseLoad * 16 + Math.random() * 4),
          diskIO: Math.max(0, baseLoad * 100 + Math.random() * 20),
        });
      }

      return data;
    },
    [timeRange],
  );

  // Load performance data
  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 500));

      const points =
        timeRange === "1h"
          ? 60
          : timeRange === "6h"
            ? 72
            : timeRange === "24h"
              ? 144
              : timeRange === "7d"
                ? 168
                : 720;

      const newData = generateMockData(points);
      setData(newData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to load performance data:", error);
    } finally {
      setIsLoading(false);
    }
  }, [generateMockData, timeRange]);

  // Auto-refresh effect
  useEffect(() => {
    loadData();

    if (autoRefresh) {
      const config = chartConfigs[selectedChart];
      const interval = setInterval(loadData, config.refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadData, autoRefresh, selectedChart, chartConfigs]);

  // Chart rendering functions
  const renderChart = () => {
    const config = chartConfigs[selectedChart];

    if (data.length === 0) {
      return (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: height - 100,
          }}
        >
          <Typography color="textSecondary">No data available</Typography>
        </Box>
      );
    }

    const formatTimestamp = (timestamp: any) => {
      const date = new Date(timestamp);
      if (timeRange === "1h" || timeRange === "6h") {
        return date.toLocaleTimeString();
      } else if (timeRange === "24h") {
        return `${date.getHours()}:00`;
      } else {
        return date.toLocaleDateString();
      }
    };

    const commonProps = {
      data,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (config.type) {
      case "line":
        return (
          <LineChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={alpha(theme.palette.divider, 0.3)}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTimestamp}
              stroke={theme.palette.text.secondary}
            />
            <YAxis stroke={theme.palette.text.secondary} />
            <RechartsTooltip
              labelFormatter={(value) =>
                `Time: ${new Date(value).toLocaleString()}`
              }
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 8,
              }}
            />
            <Legend />
            {config.metrics.map((metric) => (
              <Line
                key={metric.dataKey}
                type="monotone"
                dataKey={metric.dataKey}
                stroke={metric.color}
                strokeWidth={2}
                dot={false}
                name={metric.name}
              />
            ))}
          </LineChart>
        );

      case "area":
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={alpha(theme.palette.divider, 0.3)}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTimestamp}
              stroke={theme.palette.text.secondary}
            />
            <YAxis stroke={theme.palette.text.secondary} />
            <RechartsTooltip
              labelFormatter={(value) =>
                `Time: ${new Date(value).toLocaleString()}`
              }
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 8,
              }}
            />
            <Legend />
            {config.metrics.map((metric) => (
              <Area
                key={metric.dataKey}
                type="monotone"
                dataKey={metric.dataKey}
                stackId="1"
                stroke={metric.color}
                fill={alpha(metric.color, 0.4)}
                name={metric.name}
              />
            ))}
          </AreaChart>
        );

      case "bar":
        return (
          <BarChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={alpha(theme.palette.divider, 0.3)}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTimestamp}
              stroke={theme.palette.text.secondary}
            />
            <YAxis stroke={theme.palette.text.secondary} />
            <RechartsTooltip
              labelFormatter={(value) =>
                `Time: ${new Date(value).toLocaleString()}`
              }
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 8,
              }}
            />
            <Legend />
            {config.metrics.map((metric) => (
              <Bar
                key={metric.dataKey}
                dataKey={metric.dataKey}
                fill={metric.color}
                name={metric.name}
              />
            ))}
          </BarChart>
        );

      case "composed":
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={alpha(theme.palette.divider, 0.3)}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTimestamp}
              stroke={theme.palette.text.secondary}
            />
            <YAxis stroke={theme.palette.text.secondary} />
            <RechartsTooltip
              labelFormatter={(value) =>
                `Time: ${new Date(value).toLocaleString()}`
              }
              contentStyle={{
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 8,
              }}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="cpu"
              fill={alpha(theme.palette.primary.main, 0.3)}
              stroke={theme.palette.primary.main}
              name="CPU Usage"
            />
            <Line
              type="monotone"
              dataKey="memory"
              stroke={theme.palette.secondary.main}
              strokeWidth={2}
              name="Memory Usage"
            />
            <Bar
              dataKey="requests"
              fill={alpha(theme.palette.success.main, 0.7)}
              name="Request Rate"
            />
          </ComposedChart>
        );

      default:
        return null;
    }
  };

  const currentConfig = chartConfigs[selectedChart];

  return (
    <Box>
      {/* Controls */}
      {showControls && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 2,
            flexWrap: "wrap",
            gap: 2,
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Performance Analytics
          </Typography>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 2,
              flexWrap: "wrap",
            }}
          >
            {/* Chart Type Selector */}
            <ButtonGroup size="small" variant="outlined">
              {Object.entries(chartConfigs).map(([key, config]) => (
                <Button
                  key={key}
                  onClick={() => setSelectedChart(key as any)}
                  variant={selectedChart === key ? "contained" : "outlined"}
                  size="small"
                >
                  {config.title.split(" ")[0]}
                </Button>
              ))}
            </ButtonGroup>

            {/* Time Range Selector */}
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>Range</InputLabel>
              <Select
                value={timeRange}
                label="Range"
                onChange={(e) => setTimeRange(e.target.value as any)}
              >
                <MenuItem value="1h">1 Hour</MenuItem>
                <MenuItem value="6h">6 Hours</MenuItem>
                <MenuItem value="24h">24 Hours</MenuItem>
                <MenuItem value="7d">7 Days</MenuItem>
                <MenuItem value="30d">30 Days</MenuItem>
              </Select>
            </FormControl>

            {/* Action Buttons */}
            <Box sx={{ display: "flex", gap: 1 }}>
              <Tooltip title="Refresh Data">
                <IconButton
                  onClick={loadData}
                  size="small"
                  disabled={isLoading}
                >
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Export Data">
                <IconButton size="small">
                  <Download />
                </IconButton>
              </Tooltip>
              <Tooltip title="Chart Settings">
                <IconButton size="small">
                  <Settings />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>
      )}

      {/* Chart Container */}
      <Card
        sx={{
          bgcolor: alpha(theme.palette.background.paper, 0.7),
          backdropFilter: "blur(10px)",
          mb: 2,
        }}
      >
        <CardContent>
          {/* Chart Header */}
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              mb: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 500 }}>
              {currentConfig.title}
            </Typography>
            <Box sx={{ display: "flex", gap: 1 }}>
              {currentConfig.metrics.map((metric) => (
                <Chip
                  key={metric.dataKey}
                  label={metric.name}
                  size="small"
                  sx={{
                    backgroundColor: alpha(metric.color, 0.1),
                    color: metric.color,
                    border: `1px solid ${alpha(metric.color, 0.3)}`,
                  }}
                />
              ))}
            </Box>
          </Box>

          {/* Chart */}
          <Box sx={{ height, width: "100%" }}>
            {(() => {
              const chart = renderChart();
              return chart ? (
                <ResponsiveContainer width="100%" height="100%">
                  {chart}
                </ResponsiveContainer>
              ) : (
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    height,
                  }}
                >
                  <Typography color="textSecondary">
                    Chart type not supported
                  </Typography>
                </Box>
              );
            })()}
          </Box>
        </CardContent>
      </Card>

      {/* Chart Statistics */}
      <Grid container spacing={2}>
        {currentConfig.metrics.map((metric) => {
          const values = data
            .map((d) => d[metric.dataKey])
            .filter((v) => typeof v === "number");
          if (values.length === 0) return null;

          const avg = values.reduce((a, b) => a + b, 0) / values.length;
          const max = Math.max(...values);
          const min = Math.min(...values);

          return (
            <Grid item xs={12} sm={6} md={4} key={metric.dataKey}>
              <Card
                sx={{
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: "blur(10px)",
                }}
              >
                <CardContent>
                  <Typography
                    variant="subtitle2"
                    color="textSecondary"
                    gutterBottom
                  >
                    {metric.name}
                  </Typography>
                  <Box
                    sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}
                  >
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="caption">Average:</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>
                        {avg.toFixed(1)} {metric.unit}
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="caption">Maximum:</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>
                        {max.toFixed(1)} {metric.unit}
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="caption">Minimum:</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>
                        {min.toFixed(1)} {metric.unit}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

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
          Data points: {data.length} | Range: {timeRange}
        </Typography>
      </Box>
    </Box>
  );
};

export default PerformanceCharts;
