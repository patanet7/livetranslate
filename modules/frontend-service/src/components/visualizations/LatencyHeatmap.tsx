/**
 * LatencyHeatmap - Advanced Latency Visualization
 * 
 * Professional latency analysis and visualization providing:
 * - Real-time latency heatmap by service and time
 * - Service dependency latency tracking
 * - Percentile analysis (P50, P95, P99)
 * - Geographic latency distribution
 * - Time-based pattern analysis
 * - Anomaly detection and highlighting
 * - Performance threshold monitoring
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Grid,
  Alert,
  Switch,
  FormControlLabel,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Refresh,
  Download,
  Settings,
} from '@mui/icons-material';

// Types
interface LatencyDataPoint {
  timestamp: Date;
  service: string;
  endpoint: string;
  latency: number;
  status: 'success' | 'warning' | 'error';
  region?: string;
  method?: string;
}

interface HeatmapCell {
  x: number; // time bucket
  y: number; // service index
  value: number; // latency value
  count: number; // number of requests in bucket
  color: string;
  label: string;
}

interface LatencyStats {
  service: string;
  p50: number;
  p95: number;
  p99: number;
  avg: number;
  max: number;
  count: number;
  errorRate: number;
}

interface LatencyHeatmapProps {
  timeRange?: '1h' | '6h' | '24h' | '7d';
  services?: string[];
  height?: number;
  showControls?: boolean;
  showStats?: boolean;
  alertThreshold?: number;
  onAnomalyDetected?: (anomalies: LatencyDataPoint[]) => void;
}

const LatencyHeatmap: React.FC<LatencyHeatmapProps> = ({
  timeRange = '1h',
  services = ['whisper-service', 'translation-service', 'orchestration-service', 'database'],
  height = 400,
  showControls = true,
  showStats = true,
  alertThreshold = 1000,
  onAnomalyDetected,
}) => {
  const theme = useTheme();

  // State
  const [data, setData] = useState<LatencyDataPoint[]>([]);
  const [selectedService, setSelectedService] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'heatmap' | 'percentiles' | 'geographic'>('heatmap');
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [showAnomalies, setShowAnomalies] = useState(true);
  const [colorScale, setColorScale] = useState<'linear' | 'logarithmic'>('linear');
  const [bucketSize] = useState(5); // minutes

  // Generate mock latency data
  const generateMockData = useCallback((points: number = 1000): LatencyDataPoint[] => {
    const now = new Date();
    const data: LatencyDataPoint[] = [];
    const timeSpan = timeRange === '1h' ? 3600000 : 
                   timeRange === '6h' ? 21600000 :
                   timeRange === '24h' ? 86400000 : 604800000;
    
    const endpoints = {
      'whisper-service': ['/transcribe', '/models', '/health'],
      'translation-service': ['/translate', '/languages', '/health'],
      'orchestration-service': ['/audio/upload', '/pipeline/process', '/health'],
      'database': ['/query', '/insert', '/health'],
    };

    const regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'];

    for (let i = 0; i < points; i++) {
      const service = services[Math.floor(Math.random() * services.length)];
      const serviceEndpoints = endpoints[service as keyof typeof endpoints] || ['/default'];
      const endpoint = serviceEndpoints[Math.floor(Math.random() * serviceEndpoints.length)];
      const timestamp = new Date(now.getTime() - Math.random() * timeSpan);
      
      // Generate realistic latency based on service and time
      let baseLatency = 100;
      switch (service) {
        case 'whisper-service':
          baseLatency = 800; // Slower for AI processing
          break;
        case 'translation-service':
          baseLatency = 600;
          break;
        case 'orchestration-service':
          baseLatency = 150;
          break;
        case 'database':
          baseLatency = 50;
          break;
      }

      // Add time-based variation (higher latency during peak hours)
      const hour = timestamp.getHours();
      const peakFactor = (hour >= 9 && hour <= 17) ? 1.5 : 1.0;
      
      // Add random variation and occasional spikes
      const spike = Math.random() > 0.95 ? 3 : 1;
      const latency = baseLatency * peakFactor * spike * (0.5 + Math.random());

      // Determine status based on latency
      let status: 'success' | 'warning' | 'error' = 'success';
      if (latency > alertThreshold) status = 'error';
      else if (latency > alertThreshold * 0.7) status = 'warning';

      data.push({
        timestamp,
        service,
        endpoint,
        latency,
        status,
        region: regions[Math.floor(Math.random() * regions.length)],
        method: Math.random() > 0.5 ? 'POST' : 'GET',
      });
    }

    return data.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }, [services, timeRange, alertThreshold]);

  // Generate heatmap cells
  const heatmapCells = useMemo((): HeatmapCell[] => {
    if (data.length === 0) return [];

    const filteredData = selectedService === 'all' ? data : data.filter(d => d.service === selectedService);
    const cells: HeatmapCell[] = [];
    
    // Create time buckets
    const bucketCount = timeRange === '1h' ? 12 : timeRange === '6h' ? 36 : timeRange === '24h' ? 96 : 168;
    const bucketDuration = (timeRange === '1h' ? 3600000 : 
                           timeRange === '6h' ? 21600000 :
                           timeRange === '24h' ? 86400000 : 604800000) / bucketCount;

    const now = new Date();
    const startTime = now.getTime() - (timeRange === '1h' ? 3600000 : 
                                     timeRange === '6h' ? 21600000 :
                                     timeRange === '24h' ? 86400000 : 604800000);

    const servicesToShow = selectedService === 'all' ? services : [selectedService];

    servicesToShow.forEach((service, serviceIndex) => {
      for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex++) {
        const bucketStart = startTime + bucketIndex * bucketDuration;
        const bucketEnd = bucketStart + bucketDuration;
        
        const bucketData = filteredData.filter(d => 
          d.service === service &&
          d.timestamp.getTime() >= bucketStart && 
          d.timestamp.getTime() < bucketEnd
        );

        if (bucketData.length > 0) {
          const avgLatency = bucketData.reduce((sum, d) => sum + d.latency, 0) / bucketData.length;
          
          // Color based on latency
          let color = theme.palette.success.main;
          if (avgLatency > alertThreshold) {
            color = theme.palette.error.main;
          } else if (avgLatency > alertThreshold * 0.7) {
            color = theme.palette.warning.main;
          } else if (avgLatency > alertThreshold * 0.4) {
            color = theme.palette.primary.main;
          }

          cells.push({
            x: bucketIndex,
            y: serviceIndex,
            value: colorScale === 'logarithmic' ? Math.log10(avgLatency + 1) : avgLatency,
            count: bucketData.length,
            color: alpha(color, 0.7),
            label: `${service}\n${new Date(bucketStart).toLocaleTimeString()}\nAvg: ${avgLatency.toFixed(0)}ms\nRequests: ${bucketData.length}`,
          });
        } else {
          // Empty bucket
          cells.push({
            x: bucketIndex,
            y: serviceIndex,
            value: 0,
            count: 0,
            color: alpha(theme.palette.grey[300], 0.3),
            label: `${service}\n${new Date(bucketStart).toLocaleTimeString()}\nNo data`,
          });
        }
      }
    });

    return cells;
  }, [data, selectedService, services, timeRange, alertThreshold, colorScale, theme.palette]);

  // Calculate statistics
  const latencyStats = useMemo((): LatencyStats[] => {
    if (data.length === 0) return [];

    return services.map(service => {
      const serviceData = data.filter(d => d.service === service);
      if (serviceData.length === 0) {
        return {
          service,
          p50: 0,
          p95: 0,
          p99: 0,
          avg: 0,
          max: 0,
          count: 0,
          errorRate: 0,
        };
      }

      const latencies = serviceData.map(d => d.latency).sort((a, b) => a - b);
      const errors = serviceData.filter(d => d.status === 'error').length;

      return {
        service,
        p50: latencies[Math.floor(latencies.length * 0.5)] || 0,
        p95: latencies[Math.floor(latencies.length * 0.95)] || 0,
        p99: latencies[Math.floor(latencies.length * 0.99)] || 0,
        avg: latencies.reduce((sum, l) => sum + l, 0) / latencies.length,
        max: Math.max(...latencies),
        count: serviceData.length,
        errorRate: (errors / serviceData.length) * 100,
      };
    });
  }, [data, services]);

  // Detect anomalies
  const anomalies = useMemo((): LatencyDataPoint[] => {
    if (!showAnomalies) return [];
    
    return data.filter(d => d.latency > alertThreshold);
  }, [data, showAnomalies, alertThreshold]);

  // Load data
  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
      const newData = generateMockData();
      setData(newData);
      setLastUpdate(new Date());

      // Check for anomalies
      const currentAnomalies = newData.filter(d => d.latency > alertThreshold);
      if (onAnomalyDetected && currentAnomalies.length > 0) {
        onAnomalyDetected(currentAnomalies);
      }
    } catch (error) {
      console.error('Failed to load latency data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [generateMockData, alertThreshold, onAnomalyDetected]);

  // Auto-refresh effect
  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [loadData]);

  // Export data
  const exportData = useCallback(() => {
    const exportObj = {
      timestamp: new Date().toISOString(),
      timeRange,
      selectedService,
      data,
      stats: latencyStats,
      anomalies,
    };

    const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `latency-heatmap-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [timeRange, selectedService, data, latencyStats, anomalies]);

  const servicesToShow = selectedService === 'all' ? services : [selectedService];
  const cellSize = Math.min(40, Math.max(10, (height - 100) / servicesToShow.length));

  return (
    <Box>
      {/* Controls */}
      {showControls && (
        <Card sx={{ 
          mb: 2,
          bgcolor: alpha(theme.palette.background.paper, 0.7),
          backdropFilter: 'blur(10px)',
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: 500 }}>
                Latency Heatmap Analysis
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton onClick={loadData} disabled={isLoading}>
                  <Refresh />
                </IconButton>
                <IconButton onClick={exportData}>
                  <Download />
                </IconButton>
                <IconButton>
                  <Settings />
                </IconButton>
              </Box>
            </Box>

            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>Service</InputLabel>
                  <Select
                    value={selectedService}
                    label="Service"
                    onChange={(e) => setSelectedService(e.target.value)}
                  >
                    <MenuItem value="all">All Services</MenuItem>
                    {services.map(service => (
                      <MenuItem key={service} value={service}>{service}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>View Mode</InputLabel>
                  <Select
                    value={viewMode}
                    label="View Mode"
                    onChange={(e) => setViewMode(e.target.value as any)}
                  >
                    <MenuItem value="heatmap">Heatmap</MenuItem>
                    <MenuItem value="percentiles">Percentiles</MenuItem>
                    <MenuItem value="geographic">Geographic</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>Color Scale</InputLabel>
                  <Select
                    value={colorScale}
                    label="Color Scale"
                    onChange={(e) => setColorScale(e.target.value as any)}
                  >
                    <MenuItem value="linear">Linear</MenuItem>
                    <MenuItem value="logarithmic">Logarithmic</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showAnomalies}
                      onChange={(e) => setShowAnomalies(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Show Anomalies"
                />
              </Grid>

              <Grid item xs={12} md={4}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Chip
                    label={`${data.length} requests`}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    label={`${anomalies.length} anomalies`}
                    size="small"
                    color={anomalies.length > 0 ? 'error' : 'success'}
                    variant="outlined"
                  />
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Anomaly Alerts */}
      {anomalies.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="subtitle2">
            {anomalies.length} high-latency requests detected (&gt;{alertThreshold}ms)
          </Typography>
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* Main Heatmap */}
        <Grid item xs={12} md={showStats ? 8 : 12}>
          <Card sx={{ 
            bgcolor: alpha(theme.palette.background.paper, 0.7),
            backdropFilter: 'blur(10px)',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                {viewMode === 'heatmap' ? 'Latency Heatmap' : 
                 viewMode === 'percentiles' ? 'Percentile Analysis' : 
                 'Geographic Distribution'}
              </Typography>

              {viewMode === 'heatmap' && (
                <Box sx={{ 
                  overflowX: 'auto', 
                  overflowY: 'auto',
                  maxHeight: height,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  position: 'relative',
                }}>
                  <svg 
                    width={heatmapCells.length > 0 ? Math.max(...heatmapCells.map(c => c.x)) * cellSize + cellSize + 100 : 400}
                    height={servicesToShow.length * cellSize + 50}
                  >
                    {/* Y-axis labels (services) */}
                    {servicesToShow.map((service, index) => (
                      <text
                        key={service}
                        x={90}
                        y={index * cellSize + cellSize / 2 + 5}
                        textAnchor="end"
                        fontSize={12}
                        fill={theme.palette.text.secondary}
                      >
                        {service}
                      </text>
                    ))}

                    {/* Heatmap cells */}
                    {heatmapCells.map((cell, index) => (
                      <g key={index}>
                        <rect
                          x={100 + cell.x * cellSize}
                          y={cell.y * cellSize}
                          width={cellSize - 1}
                          height={cellSize - 1}
                          fill={cell.color}
                          stroke={alpha(theme.palette.divider, 0.3)}
                          strokeWidth={0.5}
                        >
                          <title>{cell.label}</title>
                        </rect>
                        {cell.count > 0 && cell.value > alertThreshold && showAnomalies && (
                          <circle
                            cx={100 + cell.x * cellSize + cellSize / 2}
                            cy={cell.y * cellSize + cellSize / 2}
                            r={3}
                            fill={theme.palette.error.main}
                          />
                        )}
                      </g>
                    ))}
                  </svg>
                </Box>
              )}

              {viewMode === 'percentiles' && (
                <Box sx={{ height: height - 100 }}>
                  <Alert severity="info">
                    Percentile analysis chart would be displayed here showing P50, P95, P99 latencies over time.
                  </Alert>
                </Box>
              )}

              {viewMode === 'geographic' && (
                <Box sx={{ height: height - 100 }}>
                  <Alert severity="info">
                    Geographic latency distribution map would be displayed here showing latency by region.
                  </Alert>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Statistics Panel */}
        {showStats && (
          <Grid item xs={12} md={4}>
            <Grid container spacing={2}>
              {/* Latency Stats */}
              <Grid item xs={12}>
                <Card sx={{ 
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: 'blur(10px)',
                }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                      Latency Statistics
                    </Typography>
                    <Box sx={{ maxHeight: 250, overflow: 'auto' }}>
                      {latencyStats.map(stat => (
                        <Box key={stat.service} sx={{ mb: 2, p: 1, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                            {stat.service}
                          </Typography>
                          <Grid container spacing={1} sx={{ mt: 0.5 }}>
                            <Grid item xs={6}>
                              <Typography variant="caption" color="textSecondary">P50:</Typography>
                              <Typography variant="body2">{stat.p50.toFixed(0)}ms</Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" color="textSecondary">P95:</Typography>
                              <Typography variant="body2">{stat.p95.toFixed(0)}ms</Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" color="textSecondary">P99:</Typography>
                              <Typography variant="body2">{stat.p99.toFixed(0)}ms</Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" color="textSecondary">Errors:</Typography>
                              <Typography variant="body2" color={stat.errorRate > 5 ? 'error.main' : 'textPrimary'}>
                                {stat.errorRate.toFixed(1)}%
                              </Typography>
                            </Grid>
                          </Grid>
                        </Box>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Legend */}
              <Grid item xs={12}>
                <Card sx={{ 
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: 'blur(10px)',
                }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                      Color Legend
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 16, height: 16, backgroundColor: theme.palette.success.main, borderRadius: 0.5 }} />
                        <Typography variant="body2">Good (&lt;{Math.round(alertThreshold * 0.4)}ms)</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 16, height: 16, backgroundColor: theme.palette.primary.main, borderRadius: 0.5 }} />
                        <Typography variant="body2">Fair ({Math.round(alertThreshold * 0.4)}-{Math.round(alertThreshold * 0.7)}ms)</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 16, height: 16, backgroundColor: theme.palette.warning.main, borderRadius: 0.5 }} />
                        <Typography variant="body2">Slow ({Math.round(alertThreshold * 0.7)}-{alertThreshold}ms)</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 16, height: 16, backgroundColor: theme.palette.error.main, borderRadius: 0.5 }} />
                        <Typography variant="body2">Critical (&gt;{alertThreshold}ms)</Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        )}
      </Grid>

      {/* Footer */}
      <Box sx={{ 
        mt: 2, 
        pt: 1, 
        borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="caption" color="textSecondary">
          Last updated: {lastUpdate.toLocaleTimeString()}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          Time range: {timeRange} | Bucket size: {bucketSize}min
        </Typography>
      </Box>
    </Box>
  );
};

export default LatencyHeatmap;
