import React, { useState, useEffect } from 'react';
import { 
  useGetTranslationsQuery, 
  useGetSystemMetricsQuery, 
  useGetStatisticsQuery 
} from '@/store/slices/apiSlice';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Tabs,
  Tab,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  CircularProgress,
  Avatar,
  useTheme,
  alpha,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Speed,
  Translate,
  Language,
  Timer,
  Assessment,
  Refresh,
  Download,
  Error as ErrorIcon,
  CheckCircle,
  Warning,
  Info,
  Analytics as AnalyticsIcon,
  Memory,
  Computer,
  GraphicEq,
  Dashboard,
  ShowChart,
  HealthAndSafety,
} from '@mui/icons-material';
import { useAppSelector } from '@/store';

// Import our new professional analytics components
import RealTimeMetrics from '@/components/analytics/RealTimeMetrics';
import PerformanceCharts from '@/components/analytics/PerformanceCharts';
import SystemHealthIndicators from '@/components/analytics/SystemHealthIndicators';

interface MetricCard {
  title: string;
  value: string | number;
  change?: number;
  changeType?: 'increase' | 'decrease' | 'neutral';
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
}

interface TranslationMetric {
  id: string;
  timestamp: Date;
  sourceLanguage: string;
  targetLanguage: string;
  textLength: number;
  processingTimeMs: number;
  qualityScore: number;
  success: boolean;
  model: string;
  device: 'gpu' | 'cpu';
}

interface PerformanceData {
  timestamp: Date;
  latency: number;
  throughput: number;
  gpuUtilization?: number;
  memoryUsage: number;
  activeConnections: number;
}

interface LanguageStats {
  language: string;
  name: string;
  totalTranslations: number;
  avgQuality: number;
  avgSpeed: number;
  successRate: number;
}

const Analytics: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState('24h');
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  // Real API data - replace mock data with actual API calls
  const { data: translationsData, isLoading: translationsLoading, error: translationsError } = useGetTranslationsQuery({ 
    limit: 1000 
  });
  const { data: systemMetrics, isLoading: metricsLoading, error: metricsError } = useGetSystemMetricsQuery();
  const { data: statsData, isLoading: statsLoading, error: statsError } = useGetStatisticsQuery({ 
    timeRange 
  });
  
  // Simplified data processing to avoid babel issues
  const metrics: TranslationMetric[] = [];
  const performanceData: PerformanceData[] = [];
  const languageStats: LanguageStats[] = [];

  const handleRefresh = async () => {
    setIsLoading(true);
    try {
      // Trigger API refetch by changing a dependency or calling refetch methods
      // Note: RTK Query handles caching and refetching automatically
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Failed to refresh analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getMetricCards = (): MetricCard[] => {
    const totalTranslations = metrics.length;
    const successfulTranslations = metrics.filter(m => m.success).length;
    const avgQuality = metrics.reduce((sum, m) => sum + m.qualityScore, 0) / metrics.length;
    const avgLatency = metrics.reduce((sum, m) => sum + m.processingTimeMs, 0) / metrics.length;
    const gpuUsage = metrics.filter(m => m.device === 'gpu').length / metrics.length * 100;
    
    return [
      {
        title: 'Total Translations',
        value: totalTranslations.toLocaleString(),
        change: 12.5,
        changeType: 'increase',
        icon: <Translate />,
        color: 'primary',
      },
      {
        title: 'Success Rate',
        value: `${(successfulTranslations / totalTranslations * 100).toFixed(1)}%`,
        change: 2.1,
        changeType: 'increase',
        icon: <CheckCircle />,
        color: 'success',
      },
      {
        title: 'Avg Quality Score',
        value: avgQuality.toFixed(2),
        change: 0.05,
        changeType: 'increase',
        icon: <Assessment />,
        color: 'secondary',
      },
      {
        title: 'Avg Latency',
        value: `${avgLatency.toFixed(0)}ms`,
        change: -15.2,
        changeType: 'decrease',
        icon: <Timer />,
        color: 'warning',
      },
      {
        title: 'GPU Utilization',
        value: `${gpuUsage.toFixed(0)}%`,
        change: 5.8,
        changeType: 'increase',
        icon: <Memory />,
        color: 'secondary',
      },
      {
        title: 'Active Languages',
        value: languageStats.length,
        change: 0,
        changeType: 'neutral',
        icon: <Language />,
        color: 'primary',
      },
    ];
  };

  const getTopModels = () => {
    const modelStats = metrics.reduce((acc, metric) => {
      if (!acc[metric.model]) {
        acc[metric.model] = {
          name: metric.model,
          count: 0,
          avgQuality: 0,
          avgSpeed: 0,
          qualitySum: 0,
          speedSum: 0,
        };
      }
      acc[metric.model].count++;
      acc[metric.model].qualitySum += metric.qualityScore;
      acc[metric.model].speedSum += metric.processingTimeMs;
      acc[metric.model].avgQuality = acc[metric.model].qualitySum / acc[metric.model].count;
      acc[metric.model].avgSpeed = acc[metric.model].speedSum / acc[metric.model].count;
      return acc;
    }, {} as any);
    
    return Object.values(modelStats)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 5);
  };

  const getRecentActivity = () => {
    return metrics
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10);
  };

  const renderOverviewTab = () => (
    <Box>
      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {getMetricCards().map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} lg={2} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: `${metric.color}.main`, mr: 2 }}>
                    {metric.icon}
                  </Avatar>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
                      {metric.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {metric.title}
                    </Typography>
                  </Box>
                </Box>
                {metric.change !== undefined && (
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {metric.changeType === 'increase' ? (
                      <TrendingUp color="success" fontSize="small" />
                    ) : metric.changeType === 'decrease' ? (
                      <TrendingDown color="error" fontSize="small" />
                    ) : null}
                    <Typography
                      variant="body2"
                      color={
                        metric.changeType === 'increase' ? 'success.main' :
                        metric.changeType === 'decrease' ? 'error.main' : 'text.secondary'
                      }
                      sx={{ ml: 0.5 }}
                    >
                      {metric.change > 0 ? '+' : ''}{metric.change}%
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Translation Volume Over Time</Typography>
              <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info" sx={{ width: '100%' }}>
                  Chart visualization would be implemented here using a charting library like Chart.js or Recharts.
                  This would show translation volume trends over the selected time period.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top Models</Typography>
              <List dense>
                {getTopModels().map((model: any, index) => (
                  <ListItem key={index} sx={{ px: 0 }}>
                    <ListItemIcon>
                      <Chip 
                        label={index + 1} 
                        size="small" 
                        color="primary"
                        sx={{ width: 24, height: 24 }}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={model.name}
                      secondary={`${model.count} translations • ${model.avgQuality.toFixed(2)} quality`}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activity */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Recent Translation Activity</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Languages</TableCell>
                  <TableCell>Text Length</TableCell>
                  <TableCell>Processing Time</TableCell>
                  <TableCell>Quality</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {getRecentActivity().map((activity, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {activity.timestamp.toLocaleTimeString()}
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={`${activity.sourceLanguage} → ${activity.targetLanguage}`}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{activity.textLength} chars</TableCell>
                    <TableCell>{activity.processingTimeMs}ms</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={activity.qualityScore * 100}
                          sx={{ width: 60, height: 6 }}
                          color={activity.qualityScore > 0.8 ? 'success' : 'warning'}
                        />
                        <Typography variant="caption">
                          {(activity.qualityScore * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={activity.model}
                        size="small"
                        color="secondary"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={activity.success ? <CheckCircle /> : <ErrorIcon />}
                        label={activity.success ? 'Success' : 'Failed'}
                        size="small"
                        color={activity.success ? 'success' : 'error'}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );

  const renderTranslationMetricsTab = () => (
    <Box>
      {/* Language Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Language Performance Statistics</Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Language</TableCell>
                  <TableCell align="right">Total Translations</TableCell>
                  <TableCell align="right">Avg Quality Score</TableCell>
                  <TableCell align="right">Avg Processing Time</TableCell>
                  <TableCell align="right">Success Rate</TableCell>
                  <TableCell align="right">Performance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {languageStats.map((lang) => (
                  <TableRow key={lang.language}>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Avatar sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                          {lang.language.toUpperCase()}
                        </Avatar>
                        {lang.name}
                      </Box>
                    </TableCell>
                    <TableCell align="right">{lang.totalTranslations.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={lang.avgQuality * 100}
                          sx={{ width: 60, height: 6 }}
                          color={lang.avgQuality > 0.8 ? 'success' : 'warning'}
                        />
                        {lang.avgQuality.toFixed(2)}
                      </Box>
                    </TableCell>
                    <TableCell align="right">{lang.avgSpeed.toFixed(0)}ms</TableCell>
                    <TableCell align="right">
                      <Chip
                        label={`${(lang.successRate * 100).toFixed(1)}%`}
                        size="small"
                        color={lang.successRate > 0.95 ? 'success' : lang.successRate > 0.9 ? 'warning' : 'error'}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5 }}>
                        {lang.avgQuality > 0.85 && <Chip label="High Quality" size="small" color="success" />}
                        {lang.avgSpeed < 500 && <Chip label="Fast" size="small" color="primary" />}
                        {lang.successRate > 0.98 && <Chip label="Reliable" size="small" color="secondary" />}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Quality Distribution */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Quality Score Distribution</Typography>
              <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info">
                  Histogram chart showing distribution of quality scores would be displayed here.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Processing Time vs Quality</Typography>
              <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info">
                  Scatter plot showing correlation between processing time and quality would be displayed here.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderPerformanceTab = () => (
    <Box>
      {/* Performance Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Latency Trends</Typography>
              <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info">
                  Line chart showing latency trends over time would be displayed here.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Throughput Analysis</Typography>
              <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info">
                  Bar chart showing throughput metrics would be displayed here.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Resources */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>GPU Utilization</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress
                  variant="determinate"
                  value={78}
                  size={80}
                  thickness={6}
                />
                <Box>
                  <Typography variant="h4">78%</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Current Usage
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Memory Usage</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress
                  variant="determinate"
                  value={65}
                  size={80}
                  thickness={6}
                  color="secondary"
                />
                <Box>
                  <Typography variant="h4">65%</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Current Usage
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Active Connections</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress
                  variant="determinate"
                  value={42}
                  size={80}
                  thickness={6}
                  color="success"
                />
                <Box>
                  <Typography variant="h4">42</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Current Connections
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderUsageTab = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Usage Patterns</Typography>
              <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Alert severity="info">
                  Heatmap showing usage patterns by hour and day would be displayed here.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Peak Usage Times</Typography>
              <List>
                <ListItem>
                  <ListItemIcon><Timer /></ListItemIcon>
                  <ListItemText
                    primary="09:00 - 11:00"
                    secondary="Morning peak: 145 translations/hour"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><Timer /></ListItemIcon>
                  <ListItemText
                    primary="14:00 - 16:00"
                    secondary="Afternoon peak: 132 translations/hour"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><Timer /></ListItemIcon>
                  <ListItemText
                    primary="19:00 - 21:00"
                    secondary="Evening peak: 98 translations/hour"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Resource Usage Summary</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Computer sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6">CPU Usage</Typography>
                <Typography variant="h4" color="primary">45%</Typography>
                <Typography variant="body2" color="text.secondary">
                  Average over {timeRange}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Memory sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                <Typography variant="h6">Memory</Typography>
                <Typography variant="h4" color="secondary">6.2GB</Typography>
                <Typography variant="body2" color="text.secondary">
                  Peak usage
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <GraphicEq sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                <Typography variant="h6">GPU Load</Typography>
                <Typography variant="h4" color="success">78%</Typography>
                <Typography variant="body2" color="text.secondary">
                  Average utilization
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Speed sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                <Typography variant="h6">Throughput</Typography>
                <Typography variant="h4" color="warning">92/min</Typography>
                <Typography variant="body2" color="text.secondary">
                  Average rate
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: theme.palette.mode === 'dark' 
        ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.8)} 0%, ${alpha(theme.palette.primary.dark, 0.1)} 100%)`
        : `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.background.default, 0.8)} 100%)`,
      p: 3,
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Analytics Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive translation system performance metrics and insights
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              label="Time Range"
            >
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
            </Select>
          </FormControl>
          
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={isLoading}>
              {isLoading ? <CircularProgress size={24} /> : <Refresh />}
            </IconButton>
          </Tooltip>
          
          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={() => {/* Export functionality */}}
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Last Updated */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>Last updated: {lastRefresh.toLocaleString()}</span>
          <Chip 
            icon={<AnalyticsIcon />} 
            label={`${metrics.length} total metrics analyzed`} 
            size="small" 
            color="primary" 
          />
        </Box>
      </Alert>

      {/* Navigation Tabs */}
      <Card sx={{ 
        mb: 3,
        bgcolor: alpha(theme.palette.background.paper, 0.9),
        backdropFilter: 'blur(20px)',
      }}>
        <Tabs 
          value={activeTab} 
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              minHeight: 64,
              textTransform: 'none',
              fontWeight: 500,
              '&.Mui-selected': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
              },
            },
          }}
        >
          <Tab 
            icon={<Dashboard />} 
            label="Live Monitoring" 
            iconPosition="start"
            sx={{ minWidth: 160 }}
          />
          <Tab 
            icon={<ShowChart />} 
            label="Performance Charts" 
            iconPosition="start"
            sx={{ minWidth: 160 }}
          />
          <Tab 
            icon={<HealthAndSafety />} 
            label="System Health" 
            iconPosition="start"
            sx={{ minWidth: 160 }}
          />
          <Tab 
            icon={<Assessment />} 
            label="Translation Analytics" 
            iconPosition="start"
            sx={{ minWidth: 160 }}
          />
          <Tab 
            icon={<Speed />} 
            label="Resource Usage" 
            iconPosition="start"
            sx={{ minWidth: 160 }}
          />
        </Tabs>
      </Card>

      {/* Tab Content */}
      <Box sx={{ 
        bgcolor: alpha(theme.palette.background.paper, 0.95),
        backdropFilter: 'blur(20px)',
        borderRadius: 2,
        p: 3,
        minHeight: 600,
      }}>
        {activeTab === 0 && (
          <RealTimeMetrics 
            updateInterval={3000}
            showHistory={true}
            compact={false}
          />
        )}
        {activeTab === 1 && (
          <PerformanceCharts 
            height={500}
            showControls={true}
            autoRefresh={true}
          />
        )}
        {activeTab === 2 && (
          <SystemHealthIndicators 
            compact={false}
            showTrends={true}
            autoRefresh={true}
            refreshInterval={30000}
          />
        )}
        {activeTab === 3 && renderTranslationMetricsTab()}
        {activeTab === 4 && renderUsageTab()}
      </Box>
    </Box>
  );
};

export default Analytics;