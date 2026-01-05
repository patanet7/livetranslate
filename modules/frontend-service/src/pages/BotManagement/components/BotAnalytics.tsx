import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Paper,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Alert,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  People as PeopleIcon,
  Speed as SpeedIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
} from 'recharts';
import { BotInstance, SystemStats } from '@/types';
import { format } from 'date-fns';

interface BotAnalyticsProps {
  systemStats: SystemStats;
  bots: Record<string, BotInstance>;
  onRefresh: () => void;
}

interface PerformanceMetric {
  timestamp: string;
  averageLatency: number;
  qualityScore: number;
  activeBots: number;
  totalTranslations: number;
  errorRate: number;
}

interface LanguageStats {
  language: string;
  count: number;
  percentage: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

export const BotAnalytics: React.FC<BotAnalyticsProps> = ({
  systemStats,
  bots,
  onRefresh,
}) => {
  const [timeRange, setTimeRange] = useState('24h');
  const [performanceData, setPerformanceData] = useState<PerformanceMetric[]>([]);
  const [languageStats, setLanguageStats] = useState<LanguageStats[]>([]);

  useEffect(() => {
    loadAnalyticsData();
  }, [timeRange]);

  const loadAnalyticsData = async () => {
    try {
      const response = await fetch(`/api/bot/analytics?timeRange=${timeRange}`);
      if (response.ok) {
        const data = await response.json();
        setPerformanceData(data.performance || []);
        setLanguageStats(data.languages || []);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const calculateTrend = (data: PerformanceMetric[], metric: keyof PerformanceMetric) => {
    if (data.length < 2) return 0;

    const recent = data.slice(-5);
    const older = data.slice(-10, -5);

    if (recent.length === 0 || older.length === 0) return 0;

    const recentAvg = recent.reduce((sum, item) => {
      const value = item[metric];
      return sum + (typeof value === 'number' ? value : 0);
    }, 0) / recent.length;
    const olderAvg = older.reduce((sum, item) => {
      const value = item[metric];
      return sum + (typeof value === 'number' ? value : 0);
    }, 0) / older.length;

    return ((recentAvg - olderAvg) / olderAvg) * 100;
  };

  const formatTrend = (trend: number) => {
    const isPositive = trend > 0;
    const color: 'success' | 'error' = isPositive ? 'success' : 'error';
    const icon = isPositive ? <TrendingUpIcon /> : <TrendingDownIcon />;

    return (
      <Chip
        icon={icon}
        label={`${Math.abs(trend).toFixed(1)}%`}
        color={color}
        size="small"
        variant="outlined"
      />
    );
  };

  const activeBots = Object.values(bots).filter(bot => bot?.status === 'active');
  const averageLatency = activeBots.length > 0
    ? activeBots.reduce((sum, bot) => sum + (bot?.performance?.averageLatencyMs || 0), 0) / activeBots.length
    : 0;
  const averageQuality = activeBots.length > 0
    ? activeBots.reduce((sum, bot) => sum + (bot?.audioCapture?.averageQualityScore || 0), 0) / activeBots.length
    : 0;

  const latencyTrend = calculateTrend(performanceData, 'averageLatency');
  const qualityTrend = calculateTrend(performanceData, 'qualityScore');
  const errorTrend = calculateTrend(performanceData, 'errorRate');

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Bot Performance Analytics
        </Typography>
        <Stack direction="row" spacing={2}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="1h">Last Hour</MenuItem>
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            onClick={() => {
              onRefresh();
              loadAnalyticsData();
            }}
            startIcon={<RefreshIcon />}
          >
            Refresh
          </Button>
        </Stack>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Active Bots
                  </Typography>
                  <Typography variant="h4">
                    {systemStats.activeBots}
                  </Typography>
                </Box>
                <PeopleIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Spawned: {systemStats.totalBotsSpawned}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Avg Latency
                  </Typography>
                  <Typography variant="h4">
                    {Math.round(averageLatency)}ms
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, color: 'warning.main' }} />
              </Box>
              <Box sx={{ mt: 2 }}>
                {formatTrend(latencyTrend)}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Quality Score
                  </Typography>
                  <Typography variant="h4">
                    {Math.round(averageQuality * 100)}%
                  </Typography>
                </Box>
                <AnalyticsIcon sx={{ fontSize: 40, color: 'success.main' }} />
              </Box>
              <Box sx={{ mt: 2 }}>
                {formatTrend(qualityTrend)}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Error Rate
                  </Typography>
                  <Typography variant="h4">
                    {(systemStats.errorRate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <TrendingDownIcon sx={{ fontSize: 40, color: 'error.main' }} />
              </Box>
              <Box sx={{ mt: 2 }}>
                {formatTrend(errorTrend)}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Trends
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                  />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip 
                    labelFormatter={(value) => format(new Date(value), 'MMM dd, HH:mm')}
                  />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="averageLatency"
                    stroke="#FF8042"
                    fill="#FF8042"
                    fillOpacity={0.6}
                    name="Latency (ms)"
                  />
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="qualityScore"
                    stroke="#00C49F"
                    fill="#00C49F"
                    fillOpacity={0.6}
                    name="Quality Score"
                  />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="activeBots"
                    stroke="#0088FE"
                    fill="#0088FE"
                    fillOpacity={0.6}
                    name="Active Bots"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Language Distribution
              </Typography>
              {languageStats.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={languageStats}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ language, percentage }) => `${language}: ${percentage.toFixed(1)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {languageStats.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Alert severity="info">No language data available</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Translation Volume
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => format(new Date(value), 'MMM dd, HH:mm')}
                  />
                  <Legend />
                  <Bar dataKey="totalTranslations" fill="#8884d8" name="Total Translations" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Overview */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Overview
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Session Statistics
                </Typography>
                <Stack spacing={1}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Completed Sessions:</Typography>
                    <Typography variant="body2">{systemStats.completedSessions}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Average Duration:</Typography>
                    <Typography variant="body2">{Math.round(systemStats.averageSessionDuration / 60)}min</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Success Rate:</Typography>
                    <Typography variant="body2">{((1 - systemStats.errorRate) * 100).toFixed(1)}%</Typography>
                  </Box>
                </Stack>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Current Active Bots
                </Typography>
                <Stack spacing={1}>
                  {activeBots.slice(0, 3).map((bot) => (
                    <Box key={bot.botId} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" noWrap>
                        {bot?.config?.meetingInfo?.meetingTitle || bot?.config?.meetingInfo?.meetingId || bot.botId}
                      </Typography>
                      <Chip
                        label={`${Math.round((bot?.audioCapture?.averageQualityScore || 0) * 100)}%`}
                        size="small"
                        color={(bot?.audioCapture?.averageQualityScore || 0) > 0.8 ? 'success' : 'warning'}
                      />
                    </Box>
                  ))}
                  {activeBots.length > 3 && (
                    <Typography variant="caption" color="text.secondary">
                      +{activeBots.length - 3} more bots
                    </Typography>
                  )}
                </Stack>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Performance Alerts
                </Typography>
                <Stack spacing={1}>
                  {averageLatency > 1000 && (
                    <Alert severity="warning" sx={{ py: 0 }}>
                      High latency detected
                    </Alert>
                  )}
                  {averageQuality < 0.6 && (
                    <Alert severity="error" sx={{ py: 0 }}>
                      Poor audio quality
                    </Alert>
                  )}
                  {systemStats.errorRate > 0.1 && (
                    <Alert severity="error" sx={{ py: 0 }}>
                      High error rate
                    </Alert>
                  )}
                  {systemStats.errorRate <= 0.05 && averageLatency <= 500 && averageQuality >= 0.8 && (
                    <Alert severity="success" sx={{ py: 0 }}>
                      All systems performing well
                    </Alert>
                  )}
                </Stack>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};