import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  IconButton,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Slider,
} from '@mui/material';
import {
  Assessment,
  Download,
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Warning,
  CheckCircle,
  Error,
  ExpandMore,
  Star,
  Speed,
  VolumeUp,
  GraphicEq,
  Timeline,
  CompareArrows,
  Analytics,
} from '@mui/icons-material';

export interface QualityMetricsProps {
  audioData?: {
    quality_analysis: {
      overall_score: number;
      individual_scores: {
        clarity: number;
        noise_level: number;
        dynamic_range: number;
        frequency_response: number;
        artifacts: number;
        loudness_consistency: number;
      };
      recommendations: QualityRecommendation[];
      technical_metrics: {
        snr_db: number;
        thd_percentage: number;
        peak_to_rms_ratio: number;
        crest_factor: number;
        spectral_centroid: number;
        zero_crossing_rate: number;
      };
      lufs_compliance: {
        integrated_lufs: number;
        target_lufs: number;
        compliance_status: 'compliant' | 'warning' | 'non_compliant';
        adjustment_needed_db: number;
      };
    };
    comparison_data?: {
      original_quality: number;
      processed_quality: number;
      improvement_percentage: number;
      stage_contributions: StageContribution[];
    };
    timestamp: number;
  };
  isRealTime?: boolean;
  showComparison?: boolean;
  onQualityUpdate?: (metrics: QualityMetrics) => void;
  height?: number;
  targetQuality?: number;
}

interface QualityRecommendation {
  category: 'processing' | 'settings' | 'hardware' | 'content';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  action: string;
  estimated_improvement: number;
}

interface StageContribution {
  stage_name: string;
  quality_impact: number;
  processing_time_ms: number;
  effectiveness_score: number;
}

interface QualityMetrics {
  overall_score: number;
  category_scores: Record<string, number>;
  technical_metrics: Record<string, number>;
  improvement_suggestions: QualityRecommendation[];
  compliance_status: string;
  trend: 'improving' | 'stable' | 'declining';
}

type MetricView = 'scores' | 'technical' | 'recommendations' | 'trends';

const QUALITY_CATEGORIES: Record<string, {
  name: string;
  description: string;
  icon: React.ComponentType;
  color: string;
  threshold: { good: number; fair: number };
  inverted?: boolean;
}> = {
  clarity: {
    name: 'Clarity',
    description: 'Voice intelligibility and definition',
    icon: VolumeUp,
    color: '#4dabf7',
    threshold: { good: 8.0, fair: 6.0 }
  },
  noise_level: {
    name: 'Noise Level',
    description: 'Background noise and interference',
    icon: GraphicEq,
    color: '#51cf66',
    threshold: { good: 8.5, fair: 7.0 },
    inverted: true // Lower noise is better
  },
  dynamic_range: {
    name: 'Dynamic Range',
    description: 'Difference between loudest and quietest parts',
    icon: Timeline,
    color: '#ff8787',
    threshold: { good: 7.5, fair: 6.0 }
  },
  frequency_response: {
    name: 'Frequency Response',
    description: 'Balance across frequency spectrum',
    icon: Speed,
    color: '#ffc947',
    threshold: { good: 8.0, fair: 6.5 }
  },
  artifacts: {
    name: 'Artifacts',
    description: 'Processing artifacts and distortion',
    icon: Warning,
    color: '#ff6b6b',
    threshold: { good: 9.0, fair: 7.5 },
    inverted: true // Fewer artifacts is better
  },
  loudness_consistency: {
    name: 'Loudness Consistency',
    description: 'Level stability and LUFS compliance',
    icon: CompareArrows,
    color: '#845ef7',
    threshold: { good: 8.5, fair: 7.0 }
  }
};

const SEVERITY_COLORS = {
  low: '#51cf66',
  medium: '#ffc947',
  high: '#ff8787',
  critical: '#ff6b6b'
};

export const QualityMetrics: React.FC<QualityMetricsProps> = ({
  audioData,
  isRealTime = false,
  showComparison = false,
  onQualityUpdate,
  targetQuality = 8.0,
}) => {
  const [metricView, setMetricView] = useState<MetricView>('scores');
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [qualityHistory, setQualityHistory] = useState<QualityMetrics[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(isRealTime);
  const [refreshInterval, setRefreshInterval] = useState(5); // seconds

  const processQualityData = useCallback(() => {
    if (!audioData) return null;

    const { quality_analysis } = audioData;
    const metrics: QualityMetrics = {
      overall_score: quality_analysis.overall_score,
      category_scores: quality_analysis.individual_scores,
      technical_metrics: quality_analysis.technical_metrics,
      improvement_suggestions: quality_analysis.recommendations,
      compliance_status: quality_analysis.lufs_compliance.compliance_status,
      trend: calculateTrend(quality_analysis.overall_score)
    };

    // Add to history
    setQualityHistory(prev => {
      const newHistory = [...prev, metrics];
      return newHistory.slice(-50); // Keep last 50 measurements
    });

    onQualityUpdate?.(metrics);
    return metrics;
  }, [audioData, onQualityUpdate]);

  const calculateTrend = (currentScore: number): 'improving' | 'stable' | 'declining' => {
    if (qualityHistory.length < 3) return 'stable';
    
    const recentScores = qualityHistory.slice(-3).map(h => h.overall_score);
    const average = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
    
    if (currentScore > average + 0.2) return 'improving';
    if (currentScore < average - 0.2) return 'declining';
    return 'stable';
  };

  const getScoreColor = (score: number, inverted = false): string => {
    const adjustedScore = inverted ? 10 - score : score;
    if (adjustedScore >= 8.5) return '#4caf50';
    if (adjustedScore >= 7.0) return '#ff9800';
    if (adjustedScore >= 5.0) return '#f44336';
    return '#b71c1c';
  };

  const getScoreStatus = (score: number, category: any): 'excellent' | 'good' | 'fair' | 'poor' => {
    const adjustedScore = category.inverted ? 10 - score : score;
    if (adjustedScore >= category.threshold.good) return 'excellent';
    if (adjustedScore >= category.threshold.fair) return 'good';
    if (adjustedScore >= 5.0) return 'fair';
    return 'poor';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp color="success" />;
      case 'declining': return <TrendingDown color="error" />;
      default: return <TrendingFlat color="info" />;
    }
  };

  const handleDownloadReport = () => {
    if (!audioData) return;

    const currentMetrics = processQualityData();
    if (!currentMetrics) return;

    const report = {
      timestamp: new Date().toISOString(),
      report_type: 'comprehensive_quality_analysis',
      overall_assessment: {
        score: currentMetrics.overall_score,
        grade: getQualityGrade(currentMetrics.overall_score),
        trend: currentMetrics.trend
      },
      category_breakdown: Object.entries(currentMetrics.category_scores).map(([key, score]) => ({
        category: QUALITY_CATEGORIES[key as keyof typeof QUALITY_CATEGORIES]?.name || key,
        score,
        status: getScoreStatus(score, QUALITY_CATEGORIES[key as keyof typeof QUALITY_CATEGORIES] || {}),
        threshold_met: score >= (QUALITY_CATEGORIES[key as keyof typeof QUALITY_CATEGORIES]?.threshold.good || 8.0)
      })),
      technical_analysis: currentMetrics.technical_metrics,
      recommendations: currentMetrics.improvement_suggestions.map(rec => ({
        ...rec,
        priority_score: rec.severity === 'critical' ? 4 : rec.severity === 'high' ? 3 : rec.severity === 'medium' ? 2 : 1
      })).sort((a, b) => b.priority_score - a.priority_score),
      compliance: {
        lufs_status: audioData.quality_analysis.lufs_compliance.compliance_status,
        lufs_current: audioData.quality_analysis.lufs_compliance.integrated_lufs,
        lufs_target: audioData.quality_analysis.lufs_compliance.target_lufs,
        adjustment_needed: audioData.quality_analysis.lufs_compliance.adjustment_needed_db
      },
      processing_comparison: audioData.comparison_data,
      quality_history: qualityHistory.slice(-10) // Last 10 measurements
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quality_analysis_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getQualityGrade = (score: number): string => {
    if (score >= 9.0) return 'A+';
    if (score >= 8.5) return 'A';
    if (score >= 8.0) return 'A-';
    if (score >= 7.5) return 'B+';
    if (score >= 7.0) return 'B';
    if (score >= 6.5) return 'B-';
    if (score >= 6.0) return 'C+';
    if (score >= 5.5) return 'C';
    if (score >= 5.0) return 'C-';
    return 'D';
  };

  const currentMetrics = processQualityData();

  useEffect(() => {
    if (autoRefresh && isRealTime) {
      const interval = setInterval(() => {
        processQualityData();
      }, refreshInterval * 1000);
      
      return () => clearInterval(interval);
    }
  }, [autoRefresh, isRealTime, refreshInterval, processQualityData]);

  if (!audioData) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
            <Typography variant="h6" component="h3">
              Quality Metrics
            </Typography>
            <Chip label="No Data" size="small" color="default" />
          </Box>
          
          <Alert severity="info">
            No audio data available for quality analysis. Start audio processing to see comprehensive quality metrics.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h3">
            <Assessment sx={{ verticalAlign: 'middle', mr: 1, fontSize: 24 }} />
            Quality Metrics Dashboard
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={`Score: ${(currentMetrics?.overall_score ?? 0).toFixed(1)}/10.0`}
              size="small"
              color={(currentMetrics?.overall_score ?? 0) >= targetQuality ? 'success' : 'warning'}
              variant="filled"
            />
            <Chip
              label={getQualityGrade(currentMetrics?.overall_score || 0)}
              size="small"
              color="primary"
              variant="outlined"
            />
            {currentMetrics && getTrendIcon(currentMetrics.trend)}
            <IconButton size="small" onClick={handleDownloadReport}>
              <Download />
            </IconButton>
          </Box>
        </Box>

        {/* Controls */}
        <Grid container spacing={2} mb={2}>
          <Grid item xs={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>View</InputLabel>
              <Select
                value={metricView}
                label="View"
                onChange={(e) => setMetricView(e.target.value as MetricView)}
              >
                <MenuItem value="scores">Quality Scores</MenuItem>
                <MenuItem value="technical">Technical Metrics</MenuItem>
                <MenuItem value="recommendations">Recommendations</MenuItem>
                <MenuItem value="trends">Historical Trends</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={showTechnicalDetails}
                  onChange={(e) => setShowTechnicalDetails(e.target.checked)}
                  size="small"
                />
              }
              label="Technical Details"
            />
          </Grid>

          {isRealTime && (
            <Grid item xs={12} md={6}>
              <Box display="flex" alignItems="center" gap={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Auto Refresh"
                />
                <Typography variant="body2" sx={{ minWidth: 100 }}>
                  Interval: {refreshInterval}s
                </Typography>
                <Slider
                  value={refreshInterval}
                  min={1}
                  max={30}
                  step={1}
                  onChange={(_, value) => setRefreshInterval(value as number)}
                  size="small"
                  sx={{ width: 100 }}
                />
              </Box>
            </Grid>
          )}
        </Grid>

        {/* Content based on selected view */}
        <Box flexGrow={1} overflow="auto">
          {metricView === 'scores' && (
            <Grid container spacing={2}>
              {/* Overall Score */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <Star sx={{ verticalAlign: 'middle', mr: 1 }} />
                      Overall Quality Score
                    </Typography>
                    
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                      <Typography variant="h3" component="div" color={getScoreColor(currentMetrics?.overall_score || 0)}>
                        {(currentMetrics?.overall_score || 0).toFixed(1)}
                      </Typography>
                      <Box flexGrow={1}>
                        <LinearProgress 
                          variant="determinate" 
                          value={(currentMetrics?.overall_score || 0) * 10} 
                          sx={{ 
                            height: 8, 
                            borderRadius: 4,
                            backgroundColor: 'rgba(0,0,0,0.1)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: getScoreColor(currentMetrics?.overall_score || 0)
                            }
                          }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Target: {targetQuality}/10.0
                        </Typography>
                      </Box>
                    </Box>
                    
                    {showComparison && audioData.comparison_data && (
                      <Alert severity="info" sx={{ mt: 1 }}>
                        Improvement: +{audioData.comparison_data.improvement_percentage.toFixed(1)}% 
                        from original
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Category Breakdown */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Category Breakdown
                    </Typography>
                    
                    <Box display="flex" flexDirection="column" gap={1}>
                      {Object.entries(QUALITY_CATEGORIES).map(([key, category]) => {
                        const score = currentMetrics?.category_scores[key] || 0;
                        const status = getScoreStatus(score, category);
                        
                        return (
                          <Box key={key} display="flex" alignItems="center" gap={1}>
                            {React.createElement(category.icon as any, {
                              style: { fontSize: 16, color: category.color }
                            })}
                            <Typography variant="body2" sx={{ minWidth: 120 }}>
                              {category.name}
                            </Typography>
                            <Box flexGrow={1}>
                              <LinearProgress
                                variant="determinate"
                                value={score * 10}
                                sx={{
                                  height: 4,
                                  backgroundColor: 'rgba(0,0,0,0.1)',
                                  '& .MuiLinearProgress-bar': {
                                    backgroundColor: getScoreColor(score, category.inverted || false)
                                  }
                                }}
                              />
                            </Box>
                            <Typography variant="caption" sx={{ minWidth: 30 }}>
                              {score.toFixed(1)}
                            </Typography>
                            <Chip 
                              label={status} 
                              size="small" 
                              variant="outlined"
                              color={
                                status === 'excellent' ? 'success' :
                                status === 'good' ? 'info' :
                                status === 'fair' ? 'warning' : 'error'
                              }
                            />
                          </Box>
                        );
                      })}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* LUFS Compliance */}
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <VolumeUp sx={{ verticalAlign: 'middle', mr: 1 }} />
                      LUFS Compliance Status
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="text.secondary">Current LUFS:</Typography>
                        <Typography variant="h6">
                          {audioData.quality_analysis.lufs_compliance.integrated_lufs.toFixed(1)} LUFS
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="text.secondary">Target LUFS:</Typography>
                        <Typography variant="h6">
                          {audioData.quality_analysis.lufs_compliance.target_lufs.toFixed(1)} LUFS
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="text.secondary">Adjustment Needed:</Typography>
                        <Typography variant="h6" color={Math.abs(audioData.quality_analysis.lufs_compliance.adjustment_needed_db) <= 1 ? 'success.main' : 'warning.main'}>
                          {audioData.quality_analysis.lufs_compliance.adjustment_needed_db > 0 ? '+' : ''}
                          {audioData.quality_analysis.lufs_compliance.adjustment_needed_db.toFixed(1)} dB
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="text.secondary">Status:</Typography>
                        <Box display="flex" alignItems="center" gap={1}>
                          {audioData.quality_analysis.lufs_compliance.compliance_status === 'compliant' ? 
                            <CheckCircle color="success" /> : 
                            audioData.quality_analysis.lufs_compliance.compliance_status === 'warning' ?
                            <Warning color="warning" /> :
                            <Error color="error" />
                          }
                          <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                            {audioData.quality_analysis.lufs_compliance.compliance_status}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {metricView === 'technical' && showTechnicalDetails && (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Technical Metric</strong></TableCell>
                    <TableCell align="right"><strong>Value</strong></TableCell>
                    <TableCell align="right"><strong>Unit</strong></TableCell>
                    <TableCell><strong>Assessment</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Signal-to-Noise Ratio</TableCell>
                    <TableCell align="right">{audioData.quality_analysis.technical_metrics.snr_db.toFixed(1)}</TableCell>
                    <TableCell align="right">dB</TableCell>
                    <TableCell>
                      <Chip 
                        label={audioData.quality_analysis.technical_metrics.snr_db >= 20 ? 'Excellent' : 
                               audioData.quality_analysis.technical_metrics.snr_db >= 15 ? 'Good' : 
                               audioData.quality_analysis.technical_metrics.snr_db >= 10 ? 'Fair' : 'Poor'}
                        size="small"
                        color={audioData.quality_analysis.technical_metrics.snr_db >= 20 ? 'success' : 
                               audioData.quality_analysis.technical_metrics.snr_db >= 15 ? 'info' : 
                               audioData.quality_analysis.technical_metrics.snr_db >= 10 ? 'warning' : 'error'}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Total Harmonic Distortion</TableCell>
                    <TableCell align="right">{audioData.quality_analysis.technical_metrics.thd_percentage.toFixed(2)}</TableCell>
                    <TableCell align="right">%</TableCell>
                    <TableCell>
                      <Chip 
                        label={audioData.quality_analysis.technical_metrics.thd_percentage <= 0.1 ? 'Excellent' : 
                               audioData.quality_analysis.technical_metrics.thd_percentage <= 0.5 ? 'Good' : 
                               audioData.quality_analysis.technical_metrics.thd_percentage <= 1.0 ? 'Fair' : 'Poor'}
                        size="small"
                        color={audioData.quality_analysis.technical_metrics.thd_percentage <= 0.1 ? 'success' : 
                               audioData.quality_analysis.technical_metrics.thd_percentage <= 0.5 ? 'info' : 
                               audioData.quality_analysis.technical_metrics.thd_percentage <= 1.0 ? 'warning' : 'error'}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Peak-to-RMS Ratio</TableCell>
                    <TableCell align="right">{audioData.quality_analysis.technical_metrics.peak_to_rms_ratio.toFixed(1)}</TableCell>
                    <TableCell align="right">dB</TableCell>
                    <TableCell>
                      <Chip 
                        label={audioData.quality_analysis.technical_metrics.peak_to_rms_ratio >= 12 ? 'Good Dynamic Range' : 'Compressed'}
                        size="small"
                        color={audioData.quality_analysis.technical_metrics.peak_to_rms_ratio >= 12 ? 'success' : 'warning'}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Crest Factor</TableCell>
                    <TableCell align="right">{audioData.quality_analysis.technical_metrics.crest_factor.toFixed(1)}</TableCell>
                    <TableCell align="right">dB</TableCell>
                    <TableCell>
                      <Chip 
                        label="Normal"
                        size="small"
                        color="info"
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Spectral Centroid</TableCell>
                    <TableCell align="right">{audioData.quality_analysis.technical_metrics.spectral_centroid.toFixed(0)}</TableCell>
                    <TableCell align="right">Hz</TableCell>
                    <TableCell>
                      <Chip 
                        label={audioData.quality_analysis.technical_metrics.spectral_centroid >= 1000 && 
                               audioData.quality_analysis.technical_metrics.spectral_centroid <= 3000 ? 'Voice Range' : 'Outside Voice Range'}
                        size="small"
                        color={audioData.quality_analysis.technical_metrics.spectral_centroid >= 1000 && 
                               audioData.quality_analysis.technical_metrics.spectral_centroid <= 3000 ? 'success' : 'warning'}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Zero Crossing Rate</TableCell>
                    <TableCell align="right">{(audioData.quality_analysis.technical_metrics.zero_crossing_rate * 1000).toFixed(1)}</TableCell>
                    <TableCell align="right">per ms</TableCell>
                    <TableCell>
                      <Chip 
                        label="Normal"
                        size="small"
                        color="info"
                      />
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {metricView === 'recommendations' && (
            <Box>
              {audioData.quality_analysis.recommendations.length === 0 ? (
                <Alert severity="success">
                  <Typography variant="body1">
                    No quality issues detected. Audio meets all quality thresholds.
                  </Typography>
                </Alert>
              ) : (
                audioData.quality_analysis.recommendations.map((rec, index) => (
                  <Accordion key={index}>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box display="flex" alignItems="center" gap={2} width="100%">
                        <Box 
                          sx={{ 
                            width: 8, 
                            height: 8, 
                            borderRadius: '50%', 
                            backgroundColor: SEVERITY_COLORS[rec.severity] 
                          }} 
                        />
                        <Typography variant="subtitle2" flexGrow={1}>
                          {rec.title}
                        </Typography>
                        <Chip 
                          label={rec.severity.toUpperCase()} 
                          size="small" 
                          sx={{ 
                            backgroundColor: SEVERITY_COLORS[rec.severity],
                            color: 'white'
                          }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          +{rec.estimated_improvement.toFixed(1)} score
                        </Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" paragraph>
                        <strong>Issue:</strong> {rec.description}
                      </Typography>
                      <Typography variant="body2" paragraph>
                        <strong>Recommended Action:</strong> {rec.action}
                      </Typography>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="caption" color="text.secondary">
                          Category: {rec.category.toUpperCase()}
                        </Typography>
                        <Typography variant="caption" color="success.main">
                          Estimated improvement: +{rec.estimated_improvement.toFixed(1)} points
                        </Typography>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                ))
              )}
            </Box>
          )}

          {metricView === 'trends' && qualityHistory.length > 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                <Analytics sx={{ verticalAlign: 'middle', mr: 1 }} />
                Quality Trends ({qualityHistory.length} measurements)
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom>Overall Score Trend</Typography>
                  <Box height={200} bgcolor="background.default" borderRadius={1} p={2}>
                    {/* Simplified trend visualization */}
                    <Typography variant="caption" color="text.secondary">
                      Latest: {qualityHistory[qualityHistory.length - 1]?.overall_score.toFixed(1)}/10.0
                    </Typography>
                    <br />
                    <Typography variant="caption" color="text.secondary">
                      Average: {(qualityHistory.reduce((sum, h) => sum + h.overall_score, 0) / qualityHistory.length).toFixed(1)}/10.0
                    </Typography>
                    <br />
                    <Typography variant="caption" color="text.secondary">
                      Trend: {currentMetrics?.trend.toUpperCase()}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom>Performance Summary</Typography>
                  <Box height={200} bgcolor="background.default" borderRadius={1} p={2}>
                    <Typography variant="body2" gutterBottom>
                      <CheckCircle sx={{ verticalAlign: 'middle', mr: 1, fontSize: 16 }} color="success" />
                      Measurements above target: {qualityHistory.filter(h => h.overall_score >= targetQuality).length}/{qualityHistory.length}
                    </Typography>
                    <Typography variant="body2" gutterBottom>
                      <TrendingUp sx={{ verticalAlign: 'middle', mr: 1, fontSize: 16 }} color="info" />
                      Best score: {Math.max(...qualityHistory.map(h => h.overall_score)).toFixed(1)}/10.0
                    </Typography>
                    <Typography variant="body2" gutterBottom>
                      <TrendingDown sx={{ verticalAlign: 'middle', mr: 1, fontSize: 16 }} color="warning" />
                      Lowest score: {Math.min(...qualityHistory.map(h => h.overall_score)).toFixed(1)}/10.0
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Box>
          )}
        </Box>

        {/* Stage Contributions (if comparison data available) */}
        {showComparison && audioData.comparison_data?.stage_contributions && (
          <Box mt={2}>
            <Typography variant="h6" gutterBottom>
              Processing Stage Contributions
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Stage</strong></TableCell>
                    <TableCell align="right"><strong>Quality Impact</strong></TableCell>
                    <TableCell align="right"><strong>Processing Time</strong></TableCell>
                    <TableCell align="right"><strong>Effectiveness</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {audioData.comparison_data.stage_contributions.map((stage, index) => (
                    <TableRow key={index}>
                      <TableCell>{stage.stage_name}</TableCell>
                      <TableCell align="right">
                        <Typography 
                          variant="body2" 
                          color={stage.quality_impact >= 0 ? 'success.main' : 'error.main'}
                        >
                          {stage.quality_impact >= 0 ? '+' : ''}{stage.quality_impact.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{stage.processing_time_ms.toFixed(1)}ms</TableCell>
                      <TableCell align="right">
                        <Chip 
                          label={`${(stage.effectiveness_score * 100).toFixed(0)}%`}
                          size="small"
                          color={stage.effectiveness_score >= 0.8 ? 'success' : 
                                 stage.effectiveness_score >= 0.6 ? 'info' : 
                                 stage.effectiveness_score >= 0.4 ? 'warning' : 'error'}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default QualityMetrics;