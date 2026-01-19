import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Grid,
  Alert,
  Tooltip,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  Timeline,
  Speed,
  ShowChart,
  Warning,
  CheckCircle,
  Error,
  ExpandMore,
  Refresh,
} from "@mui/icons-material";

interface StageMetricsProps {
  stageName: string;
  stageDisplayName: string;
  metrics?: StagePerformanceMetrics;
  isActive?: boolean;
  refreshInterval?: number;
  onRefresh?: () => void;
}

interface StagePerformanceMetrics {
  processing_time_ms: number;
  target_latency_ms: number;
  max_latency_ms: number;
  quality_metrics: {
    input_rms: number;
    output_rms: number;
    level_change_db: number;
    estimated_snr_db: number;
    dynamic_range: number;
  };
  status: "success" | "warning" | "error";
  error_message?: string;
  processing_count: number;
  average_latency_ms: number;
  peak_latency_ms: number;
  success_rate: number;
  last_updated: string;
}

interface QualityIndicatorProps {
  label: string;
  value: number;
  unit: string;
  range: [number, number];
  optimal?: [number, number];
  format?: "decimal" | "integer" | "db";
}

const QualityIndicator: React.FC<QualityIndicatorProps> = ({
  label,
  value,
  unit,
  range,
  optimal,
  format = "decimal",
}) => {
  const formatValue = (val: number) => {
    switch (format) {
      case "integer":
        return Math.round(val).toString();
      case "db":
        return `${val > 0 ? "+" : ""}${val.toFixed(1)}`;
      default:
        return val.toFixed(3);
    }
  };

  const getIndicatorColor = () => {
    if (optimal) {
      if (value >= optimal[0] && value <= optimal[1]) return "success";
    }
    if (value >= range[0] && value <= range[1]) return "warning";
    return "error";
  };

  const getProgressValue = () => {
    const [min, max] = range;
    return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  };

  return (
    <Box mb={2}>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={1}
      >
        <Typography variant="body2">{label}</Typography>
        <Chip
          label={`${formatValue(value)} ${unit}`}
          size="small"
          color={getIndicatorColor()}
          variant="outlined"
        />
      </Box>
      <LinearProgress
        variant="determinate"
        value={getProgressValue()}
        color={getIndicatorColor()}
        sx={{ height: 6, borderRadius: 3 }}
      />
      <Box display="flex" justifyContent="space-between" mt={0.5}>
        <Typography variant="caption" color="text.secondary">
          {formatValue(range[0])} {unit}
        </Typography>
        {optimal && (
          <Typography variant="caption" color="success.main">
            Optimal: {formatValue(optimal[0])}-{formatValue(optimal[1])} {unit}
          </Typography>
        )}
        <Typography variant="caption" color="text.secondary">
          {formatValue(range[1])} {unit}
        </Typography>
      </Box>
    </Box>
  );
};

const STAGE_PERFORMANCE_TARGETS = {
  vad: { target: 5.0, max: 10.0 },
  voice_filter: { target: 8.0, max: 15.0 },
  noise_reduction: { target: 15.0, max: 25.0 },
  voice_enhancement: { target: 10.0, max: 20.0 },
  equalizer: { target: 12.0, max: 22.0 },
  spectral_denoising: { target: 20.0, max: 35.0 },
  conventional_denoising: { target: 8.0, max: 15.0 },
  lufs_normalization: { target: 18.0, max: 30.0 },
  agc: { target: 12.0, max: 20.0 },
  compression: { target: 8.0, max: 15.0 },
  limiter: { target: 6.0, max: 12.0 },
};

export const StageMetrics: React.FC<StageMetricsProps> = ({
  stageName,
  stageDisplayName,
  metrics,
  isActive = false,
  refreshInterval = 1000,
  onRefresh,
}) => {
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [expanded, setExpanded] = useState(false);

  const targets = (
    STAGE_PERFORMANCE_TARGETS as Record<string, { target: number; max: number }>
  )[stageName] || { target: 10.0, max: 20.0 };

  useEffect(() => {
    if (isActive && onRefresh) {
      const interval = setInterval(() => {
        onRefresh();
        setLastRefresh(new Date());
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [isActive, onRefresh, refreshInterval]);

  const getLatencyStatus = () => {
    if (!metrics) return "default";

    if (metrics.processing_time_ms <= targets.target) return "success";
    if (metrics.processing_time_ms <= targets.max) return "warning";
    return "error";
  };

  const getLatencyIcon = () => {
    const status = getLatencyStatus();
    switch (status) {
      case "success":
        return <CheckCircle color="success" />;
      case "warning":
        return <Warning color="warning" />;
      case "error":
        return <Error color="error" />;
      default:
        return <Speed />;
    }
  };

  if (!metrics) {
    return (
      <Card sx={{ height: "100%" }}>
        <CardContent>
          <Box
            display="flex"
            alignItems="center"
            justifyContent="between"
            mb={2}
          >
            <Typography variant="h6" component="h3">
              {stageDisplayName} Metrics
            </Typography>
            <Chip label="No Data" size="small" color="default" />
          </Box>

          <Alert severity="info">
            No metrics available. Process audio through this stage to see
            performance data.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: "100%" }}>
      <CardContent>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="space-between"
          mb={2}
        >
          <Typography variant="h6" component="h3">
            {stageDisplayName} Metrics
          </Typography>

          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={isActive ? "Live" : "Static"}
              size="small"
              color={isActive ? "success" : "default"}
              variant={isActive ? "filled" : "outlined"}
            />
            {onRefresh && (
              <Tooltip title="Refresh metrics">
                <IconButton size="small" onClick={onRefresh}>
                  <Refresh />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {/* Primary Performance Metrics */}
        <Grid container spacing={2} mb={3}>
          <Grid item xs={6}>
            <Box
              textAlign="center"
              p={2}
              bgcolor="background.default"
              borderRadius={1}
            >
              <Box display="flex" justifyContent="center" mb={1}>
                {getLatencyIcon()}
              </Box>
              <Typography variant="h4" color={`${getLatencyStatus()}.main`}>
                {metrics.processing_time_ms.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                ms processing time
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Target: {targets.target}ms | Max: {targets.max}ms
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box
              textAlign="center"
              p={2}
              bgcolor="background.default"
              borderRadius={1}
            >
              <Box display="flex" justifyContent="center" mb={1}>
                <ShowChart
                  color={
                    metrics.success_rate >= 95
                      ? "success"
                      : metrics.success_rate >= 90
                        ? "warning"
                        : "error"
                  }
                />
              </Box>
              <Typography
                variant="h4"
                color={
                  metrics.success_rate >= 95
                    ? "success.main"
                    : metrics.success_rate >= 90
                      ? "warning.main"
                      : "error.main"
                }
              >
                {metrics.success_rate.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                success rate
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {metrics.processing_count} processed
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Status Alert */}
        {metrics.status === "error" && metrics.error_message && (
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Processing Error:</strong> {metrics.error_message}
            </Typography>
          </Alert>
        )}

        {metrics.status === "warning" && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Performance degradation detected. Processing time exceeds target
            latency.
          </Alert>
        )}

        {/* Latency Statistics */}
        <Accordion expanded={expanded} onChange={() => setExpanded(!expanded)}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">
              <Timeline sx={{ verticalAlign: "middle", mr: 1, fontSize: 20 }} />
              Detailed Performance Analysis
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" gutterBottom>
                  Latency Statistics
                </Typography>
                <Box mb={2}>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Current:</Typography>
                    <Typography variant="caption">
                      {metrics.processing_time_ms.toFixed(1)}ms
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Average:</Typography>
                    <Typography variant="caption">
                      {metrics.average_latency_ms.toFixed(1)}ms
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Peak:</Typography>
                    <Typography variant="caption">
                      {metrics.peak_latency_ms.toFixed(1)}ms
                    </Typography>
                  </Box>
                </Box>
              </Grid>

              <Grid item xs={6}>
                <Typography variant="body2" gutterBottom>
                  Session Info
                </Typography>
                <Box mb={2}>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Processed:</Typography>
                    <Typography variant="caption">
                      {metrics.processing_count}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Success Rate:</Typography>
                    <Typography variant="caption">
                      {metrics.success_rate.toFixed(1)}%
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="caption">Last Updated:</Typography>
                    <Typography variant="caption">
                      {new Date(metrics.last_updated).toLocaleTimeString()}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            </Grid>

            {/* Audio Quality Metrics */}
            <Typography variant="body2" gutterBottom mt={2}>
              Audio Quality Analysis
            </Typography>

            <QualityIndicator
              label="Input RMS Level"
              value={metrics.quality_metrics.input_rms}
              unit=""
              range={[0, 1]}
              optimal={[0.1, 0.7]}
              format="decimal"
            />

            <QualityIndicator
              label="Output RMS Level"
              value={metrics.quality_metrics.output_rms}
              unit=""
              range={[0, 1]}
              optimal={[0.1, 0.7]}
              format="decimal"
            />

            <QualityIndicator
              label="Level Change"
              value={metrics.quality_metrics.level_change_db}
              unit="dB"
              range={[-20, 20]}
              optimal={[-3, 3]}
              format="db"
            />

            <QualityIndicator
              label="Estimated SNR"
              value={metrics.quality_metrics.estimated_snr_db}
              unit="dB"
              range={[0, 60]}
              optimal={[20, 50]}
              format="db"
            />

            <QualityIndicator
              label="Dynamic Range"
              value={metrics.quality_metrics.dynamic_range}
              unit=""
              range={[0, 2]}
              optimal={[0.5, 1.5]}
              format="decimal"
            />
          </AccordionDetails>
        </Accordion>

        {/* Refresh Info */}
        <Box mt={2} textAlign="center">
          <Typography variant="caption" color="text.secondary">
            {isActive ? (
              <>
                Refreshing every {refreshInterval / 1000}s • Last:{" "}
                {lastRefresh.toLocaleTimeString()}
              </>
            ) : (
              <>
                Static metrics • Last updated:{" "}
                {new Date(metrics.last_updated).toLocaleTimeString()}
              </>
            )}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
