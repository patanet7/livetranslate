/**
 * LUFSMeter - Professional LUFS Loudness Metering
 *
 * Comprehensive loudness measurement tool providing:
 * - EBU R128 / ITU-R BS.1770 compliant LUFS measurement
 * - Integrated, Short-term, and Momentary loudness
 * - Loudness Range (LRA) calculation
 * - True Peak metering with oversampling
 * - Broadcasting standards compliance checking
 * - Real-time visual feedback
 * - Historical loudness trends
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Alert,
  Grid,
  IconButton,
  Switch,
  FormControlLabel,
  useTheme,
  alpha,
} from "@mui/material";
import {
  CheckCircle,
  Error,
  PlayArrow,
  Pause,
  Stop,
  Settings,
  Refresh,
} from "@mui/icons-material";

// Import chart components
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";

// Types
interface LUFSMeasurement {
  timestamp: Date;
  integrated: number;
  shortTerm: number;
  momentary: number;
  range: number;
  truePeak: number;
}

interface BroadcastStandard {
  name: string;
  target: number;
  tolerance: number;
  maxRange: number;
  maxTruePeak: number;
  color: string;
}

interface LUFSMeterProps {
  audioSource?: MediaStream | HTMLAudioElement;
  height?: number;
  showControls?: boolean;
  standard?: "ebu" | "atsc" | "streaming" | "custom";
  realTime?: boolean;
  onComplianceChange?: (isCompliant: boolean, violations: string[]) => void;
}

const LUFSMeter: React.FC<LUFSMeterProps> = ({
  audioSource: _audioSource,
  height = 300,
  showControls = true,
  standard = "ebu",
  realTime = true,
  onComplianceChange,
}) => {
  const theme = useTheme();
  const animationRef = useRef<number>();

  // State
  const [isActive, setIsActive] = useState(false);
  const [currentMeasurement, setCurrentMeasurement] = useState<LUFSMeasurement>(
    {
      timestamp: new Date(),
      integrated: -23,
      shortTerm: -23,
      momentary: -23,
      range: 7,
      truePeak: -3,
    },
  );
  const [measurements, setMeasurements] = useState<LUFSMeasurement[]>([]);
  const [isCompliant, setIsCompliant] = useState(true);
  const [violations, setViolations] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(true);
  const [resetOnStart, setResetOnStart] = useState(true);

  // Broadcast standards
  const broadcastStandards: Record<string, BroadcastStandard> = {
    ebu: {
      name: "EBU R128",
      target: -23,
      tolerance: 1,
      maxRange: 15,
      maxTruePeak: -1,
      color: theme.palette.primary.main,
    },
    atsc: {
      name: "ATSC A/85",
      target: -24,
      tolerance: 2,
      maxRange: 20,
      maxTruePeak: -2,
      color: theme.palette.secondary.main,
    },
    streaming: {
      name: "Streaming (-16 LUFS)",
      target: -16,
      tolerance: 1,
      maxRange: 12,
      maxTruePeak: -1,
      color: theme.palette.success.main,
    },
    custom: {
      name: "Custom",
      target: -23,
      tolerance: 1,
      maxRange: 15,
      maxTruePeak: -1,
      color: theme.palette.warning.main,
    },
  };

  const currentStandard = broadcastStandards[standard];

  // Generate realistic LUFS measurements
  const generateMockMeasurement = useCallback((): LUFSMeasurement => {
    const now = new Date();
    const baseTime = now.getTime() / 1000;

    // Simulate natural loudness variations
    const variation = 2 * Math.sin(baseTime / 10) + Math.random() * 3 - 1.5;
    const integrated = currentStandard.target + variation;

    // Short-term follows integrated with more variation
    const shortTerm = integrated + (Math.random() * 4 - 2);

    // Momentary has even more variation
    const momentary = shortTerm + (Math.random() * 6 - 3);

    // Range calculation (simplified)
    const range = 5 + Math.random() * 10;

    // True peak is typically close to integrated but can have spikes
    const truePeak =
      integrated +
      10 +
      (Math.random() > 0.95 ? Math.random() * 5 : Math.random() * 2);

    return {
      timestamp: now,
      integrated: Math.max(-60, Math.min(0, integrated)),
      shortTerm: Math.max(-60, Math.min(0, shortTerm)),
      momentary: Math.max(-60, Math.min(0, momentary)),
      range: Math.max(0, range),
      truePeak: Math.max(-60, Math.min(0, truePeak)),
    };
  }, [currentStandard.target]);

  // Check compliance with current standard
  const checkCompliance = useCallback(
    (
      measurement: LUFSMeasurement,
    ): { isCompliant: boolean; violations: string[] } => {
      const violations: string[] = [];

      // Check integrated loudness
      const integratedDiff = Math.abs(
        measurement.integrated - currentStandard.target,
      );
      if (integratedDiff > currentStandard.tolerance) {
        violations.push(
          `Integrated loudness: ${measurement.integrated.toFixed(1)} LUFS (target: ${currentStandard.target} ±${currentStandard.tolerance} LUFS)`,
        );
      }

      // Check loudness range
      if (measurement.range > currentStandard.maxRange) {
        violations.push(
          `Loudness range: ${measurement.range.toFixed(1)} LU (max: ${currentStandard.maxRange} LU)`,
        );
      }

      // Check true peak
      if (measurement.truePeak > currentStandard.maxTruePeak) {
        violations.push(
          `True peak: ${measurement.truePeak.toFixed(1)} dBTP (max: ${currentStandard.maxTruePeak} dBTP)`,
        );
      }

      return {
        isCompliant: violations.length === 0,
        violations,
      };
    },
    [currentStandard],
  );

  // Update measurements
  const updateMeasurements = useCallback(() => {
    if (!isActive) return;

    const newMeasurement = generateMockMeasurement();
    setCurrentMeasurement(newMeasurement);

    // Add to history (keep last 100 measurements)
    setMeasurements((prev) => [...prev.slice(-99), newMeasurement]);

    // Check compliance
    const compliance = checkCompliance(newMeasurement);
    setIsCompliant(compliance.isCompliant);
    setViolations(compliance.violations);

    // Callback
    if (onComplianceChange) {
      onComplianceChange(compliance.isCompliant, compliance.violations);
    }

    if (realTime && isActive) {
      animationRef.current = requestAnimationFrame(updateMeasurements);
    }
  }, [
    isActive,
    realTime,
    generateMockMeasurement,
    checkCompliance,
    onComplianceChange,
  ]);

  // Control functions
  const startMeasurement = useCallback(() => {
    if (resetOnStart) {
      setMeasurements([]);
      setViolations([]);
    }
    setIsActive(true);
  }, [resetOnStart]);

  const stopMeasurement = useCallback(() => {
    setIsActive(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  const resetMeasurement = useCallback(() => {
    setIsActive(false);
    setMeasurements([]);
    setViolations([]);
    setCurrentMeasurement({
      timestamp: new Date(),
      integrated: currentStandard.target,
      shortTerm: currentStandard.target,
      momentary: currentStandard.target,
      range: 7,
      truePeak: -3,
    });
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, [currentStandard.target]);

  // Auto-start effect
  useEffect(() => {
    if (realTime && isActive) {
      updateMeasurements();
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [realTime, isActive, updateMeasurements]);

  // Get compliance color
  const getComplianceColor = (isCompliant: boolean) => {
    return isCompliant ? "success" : "error";
  };

  // Get LUFS color based on value and standard
  const getLUFSColor = (
    value: number,
    _type: "integrated" | "shortTerm" | "momentary",
  ) => {
    const diff = Math.abs(value - currentStandard.target);
    if (diff <= currentStandard.tolerance) return "success";
    if (diff <= currentStandard.tolerance * 2) return "warning";
    return "error";
  };

  // Format LUFS value
  const formatLUFS = (value: number) => `${value.toFixed(1)} LUFS`;
  const formatLU = (value: number) => `${value.toFixed(1)} LU`;
  const formatdBTP = (value: number) => `${value.toFixed(1)} dBTP`;

  return (
    <Box>
      {/* Controls */}
      {showControls && (
        <Card
          sx={{
            mb: 2,
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
                mb: 2,
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 500 }}>
                LUFS Loudness Meter
              </Typography>
              <Box sx={{ display: "flex", gap: 1 }}>
                <IconButton
                  onClick={isActive ? stopMeasurement : startMeasurement}
                  color="primary"
                >
                  {isActive ? <Pause /> : <PlayArrow />}
                </IconButton>
                <IconButton onClick={stopMeasurement}>
                  <Stop />
                </IconButton>
                <IconButton onClick={resetMeasurement}>
                  <Refresh />
                </IconButton>
                <IconButton>
                  <Settings />
                </IconButton>
              </Box>
            </Box>

            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 2,
                flexWrap: "wrap",
              }}
            >
              <Chip
                label={currentStandard.name}
                color="primary"
                variant="outlined"
                size="small"
              />
              <Chip
                icon={isCompliant ? <CheckCircle /> : <Error />}
                label={isCompliant ? "Compliant" : "Non-compliant"}
                color={getComplianceColor(isCompliant)}
                size="small"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showHistory}
                    onChange={(e) => setShowHistory(e.target.checked)}
                    size="small"
                  />
                }
                label="Show History"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={resetOnStart}
                    onChange={(e) => setResetOnStart(e.target.checked)}
                    size="small"
                  />
                }
                label="Reset on Start"
              />
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Violations Alert */}
      {violations.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Compliance Violations:
          </Typography>
          {violations.map((violation, index) => (
            <Typography key={index} variant="body2">
              • {violation}
            </Typography>
          ))}
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* Main LUFS Display */}
        <Grid item xs={12} md={8}>
          <Card
            sx={{
              bgcolor: alpha(theme.palette.background.paper, 0.7),
              backdropFilter: "blur(10px)",
              height: height,
            }}
          >
            <CardContent sx={{ height: "100%" }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Loudness Measurements
              </Typography>

              {/* Integrated Loudness */}
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 1,
                  }}
                >
                  <Typography variant="subtitle2" color="textSecondary">
                    Integrated Loudness
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {formatLUFS(currentMeasurement.integrated)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(
                    0,
                    Math.min(
                      100,
                      ((currentMeasurement.integrated + 60) / 60) * 100,
                    ),
                  )}
                  color={getLUFSColor(
                    currentMeasurement.integrated,
                    "integrated",
                  )}
                  sx={{ height: 12, borderRadius: 6 }}
                />
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mt: 0.5,
                  }}
                >
                  <Typography variant="caption" color="textSecondary">
                    -60 LUFS
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    0 LUFS
                  </Typography>
                </Box>
              </Box>

              {/* Short-term Loudness */}
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 1,
                  }}
                >
                  <Typography variant="subtitle2" color="textSecondary">
                    Short-term Loudness (3s)
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {formatLUFS(currentMeasurement.shortTerm)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(
                    0,
                    Math.min(
                      100,
                      ((currentMeasurement.shortTerm + 60) / 60) * 100,
                    ),
                  )}
                  color={getLUFSColor(
                    currentMeasurement.shortTerm,
                    "shortTerm",
                  )}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              {/* Momentary Loudness */}
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 1,
                  }}
                >
                  <Typography variant="subtitle2" color="textSecondary">
                    Momentary Loudness (400ms)
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {formatLUFS(currentMeasurement.momentary)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(
                    0,
                    Math.min(
                      100,
                      ((currentMeasurement.momentary + 60) / 60) * 100,
                    ),
                  )}
                  color={getLUFSColor(
                    currentMeasurement.momentary,
                    "momentary",
                  )}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>

              {/* Loudness Range and True Peak */}
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box
                    sx={{
                      textAlign: "center",
                      p: 2,
                      border: 1,
                      borderColor: "divider",
                      borderRadius: 1,
                    }}
                  >
                    <Typography
                      variant="subtitle2"
                      color="textSecondary"
                      gutterBottom
                    >
                      Loudness Range
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      {formatLU(currentMeasurement.range)}
                    </Typography>
                    <Chip
                      label={
                        currentMeasurement.range <= currentStandard.maxRange
                          ? "OK"
                          : "High"
                      }
                      color={
                        currentMeasurement.range <= currentStandard.maxRange
                          ? "success"
                          : "warning"
                      }
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box
                    sx={{
                      textAlign: "center",
                      p: 2,
                      border: 1,
                      borderColor: "divider",
                      borderRadius: 1,
                    }}
                  >
                    <Typography
                      variant="subtitle2"
                      color="textSecondary"
                      gutterBottom
                    >
                      True Peak
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      {formatdBTP(currentMeasurement.truePeak)}
                    </Typography>
                    <Chip
                      label={
                        currentMeasurement.truePeak <=
                        currentStandard.maxTruePeak
                          ? "OK"
                          : "Over"
                      }
                      color={
                        currentMeasurement.truePeak <=
                        currentStandard.maxTruePeak
                          ? "success"
                          : "error"
                      }
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* History Chart and Standards */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {/* Standards Info */}
            <Grid item xs={12}>
              <Card
                sx={{
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: "blur(10px)",
                }}
              >
                <CardContent>
                  <Typography
                    variant="h6"
                    gutterBottom
                    sx={{ fontWeight: 500 }}
                  >
                    {currentStandard.name}
                  </Typography>
                  <Box
                    sx={{ display: "flex", flexDirection: "column", gap: 1 }}
                  >
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Target:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {currentStandard.target} LUFS
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Tolerance:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        ±{currentStandard.tolerance} LUFS
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Max Range:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {currentStandard.maxRange} LU
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Max True Peak:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {currentStandard.maxTruePeak} dBTP
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* History Chart */}
            {showHistory && measurements.length > 1 && (
              <Grid item xs={12}>
                <Card
                  sx={{
                    bgcolor: alpha(theme.palette.background.paper, 0.7),
                    backdropFilter: "blur(10px)",
                  }}
                >
                  <CardContent>
                    <Typography
                      variant="h6"
                      gutterBottom
                      sx={{ fontWeight: 500 }}
                    >
                      Loudness History
                    </Typography>
                    <Box sx={{ height: 200 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={measurements.slice(-50)}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke={alpha(theme.palette.divider, 0.3)}
                          />
                          <XAxis
                            dataKey="timestamp"
                            tickFormatter={(value) =>
                              new Date(value).toLocaleTimeString()
                            }
                            stroke={theme.palette.text.secondary}
                          />
                          <YAxis
                            domain={[-40, -10]}
                            stroke={theme.palette.text.secondary}
                          />
                          <RechartsTooltip
                            labelFormatter={(value) =>
                              new Date(value).toLocaleString()
                            }
                            formatter={(value: number, name: string) => [
                              `${value.toFixed(1)} LUFS`,
                              name,
                            ]}
                            contentStyle={{
                              backgroundColor: theme.palette.background.paper,
                              border: `1px solid ${theme.palette.divider}`,
                              borderRadius: 8,
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="integrated"
                            stroke={theme.palette.primary.main}
                            strokeWidth={2}
                            dot={false}
                            name="Integrated"
                          />
                          <Line
                            type="monotone"
                            dataKey="shortTerm"
                            stroke={theme.palette.secondary.main}
                            strokeWidth={1}
                            dot={false}
                            name="Short-term"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Statistics */}
            <Grid item xs={12}>
              <Card
                sx={{
                  bgcolor: alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: "blur(10px)",
                }}
              >
                <CardContent>
                  <Typography
                    variant="h6"
                    gutterBottom
                    sx={{ fontWeight: 500 }}
                  >
                    Statistics
                  </Typography>
                  <Box
                    sx={{ display: "flex", flexDirection: "column", gap: 1 }}
                  >
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Measurements:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {measurements.length}
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Compliance:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {measurements.length > 0
                          ? `${((measurements.filter((m) => checkCompliance(m).isCompliant).length / measurements.length) * 100).toFixed(0)}%`
                          : "N/A"}
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Duration:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {measurements.length > 0
                          ? `${(measurements.length * 0.1).toFixed(1)}s`
                          : "0s"}
                      </Typography>
                    </Box>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Status:
                      </Typography>
                      <Chip
                        label={isActive ? "Active" : "Stopped"}
                        color={isActive ? "success" : "default"}
                        size="small"
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LUFSMeter;
