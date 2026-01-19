import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Alert,
  Grid,
} from "@mui/material";
import {
  VolumeUp,
  Download,
  Warning,
  CheckCircle,
  Error,
  Info,
} from "@mui/icons-material";

export interface LUFSMeterProps {
  audioData?: {
    lufs_measurements: {
      integrated_lufs: number;
      lufs_range: number;
      momentary_lufs: number[];
      short_term_lufs: number[];
      true_peak_db: number;
    };
    compliance: {
      ebu_r128_compliant: boolean;
      broadcast_safe: boolean;
      recommended_adjustment_db: number;
    };
    timestamp: number;
  };
  isRealTime?: boolean;
  targetLufs?: number;
  onMeasurementUpdate?: (measurement: LUFSMeasurement) => void;
  height?: number;
  showCompliance?: boolean;
}

interface LUFSMeasurement {
  integrated_lufs: number;
  momentary_lufs: number;
  short_term_lufs: number;
  lufs_range: number;
  true_peak_db: number;
  loudness_units: number;
  ebu_r128_compliant: boolean;
  broadcast_safe: boolean;
  recommended_adjustment: number;
}

type MeasurementType = "momentary" | "short_term" | "integrated";
type BroadcastStandard = "ebu_r128" | "atsc_a85" | "arib_tr_b32" | "custom";

const BROADCAST_STANDARDS = {
  ebu_r128: {
    name: "EBU R128",
    target_lufs: -23,
    max_true_peak: -1,
    max_range: 20,
    description: "European Broadcasting Union standard",
  },
  atsc_a85: {
    name: "ATSC A/85",
    target_lufs: -24,
    max_true_peak: -2,
    max_range: 18,
    description: "US broadcast standard",
  },
  arib_tr_b32: {
    name: "ARIB TR-B32",
    target_lufs: -24,
    max_true_peak: -1,
    max_range: 18,
    description: "Japanese broadcast standard",
  },
  custom: {
    name: "Custom",
    target_lufs: -23,
    max_true_peak: -1,
    max_range: 15,
    description: "Custom target levels",
  },
};

export const LUFSMeter: React.FC<LUFSMeterProps> = ({
  audioData,
  isRealTime = false,
  targetLufs = -23,
  onMeasurementUpdate,
  height = 400,
  showCompliance = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  const [measurementType, setMeasurementType] =
    useState<MeasurementType>("integrated");
  const [broadcastStandard, setBroadcastStandard] =
    useState<BroadcastStandard>("ebu_r128");
  const [showMomentary, setShowMomentary] = useState(true);
  const [showShortTerm, setShowShortTerm] = useState(true);
  const [showTruePeak, setShowTruePeak] = useState(true);
  const [historyLength] = useState(30); // seconds

  const [currentMeasurement, setCurrentMeasurement] =
    useState<LUFSMeasurement | null>(null);
  const [measurementHistory, setMeasurementHistory] = useState<
    LUFSMeasurement[]
  >([]);
  const [peakHold, setPeakHold] = useState(-60);

  const standard = BROADCAST_STANDARDS[broadcastStandard];
  const currentTarget = targetLufs || standard.target_lufs;

  const drawLUFSMeter = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioData) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height: canvasHeight } = canvas;
    const { lufs_measurements } = audioData;

    // Clear canvas
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, width, canvasHeight);

    // Meter dimensions
    const meterWidth = 60;
    const meterHeight = canvasHeight - 80;
    const meterX = 50;
    const meterY = 40;

    // LUFS scale (-60 to 0)
    const lufsMin = -60;
    const lufsMax = 0;
    const lufsRange = lufsMax - lufsMin;

    // Draw meter background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(meterX, meterY, meterWidth, meterHeight);

    // Draw scale markings
    ctx.strokeStyle = "#444444";
    ctx.lineWidth = 1;
    ctx.font = "10px monospace";
    ctx.fillStyle = "#888888";
    ctx.textAlign = "right";

    for (let lufs = lufsMin; lufs <= lufsMax; lufs += 6) {
      const y =
        meterY + meterHeight - ((lufs - lufsMin) / lufsRange) * meterHeight;

      // Scale line
      ctx.beginPath();
      ctx.moveTo(meterX - 5, y);
      ctx.lineTo(meterX + meterWidth + 5, y);
      ctx.stroke();

      // Scale label
      ctx.fillText(lufs.toString(), meterX - 8, y + 3);
    }

    // Draw target level line
    const targetY =
      meterY +
      meterHeight -
      ((currentTarget - lufsMin) / lufsRange) * meterHeight;
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(meterX - 10, targetY);
    ctx.lineTo(meterX + meterWidth + 10, targetY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Target label
    ctx.fillStyle = "#00ff88";
    ctx.textAlign = "left";
    ctx.fillText(
      `${currentTarget} LUFS`,
      meterX + meterWidth + 15,
      targetY + 3,
    );

    // Draw warning zones
    const warningZones = [
      { min: -9, max: 0, color: "rgba(255, 0, 0, 0.2)" }, // Red zone (too loud)
      { min: -18, max: -9, color: "rgba(255, 193, 7, 0.2)" }, // Yellow zone (caution)
      {
        min: currentTarget - 3,
        max: currentTarget + 3,
        color: "rgba(0, 255, 136, 0.2)",
      }, // Green zone (target)
    ];

    warningZones.forEach((zone) => {
      const zoneTop =
        meterY + meterHeight - ((zone.max - lufsMin) / lufsRange) * meterHeight;
      const zoneBottom =
        meterY + meterHeight - ((zone.min - lufsMin) / lufsRange) * meterHeight;

      ctx.fillStyle = zone.color;
      ctx.fillRect(meterX, zoneTop, meterWidth, zoneBottom - zoneTop);
    });

    // Draw current level bars
    const measurements = {
      integrated: lufs_measurements.integrated_lufs,
      momentary:
        lufs_measurements.momentary_lufs[
          lufs_measurements.momentary_lufs.length - 1
        ] || -60,
      short_term:
        lufs_measurements.short_term_lufs[
          lufs_measurements.short_term_lufs.length - 1
        ] || -60,
    };

    const barColors = {
      integrated: "#00ff88",
      momentary: "#ff6b6b",
      short_term: "#4dabf7",
    };

    Object.entries(measurements).forEach(([type, value], index) => {
      if (
        type === "integrated" ||
        (type === "momentary" && showMomentary) ||
        (type === "short_term" && showShortTerm)
      ) {
        const barX = meterX + index * (meterWidth / 3);
        const barWidth = meterWidth / 3 - 2;
        const barHeight = Math.max(
          1,
          ((Math.max(value, lufsMin) - lufsMin) / lufsRange) * meterHeight,
        );
        const barY = meterY + meterHeight - barHeight;

        // Draw level bar
        ctx.fillStyle = barColors[type as keyof typeof barColors];
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Peak hold indicator
        if (value > peakHold && type === measurementType) {
          setPeakHold(value);
        }

        if (type === measurementType) {
          const peakY =
            meterY +
            meterHeight -
            ((peakHold - lufsMin) / lufsRange) * meterHeight;
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(barX, peakY - 1, barWidth, 2);
        }
      }
    });

    // Draw history graph (right side)
    if (measurementHistory.length > 1) {
      const graphX = meterX + meterWidth + 80;
      const graphWidth = width - graphX - 20;
      const graphHeight = meterHeight;
      const graphY = meterY;

      // Graph background
      ctx.fillStyle = "rgba(26, 26, 26, 0.8)";
      ctx.fillRect(graphX, graphY, graphWidth, graphHeight);

      // Graph border
      ctx.strokeStyle = "#444444";
      ctx.lineWidth = 1;
      ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);

      // Draw measurement history
      const pointSpacing =
        graphWidth / Math.max(measurementHistory.length - 1, 1);

      ["integrated", "momentary", "short_term"].forEach((type) => {
        if (
          type === "integrated" ||
          (type === "momentary" && showMomentary) ||
          (type === "short_term" && showShortTerm)
        ) {
          ctx.strokeStyle = barColors[type as keyof typeof barColors];
          ctx.lineWidth = 2;
          ctx.beginPath();

          measurementHistory.forEach((measurement, index) => {
            const x = graphX + index * pointSpacing;
            const value =
              type === "integrated"
                ? measurement.integrated_lufs
                : type === "momentary"
                  ? measurement.momentary_lufs
                  : measurement.short_term_lufs;
            const y =
              graphY +
              graphHeight -
              ((Math.max(value, lufsMin) - lufsMin) / lufsRange) * graphHeight;

            if (index === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          });

          ctx.stroke();
        }
      });

      // Target line on graph
      const targetGraphY =
        graphY +
        graphHeight -
        ((currentTarget - lufsMin) / lufsRange) * graphHeight;
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(graphX, targetGraphY);
      ctx.lineTo(graphX + graphWidth, targetGraphY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Labels and legend
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px monospace";
    ctx.textAlign = "left";

    // Current values display
    const valueY = meterY + meterHeight + 30;

    ctx.fillStyle = barColors.integrated;
    ctx.fillText(
      `Integrated: ${measurements.integrated.toFixed(1)} LUFS`,
      meterX,
      valueY,
    );

    if (showMomentary) {
      ctx.fillStyle = barColors.momentary;
      ctx.fillText(
        `Momentary: ${measurements.momentary.toFixed(1)} LUFS`,
        meterX,
        valueY + 15,
      );
    }

    if (showShortTerm) {
      ctx.fillStyle = barColors.short_term;
      ctx.fillText(
        `Short-term: ${measurements.short_term.toFixed(1)} LUFS`,
        meterX,
        valueY + 30,
      );
    }

    if (showTruePeak) {
      ctx.fillStyle = "#ffff00";
      ctx.fillText(
        `True Peak: ${lufs_measurements.true_peak_db.toFixed(1)} dBTP`,
        meterX,
        valueY + 45,
      );
    }
  }, [
    audioData,
    currentTarget,
    showMomentary,
    showShortTerm,
    showTruePeak,
    measurementHistory,
    peakHold,
    measurementType,
  ]);

  const updateMeasurement = useCallback(() => {
    if (!audioData) return;

    const { lufs_measurements, compliance } = audioData;

    const measurement: LUFSMeasurement = {
      integrated_lufs: lufs_measurements.integrated_lufs,
      momentary_lufs:
        lufs_measurements.momentary_lufs[
          lufs_measurements.momentary_lufs.length - 1
        ] || -60,
      short_term_lufs:
        lufs_measurements.short_term_lufs[
          lufs_measurements.short_term_lufs.length - 1
        ] || -60,
      lufs_range: lufs_measurements.lufs_range,
      true_peak_db: lufs_measurements.true_peak_db,
      loudness_units: lufs_measurements.integrated_lufs - currentTarget,
      ebu_r128_compliant: compliance.ebu_r128_compliant,
      broadcast_safe: compliance.broadcast_safe,
      recommended_adjustment: compliance.recommended_adjustment_db,
    };

    setCurrentMeasurement(measurement);

    // Add to history (keep last N measurements)
    setMeasurementHistory((prev) => {
      const newHistory = [...prev, measurement];
      const maxPoints = Math.floor(historyLength * 10); // 10 measurements per second
      return newHistory.slice(-maxPoints);
    });

    onMeasurementUpdate?.(measurement);
  }, [audioData, currentTarget, historyLength, onMeasurementUpdate]);

  useEffect(() => {
    if (isRealTime && audioData) {
      const animate = () => {
        updateMeasurement();
        drawLUFSMeter();

        if (isRealTime) {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      };

      animate();
    } else if (audioData) {
      updateMeasurement();
      drawLUFSMeter();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRealTime, audioData, updateMeasurement, drawLUFSMeter]);

  const handleDownloadReport = () => {
    if (!currentMeasurement) return;

    const report = {
      timestamp: new Date().toISOString(),
      broadcast_standard: standard.name,
      target_lufs: currentTarget,
      measurement: currentMeasurement,
      compliance_analysis: {
        meets_target:
          Math.abs(currentMeasurement.integrated_lufs - currentTarget) <= 1,
        true_peak_safe:
          currentMeasurement.true_peak_db <= standard.max_true_peak,
        range_acceptable: currentMeasurement.lufs_range <= standard.max_range,
        overall_compliant:
          currentMeasurement.ebu_r128_compliant &&
          currentMeasurement.broadcast_safe,
      },
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lufs_measurement_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getComplianceStatus = () => {
    if (!currentMeasurement) return "unknown";

    if (
      currentMeasurement.ebu_r128_compliant &&
      currentMeasurement.broadcast_safe
    ) {
      return "compliant";
    } else if (
      Math.abs(currentMeasurement.integrated_lufs - currentTarget) <= 3
    ) {
      return "warning";
    } else {
      return "non_compliant";
    }
  };

  const getComplianceIcon = () => {
    const status = getComplianceStatus();
    switch (status) {
      case "compliant":
        return <CheckCircle color="success" />;
      case "warning":
        return <Warning color="warning" />;
      case "non_compliant":
        return <Error color="error" />;
      default:
        return <Info color="info" />;
    }
  };

  if (!audioData) {
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
              LUFS Meter
            </Typography>
            <Chip label="No Data" size="small" color="default" />
          </Box>

          <Alert severity="info">
            No audio data available for LUFS measurement. Start audio processing
            to see broadcast-compliant loudness metering.
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
            <VolumeUp sx={{ verticalAlign: "middle", mr: 1, fontSize: 24 }} />
            LUFS Meter (ITU-R BS.1770-4)
          </Typography>

          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={isRealTime ? "Real-time" : "Static"}
              size="small"
              color={isRealTime ? "success" : "default"}
              variant={isRealTime ? "filled" : "outlined"}
            />
            <IconButton size="small" onClick={handleDownloadReport}>
              <Download />
            </IconButton>
          </Box>
        </Box>

        {/* Controls */}
        <Grid container spacing={2} mb={2}>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Broadcast Standard</InputLabel>
              <Select
                value={broadcastStandard}
                label="Broadcast Standard"
                onChange={(e) =>
                  setBroadcastStandard(e.target.value as BroadcastStandard)
                }
              >
                {Object.entries(BROADCAST_STANDARDS).map(([key, std]) => (
                  <MenuItem key={key} value={key}>
                    {std.name} ({std.target_lufs} LUFS)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Primary Display</InputLabel>
              <Select
                value={measurementType}
                label="Primary Display"
                onChange={(e) =>
                  setMeasurementType(e.target.value as MeasurementType)
                }
              >
                <MenuItem value="integrated">Integrated</MenuItem>
                <MenuItem value="momentary">Momentary</MenuItem>
                <MenuItem value="short_term">Short-term</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box display="flex" gap={1} flexWrap="wrap">
              <FormControlLabel
                control={
                  <Switch
                    checked={showMomentary}
                    onChange={(e) => setShowMomentary(e.target.checked)}
                    size="small"
                  />
                }
                label="Momentary"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showShortTerm}
                    onChange={(e) => setShowShortTerm(e.target.checked)}
                    size="small"
                  />
                }
                label="Short-term"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showTruePeak}
                    onChange={(e) => setShowTruePeak(e.target.checked)}
                    size="small"
                  />
                }
                label="True Peak"
              />
            </Box>
          </Grid>
        </Grid>

        {/* LUFS Meter Canvas */}
        <Box
          position="relative"
          height={height}
          bgcolor="#000"
          borderRadius={1}
          overflow="hidden"
        >
          <canvas
            ref={canvasRef}
            width={800}
            height={height}
            style={{
              width: "100%",
              height: "100%",
              display: "block",
            }}
          />
        </Box>

        {/* Compliance Status */}
        {showCompliance && currentMeasurement && (
          <Box mt={2} p={2} bgcolor="background.default" borderRadius={1}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              {getComplianceIcon()}
              <Typography variant="subtitle2">
                {standard.name} Compliance Status
              </Typography>
            </Box>

            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Integrated LUFS:
                </Typography>
                <Typography variant="body2">
                  {currentMeasurement.integrated_lufs.toFixed(1)} /{" "}
                  {currentTarget} LUFS
                </Typography>
              </Grid>

              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Loudness Units:
                </Typography>
                <Typography variant="body2">
                  {currentMeasurement.loudness_units > 0 ? "+" : ""}
                  {currentMeasurement.loudness_units.toFixed(1)} LU
                </Typography>
              </Grid>

              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  True Peak:
                </Typography>
                <Typography variant="body2">
                  {currentMeasurement.true_peak_db.toFixed(1)} /{" "}
                  {standard.max_true_peak} dBTP
                </Typography>
              </Grid>

              <Grid item xs={6} md={3}>
                <Typography variant="caption" color="text.secondary">
                  Range:
                </Typography>
                <Typography variant="body2">
                  {currentMeasurement.lufs_range.toFixed(1)} /{" "}
                  {standard.max_range} LU
                </Typography>
              </Grid>
            </Grid>

            {Math.abs(currentMeasurement.recommended_adjustment) > 0.5 && (
              <Alert severity="info" sx={{ mt: 1 }}>
                <Typography variant="body2">
                  Recommended adjustment:{" "}
                  {currentMeasurement.recommended_adjustment > 0 ? "+" : ""}
                  {currentMeasurement.recommended_adjustment.toFixed(1)} dB to
                  meet target
                </Typography>
              </Alert>
            )}
          </Box>
        )}

        {/* Technical Info */}
        <Box mt={2} textAlign="center">
          <Typography variant="caption" color="text.secondary">
            Standard: {standard.description} | Target: {currentTarget} LUFS |
            Max Peak: {standard.max_true_peak} dBTP | Max Range:{" "}
            {standard.max_range} LU
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default LUFSMeter;
