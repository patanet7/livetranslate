import React from "react";
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Stack,
  IconButton,
  Collapse,
  Paper,
} from "@mui/material";
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as ProcessingIcon,
} from "@mui/icons-material";
import { ProcessingStage } from "@/types/audio";

interface PipelineProcessingProps {
  onRunPipeline: () => void;
  onRunStepByStep: () => void;
  onResetPipeline: () => void;
  onExportResults: () => void;
  isProcessing: boolean;
  progress: number;
  hasRecording: boolean;
  stages: ProcessingStage[];
}

// 10-Stage Meeting Audio Processing Pipeline (matches orchestration service)
const pipelineStages = [
  {
    id: "original_audio",
    name: "Original Audio Analysis",
    description: "Input audio validation and initial analysis",
    icon: "🎤",
    metrics: [
      "Input Level (dB)",
      "Sample Rate",
      "Duration (s)",
      "Format",
      "Clipping Detection",
    ],
  },
  {
    id: "decoded_audio",
    name: "Audio Decoding",
    description: "Decode and prepare audio for processing",
    icon: "🔧",
    metrics: [
      "Decode Time (ms)",
      "Memory Usage",
      "Channel Count",
      "Bit Depth",
      "Compression Ratio",
    ],
  },
  {
    id: "voice_frequency_filter",
    name: "Voice Frequency Filter",
    description: "Focus on human speech frequencies (85-300Hz fundamental)",
    icon: "🎵",
    metrics: [
      "Filter Range (Hz)",
      "Voice Band Gain",
      "Frequency Response",
      "Formant Preservation",
      "Sibilance Enhancement",
    ],
  },
  {
    id: "voice_activity_detection",
    name: "Voice Activity Detection",
    description: "Detect and segment speech vs silence",
    icon: "🗣️",
    metrics: [
      "Speech Segments",
      "Voice Activity Ratio",
      "Silence Removed (ms)",
      "VAD Confidence",
      "Energy Threshold",
    ],
  },
  {
    id: "voice_aware_noise_reduction",
    name: "Voice-Aware Noise Reduction",
    description: "Remove noise while preserving speech characteristics",
    icon: "🔇",
    metrics: [
      "Noise Reduction (dB)",
      "Speech Preservation",
      "SNR Improvement",
      "Artifact Detection",
      "Voice Protection",
    ],
  },
  {
    id: "voice_enhancement",
    name: "Voice Enhancement",
    description: "Enhance clarity and intelligibility for meetings",
    icon: "✨",
    metrics: [
      "Clarity Score",
      "Compression Ratio",
      "Gain Applied (dB)",
      "Presence Boost",
      "Dynamic Range",
    ],
  },
  {
    id: "advanced_voice_processing",
    name: "Advanced Voice Processing",
    description: "Meeting-specific voice optimization",
    icon: "🎯",
    metrics: [
      "De-essing (dB)",
      "EQ Adjustment",
      "Warmth Factor",
      "Articulation",
      "Meeting Optimization",
    ],
  },
  {
    id: "voice_aware_silence_trimming",
    name: "Intelligent Silence Trimming",
    description: "Remove excessive silence while preserving natural pauses",
    icon: "✂️",
    metrics: [
      "Trimmed Duration (ms)",
      "Pause Preservation",
      "Natural Rhythm",
      "Speaking Rate",
      "Silence Threshold",
    ],
  },
  {
    id: "high_quality_resampling",
    name: "High-Quality Resampling",
    description: "Resample to target rate with anti-aliasing",
    icon: "🔄",
    metrics: [
      "Target Sample Rate",
      "Resampling Quality",
      "Anti-aliasing",
      "Phase Coherence",
      "Frequency Response",
    ],
  },
  {
    id: "final_output",
    name: "Final Output",
    description: "Quality validation and final processing",
    icon: "📤",
    metrics: [
      "Final Level (dB)",
      "Quality Score",
      "Processing Time (ms)",
      "File Size",
      "Meeting Readiness",
    ],
  },
];

const StageCard: React.FC<{
  stage: (typeof pipelineStages)[0];
  stageState?: ProcessingStage;
  isActive: boolean;
}> = ({ stage, stageState, isActive }) => {
  const [expanded, setExpanded] = React.useState(false);

  const getStatusIcon = () => {
    if (!stageState) return <ProcessingIcon color="disabled" />;

    switch (stageState.status) {
      case "completed":
        return <CheckCircleIcon color="success" />;
      case "processing":
        return (
          <ProcessingIcon
            color="primary"
            sx={{ animation: "spin 1s linear infinite" }}
          />
        );
      case "error":
        return <ErrorIcon color="error" />;
      default:
        return <ProcessingIcon color="disabled" />;
    }
  };

  const getStatusColor = () => {
    if (!stageState) return "default";

    switch (stageState.status) {
      case "completed":
        return "success";
      case "processing":
        return "primary";
      case "error":
        return "error";
      default:
        return "default";
    }
  };

  return (
    <Card
      sx={{
        border: isActive ? 2 : 1,
        borderColor: isActive ? "primary.main" : "divider",
        transition: "all 0.3s ease",
      }}
    >
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              backgroundColor: "primary.main",
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: "bold",
            }}
          >
            {stage.icon}
          </Box>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" component="h3">
              {stage.name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {stage.description}
            </Typography>
          </Box>
          {getStatusIcon()}
        </Stack>

        <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
          <Chip
            label={stageState?.status || "Ready"}
            color={getStatusColor()}
            size="small"
            variant="outlined"
          />
          {stageState?.processingTime && (
            <Chip
              label={`${stageState.processingTime}ms`}
              size="small"
              variant="outlined"
            />
          )}
        </Stack>
      </CardContent>

      <CardActions sx={{ justifyContent: "space-between" }}>
        <Stack direction="row" spacing={1}>
          <Button
            size="small"
            startIcon={<PlayIcon />}
            disabled={!stageState || stageState.status !== "completed"}
            onClick={() => {
              // TODO: Implement individual stage replay
              console.log(`Replaying stage: ${stage.id}`);
            }}
          >
            Replay
          </Button>
          <Button
            size="small"
            startIcon={<PlayIcon />}
            disabled={!stageState || stageState.status !== "completed"}
            onClick={() => {
              // TODO: Implement stage audio playback
              console.log(`Playing audio from stage: ${stage.id}`);
            }}
          >
            Play Audio
          </Button>
          <Button
            size="small"
            startIcon={<DownloadIcon />}
            disabled={!stageState || stageState.status !== "completed"}
            onClick={() => {
              // TODO: Implement stage result export
              console.log(`Exporting stage: ${stage.id}`);
            }}
          >
            Export
          </Button>
        </Stack>

        <IconButton
          size="small"
          onClick={() => setExpanded(!expanded)}
          disabled={!stageState}
          sx={{
            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.3s",
          }}
        >
          <ExpandMoreIcon />
        </IconButton>
      </CardActions>

      <Collapse in={expanded && !!stageState}>
        <CardContent sx={{ pt: 0 }}>
          <Typography variant="subtitle2" gutterBottom>
            📊 Processing Metrics
          </Typography>
          <Grid container spacing={1} sx={{ mb: 2 }}>
            {stage.metrics.map((metric) => (
              <Grid item xs={6} key={metric}>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption">{metric}:</Typography>
                  <Typography variant="caption" fontWeight="bold">
                    {stageState?.metrics?.[metric] || "--"}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>

          {/* Stage-specific results from orchestration service */}
          {stageState?.result && (
            <>
              <Typography variant="subtitle2" gutterBottom>
                🔍 Stage Results
              </Typography>
              <Paper sx={{ p: 1.5, mb: 2, backgroundColor: "grey.50" }}>
                <Grid container spacing={1}>
                  {stageState.result.input_level_db && (
                    <Grid item xs={6}>
                      <Typography variant="caption">Input Level:</Typography>
                      <Typography
                        variant="caption"
                        fontWeight="bold"
                        sx={{ ml: 1 }}
                      >
                        {stageState.result.input_level_db.toFixed(1)} dB
                      </Typography>
                    </Grid>
                  )}
                  {stageState.result.output_level_db && (
                    <Grid item xs={6}>
                      <Typography variant="caption">Output Level:</Typography>
                      <Typography
                        variant="caption"
                        fontWeight="bold"
                        sx={{ ml: 1 }}
                      >
                        {stageState.result.output_level_db.toFixed(1)} dB
                      </Typography>
                    </Grid>
                  )}
                  {stageState.result.quality_score && (
                    <Grid item xs={6}>
                      <Typography variant="caption">Quality Score:</Typography>
                      <Typography
                        variant="caption"
                        fontWeight="bold"
                        sx={{ ml: 1 }}
                      >
                        {(stageState.result.quality_score * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                  )}
                  {stageState.result.artifacts_detected !== undefined && (
                    <Grid item xs={6}>
                      <Typography variant="caption">Artifacts:</Typography>
                      <Typography
                        variant="caption"
                        fontWeight="bold"
                        sx={{ ml: 1 }}
                      >
                        {stageState.result.artifacts_detected
                          ? "⚠️ Detected"
                          : "✅ None"}
                      </Typography>
                    </Grid>
                  )}
                </Grid>

                {/* Stage-specific data */}
                {stageState.result.stage_specific_data && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" fontWeight="bold">
                      Stage Data:
                    </Typography>
                    <Grid container spacing={1} sx={{ mt: 0.5 }}>
                      {Object.entries(
                        stageState.result.stage_specific_data,
                      ).map(([key, value]) => (
                        <Grid item xs={12} key={key}>
                          <Box
                            sx={{
                              display: "flex",
                              justifyContent: "space-between",
                            }}
                          >
                            <Typography variant="caption">
                              {key.replace(/_/g, " ")}:
                            </Typography>
                            <Typography variant="caption" fontWeight="bold">
                              {typeof value === "number"
                                ? value.toFixed(2)
                                : String(value)}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}
              </Paper>
            </>
          )}

          {/* Error details */}
          {stageState?.error && (
            <>
              <Typography variant="subtitle2" gutterBottom color="error">
                ❌ Error Details
              </Typography>
              <Paper
                sx={{
                  p: 1.5,
                  backgroundColor: "error.light",
                  color: "error.contrastText",
                }}
              >
                <Typography variant="caption">{stageState.error}</Typography>
              </Paper>
            </>
          )}
        </CardContent>
      </Collapse>
    </Card>
  );
};

export const PipelineProcessing: React.FC<PipelineProcessingProps> = ({
  onRunPipeline,
  onRunStepByStep,
  onResetPipeline,
  onExportResults,
  isProcessing,
  progress,
  hasRecording,
  stages,
}) => {
  return (
    <Box>
      {/* Pipeline Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          🔧 Audio Processing Pipeline
        </Typography>

        <Stack direction="row" spacing={2} sx={{ mb: 3 }} flexWrap="wrap">
          <Button
            variant="contained"
            startIcon={<PlayIcon />}
            onClick={onRunPipeline}
            disabled={!hasRecording || isProcessing}
            size="large"
          >
            🚀 Run Full Pipeline
          </Button>
          <Button
            variant="outlined"
            startIcon={<PlayIcon />}
            onClick={onRunStepByStep}
            disabled={!hasRecording || isProcessing}
          >
            ➡️ Run Step by Step
          </Button>
          <Button
            variant="outlined"
            startIcon={<StopIcon />}
            disabled={!isProcessing}
          >
            ⏸️ Pause Pipeline
          </Button>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={onResetPipeline}
          >
            🔄 Reset Pipeline
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={onExportResults}
            disabled={stages.length === 0}
          >
            📤 Export Results
          </Button>
        </Stack>

        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
            <Typography variant="body2">Processing Progress</Typography>
            <Typography variant="body2" fontWeight="bold">
              {Math.round(progress * 100)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={progress * 100}
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="caption" color="text.secondary">
            {isProcessing
              ? `Processing: ${Math.round(progress * 100)}% complete`
              : hasRecording
                ? "Ready to process audio"
                : "Please record or load audio first"}
          </Typography>
        </Box>

        {!hasRecording && (
          <Box
            sx={{
              p: 2,
              backgroundColor: "warning.light",
              borderRadius: 1,
              mb: 2,
            }}
          >
            <Typography variant="body2" color="warning.contrastText">
              ⚠️ No audio available for processing. Please record audio or
              upload a file first.
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Pipeline Summary (when processing is complete) */}
      {stages.length > 0 && stages.some((s) => s.status === "completed") && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            📈 Pipeline Summary
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h4" color="primary.main">
                  {stages.filter((s) => s.status === "completed").length}
                </Typography>
                <Typography variant="caption">Stages Completed</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h4" color="success.main">
                  {stages
                    .reduce(
                      (total, stage) => total + (stage.processingTime || 0),
                      0,
                    )
                    .toFixed(0)}
                  ms
                </Typography>
                <Typography variant="caption">Total Processing Time</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h4" color="warning.main">
                  {stages.filter((s) => s.status === "error").length}
                </Typography>
                <Typography variant="caption">Errors</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h4" color="info.main">
                  {stages.length > 0
                    ? (
                        (stages.filter((s) => s.status === "completed").length /
                          stages.length) *
                        100
                      ).toFixed(0) + "%"
                    : "0%"}
                </Typography>
                <Typography variant="caption">Success Rate</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Pipeline Stages */}
      <Typography variant="h6" gutterBottom>
        🔄 Processing Stages ({pipelineStages.length} Total)
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Each stage processes audio with specific optimizations for meeting voice
        isolation and clarity enhancement.
      </Typography>
      <Grid container spacing={2}>
        {pipelineStages.map((stage, index) => {
          const stageState = stages.find((s) => s.id === stage.id);
          const isActive =
            isProcessing &&
            Math.floor(progress * pipelineStages.length) === index;

          return (
            <Grid item xs={12} md={6} lg={4} key={stage.id}>
              <StageCard
                stage={stage}
                stageState={stageState}
                isActive={isActive}
              />
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};
