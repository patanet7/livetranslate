import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Switch,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RestoreIcon from "@mui/icons-material/Restore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";

interface CorrelationConfig {
  // General Correlation Settings
  general: {
    enabled: boolean;
    correlation_mode: "manual" | "automatic" | "hybrid";
    fallback_to_acoustic: boolean;
    confidence_threshold: number;
    auto_correlation_timeout: number;
  };

  // Manual Correlation Settings
  manual: {
    enabled: boolean;
    allow_manual_override: boolean;
    manual_mapping_priority: boolean;
    require_confirmation: boolean;
    default_speaker_names: string[];
  };

  // Acoustic Correlation Settings
  acoustic: {
    enabled: boolean;
    algorithm: "cosine_similarity" | "euclidean_distance" | "neural_embedding";
    similarity_threshold: number;
    voice_embedding_model: string;
    speaker_identification_confidence: number;
    adaptive_threshold: boolean;
  };

  // Google Meet Integration
  google_meet: {
    enabled: boolean;
    api_correlation: boolean;
    caption_correlation: boolean;
    participant_matching: boolean;
    use_display_names: boolean;
    fallback_on_api_failure: boolean;
  };

  // Timing and Synchronization
  timing: {
    time_drift_correction: boolean;
    max_time_drift_ms: number;
    correlation_window_ms: number;
    timestamp_alignment: "strict" | "flexible" | "adaptive";
    sync_quality_threshold: number;
  };

  // Database Storage
  database: {
    store_correlations: boolean;
    store_confidence_scores: boolean;
    store_timing_data: boolean;
    correlation_history: boolean;
    performance_metrics: boolean;
  };

  // Performance Settings
  performance: {
    max_concurrent_correlations: number;
    correlation_timeout_ms: number;
    cache_correlations: boolean;
    cache_duration_minutes: number;
    batch_processing: boolean;
  };
}

interface ManualSpeakerMapping {
  whisper_speaker_id: string;
  google_meet_speaker_id: string;
  display_name: string;
  confidence: number;
  is_confirmed: boolean;
}

interface CorrelationSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const defaultCorrelationConfig: CorrelationConfig = {
  general: {
    enabled: true,
    correlation_mode: "hybrid",
    fallback_to_acoustic: true,
    confidence_threshold: 0.7,
    auto_correlation_timeout: 30000,
  },
  manual: {
    enabled: true,
    allow_manual_override: true,
    manual_mapping_priority: true,
    require_confirmation: false,
    default_speaker_names: ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"],
  },
  acoustic: {
    enabled: true,
    algorithm: "cosine_similarity",
    similarity_threshold: 0.8,
    voice_embedding_model: "resemblyzer",
    speaker_identification_confidence: 0.75,
    adaptive_threshold: true,
  },
  google_meet: {
    enabled: true,
    api_correlation: true,
    caption_correlation: true,
    participant_matching: true,
    use_display_names: true,
    fallback_on_api_failure: true,
  },
  timing: {
    time_drift_correction: true,
    max_time_drift_ms: 1000,
    correlation_window_ms: 5000,
    timestamp_alignment: "adaptive",
    sync_quality_threshold: 0.8,
  },
  database: {
    store_correlations: true,
    store_confidence_scores: true,
    store_timing_data: true,
    correlation_history: true,
    performance_metrics: true,
  },
  performance: {
    max_concurrent_correlations: 5,
    correlation_timeout_ms: 10000,
    cache_correlations: true,
    cache_duration_minutes: 30,
    batch_processing: true,
  },
};

const CorrelationSettings: React.FC<CorrelationSettingsProps> = ({
  onSave,
}) => {
  const [config, setConfig] = useState<CorrelationConfig>(
    defaultCorrelationConfig,
  );
  const [manualMappings, setManualMappings] = useState<ManualSpeakerMapping[]>(
    [],
  );
  const [correlationStats, setCorrelationStats] = useState({
    total_correlations: 0,
    successful_correlations: 0,
    manual_correlations: 0,
    acoustic_correlations: 0,
    average_confidence: 0,
  });
  const [editMappingDialog, setEditMappingDialog] = useState(false);
  const [editingMapping, setEditingMapping] =
    useState<ManualSpeakerMapping | null>(null);

  // Load current configuration
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch("/api/settings/correlation");
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultCorrelationConfig, ...currentConfig });
        }
      } catch (error) {
        console.error("Failed to load correlation configuration:", error);
      }
    };

    const loadMappings = async () => {
      try {
        const response = await fetch(
          "/api/settings/correlation/manual-mappings",
        );
        if (response.ok) {
          const mappings = await response.json();
          setManualMappings(mappings);
        }
      } catch (error) {
        console.error("Failed to load manual mappings:", error);
      }
    };

    const loadStats = async () => {
      try {
        const response = await fetch("/api/settings/correlation/stats");
        if (response.ok) {
          const stats = await response.json();
          setCorrelationStats(stats);
        }
      } catch (error) {
        console.error("Failed to load correlation stats:", error);
      }
    };

    loadConfiguration();
    loadMappings();
    loadStats();
  }, []);

  const handleConfigChange = (
    section: keyof CorrelationConfig,
    key: string,
    value: any,
  ) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  const handleSave = async () => {
    try {
      const response = await fetch("/api/settings/correlation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        onSave("Speaker correlation settings saved successfully");
      } else {
        onSave("Failed to save speaker correlation settings", false);
      }
    } catch (error) {
      onSave("Error saving speaker correlation settings", false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultCorrelationConfig);
    onSave("Correlation configuration reset to defaults");
  };

  const handleAddManualMapping = () => {
    setEditingMapping({
      whisper_speaker_id: "",
      google_meet_speaker_id: "",
      display_name: "",
      confidence: 1.0,
      is_confirmed: false,
    });
    setEditMappingDialog(true);
  };

  const handleEditMapping = (mapping: ManualSpeakerMapping) => {
    setEditingMapping(mapping);
    setEditMappingDialog(true);
  };

  const handleSaveMapping = async () => {
    if (!editingMapping) return;

    try {
      const response = await fetch(
        "/api/settings/correlation/manual-mappings",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(editingMapping),
        },
      );

      if (response.ok) {
        // Reload mappings
        const mappingsResponse = await fetch(
          "/api/settings/correlation/manual-mappings",
        );
        if (mappingsResponse.ok) {
          const mappings = await mappingsResponse.json();
          setManualMappings(mappings);
        }
        onSave("Manual speaker mapping saved successfully");
        setEditMappingDialog(false);
      } else {
        onSave("Failed to save manual speaker mapping", false);
      }
    } catch (error) {
      onSave("Error saving manual speaker mapping", false);
    }
  };

  const handleDeleteMapping = async (mappingId: string) => {
    try {
      const response = await fetch(
        `/api/settings/correlation/manual-mappings/${mappingId}`,
        {
          method: "DELETE",
        },
      );

      if (response.ok) {
        setManualMappings((prev) =>
          prev.filter((m) => m.whisper_speaker_id !== mappingId),
        );
        onSave("Manual speaker mapping deleted successfully");
      } else {
        onSave("Failed to delete manual speaker mapping", false);
      }
    } catch (error) {
      onSave("Error deleting manual speaker mapping", false);
    }
  };

  const handleTestCorrelation = async () => {
    try {
      const response = await fetch("/api/settings/correlation/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        onSave("Speaker correlation test completed successfully");
      } else {
        onSave("Speaker correlation test failed", false);
      }
    } catch (error) {
      onSave("Error testing speaker correlation", false);
    }
  };

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={3}
      >
        <Typography variant="h5" component="h2">
          Speaker Correlation Configuration
        </Typography>
        <Box>
          <Tooltip title="Reset to defaults">
            <IconButton onClick={handleResetToDefaults} color="secondary">
              <RestoreIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<PlayArrowIcon />}
            onClick={handleTestCorrelation}
            sx={{ mr: 2 }}
          >
            Test Correlation
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure speaker correlation between Whisper transcriptions and Google
        Meet participants. This system links speaker IDs to actual participant
        names for accurate attribution.
      </Alert>

      {/* Correlation Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Correlation Statistics" />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {correlationStats.total_correlations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Correlations
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="success.main">
                  {correlationStats.successful_correlations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Successful
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="info.main">
                  {correlationStats.manual_correlations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Manual
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="secondary.main">
                  {correlationStats.acoustic_correlations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Acoustic
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {(correlationStats.average_confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Confidence
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* General Configuration */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">General Correlation Settings</Typography>
          <Chip
            label={config.general.enabled ? "Enabled" : "Disabled"}
            color={config.general.enabled ? "success" : "default"}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.general.enabled}
                    onChange={(e) =>
                      handleConfigChange("general", "enabled", e.target.checked)
                    }
                  />
                }
                label="Enable Speaker Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Correlation Mode</InputLabel>
                <Select
                  value={config.general.correlation_mode}
                  label="Correlation Mode"
                  onChange={(e) =>
                    handleConfigChange(
                      "general",
                      "correlation_mode",
                      e.target.value,
                    )
                  }
                  disabled={!config.general.enabled}
                >
                  <MenuItem value="manual">Manual Only</MenuItem>
                  <MenuItem value="automatic">Automatic Only</MenuItem>
                  <MenuItem value="hybrid">
                    Hybrid (Manual + Automatic)
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.general.fallback_to_acoustic}
                    onChange={(e) =>
                      handleConfigChange(
                        "general",
                        "fallback_to_acoustic",
                        e.target.checked,
                      )
                    }
                    disabled={!config.general.enabled}
                  />
                }
                label="Fallback to Acoustic Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Confidence Threshold: {config.general.confidence_threshold}
              </Typography>
              <Slider
                value={config.general.confidence_threshold}
                onChange={(_, value) =>
                  handleConfigChange("general", "confidence_threshold", value)
                }
                min={0}
                max={1}
                step={0.1}
                disabled={!config.general.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Auto Correlation Timeout (ms)"
                type="number"
                value={config.general.auto_correlation_timeout}
                onChange={(e) =>
                  handleConfigChange(
                    "general",
                    "auto_correlation_timeout",
                    Number(e.target.value),
                  )
                }
                disabled={!config.general.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Manual Correlation */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Manual Speaker Mappings</Typography>
          <Chip
            label={`${manualMappings.length} mappings`}
            color="info"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.manual.enabled}
                    onChange={(e) =>
                      handleConfigChange("manual", "enabled", e.target.checked)
                    }
                  />
                }
                label="Enable Manual Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.manual.allow_manual_override}
                    onChange={(e) =>
                      handleConfigChange(
                        "manual",
                        "allow_manual_override",
                        e.target.checked,
                      )
                    }
                    disabled={!config.manual.enabled}
                  />
                }
                label="Allow Manual Override"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.manual.manual_mapping_priority}
                    onChange={(e) =>
                      handleConfigChange(
                        "manual",
                        "manual_mapping_priority",
                        e.target.checked,
                      )
                    }
                    disabled={!config.manual.enabled}
                  />
                }
                label="Manual Mapping Priority"
              />
            </Grid>
            <Grid item xs={12}>
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                mb={2}
              >
                <Typography variant="h6">Current Manual Mappings</Typography>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={handleAddManualMapping}
                  disabled={!config.manual.enabled}
                >
                  Add Mapping
                </Button>
              </Box>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Whisper Speaker</TableCell>
                      <TableCell>Google Meet Speaker</TableCell>
                      <TableCell>Display Name</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {manualMappings.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6} align="center">
                          <Typography color="text.secondary">
                            No manual mappings configured
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      manualMappings.map((mapping) => (
                        <TableRow key={mapping.whisper_speaker_id}>
                          <TableCell>{mapping.whisper_speaker_id}</TableCell>
                          <TableCell>
                            {mapping.google_meet_speaker_id}
                          </TableCell>
                          <TableCell>{mapping.display_name}</TableCell>
                          <TableCell>
                            {(mapping.confidence * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={
                                mapping.is_confirmed ? "Confirmed" : "Pending"
                              }
                              color={
                                mapping.is_confirmed ? "success" : "warning"
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            <IconButton
                              size="small"
                              onClick={() => handleEditMapping(mapping)}
                              disabled={!config.manual.enabled}
                            >
                              <EditIcon />
                            </IconButton>
                            <IconButton
                              size="small"
                              onClick={() =>
                                handleDeleteMapping(mapping.whisper_speaker_id)
                              }
                              disabled={!config.manual.enabled}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Acoustic Correlation */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Acoustic Correlation</Typography>
          <Chip
            label={config.acoustic.enabled ? "Enabled" : "Disabled"}
            color={config.acoustic.enabled ? "success" : "default"}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.acoustic.enabled}
                    onChange={(e) =>
                      handleConfigChange(
                        "acoustic",
                        "enabled",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Acoustic Speaker Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Correlation Algorithm</InputLabel>
                <Select
                  value={config.acoustic.algorithm}
                  label="Correlation Algorithm"
                  onChange={(e) =>
                    handleConfigChange("acoustic", "algorithm", e.target.value)
                  }
                  disabled={!config.acoustic.enabled}
                >
                  <MenuItem value="cosine_similarity">
                    Cosine Similarity
                  </MenuItem>
                  <MenuItem value="euclidean_distance">
                    Euclidean Distance
                  </MenuItem>
                  <MenuItem value="neural_embedding">Neural Embedding</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Voice Embedding Model</InputLabel>
                <Select
                  value={config.acoustic.voice_embedding_model}
                  label="Voice Embedding Model"
                  onChange={(e) =>
                    handleConfigChange(
                      "acoustic",
                      "voice_embedding_model",
                      e.target.value,
                    )
                  }
                  disabled={!config.acoustic.enabled}
                >
                  <MenuItem value="resemblyzer">Resemblyzer</MenuItem>
                  <MenuItem value="speechbrain">SpeechBrain</MenuItem>
                  <MenuItem value="pyannote">Pyannote Audio</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Similarity Threshold: {config.acoustic.similarity_threshold}
              </Typography>
              <Slider
                value={config.acoustic.similarity_threshold}
                onChange={(_, value) =>
                  handleConfigChange("acoustic", "similarity_threshold", value)
                }
                min={0}
                max={1}
                step={0.05}
                disabled={!config.acoustic.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Speaker ID Confidence:{" "}
                {config.acoustic.speaker_identification_confidence}
              </Typography>
              <Slider
                value={config.acoustic.speaker_identification_confidence}
                onChange={(_, value) =>
                  handleConfigChange(
                    "acoustic",
                    "speaker_identification_confidence",
                    value,
                  )
                }
                min={0}
                max={1}
                step={0.05}
                disabled={!config.acoustic.enabled}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.acoustic.adaptive_threshold}
                    onChange={(e) =>
                      handleConfigChange(
                        "acoustic",
                        "adaptive_threshold",
                        e.target.checked,
                      )
                    }
                    disabled={!config.acoustic.enabled}
                  />
                }
                label="Adaptive Threshold (Auto-adjust based on audio quality)"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Google Meet Integration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Google Meet Integration</Typography>
          <Chip
            label={config.google_meet.enabled ? "Enabled" : "Disabled"}
            color={config.google_meet.enabled ? "success" : "default"}
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.enabled}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "enabled",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Google Meet Integration"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.api_correlation}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "api_correlation",
                        e.target.checked,
                      )
                    }
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="API-based Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.caption_correlation}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "caption_correlation",
                        e.target.checked,
                      )
                    }
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Caption-based Correlation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.participant_matching}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "participant_matching",
                        e.target.checked,
                      )
                    }
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Participant Matching"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.use_display_names}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "use_display_names",
                        e.target.checked,
                      )
                    }
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Use Display Names"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.google_meet.fallback_on_api_failure}
                    onChange={(e) =>
                      handleConfigChange(
                        "google_meet",
                        "fallback_on_api_failure",
                        e.target.checked,
                      )
                    }
                    disabled={!config.google_meet.enabled}
                  />
                }
                label="Fallback on API Failure"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Timing and Synchronization */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Timing and Synchronization</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.timing.time_drift_correction}
                    onChange={(e) =>
                      handleConfigChange(
                        "timing",
                        "time_drift_correction",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Time Drift Correction"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Time Drift (ms)"
                type="number"
                value={config.timing.max_time_drift_ms}
                onChange={(e) =>
                  handleConfigChange(
                    "timing",
                    "max_time_drift_ms",
                    Number(e.target.value),
                  )
                }
                disabled={!config.timing.time_drift_correction}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Correlation Window (ms)"
                type="number"
                value={config.timing.correlation_window_ms}
                onChange={(e) =>
                  handleConfigChange(
                    "timing",
                    "correlation_window_ms",
                    Number(e.target.value),
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Timestamp Alignment</InputLabel>
                <Select
                  value={config.timing.timestamp_alignment}
                  label="Timestamp Alignment"
                  onChange={(e) =>
                    handleConfigChange(
                      "timing",
                      "timestamp_alignment",
                      e.target.value,
                    )
                  }
                >
                  <MenuItem value="strict">Strict</MenuItem>
                  <MenuItem value="flexible">Flexible</MenuItem>
                  <MenuItem value="adaptive">Adaptive</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Sync Quality Threshold: {config.timing.sync_quality_threshold}
              </Typography>
              <Slider
                value={config.timing.sync_quality_threshold}
                onChange={(_, value) =>
                  handleConfigChange("timing", "sync_quality_threshold", value)
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Manual Mapping Dialog */}
      <Dialog
        open={editMappingDialog}
        onClose={() => setEditMappingDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingMapping?.whisper_speaker_id
            ? "Edit Speaker Mapping"
            : "Add Speaker Mapping"}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Whisper Speaker ID"
                value={editingMapping?.whisper_speaker_id || ""}
                onChange={(e) =>
                  setEditingMapping((prev) =>
                    prev
                      ? { ...prev, whisper_speaker_id: e.target.value }
                      : null,
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Google Meet Speaker ID"
                value={editingMapping?.google_meet_speaker_id || ""}
                onChange={(e) =>
                  setEditingMapping((prev) =>
                    prev
                      ? { ...prev, google_meet_speaker_id: e.target.value }
                      : null,
                  )
                }
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Display Name"
                value={editingMapping?.display_name || ""}
                onChange={(e) =>
                  setEditingMapping((prev) =>
                    prev ? { ...prev, display_name: e.target.value } : null,
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Confidence:{" "}
                {((editingMapping?.confidence || 0) * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={editingMapping?.confidence || 0}
                onChange={(_, value) =>
                  setEditingMapping((prev) =>
                    prev ? { ...prev, confidence: value as number } : null,
                  )
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editingMapping?.is_confirmed || false}
                    onChange={(e) =>
                      setEditingMapping((prev) =>
                        prev
                          ? { ...prev, is_confirmed: e.target.checked }
                          : null,
                      )
                    }
                  />
                }
                label="Confirmed Mapping"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditMappingDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveMapping} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CorrelationSettings;
