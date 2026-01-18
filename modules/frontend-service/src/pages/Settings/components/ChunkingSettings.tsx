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
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RestoreIcon from "@mui/icons-material/Restore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import InfoIcon from "@mui/icons-material/Info";
import StorageIcon from "@mui/icons-material/Storage";

interface ChunkingConfig {
  // Basic Chunking Configuration
  chunking: {
    chunk_duration: number;
    overlap_duration: number;
    overlap_mode: "fixed" | "adaptive" | "voice_based";
    min_chunk_duration: number;
    max_chunk_duration: number;
    buffer_duration: number;
    silence_threshold: number;
  };

  // Quality and Processing
  quality: {
    quality_threshold: number;
    auto_reject_low_quality: boolean;
    quality_analysis_enabled: boolean;
    snr_threshold: number;
    voice_activity_threshold: number;
  };

  // File Management
  storage: {
    audio_storage_path: string;
    file_format: "wav" | "mp3" | "flac" | "ogg";
    compression_level: number;
    cleanup_files_on_stop: boolean;
    max_storage_size_gb: number;
    retention_days: number;
  };

  // Performance Configuration
  performance: {
    max_concurrent_chunks: number;
    processing_threads: number;
    memory_limit_mb: number;
    batch_processing: boolean;
    adaptive_buffering: boolean;
  };

  // Database Configuration
  database: {
    chunk_metadata_enabled: boolean;
    store_quality_metrics: boolean;
    store_processing_stats: boolean;
    correlation_tracking: boolean;
    speaker_tracking: boolean;
  };

  // Source Configuration
  source: {
    source_type: "bot_audio" | "meeting_test" | "microphone" | "file_upload";
    session_timeout: number;
    auto_restart: boolean;
    error_recovery: boolean;
    max_retries: number;
  };
}

interface ChunkingSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const defaultChunkingConfig: ChunkingConfig = {
  chunking: {
    chunk_duration: 3.0,
    overlap_duration: 0.5,
    overlap_mode: "fixed",
    min_chunk_duration: 1.0,
    max_chunk_duration: 10.0,
    buffer_duration: 5.0,
    silence_threshold: 0.001,
  },
  quality: {
    quality_threshold: 0.3,
    auto_reject_low_quality: false,
    quality_analysis_enabled: true,
    snr_threshold: 10.0,
    voice_activity_threshold: 0.5,
  },
  storage: {
    audio_storage_path: "/data/audio",
    file_format: "wav",
    compression_level: 5,
    cleanup_files_on_stop: false,
    max_storage_size_gb: 10,
    retention_days: 30,
  },
  performance: {
    max_concurrent_chunks: 5,
    processing_threads: 2,
    memory_limit_mb: 512,
    batch_processing: true,
    adaptive_buffering: true,
  },
  database: {
    chunk_metadata_enabled: true,
    store_quality_metrics: true,
    store_processing_stats: true,
    correlation_tracking: true,
    speaker_tracking: true,
  },
  source: {
    source_type: "bot_audio",
    session_timeout: 3600,
    auto_restart: true,
    error_recovery: true,
    max_retries: 3,
  },
};

const ChunkingSettings: React.FC<ChunkingSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<ChunkingConfig>(defaultChunkingConfig);
  const [testingChunking, setTestingChunking] = useState(false);
  const [storageStats, setStorageStats] = useState({
    used_space_gb: 0,
    total_chunks: 0,
    avg_chunk_size_mb: 0,
  });

  // Load current configuration
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch("/api/settings/chunking");
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultChunkingConfig, ...currentConfig });
        }
      } catch (error) {
        console.error("Failed to load chunking configuration:", error);
      }
    };

    const loadStorageStats = async () => {
      try {
        const response = await fetch("/api/settings/chunking/storage-stats");
        if (response.ok) {
          const stats = await response.json();
          setStorageStats(stats);
        }
      } catch (error) {
        console.error("Failed to load storage stats:", error);
      }
    };

    loadConfiguration();
    loadStorageStats();
  }, []);

  const handleConfigChange = (
    section: keyof ChunkingConfig,
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
      const response = await fetch("/api/settings/chunking", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        onSave("Chunking settings saved successfully");
      } else {
        onSave("Failed to save chunking settings", false);
      }
    } catch (error) {
      onSave("Error saving chunking settings", false);
    }
  };

  const handleTestChunking = async () => {
    setTestingChunking(true);
    try {
      const response = await fetch("/api/settings/chunking/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        onSave("Chunking configuration test completed successfully");
      } else {
        onSave("Chunking configuration test failed", false);
      }
    } catch (error) {
      onSave("Error testing chunking configuration", false);
    } finally {
      setTestingChunking(false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultChunkingConfig);
    onSave("Chunking configuration reset to defaults");
  };

  const handleCleanupStorage = async () => {
    try {
      const response = await fetch("/api/settings/chunking/cleanup-storage", {
        method: "POST",
      });

      if (response.ok) {
        onSave("Storage cleanup completed successfully");
        // Reload storage stats
        const statsResponse = await fetch(
          "/api/settings/chunking/storage-stats",
        );
        if (statsResponse.ok) {
          const stats = await statsResponse.json();
          setStorageStats(stats);
        }
      } else {
        onSave("Storage cleanup failed", false);
      }
    } catch (error) {
      onSave("Error during storage cleanup", false);
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
          Audio Chunking Configuration
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
            onClick={handleTestChunking}
            disabled={testingChunking}
            sx={{ mr: 2 }}
          >
            {testingChunking ? "Testing..." : "Test Configuration"}
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure audio chunking parameters for optimal processing performance
        and storage efficiency. Overlap handling is managed by the orchestration
        service.
      </Alert>

      {/* Storage Overview */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title="Storage Overview"
          action={
            <Button
              variant="outlined"
              startIcon={<StorageIcon />}
              onClick={handleCleanupStorage}
              size="small"
            >
              Cleanup Storage
            </Button>
          }
        />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {storageStats.used_space_gb.toFixed(2)} GB
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Used Space
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {storageStats.total_chunks.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Chunks
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {storageStats.avg_chunk_size_mb.toFixed(2)} MB
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Chunk Size
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Basic Chunking Configuration */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Basic Chunking Parameters</Typography>
          <Chip
            label={`${config.chunking.chunk_duration}s chunks`}
            color="primary"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Chunk Duration: {config.chunking.chunk_duration}s
                <Tooltip title="Duration of each audio chunk">
                  <InfoIcon
                    fontSize="small"
                    sx={{ ml: 1, color: "text.secondary" }}
                  />
                </Tooltip>
              </Typography>
              <Slider
                value={config.chunking.chunk_duration}
                onChange={(_, value) =>
                  handleConfigChange("chunking", "chunk_duration", value)
                }
                min={0.5}
                max={30}
                step={0.5}
                marks={[
                  { value: 1, label: "1s" },
                  { value: 3, label: "3s" },
                  { value: 5, label: "5s" },
                  { value: 10, label: "10s" },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Overlap Duration: {config.chunking.overlap_duration}s
                <Tooltip title="Overlap between consecutive chunks">
                  <InfoIcon
                    fontSize="small"
                    sx={{ ml: 1, color: "text.secondary" }}
                  />
                </Tooltip>
              </Typography>
              <Slider
                value={config.chunking.overlap_duration}
                onChange={(_, value) =>
                  handleConfigChange("chunking", "overlap_duration", value)
                }
                min={0}
                max={5}
                step={0.1}
                marks={[
                  { value: 0, label: "0s" },
                  { value: 0.5, label: "0.5s" },
                  { value: 1, label: "1s" },
                  { value: 2, label: "2s" },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Overlap Mode</InputLabel>
                <Select
                  value={config.chunking.overlap_mode}
                  label="Overlap Mode"
                  onChange={(e) =>
                    handleConfigChange(
                      "chunking",
                      "overlap_mode",
                      e.target.value,
                    )
                  }
                >
                  <MenuItem value="fixed">Fixed Overlap</MenuItem>
                  <MenuItem value="adaptive">Adaptive Overlap</MenuItem>
                  <MenuItem value="voice_based">Voice-based Overlap</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Buffer Duration: {config.chunking.buffer_duration}s
                <Tooltip title="Internal audio buffer size">
                  <InfoIcon
                    fontSize="small"
                    sx={{ ml: 1, color: "text.secondary" }}
                  />
                </Tooltip>
              </Typography>
              <Slider
                value={config.chunking.buffer_duration}
                onChange={(_, value) =>
                  handleConfigChange("chunking", "buffer_duration", value)
                }
                min={1}
                max={30}
                step={1}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Min Chunk Duration (s)"
                type="number"
                value={config.chunking.min_chunk_duration}
                onChange={(e) =>
                  handleConfigChange(
                    "chunking",
                    "min_chunk_duration",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 0.1, max: 10, step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Chunk Duration (s)"
                type="number"
                value={config.chunking.max_chunk_duration}
                onChange={(e) =>
                  handleConfigChange(
                    "chunking",
                    "max_chunk_duration",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 1, max: 60, step: 1 }}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Quality Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Quality and Filtering</Typography>
          <Chip
            label={
              config.quality.quality_analysis_enabled
                ? "Analysis Enabled"
                : "Analysis Disabled"
            }
            color={
              config.quality.quality_analysis_enabled ? "success" : "default"
            }
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
                    checked={config.quality.quality_analysis_enabled}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "quality_analysis_enabled",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Quality Analysis"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Quality Threshold: {config.quality.quality_threshold}
              </Typography>
              <Slider
                value={config.quality.quality_threshold}
                onChange={(_, value) =>
                  handleConfigChange("quality", "quality_threshold", value)
                }
                min={0}
                max={1}
                step={0.1}
                disabled={!config.quality.quality_analysis_enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                SNR Threshold: {config.quality.snr_threshold} dB
              </Typography>
              <Slider
                value={config.quality.snr_threshold}
                onChange={(_, value) =>
                  handleConfigChange("quality", "snr_threshold", value)
                }
                min={0}
                max={40}
                step={1}
                disabled={!config.quality.quality_analysis_enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.quality.auto_reject_low_quality}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "auto_reject_low_quality",
                        e.target.checked,
                      )
                    }
                    disabled={!config.quality.quality_analysis_enabled}
                  />
                }
                label="Auto-reject Low Quality Chunks"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Voice Activity Threshold:{" "}
                {config.quality.voice_activity_threshold}
              </Typography>
              <Slider
                value={config.quality.voice_activity_threshold}
                onChange={(_, value) =>
                  handleConfigChange(
                    "quality",
                    "voice_activity_threshold",
                    value,
                  )
                }
                min={0}
                max={1}
                step={0.1}
                disabled={!config.quality.quality_analysis_enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Storage Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Storage Configuration</Typography>
          <Chip
            label={config.storage.file_format.toUpperCase()}
            color="primary"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Audio Storage Path"
                value={config.storage.audio_storage_path}
                onChange={(e) =>
                  handleConfigChange(
                    "storage",
                    "audio_storage_path",
                    e.target.value,
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>File Format</InputLabel>
                <Select
                  value={config.storage.file_format}
                  label="File Format"
                  onChange={(e) =>
                    handleConfigChange("storage", "file_format", e.target.value)
                  }
                >
                  <MenuItem value="wav">WAV (Uncompressed)</MenuItem>
                  <MenuItem value="flac">FLAC (Lossless)</MenuItem>
                  <MenuItem value="mp3">MP3 (Compressed)</MenuItem>
                  <MenuItem value="ogg">OGG (Compressed)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Compression Level: {config.storage.compression_level}
              </Typography>
              <Slider
                value={config.storage.compression_level}
                onChange={(_, value) =>
                  handleConfigChange("storage", "compression_level", value)
                }
                min={0}
                max={9}
                step={1}
                disabled={config.storage.file_format === "wav"}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Storage Size (GB)"
                type="number"
                value={config.storage.max_storage_size_gb}
                onChange={(e) =>
                  handleConfigChange(
                    "storage",
                    "max_storage_size_gb",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 1, max: 1000 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Retention Days"
                type="number"
                value={config.storage.retention_days}
                onChange={(e) =>
                  handleConfigChange(
                    "storage",
                    "retention_days",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 1, max: 365 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.storage.cleanup_files_on_stop}
                    onChange={(e) =>
                      handleConfigChange(
                        "storage",
                        "cleanup_files_on_stop",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Cleanup Files on Session Stop"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Performance Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Performance Settings</Typography>
          <Chip
            label={`${config.performance.max_concurrent_chunks} concurrent`}
            color="secondary"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Max Concurrent Chunks:{" "}
                {config.performance.max_concurrent_chunks}
              </Typography>
              <Slider
                value={config.performance.max_concurrent_chunks}
                onChange={(_, value) =>
                  handleConfigChange(
                    "performance",
                    "max_concurrent_chunks",
                    value,
                  )
                }
                min={1}
                max={20}
                step={1}
                marks={[
                  { value: 1, label: "1" },
                  { value: 5, label: "5" },
                  { value: 10, label: "10" },
                  { value: 20, label: "20" },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Processing Threads: {config.performance.processing_threads}
              </Typography>
              <Slider
                value={config.performance.processing_threads}
                onChange={(_, value) =>
                  handleConfigChange("performance", "processing_threads", value)
                }
                min={1}
                max={8}
                step={1}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Memory Limit (MB)"
                type="number"
                value={config.performance.memory_limit_mb}
                onChange={(e) =>
                  handleConfigChange(
                    "performance",
                    "memory_limit_mb",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 128, max: 8192 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.batch_processing}
                    onChange={(e) =>
                      handleConfigChange(
                        "performance",
                        "batch_processing",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Batch Processing"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.adaptive_buffering}
                    onChange={(e) =>
                      handleConfigChange(
                        "performance",
                        "adaptive_buffering",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Adaptive Buffering"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Database Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Database Integration</Typography>
          <Chip
            label={
              config.database.chunk_metadata_enabled
                ? "Metadata Enabled"
                : "Metadata Disabled"
            }
            color={
              config.database.chunk_metadata_enabled ? "success" : "default"
            }
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.chunk_metadata_enabled}
                    onChange={(e) =>
                      handleConfigChange(
                        "database",
                        "chunk_metadata_enabled",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Store Chunk Metadata"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.store_quality_metrics}
                    onChange={(e) =>
                      handleConfigChange(
                        "database",
                        "store_quality_metrics",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Store Quality Metrics"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.store_processing_stats}
                    onChange={(e) =>
                      handleConfigChange(
                        "database",
                        "store_processing_stats",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Store Processing Statistics"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.correlation_tracking}
                    onChange={(e) =>
                      handleConfigChange(
                        "database",
                        "correlation_tracking",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Correlation Tracking"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.speaker_tracking}
                    onChange={(e) =>
                      handleConfigChange(
                        "database",
                        "speaker_tracking",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Speaker Tracking"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Source Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Source Configuration</Typography>
          <Chip
            label={config.source.source_type.replace("_", " ").toUpperCase()}
            color="info"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Source Type</InputLabel>
                <Select
                  value={config.source.source_type}
                  label="Source Type"
                  onChange={(e) =>
                    handleConfigChange("source", "source_type", e.target.value)
                  }
                >
                  <MenuItem value="bot_audio">Bot Audio Stream</MenuItem>
                  <MenuItem value="meeting_test">Meeting Test</MenuItem>
                  <MenuItem value="microphone">Microphone Input</MenuItem>
                  <MenuItem value="file_upload">File Upload</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Session Timeout (seconds)"
                type="number"
                value={config.source.session_timeout}
                onChange={(e) =>
                  handleConfigChange(
                    "source",
                    "session_timeout",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 60, max: 86400 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.source.auto_restart}
                    onChange={(e) =>
                      handleConfigChange(
                        "source",
                        "auto_restart",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Auto-restart on Failure"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Retries"
                type="number"
                value={config.source.max_retries}
                onChange={(e) =>
                  handleConfigChange(
                    "source",
                    "max_retries",
                    Number(e.target.value),
                  )
                }
                inputProps={{ min: 0, max: 10 }}
                disabled={!config.source.auto_restart}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.source.error_recovery}
                    onChange={(e) =>
                      handleConfigChange(
                        "source",
                        "error_recovery",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Enable Error Recovery"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default ChunkingSettings;
