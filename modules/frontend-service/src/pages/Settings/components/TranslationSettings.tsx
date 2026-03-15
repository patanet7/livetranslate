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
  Checkbox,
  Collapse,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
  InputAdornment,
  Autocomplete,
  Divider,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RestoreIcon from "@mui/icons-material/Restore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import LinkIcon from "@mui/icons-material/Link";
import { SUPPORTED_LANGUAGES } from "@/constants/languages";
import {
  DEFAULT_SOURCE_LANGUAGE,
  DEFAULT_TARGET_LANGUAGES,
  DEFAULT_CONFIDENCE_THRESHOLD,
} from "@/config/translation";

interface TranslationConfig {
  // Service Configuration
  service: {
    enabled: boolean;
    service_url: string;
    inference_engine: "vllm" | "ollama" | "triton" | "openai_compatible";
    model_name: string;
    fallback_model: string;
    timeout_ms: number;
    max_retries: number;
    api_key: string;
  };

  // Language Configuration
  languages: {
    auto_detect: boolean;
    default_source_language: string;
    target_languages: string[];
    supported_languages: string[];
    confidence_threshold: number;
  };

  // Quality and Performance
  quality: {
    quality_threshold: number;
    confidence_scoring: boolean;
    translation_validation: boolean;
    context_preservation: boolean;
    speaker_attribution: boolean;
  };

  // Model Parameters
  model: {
    temperature: number;
    max_tokens: number;
    top_p: number;
    repetition_penalty: number;
    context_window: number;
    batch_size: number;
  };

  // Real-time Configuration
  realtime: {
    streaming_translation: boolean;
    partial_results: boolean;
    translation_delay_ms: number;
    batch_processing: boolean;
    adaptive_batching: boolean;
  };

  // Caching and Performance
  caching: {
    enabled: boolean;
    cache_duration_minutes: number;
    similarity_threshold: number;
    memory_limit_mb: number;
    cache_cleanup_interval: number;
  };
}

interface TranslationSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const availableLanguages = SUPPORTED_LANGUAGES;


const engineDefaults: Record<
  string,
  {
    url: string;
    placeholder: string;
    modelPlaceholder: string;
    helperText: string;
  }
> = {
  vllm: {
    url: "http://localhost:8000",
    placeholder: "http://localhost:8000",
    modelPlaceholder: "meta-llama/Llama-2-7b-chat-hf",
    helperText:
      "vLLM serves OpenAI-compatible endpoints at /v1/chat/completions",
  },
  ollama: {
    url: "http://localhost:11434",
    placeholder: "http://localhost:11434",
    modelPlaceholder: "llama2:7b",
    helperText: "Ollama API serves models at /api/chat",
  },
  triton: {
    url: "http://localhost:8001",
    placeholder: "http://localhost:8001",
    modelPlaceholder: "ensemble_model",
    helperText: "NVIDIA Triton Inference Server",
  },
  openai_compatible: {
    url: "https://api.openai.com/v1",
    placeholder: "https://api.openai.com/v1",
    modelPlaceholder: "gpt-4",
    helperText: "Any OpenAI-compatible API endpoint",
  },
};

const defaultTranslationConfig: TranslationConfig = {
  service: {
    enabled: true,
    service_url: "http://localhost:8000",
    inference_engine: "vllm",
    model_name: "meta-llama/Llama-2-7b-chat-hf",
    fallback_model: "mistral-7b-instruct",
    timeout_ms: 30000,
    max_retries: 3,
    api_key: "",
  },
  languages: {
    auto_detect: true,
    default_source_language: DEFAULT_SOURCE_LANGUAGE,
    target_languages: [...DEFAULT_TARGET_LANGUAGES],
    supported_languages: availableLanguages.map((l) => l.code),
    confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
  },
  quality: {
    quality_threshold: 0.7,
    confidence_scoring: true,
    translation_validation: true,
    context_preservation: true,
    speaker_attribution: true,
  },
  model: {
    temperature: 0.1,
    max_tokens: 512,
    top_p: 0.9,
    repetition_penalty: 1.1,
    context_window: 2048,
    batch_size: 4,
  },
  realtime: {
    streaming_translation: true,
    partial_results: true,
    translation_delay_ms: 500,
    batch_processing: true,
    adaptive_batching: true,
  },
  caching: {
    enabled: true,
    cache_duration_minutes: 60,
    similarity_threshold: 0.95,
    memory_limit_mb: 256,
    cache_cleanup_interval: 300,
  },
};

const TranslationSettings: React.FC<TranslationSettingsProps> = ({
  onSave,
}) => {
  const [config, setConfig] = useState<TranslationConfig>(
    defaultTranslationConfig,
  );
  const [testingTranslation, setTestingTranslation] = useState(false);
  const [translationStats, setTranslationStats] = useState({
    total_translations: 0,
    successful_translations: 0,
    cache_hits: 0,
    average_quality: 0,
    average_latency_ms: 0,
  });

  // Connection panel state
  const [connectionStatus, setConnectionStatus] = useState<
    "unknown" | "connected" | "error" | "verifying"
  >("unknown");
  const [connectionMessage, setConnectionMessage] = useState<string>("");
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([]);
  const [showApiKey, setShowApiKey] = useState(false);
  const [useAuth, setUseAuth] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Load current configuration
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch("/api/settings/translation");
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultTranslationConfig, ...currentConfig });
        }
      } catch (error) {
        console.error("Failed to load translation configuration:", error);
      }
    };

    const loadStats = async () => {
      try {
        const response = await fetch("/api/settings/translation/stats");
        if (response.ok) {
          const stats = await response.json();
          setTranslationStats(stats);
        }
      } catch (error) {
        console.error("Failed to load translation stats:", error);
      }
    };

    loadConfiguration();
    loadStats();
  }, []);

  const handleConfigChange = (
    section: keyof TranslationConfig,
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

  const handleLanguageToggle = (
    languageCode: string,
    isTarget: boolean = false,
  ) => {
    const key = isTarget ? "target_languages" : "supported_languages";
    const currentLanguages = config.languages[key];

    const updatedLanguages = currentLanguages.includes(languageCode)
      ? currentLanguages.filter((code) => code !== languageCode)
      : [...currentLanguages, languageCode];

    handleConfigChange("languages", key, updatedLanguages);
  };

  const handleEngineChange = (
    _event: React.MouseEvent<HTMLElement>,
    newEngine: string | null,
  ) => {
    if (!newEngine) return;
    const engine = newEngine as TranslationConfig["service"]["inference_engine"];
    handleConfigChange("service", "inference_engine", engine);
    const defaults = engineDefaults[engine];
    if (defaults) {
      handleConfigChange("service", "service_url", defaults.url);
    }
    // Reset connection status when engine changes
    setConnectionStatus("unknown");
    setConnectionMessage("");
    setDiscoveredModels([]);
  };

  const handleVerifyConnection = async () => {
    setConnectionStatus("verifying");
    setConnectionMessage("");
    try {
      const response = await fetch(
        "/api/settings/translation/verify-connection",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: config.service.service_url,
            engine: config.service.inference_engine,
            api_key: config.service.api_key,
          }),
        },
      );
      if (response.ok) {
        const result = await response.json();
        setConnectionStatus("connected");
        const models: string[] = result.models ?? [];
        setDiscoveredModels(models);
        setConnectionMessage(
          models.length > 0
            ? `Connected. ${models.length} model${models.length !== 1 ? "s" : ""} available.`
            : "Connected successfully.",
        );
      } else {
        const errorData = await response.json().catch(() => ({}));
        setConnectionStatus("error");
        setConnectionMessage(
          (errorData as { detail?: string }).detail ??
            `Connection failed (HTTP ${response.status})`,
        );
      }
    } catch (error) {
      setConnectionStatus("error");
      setConnectionMessage(
        error instanceof Error ? error.message : "Unable to reach endpoint",
      );
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch("/api/settings/translation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        onSave("Translation settings saved successfully");
      } else {
        onSave("Failed to save translation settings", false);
      }
    } catch (error) {
      onSave("Error saving translation settings", false);
    }
  };

  const handleTestTranslation = async () => {
    setTestingTranslation(true);
    try {
      const response = await fetch("/api/settings/translation/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "This is a test translation.",
          target_language: config.languages.target_languages[0] || "es",
          config: config,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        onSave(`Translation test successful: "${result.translated_text}"`);
      } else {
        onSave("Translation test failed", false);
      }
    } catch (error) {
      onSave("Error testing translation", false);
    } finally {
      setTestingTranslation(false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultTranslationConfig);
    onSave("Translation configuration reset to defaults");
  };

  const handleClearCache = async () => {
    try {
      const response = await fetch("/api/settings/translation/clear-cache", {
        method: "POST",
      });

      if (response.ok) {
        onSave("Translation cache cleared successfully");
      } else {
        onSave("Failed to clear translation cache", false);
      }
    } catch (error) {
      onSave("Error clearing translation cache", false);
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
          Translation Service Configuration
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
            onClick={handleTestTranslation}
            disabled={testingTranslation || !config.service.enabled}
            sx={{ mr: 2 }}
          >
            {testingTranslation ? "Testing..." : "Test Translation"}
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure the translation service for real-time multi-language
        translation. Supports local LLM inference with vLLM, Ollama, and Triton
        backends.
      </Alert>

      {/* Translation Service Connection Card */}
      <Card
        variant="outlined"
        sx={{
          mb: 3,
          borderColor:
            connectionStatus === "connected"
              ? "success.main"
              : connectionStatus === "error"
                ? "error.main"
                : connectionStatus === "verifying"
                  ? "warning.main"
                  : "divider",
          borderWidth: connectionStatus !== "unknown" ? 2 : 1,
          transition: "border-color 0.3s ease",
        }}
      >
        <CardHeader
          avatar={<LinkIcon color="action" />}
          title={
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="h6">
                Translation Service Connection
              </Typography>
              {connectionStatus === "connected" && (
                <Chip
                  icon={<CheckCircleIcon />}
                  label="Connected"
                  color="success"
                  size="small"
                />
              )}
              {connectionStatus === "error" && (
                <Chip
                  icon={<ErrorIcon />}
                  label="Disconnected"
                  color="error"
                  size="small"
                />
              )}
              {connectionStatus === "verifying" && (
                <Chip
                  icon={
                    <CircularProgress size={12} color="inherit" sx={{ mr: 0 }} />
                  }
                  label="Verifying..."
                  color="warning"
                  size="small"
                />
              )}
              {connectionStatus === "unknown" && (
                <Chip label="Not verified" size="small" variant="outlined" />
              )}
            </Box>
          }
          action={
            <FormControlLabel
              control={
                <Switch
                  checked={config.service.enabled}
                  onChange={(e) =>
                    handleConfigChange(
                      "service",
                      "enabled",
                      e.target.checked,
                    )
                  }
                />
              }
              label="Enable"
              labelPlacement="start"
            />
          }
        />

        <CardContent>
          <Grid container spacing={3}>
            {/* Inference Engine Toggle */}
            <Grid item xs={12}>
              <Typography
                variant="body2"
                color="text.secondary"
                gutterBottom
                fontWeight={500}
              >
                Inference Engine
              </Typography>
              <ToggleButtonGroup
                value={config.service.inference_engine}
                exclusive
                onChange={handleEngineChange}
                disabled={!config.service.enabled}
                size="small"
                sx={{ flexWrap: "wrap", gap: 0.5 }}
              >
                <ToggleButton value="vllm" sx={{ px: 2 }}>
                  vLLM
                </ToggleButton>
                <ToggleButton value="ollama" sx={{ px: 2 }}>
                  Ollama
                </ToggleButton>
                <ToggleButton value="triton" sx={{ px: 2 }}>
                  Triton
                </ToggleButton>
                <ToggleButton value="openai_compatible" sx={{ px: 2 }}>
                  OpenAI Compatible
                </ToggleButton>
              </ToggleButtonGroup>
            </Grid>

            {/* Service URL with Verify button */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Service URL"
                value={config.service.service_url}
                onChange={(e) => {
                  handleConfigChange(
                    "service",
                    "service_url",
                    e.target.value,
                  );
                  setConnectionStatus("unknown");
                }}
                disabled={!config.service.enabled}
                placeholder={
                  engineDefaults[config.service.inference_engine]?.placeholder
                }
                helperText={
                  connectionStatus === "error" && connectionMessage
                    ? connectionMessage
                    : (engineDefaults[config.service.inference_engine]
                        ?.helperText ?? "")
                }
                error={connectionStatus === "error"}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <Button
                        variant="contained"
                        size="small"
                        onClick={handleVerifyConnection}
                        disabled={
                          !config.service.enabled ||
                          connectionStatus === "verifying" ||
                          !config.service.service_url
                        }
                        startIcon={
                          connectionStatus === "verifying" ? (
                            <CircularProgress size={14} color="inherit" />
                          ) : undefined
                        }
                        sx={{ whiteSpace: "nowrap", minWidth: 80 }}
                      >
                        {connectionStatus === "verifying"
                          ? "Verifying"
                          : "Verify"}
                      </Button>
                    </InputAdornment>
                  ),
                }}
              />
              {connectionStatus === "connected" && connectionMessage && (
                <Typography
                  variant="caption"
                  color="success.main"
                  sx={{ mt: 0.5, display: "block" }}
                >
                  {connectionMessage}
                </Typography>
              )}
            </Grid>

            {/* Authentication toggle + API key */}
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={
                      useAuth ||
                      config.service.inference_engine === "openai_compatible"
                    }
                    onChange={(e) => setUseAuth(e.target.checked)}
                    disabled={
                      config.service.inference_engine === "openai_compatible" ||
                      !config.service.enabled
                    }
                  />
                }
                label="Authentication required"
              />
            </Grid>

            <Grid item xs={12}>
              <Collapse
                in={
                  useAuth ||
                  config.service.inference_engine === "openai_compatible"
                }
              >
                <TextField
                  fullWidth
                  label="API Key"
                  type={showApiKey ? "text" : "password"}
                  value={config.service.api_key}
                  onChange={(e) =>
                    handleConfigChange("service", "api_key", e.target.value)
                  }
                  disabled={!config.service.enabled}
                  placeholder="sk-... or Bearer token"
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowApiKey((prev) => !prev)}
                          edge="end"
                          size="small"
                          aria-label={
                            showApiKey ? "Hide API key" : "Show API key"
                          }
                        >
                          {showApiKey ? (
                            <VisibilityOffIcon fontSize="small" />
                          ) : (
                            <VisibilityIcon fontSize="small" />
                          )}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
              </Collapse>
            </Grid>

            {/* Primary Model */}
            <Grid item xs={12} md={6}>
              <Autocomplete
                freeSolo
                options={discoveredModels}
                value={config.service.model_name}
                onInputChange={(_event, newValue) =>
                  handleConfigChange("service", "model_name", newValue)
                }
                disabled={!config.service.enabled}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    label="Model"
                    placeholder={
                      engineDefaults[config.service.inference_engine]
                        ?.modelPlaceholder
                    }
                    helperText={
                      discoveredModels.length > 0
                        ? `${discoveredModels.length} model${discoveredModels.length !== 1 ? "s" : ""} discovered`
                        : "Type any model name or verify connection to discover models"
                    }
                  />
                )}
              />
            </Grid>

            {/* Fallback Model */}
            <Grid item xs={12} md={6}>
              <Autocomplete
                freeSolo
                options={discoveredModels}
                value={config.service.fallback_model}
                onInputChange={(_event, newValue) =>
                  handleConfigChange("service", "fallback_model", newValue)
                }
                disabled={!config.service.enabled}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    label="Fallback Model"
                    placeholder={
                      engineDefaults[config.service.inference_engine]
                        ?.modelPlaceholder
                    }
                    helperText="Used when the primary model is unavailable"
                  />
                )}
              />
            </Grid>

            {/* Advanced section */}
            <Grid item xs={12}>
              <Divider sx={{ mb: 1 }} />
              <Button
                size="small"
                onClick={() => setShowAdvanced((prev) => !prev)}
                endIcon={
                  <ExpandMoreIcon
                    sx={{
                      transform: showAdvanced
                        ? "rotate(180deg)"
                        : "rotate(0deg)",
                      transition: "transform 0.2s",
                    }}
                  />
                }
                sx={{ color: "text.secondary", textTransform: "none" }}
              >
                {showAdvanced ? "Hide advanced" : "Show advanced"}
              </Button>
              <Collapse in={showAdvanced}>
                <Grid container spacing={3} sx={{ mt: 0.5 }}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Timeout (ms)"
                      type="number"
                      value={config.service.timeout_ms}
                      onChange={(e) =>
                        handleConfigChange(
                          "service",
                          "timeout_ms",
                          Number(e.target.value),
                        )
                      }
                      disabled={!config.service.enabled}
                      helperText="Request timeout in milliseconds"
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Max Retries"
                      type="number"
                      value={config.service.max_retries}
                      onChange={(e) =>
                        handleConfigChange(
                          "service",
                          "max_retries",
                          Number(e.target.value),
                        )
                      }
                      disabled={!config.service.enabled}
                      helperText="Number of retry attempts on failure"
                    />
                  </Grid>
                </Grid>
              </Collapse>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Translation Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title="Translation Statistics"
          action={
            <Button
              variant="outlined"
              onClick={handleClearCache}
              size="small"
              disabled={!config.caching.enabled}
            >
              Clear Cache
            </Button>
          }
        />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {translationStats.total_translations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Translations
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="success.main">
                  {translationStats.successful_translations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Successful
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="info.main">
                  {translationStats.cache_hits}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Cache Hits
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="secondary.main">
                  {(translationStats.average_quality * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Quality
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6" color="primary">
                  {translationStats.average_latency_ms.toFixed(0)}ms
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Latency
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Language Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Language Configuration</Typography>
          <Chip
            label={`${config.languages?.target_languages?.length || 0} target languages`}
            color="primary"
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
                    checked={config.languages.auto_detect}
                    onChange={(e) =>
                      handleConfigChange(
                        "languages",
                        "auto_detect",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Auto-detect Source Language"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Default Source Language</InputLabel>
                <Select
                  value={config.languages.default_source_language}
                  label="Default Source Language"
                  onChange={(e) =>
                    handleConfigChange(
                      "languages",
                      "default_source_language",
                      e.target.value,
                    )
                  }
                  disabled={config.languages.auto_detect}
                >
                  {availableLanguages.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Confidence Threshold: {config.languages.confidence_threshold}
              </Typography>
              <Slider
                value={config.languages.confidence_threshold}
                onChange={(_, value) =>
                  handleConfigChange("languages", "confidence_threshold", value)
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Target Languages
              </Typography>
              <Paper
                variant="outlined"
                sx={{ p: 2, maxHeight: 200, overflow: "auto" }}
              >
                <Grid container spacing={1}>
                  {availableLanguages.map((lang) => (
                    <Grid item xs={6} md={4} lg={3} key={lang.code}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={config.languages.target_languages.includes(
                              lang.code,
                            )}
                            onChange={() =>
                              handleLanguageToggle(lang.code, true)
                            }
                          />
                        }
                        label={lang.name}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Model Parameters */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Model Parameters</Typography>
          <Chip
            label={`temp: ${config.model.temperature}`}
            color="secondary"
            size="small"
            sx={{ ml: 2 }}
          />
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Temperature: {config.model.temperature}
              </Typography>
              <Slider
                value={config.model.temperature}
                onChange={(_, value) =>
                  handleConfigChange("model", "temperature", value)
                }
                min={0}
                max={2}
                step={0.1}
                marks={[
                  { value: 0, label: "0" },
                  { value: 0.5, label: "0.5" },
                  { value: 1, label: "1" },
                  { value: 2, label: "2" },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Top P: {config.model.top_p}</Typography>
              <Slider
                value={config.model.top_p}
                onChange={(_, value) =>
                  handleConfigChange("model", "top_p", value)
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Tokens"
                type="number"
                value={config.model.max_tokens}
                onChange={(e) =>
                  handleConfigChange(
                    "model",
                    "max_tokens",
                    Number(e.target.value),
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Repetition Penalty: {config.model.repetition_penalty}
              </Typography>
              <Slider
                value={config.model.repetition_penalty}
                onChange={(_, value) =>
                  handleConfigChange("model", "repetition_penalty", value)
                }
                min={1}
                max={2}
                step={0.1}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Context Window"
                type="number"
                value={config.model.context_window}
                onChange={(e) =>
                  handleConfigChange(
                    "model",
                    "context_window",
                    Number(e.target.value),
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Batch Size"
                type="number"
                value={config.model.batch_size}
                onChange={(e) =>
                  handleConfigChange(
                    "model",
                    "batch_size",
                    Number(e.target.value),
                  )
                }
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Quality and Performance */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Quality and Performance</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
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
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.quality.confidence_scoring}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "confidence_scoring",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Confidence Scoring"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.quality.translation_validation}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "translation_validation",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Translation Validation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.quality.context_preservation}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "context_preservation",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Context Preservation"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.quality.speaker_attribution}
                    onChange={(e) =>
                      handleConfigChange(
                        "quality",
                        "speaker_attribution",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Speaker Attribution"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Real-time Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Real-time Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.realtime.streaming_translation}
                    onChange={(e) =>
                      handleConfigChange(
                        "realtime",
                        "streaming_translation",
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Streaming Translation"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.realtime.partial_results}
                    onChange={(e) =>
                      handleConfigChange(
                        "realtime",
                        "partial_results",
                        e.target.checked,
                      )
                    }
                    disabled={!config.realtime.streaming_translation}
                  />
                }
                label="Partial Results"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Translation Delay (ms)"
                type="number"
                value={config.realtime.translation_delay_ms}
                onChange={(e) =>
                  handleConfigChange(
                    "realtime",
                    "translation_delay_ms",
                    Number(e.target.value),
                  )
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.realtime.batch_processing}
                    onChange={(e) =>
                      handleConfigChange(
                        "realtime",
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
                    checked={config.realtime.adaptive_batching}
                    onChange={(e) =>
                      handleConfigChange(
                        "realtime",
                        "adaptive_batching",
                        e.target.checked,
                      )
                    }
                    disabled={!config.realtime.batch_processing}
                  />
                }
                label="Adaptive Batching"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Caching Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Caching and Optimization</Typography>
          <Chip
            label={config.caching.enabled ? "Enabled" : "Disabled"}
            color={config.caching.enabled ? "success" : "default"}
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
                    checked={config.caching.enabled}
                    onChange={(e) =>
                      handleConfigChange("caching", "enabled", e.target.checked)
                    }
                  />
                }
                label="Enable Translation Caching"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Cache Duration (minutes)"
                type="number"
                value={config.caching.cache_duration_minutes}
                onChange={(e) =>
                  handleConfigChange(
                    "caching",
                    "cache_duration_minutes",
                    Number(e.target.value),
                  )
                }
                disabled={!config.caching.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Similarity Threshold: {config.caching.similarity_threshold}
              </Typography>
              <Slider
                value={config.caching.similarity_threshold}
                onChange={(_, value) =>
                  handleConfigChange("caching", "similarity_threshold", value)
                }
                min={0.8}
                max={1}
                step={0.01}
                disabled={!config.caching.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Memory Limit (MB)"
                type="number"
                value={config.caching.memory_limit_mb}
                onChange={(e) =>
                  handleConfigChange(
                    "caching",
                    "memory_limit_mb",
                    Number(e.target.value),
                  )
                }
                disabled={!config.caching.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Cleanup Interval (seconds)"
                type="number"
                value={config.caching.cache_cleanup_interval}
                onChange={(e) =>
                  handleConfigChange(
                    "caching",
                    "cache_cleanup_interval",
                    Number(e.target.value),
                  )
                }
                disabled={!config.caching.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default TranslationSettings;
