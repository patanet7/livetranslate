import React, { useState, useEffect } from 'react';
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
  List,
  ListItem,
  ListItemText,
  Checkbox,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RestoreIcon from '@mui/icons-material/Restore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TranslateIcon from '@mui/icons-material/Translate';

interface TranslationConfig {
  // Service Configuration
  service: {
    enabled: boolean;
    service_url: string;
    inference_engine: 'vllm' | 'ollama' | 'triton';
    model_name: string;
    fallback_model: string;
    timeout_ms: number;
    max_retries: number;
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

import { SUPPORTED_LANGUAGES } from '@/constants/languages';

interface TranslationSettingsProps {
  onSave: (message: string, success?: boolean) => void;
}

const availableLanguages = SUPPORTED_LANGUAGES;

const availableModels = [
  { id: 'llama2-7b-chat', name: 'Llama 2 7B Chat' },
  { id: 'mistral-7b-instruct', name: 'Mistral 7B Instruct' },
  { id: 'codellama-7b-instruct', name: 'CodeLlama 7B Instruct' },
  { id: 'vicuna-7b-v1.5', name: 'Vicuna 7B v1.5' },
  { id: 'orca-mini-3b', name: 'Orca Mini 3B' },
];

const defaultTranslationConfig: TranslationConfig = {
  service: {
    enabled: true,
    service_url: 'http://localhost:5003',
    inference_engine: 'vllm',
    model_name: 'llama2-7b-chat',
    fallback_model: 'orca-mini-3b',
    timeout_ms: 30000,
    max_retries: 3,
  },
  languages: {
    auto_detect: true,
    default_source_language: 'en',
    target_languages: ['es', 'fr', 'de'],
    supported_languages: availableLanguages.map(l => l.code),
    confidence_threshold: 0.8,
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

const TranslationSettings: React.FC<TranslationSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<TranslationConfig>(defaultTranslationConfig);
  const [testingTranslation, setTestingTranslation] = useState(false);
  const [translationStats, setTranslationStats] = useState({
    total_translations: 0,
    successful_translations: 0,
    cache_hits: 0,
    average_quality: 0,
    average_latency_ms: 0,
  });

  // Load current configuration
  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const response = await fetch('/api/settings/translation');
        if (response.ok) {
          const currentConfig = await response.json();
          setConfig({ ...defaultTranslationConfig, ...currentConfig });
        }
      } catch (error) {
        console.error('Failed to load translation configuration:', error);
      }
    };

    const loadStats = async () => {
      try {
        const response = await fetch('/api/settings/translation/stats');
        if (response.ok) {
          const stats = await response.json();
          setTranslationStats(stats);
        }
      } catch (error) {
        console.error('Failed to load translation stats:', error);
      }
    };
    
    loadConfiguration();
    loadStats();
  }, []);

  const handleConfigChange = (section: keyof TranslationConfig, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  const handleLanguageToggle = (languageCode: string, isTarget: boolean = false) => {
    const key = isTarget ? 'target_languages' : 'supported_languages';
    const currentLanguages = config.languages[key];
    
    const updatedLanguages = currentLanguages.includes(languageCode)
      ? currentLanguages.filter(code => code !== languageCode)
      : [...currentLanguages, languageCode];
    
    handleConfigChange('languages', key, updatedLanguages);
  };

  const handleSave = async () => {
    try {
      const response = await fetch('/api/settings/translation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        onSave('Translation settings saved successfully');
      } else {
        onSave('Failed to save translation settings', false);
      }
    } catch (error) {
      onSave('Error saving translation settings', false);
    }
  };

  const handleTestTranslation = async () => {
    setTestingTranslation(true);
    try {
      const response = await fetch('/api/settings/translation/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: 'This is a test translation.',
          target_language: config.languages.target_languages[0] || 'es',
          config: config,
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        onSave(`Translation test successful: "${result.translated_text}"`);
      } else {
        onSave('Translation test failed', false);
      }
    } catch (error) {
      onSave('Error testing translation', false);
    } finally {
      setTestingTranslation(false);
    }
  };

  const handleResetToDefaults = () => {
    setConfig(defaultTranslationConfig);
    onSave('Translation configuration reset to defaults');
  };

  const handleClearCache = async () => {
    try {
      const response = await fetch('/api/settings/translation/clear-cache', {
        method: 'POST',
      });
      
      if (response.ok) {
        onSave('Translation cache cleared successfully');
      } else {
        onSave('Failed to clear translation cache', false);
      }
    } catch (error) {
      onSave('Error clearing translation cache', false);
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
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
            {testingTranslation ? 'Testing...' : 'Test Translation'}
          </Button>
          <Button variant="contained" onClick={handleSave}>
            Save Settings
          </Button>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Configure the translation service for real-time multi-language translation.
        Supports local LLM inference with vLLM, Ollama, and Triton backends.
      </Alert>

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
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {translationStats.total_translations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Translations
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="success.main">
                  {translationStats.successful_translations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Successful
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="info.main">
                  {translationStats.cache_hits}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Cache Hits
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" color="secondary.main">
                  {(translationStats.average_quality * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Quality
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6} lg={2.4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
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

      {/* Service Configuration */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Service Configuration</Typography>
          <Chip 
            label={config.service.enabled ? 'Enabled' : 'Disabled'} 
            color={config.service.enabled ? 'success' : 'default'}
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
                    checked={config.service.enabled}
                    onChange={(e) => handleConfigChange('service', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Translation Service"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Service URL"
                value={config.service.service_url}
                onChange={(e) => handleConfigChange('service', 'service_url', e.target.value)}
                disabled={!config.service.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Inference Engine</InputLabel>
                <Select
                  value={config.service.inference_engine}
                  label="Inference Engine"
                  onChange={(e) => handleConfigChange('service', 'inference_engine', e.target.value)}
                  disabled={!config.service.enabled}
                >
                  <MenuItem value="vllm">vLLM (GPU Optimized)</MenuItem>
                  <MenuItem value="ollama">Ollama (CPU/GPU)</MenuItem>
                  <MenuItem value="triton">Triton Inference Server</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Primary Model</InputLabel>
                <Select
                  value={config.service.model_name}
                  label="Primary Model"
                  onChange={(e) => handleConfigChange('service', 'model_name', e.target.value)}
                  disabled={!config.service.enabled}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model.id} value={model.id}>{model.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Fallback Model</InputLabel>
                <Select
                  value={config.service.fallback_model}
                  label="Fallback Model"
                  onChange={(e) => handleConfigChange('service', 'fallback_model', e.target.value)}
                  disabled={!config.service.enabled}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model.id} value={model.id}>{model.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Timeout (ms)"
                type="number"
                value={config.service.timeout_ms}
                onChange={(e) => handleConfigChange('service', 'timeout_ms', Number(e.target.value))}
                disabled={!config.service.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Retries"
                type="number"
                value={config.service.max_retries}
                onChange={(e) => handleConfigChange('service', 'max_retries', Number(e.target.value))}
                disabled={!config.service.enabled}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

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
                    onChange={(e) => handleConfigChange('languages', 'auto_detect', e.target.checked)}
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
                  onChange={(e) => handleConfigChange('languages', 'default_source_language', e.target.value)}
                  disabled={config.languages.auto_detect}
                >
                  {availableLanguages.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>{lang.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Confidence Threshold: {config.languages.confidence_threshold}</Typography>
              <Slider
                value={config.languages.confidence_threshold}
                onChange={(_, value) => handleConfigChange('languages', 'confidence_threshold', value)}
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Target Languages</Typography>
              <Paper variant="outlined" sx={{ p: 2, maxHeight: 200, overflow: 'auto' }}>
                <Grid container spacing={1}>
                  {availableLanguages.map((lang) => (
                    <Grid item xs={6} md={4} lg={3} key={lang.code}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={config.languages.target_languages.includes(lang.code)}
                            onChange={() => handleLanguageToggle(lang.code, true)}
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
              <Typography gutterBottom>Temperature: {config.model.temperature}</Typography>
              <Slider
                value={config.model.temperature}
                onChange={(_, value) => handleConfigChange('model', 'temperature', value)}
                min={0}
                max={2}
                step={0.1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Top P: {config.model.top_p}</Typography>
              <Slider
                value={config.model.top_p}
                onChange={(_, value) => handleConfigChange('model', 'top_p', value)}
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
                onChange={(e) => handleConfigChange('model', 'max_tokens', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Repetition Penalty: {config.model.repetition_penalty}</Typography>
              <Slider
                value={config.model.repetition_penalty}
                onChange={(_, value) => handleConfigChange('model', 'repetition_penalty', value)}
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
                onChange={(e) => handleConfigChange('model', 'context_window', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Batch Size"
                type="number"
                value={config.model.batch_size}
                onChange={(e) => handleConfigChange('model', 'batch_size', Number(e.target.value))}
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
              <Typography gutterBottom>Quality Threshold: {config.quality.quality_threshold}</Typography>
              <Slider
                value={config.quality.quality_threshold}
                onChange={(_, value) => handleConfigChange('quality', 'quality_threshold', value)}
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
                    onChange={(e) => handleConfigChange('quality', 'confidence_scoring', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('quality', 'translation_validation', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('quality', 'context_preservation', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('quality', 'speaker_attribution', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('realtime', 'streaming_translation', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('realtime', 'partial_results', e.target.checked)}
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
                onChange={(e) => handleConfigChange('realtime', 'translation_delay_ms', Number(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.realtime.batch_processing}
                    onChange={(e) => handleConfigChange('realtime', 'batch_processing', e.target.checked)}
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
                    onChange={(e) => handleConfigChange('realtime', 'adaptive_batching', e.target.checked)}
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
            label={config.caching.enabled ? 'Enabled' : 'Disabled'} 
            color={config.caching.enabled ? 'success' : 'default'}
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
                    onChange={(e) => handleConfigChange('caching', 'enabled', e.target.checked)}
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
                onChange={(e) => handleConfigChange('caching', 'cache_duration_minutes', Number(e.target.value))}
                disabled={!config.caching.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Similarity Threshold: {config.caching.similarity_threshold}</Typography>
              <Slider
                value={config.caching.similarity_threshold}
                onChange={(_, value) => handleConfigChange('caching', 'similarity_threshold', value)}
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
                onChange={(e) => handleConfigChange('caching', 'memory_limit_mb', Number(e.target.value))}
                disabled={!config.caching.enabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Cleanup Interval (seconds)"
                type="number"
                value={config.caching.cache_cleanup_interval}
                onChange={(e) => handleConfigChange('caching', 'cache_cleanup_interval', Number(e.target.value))}
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