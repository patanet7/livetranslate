import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  LinearProgress,
  Fade,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Code as CodeIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  ContentCopy as CopyIcon,
  PlayArrow as TestIcon,
  History as HistoryIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useApiClient } from '@/hooks/useApiClient';

interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  template: string;
  systemMessage?: string;
  languagePairs: string[];
  category: 'general' | 'technical' | 'medical' | 'legal' | 'conversational' | 'creative' | 'formal';
  version: string;
  isActive: boolean;
  isDefault: boolean;
  metadata: {
    createdAt: number;
    updatedAt: number;
    createdBy: string;
    tags: string[];
  };
  performanceMetrics?: {
    avgQuality: number;
    avgSpeed: number;
    avgConfidence: number;
    usageCount: number;
    successRate: number;
    lastUsed?: number;
  };
  testResults?: {
    testCases: number;
    avgScore: number;
    lastTested: number;
  };
}

interface PromptVariable {
  name: string;
  description: string;
  type: 'text' | 'language' | 'number' | 'boolean';
  required: boolean;
  defaultValue?: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`prompt-management-tabpanel-${index}`}
      aria-labelledby={`prompt-management-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Fade in={true} timeout={300}>
          <Box sx={{ py: 3 }}>
            {children}
          </Box>
        </Fade>
      )}
    </div>
  );
}

export const PromptManagementSettings: React.FC = () => {
  const { apiRequest } = useApiClient();
  
  const [tabValue, setTabValue] = useState(0);
  const [prompts, setPrompts] = useState<PromptTemplate[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<PromptTemplate | null>(null);
  const [editingPrompt, setEditingPrompt] = useState<PromptTemplate | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [saveInProgress, setSaveInProgress] = useState(false);
  const [testInProgress, setTestInProgress] = useState(false);
  
  // Dialog states
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [promptToDelete, setPromptToDelete] = useState<PromptTemplate | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  
  // Test configuration
  const [testText, setTestText] = useState('Hello world! How are you today?');
  const [testSourceLang, setTestSourceLang] = useState('en');
  const [testTargetLang, setTestTargetLang] = useState('es');
  const [testResults, setTestResults] = useState<any>(null);

  const supportedLanguages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'it', name: 'Italian', flag: '🇮🇹' },
    { code: 'pt', name: 'Portuguese', flag: '🇵🇹' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵' },
    { code: 'ko', name: 'Korean', flag: '🇰🇷' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
    { code: 'ru', name: 'Russian', flag: '🇷🇺' },
    { code: 'ar', name: 'Arabic', flag: '🇸🇦' },
  ];

  const promptCategories = [
    { value: 'general', label: 'General Purpose', color: 'primary' },
    { value: 'conversational', label: 'Conversational', color: 'success' },
    { value: 'technical', label: 'Technical', color: 'info' },
    { value: 'medical', label: 'Medical', color: 'warning' },
    { value: 'legal', label: 'Legal', color: 'error' },
    { value: 'creative', label: 'Creative', color: 'secondary' },
    { value: 'formal', label: 'Formal', color: 'default' },
  ];

  const defaultPrompts: PromptTemplate[] = [
    {
      id: 'default',
      name: 'Default Translation',
      description: 'Standard translation prompt for general use',
      template: 'Translate the following text from {source_language} to {target_language}. Provide only the translation without any explanation:\n\n{text}',
      languagePairs: ['*'],
      category: 'general',
      version: '1.0',
      isActive: true,
      isDefault: true,
      metadata: {
        createdAt: Date.now() - 86400000 * 30,
        updatedAt: Date.now() - 86400000 * 7,
        createdBy: 'system',
        tags: ['basic', 'general']
      },
      performanceMetrics: {
        avgQuality: 0.85,
        avgSpeed: 450,
        avgConfidence: 0.82,
        usageCount: 156,
        successRate: 0.94,
        lastUsed: Date.now() - 3600000
      }
    },
    {
      id: 'conversational',
      name: 'Conversational Style',
      description: 'Natural conversational translation with context awareness',
      template: 'Translate this conversational text naturally, maintaining the tone and style from {source_language} to {target_language}:\n\n{text}\n\nKeep the natural flow and cultural context appropriate for casual conversation.',
      systemMessage: 'You are a skilled translator specializing in natural, conversational language. Maintain cultural context and informal tone.',
      languagePairs: ['en-es', 'en-fr', 'zh-en', 'ja-en'],
      category: 'conversational',
      version: '1.2',
      isActive: true,
      isDefault: false,
      metadata: {
        createdAt: Date.now() - 86400000 * 15,
        updatedAt: Date.now() - 86400000 * 2,
        createdBy: 'admin',
        tags: ['conversational', 'natural', 'context-aware']
      },
      performanceMetrics: {
        avgQuality: 0.92,
        avgSpeed: 380,
        avgConfidence: 0.89,
        usageCount: 89,
        successRate: 0.96,
        lastUsed: Date.now() - 1800000
      }
    },
    {
      id: 'technical',
      name: 'Technical Documentation',
      description: 'Specialized translation for technical documentation and manuals',
      template: 'Translate this technical documentation from {source_language} to {target_language}, maintaining technical accuracy and terminology:\n\n{text}\n\nPreserve technical terms, maintain precision, and ensure clarity for technical audience.',
      systemMessage: 'You are a technical translator with expertise in software, engineering, and scientific documentation. Prioritize accuracy and consistency of technical terminology.',
      languagePairs: ['en-de', 'en-ja', 'en-zh'],
      category: 'technical',
      version: '2.0',
      isActive: true,
      isDefault: false,
      metadata: {
        createdAt: Date.now() - 86400000 * 20,
        updatedAt: Date.now() - 86400000 * 1,
        createdBy: 'tech_team',
        tags: ['technical', 'documentation', 'precision']
      },
      performanceMetrics: {
        avgQuality: 0.94,
        avgSpeed: 520,
        avgConfidence: 0.91,
        usageCount: 67,
        successRate: 0.98,
        lastUsed: Date.now() - 7200000
      }
    }
  ];

  const availableVariables: PromptVariable[] = [
    { name: 'text', description: 'The text to be translated', type: 'text', required: true },
    { name: 'source_language', description: 'Source language code or name', type: 'language', required: true },
    { name: 'target_language', description: 'Target language code or name', type: 'language', required: true },
    { name: 'context', description: 'Additional context for translation', type: 'text', required: false },
    { name: 'style', description: 'Translation style (formal, casual, etc.)', type: 'text', required: false },
    { name: 'domain', description: 'Domain/field (medical, legal, technical)', type: 'text', required: false },
    { name: 'max_length', description: 'Maximum length of translation', type: 'number', required: false },
    { name: 'preserve_formatting', description: 'Whether to preserve text formatting', type: 'boolean', required: false, defaultValue: 'true' },
  ];

  // Initialize with default prompts
  useEffect(() => {
    loadPrompts();
  }, []);

  const loadPrompts = useCallback(async () => {
    try {
      // Try to load from API first
      const response = await apiRequest('/api/settings/prompts');
      if (response && response.prompts) {
        setPrompts(response.prompts);
      } else {
        // Fall back to default prompts
        setPrompts(defaultPrompts);
      }
    } catch (error) {
      console.error('Failed to load prompts, using defaults:', error);
      setPrompts(defaultPrompts);
    }
  }, [apiRequest]);

  const savePrompts = useCallback(async (updatedPrompts: PromptTemplate[]) => {
    setSaveInProgress(true);
    try {
      await apiRequest('/api/settings/prompts', {
        method: 'POST',
        data: { prompts: updatedPrompts }
      });
      setPrompts(updatedPrompts);
    } catch (error) {
      console.error('Failed to save prompts:', error);
    } finally {
      setSaveInProgress(false);
    }
  }, [apiRequest]);

  const handleTabChange = useCallback((_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  }, []);

  const handleCreatePrompt = useCallback(() => {
    const newPrompt: PromptTemplate = {
      id: `custom-${Date.now()}`,
      name: 'New Prompt Template',
      description: 'Custom translation prompt',
      template: 'Translate the following text from {source_language} to {target_language}:\n\n{text}',
      languagePairs: ['*'],
      category: 'general',
      version: '1.0',
      isActive: true,
      isDefault: false,
      metadata: {
        createdAt: Date.now(),
        updatedAt: Date.now(),
        createdBy: 'user',
        tags: ['custom']
      }
    };
    
    setEditingPrompt(newPrompt);
    setIsCreating(true);
    setCreateDialogOpen(true);
  }, []);

  const handleSavePrompt = useCallback(async () => {
    if (!editingPrompt) return;
    
    const updatedPrompts = isCreating 
      ? [...prompts, { ...editingPrompt, metadata: { ...editingPrompt.metadata, updatedAt: Date.now() } }]
      : prompts.map(p => p.id === editingPrompt.id 
          ? { ...editingPrompt, metadata: { ...editingPrompt.metadata, updatedAt: Date.now() } }
          : p
        );
    
    await savePrompts(updatedPrompts);
    setEditingPrompt(null);
    setIsCreating(false);
    setCreateDialogOpen(false);
  }, [editingPrompt, isCreating, prompts, savePrompts]);

  const handleDeletePrompt = useCallback(async () => {
    if (!promptToDelete) return;
    
    const updatedPrompts = prompts.filter(p => p.id !== promptToDelete.id);
    await savePrompts(updatedPrompts);
    setPromptToDelete(null);
    setDeleteDialogOpen(false);
  }, [promptToDelete, prompts, savePrompts]);

  const handleTestPrompt = useCallback(async () => {
    if (!selectedPrompt) return;
    
    setTestInProgress(true);
    try {
      const response = await apiRequest('/api/translation/test', {
        method: 'POST',
        data: {
          text: testText,
          source_language: testSourceLang,
          target_language: testTargetLang,
          prompt_id: selectedPrompt.id,
          prompt_template: selectedPrompt.template,
          system_message: selectedPrompt.systemMessage
        }
      });
      
      setTestResults(response);
    } catch (error) {
      console.error('Prompt test failed:', error);
      setTestResults({ error: 'Test failed. Please check your configuration.' });
    } finally {
      setTestInProgress(false);
    }
  }, [selectedPrompt, testText, testSourceLang, testTargetLang, apiRequest]);

  const handleTogglePrompt = useCallback(async (promptId: string, isActive: boolean) => {
    const updatedPrompts = prompts.map(p => 
      p.id === promptId ? { ...p, isActive } : p
    );
    await savePrompts(updatedPrompts);
  }, [prompts, savePrompts]);

  const handleCopyPrompt = useCallback((prompt: PromptTemplate) => {
    navigator.clipboard.writeText(prompt.template);
  }, []);

  const exportPrompts = useCallback(() => {
    const dataStr = JSON.stringify(prompts, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `translation-prompts-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }, [prompts]);

  const importPrompts = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedPrompts = JSON.parse(e.target?.result as string);
        if (Array.isArray(importedPrompts)) {
          setPrompts(prev => [...prev, ...importedPrompts]);
        }
      } catch (error) {
        console.error('Failed to import prompts:', error);
      }
    };
    reader.readAsText(file);
  }, []);

  const getLanguageName = useCallback((code: string) => {
    const lang = supportedLanguages.find(l => l.code === code);
    return lang ? `${lang.flag} ${lang.name}` : code;
  }, []);

  const getCategoryConfig = useCallback((category: string) => {
    return promptCategories.find(c => c.value === category) || promptCategories[0];
  }, []);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        🛠️ Prompt Management Settings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Create, edit, and manage translation prompt templates with advanced testing and performance analytics
      </Typography>

      <Paper sx={{ width: '100%', mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="prompt management tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab 
              label="Prompt Library" 
              id="prompt-management-tab-0"
              icon={<CodeIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Create/Edit" 
              id="prompt-management-tab-1"
              icon={<EditIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Testing" 
              id="prompt-management-tab-2"
              icon={<TestIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Analytics" 
              id="prompt-management-tab-3"
              icon={<TrendingUpIcon />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
            <Button
              variant="contained"
              onClick={handleCreatePrompt}
              startIcon={<AddIcon />}
            >
              Create New Prompt
            </Button>
            <Button
              variant="outlined"
              onClick={exportPrompts}
              startIcon={<DownloadIcon />}
            >
              Export Prompts
            </Button>
            <Button
              variant="outlined"
              component="label"
              startIcon={<UploadIcon />}
            >
              Import Prompts
              <input
                type="file"
                hidden
                accept=".json"
                onChange={importPrompts}
              />
            </Button>
            {saveInProgress && <LinearProgress sx={{ flexGrow: 1 }} />}
          </Box>

          <Grid container spacing={2}>
            {prompts.map((prompt) => {
              const categoryConfig = getCategoryConfig(prompt.category);
              return (
                <Grid item xs={12} key={prompt.id}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                        <Typography variant="h6" fontWeight="bold">
                          {prompt.name}
                        </Typography>
                        {prompt.isDefault && (
                          <Chip label="Default" size="small" color="primary" />
                        )}
                        <Chip 
                          label={categoryConfig.label} 
                          size="small" 
                          color={categoryConfig.color as any}
                        />
                        <Chip 
                          label={`v${prompt.version}`} 
                          size="small"
                        />
                        <Switch
                          checked={prompt.isActive}
                          onChange={(e) => handleTogglePrompt(prompt.id, e.target.checked)}
                          size="small"
                          onClick={(e) => e.stopPropagation()}
                        />
                        {prompt.performanceMetrics && (
                          <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
                            <Chip 
                              label={`Quality: ${(prompt.performanceMetrics.avgQuality * 100).toFixed(0)}%`}
                              size="small"
                              color={prompt.performanceMetrics.avgQuality > 0.8 ? "success" : "warning"}
                            />
                            <Chip 
                              label={`${prompt.performanceMetrics.avgSpeed}ms`}
                              size="small"
                            />
                            <Chip 
                              label={`${prompt.performanceMetrics.usageCount} uses`}
                              size="small"
                            />
                          </Box>
                        )}
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={8}>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {prompt.description}
                          </Typography>
                          
                          <TextField
                            fullWidth
                            multiline
                            rows={8}
                            label="Prompt Template"
                            value={prompt.template}
                            disabled
                            sx={{ mb: 2 }}
                            variant="outlined"
                          />
                          
                          {prompt.systemMessage && (
                            <TextField
                              fullWidth
                              multiline
                              rows={3}
                              label="System Message"
                              value={prompt.systemMessage}
                              disabled
                              sx={{ mb: 2 }}
                              variant="outlined"
                            />
                          )}
                          
                          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                            <Typography variant="caption" sx={{ mr: 1 }}>Languages:</Typography>
                            {prompt.languagePairs.map((pair, index) => (
                              <Chip key={index} label={pair} size="small" />
                            ))}
                          </Box>
                          
                          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                            <Typography variant="caption" sx={{ mr: 1 }}>Tags:</Typography>
                            {prompt.metadata.tags.map((tag, index) => (
                              <Chip key={index} label={tag} size="small" variant="outlined" />
                            ))}
                          </Box>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          {prompt.performanceMetrics && (
                            <Card variant="outlined" sx={{ mb: 2 }}>
                              <CardContent>
                                <Typography variant="subtitle2" gutterBottom>
                                  Performance Metrics
                                </Typography>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">Quality:</Typography>
                                    <Typography variant="caption" fontWeight="bold">
                                      {(prompt.performanceMetrics.avgQuality * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">Speed:</Typography>
                                    <Typography variant="caption" fontWeight="bold">
                                      {prompt.performanceMetrics.avgSpeed}ms
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">Confidence:</Typography>
                                    <Typography variant="caption" fontWeight="bold">
                                      {(prompt.performanceMetrics.avgConfidence * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">Success Rate:</Typography>
                                    <Typography variant="caption" fontWeight="bold">
                                      {(prompt.performanceMetrics.successRate * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption">Usage:</Typography>
                                    <Typography variant="caption" fontWeight="bold">
                                      {prompt.performanceMetrics.usageCount}
                                    </Typography>
                                  </Box>
                                </Box>
                              </CardContent>
                            </Card>
                          )}
                          
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Metadata
                              </Typography>
                              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Created:</Typography>
                                  <Typography variant="caption" display="block">
                                    {new Date(prompt.metadata.createdAt).toLocaleDateString()}
                                  </Typography>
                                </Box>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Updated:</Typography>
                                  <Typography variant="caption" display="block">
                                    {new Date(prompt.metadata.updatedAt).toLocaleDateString()}
                                  </Typography>
                                </Box>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Author:</Typography>
                                  <Typography variant="caption" display="block">
                                    {prompt.metadata.createdBy}
                                  </Typography>
                                </Box>
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      </Grid>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                          variant="outlined"
                          onClick={() => {
                            setEditingPrompt(prompt);
                            setIsCreating(false);
                            setCreateDialogOpen(true);
                          }}
                          startIcon={<EditIcon />}
                          disabled={prompt.isDefault}
                        >
                          Edit
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={() => handleCopyPrompt(prompt)}
                          startIcon={<CopyIcon />}
                        >
                          Copy Template
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={() => {
                            setSelectedPrompt(prompt);
                            setTestDialogOpen(true);
                          }}
                          startIcon={<TestIcon />}
                        >
                          Test
                        </Button>
                        {!prompt.isDefault && (
                          <Button
                            variant="outlined"
                            color="error"
                            onClick={() => {
                              setPromptToDelete(prompt);
                              setDeleteDialogOpen(true);
                            }}
                            startIcon={<DeleteIcon />}
                          >
                            Delete
                          </Button>
                        )}
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Grid>
              );
            })}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📝 Available Template Variables
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Use these variables in your prompt templates. Required variables must be included.
              </Typography>
              
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Variable</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Required</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Default</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {availableVariables.map((variable) => (
                      <TableRow key={variable.name}>
                        <TableCell>
                          <Typography variant="body2" fontFamily="monospace">
                            {`{${variable.name}}`}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip label={variable.type} size="small" />
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={variable.required ? "Required" : "Optional"} 
                            size="small"
                            color={variable.required ? "error" : "default"}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {variable.description}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" fontFamily="monospace">
                            {variable.defaultValue || '-'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Template Tips:</strong><br />
                  • Use {`{text}`} for the source text to translate<br />
                  • Use {`{source_language}`} and {`{target_language}`} for language context<br />
                  • Add specific instructions for your use case<br />
                  • Test your prompts thoroughly before using in production
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🧪 Prompt Testing Interface
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Test your prompt templates with sample text to evaluate performance
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Select Prompt to Test</InputLabel>
                    <Select
                      value={selectedPrompt?.id || ''}
                      label="Select Prompt to Test"
                      onChange={(e) => {
                        const prompt = prompts.find(p => p.id === e.target.value);
                        setSelectedPrompt(prompt || null);
                      }}
                    >
                      {prompts.filter(p => p.isActive).map((prompt) => (
                        <MenuItem key={prompt.id} value={prompt.id}>
                          {prompt.name} (v{prompt.version})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    label="Test Text"
                    value={testText}
                    onChange={(e) => setTestText(e.target.value)}
                    sx={{ mb: 2 }}
                  />
                  
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Source Language</InputLabel>
                        <Select
                          value={testSourceLang}
                          label="Source Language"
                          onChange={(e) => setTestSourceLang(e.target.value)}
                        >
                          {supportedLanguages.map((lang) => (
                            <MenuItem key={lang.code} value={lang.code}>
                              {lang.flag} {lang.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Target Language</InputLabel>
                        <Select
                          value={testTargetLang}
                          label="Target Language"
                          onChange={(e) => setTestTargetLang(e.target.value)}
                        >
                          {supportedLanguages.map((lang) => (
                            <MenuItem key={lang.code} value={lang.code}>
                              {lang.flag} {lang.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                  
                  <Button
                    variant="contained"
                    onClick={handleTestPrompt}
                    disabled={!selectedPrompt || !testText.trim() || testInProgress}
                    fullWidth
                    startIcon={testInProgress ? <LinearProgress /> : <TestIcon />}
                  >
                    {testInProgress ? 'Testing...' : 'Test Prompt'}
                  </Button>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Test Results
                  </Typography>
                  
                  {testResults ? (
                    testResults.error ? (
                      <Alert severity="error">
                        {testResults.error}
                      </Alert>
                    ) : (
                      <Box>
                        <Paper sx={{ p: 2, mb: 2, backgroundColor: 'primary.50' }}>
                          <Typography variant="h6" gutterBottom>
                            Translation Result
                          </Typography>
                          <Typography variant="body1" sx={{ mb: 2 }}>
                            {testResults.translated_text}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            <Chip 
                              label={`Confidence: ${(testResults.confidence * 100).toFixed(1)}%`}
                              color={testResults.confidence > 0.8 ? "success" : "warning"}
                              size="small"
                            />
                            <Chip 
                              label={`Time: ${testResults.processing_time}ms`}
                              size="small"
                            />
                            {testResults.quality_score && (
                              <Chip 
                                label={`Quality: ${(testResults.quality_score * 100).toFixed(1)}%`}
                                color={testResults.quality_score > 0.8 ? "success" : "warning"}
                                size="small"
                              />
                            )}
                          </Box>
                        </Paper>
                        
                        {testResults.prompt_analysis && (
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Prompt Analysis
                              </Typography>
                              <Typography variant="body2">
                                {testResults.prompt_analysis}
                              </Typography>
                            </CardContent>
                          </Card>
                        )}
                      </Box>
                    )
                  ) : (
                    <Alert severity="info">
                      Select a prompt and run a test to see results here.
                    </Alert>
                  )}
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="primary">
                    {prompts.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Prompts
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="success.main">
                    {prompts.filter(p => p.isActive).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Prompts
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="info.main">
                    {prompts.reduce((sum, p) => sum + (p.performanceMetrics?.usageCount || 0), 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Usage
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 Prompt Performance Comparison
                  </Typography>
                  
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Prompt Name</TableCell>
                          <TableCell>Category</TableCell>
                          <TableCell>Usage Count</TableCell>
                          <TableCell>Avg Quality</TableCell>
                          <TableCell>Avg Speed</TableCell>
                          <TableCell>Success Rate</TableCell>
                          <TableCell>Last Used</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {prompts
                          .filter(p => p.performanceMetrics)
                          .sort((a, b) => (b.performanceMetrics?.usageCount || 0) - (a.performanceMetrics?.usageCount || 0))
                          .map((prompt) => (
                            <TableRow key={prompt.id}>
                              <TableCell>
                                <Typography variant="body2" fontWeight="bold">
                                  {prompt.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  v{prompt.version}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={getCategoryConfig(prompt.category).label}
                                  size="small"
                                  color={getCategoryConfig(prompt.category).color as any}
                                />
                              </TableCell>
                              <TableCell>{prompt.performanceMetrics?.usageCount || 0}</TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${((prompt.performanceMetrics?.avgQuality || 0) * 100).toFixed(1)}%`}
                                  size="small"
                                  color={(prompt.performanceMetrics?.avgQuality || 0) > 0.8 ? "success" : "warning"}
                                />
                              </TableCell>
                              <TableCell>{prompt.performanceMetrics?.avgSpeed || 0}ms</TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${((prompt.performanceMetrics?.successRate || 0) * 100).toFixed(1)}%`}
                                  size="small"
                                  color={(prompt.performanceMetrics?.successRate || 0) > 0.9 ? "success" : "warning"}
                                />
                              </TableCell>
                              <TableCell>
                                <Typography variant="caption">
                                  {prompt.performanceMetrics?.lastUsed 
                                    ? new Date(prompt.performanceMetrics.lastUsed).toLocaleDateString()
                                    : 'Never'
                                  }
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* Create/Edit Dialog */}
      <Dialog 
        open={createDialogOpen} 
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {isCreating ? 'Create New Prompt Template' : 'Edit Prompt Template'}
        </DialogTitle>
        <DialogContent>
          {editingPrompt && (
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                label="Prompt Name"
                value={editingPrompt.name}
                onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, name: e.target.value } : null)}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Description"
                value={editingPrompt.description}
                onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, description: e.target.value } : null)}
                sx={{ mb: 2 }}
              />
              
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel>Category</InputLabel>
                    <Select
                      value={editingPrompt.category}
                      label="Category"
                      onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, category: e.target.value as any } : null)}
                    >
                      {promptCategories.map((category) => (
                        <MenuItem key={category.value} value={category.value}>
                          {category.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Version"
                    value={editingPrompt.version}
                    onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, version: e.target.value } : null)}
                  />
                </Grid>
              </Grid>
              
              <TextField
                fullWidth
                multiline
                rows={8}
                label="Prompt Template"
                value={editingPrompt.template}
                onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, template: e.target.value } : null)}
                sx={{ mb: 2 }}
                helperText="Use {text}, {source_language}, {target_language} as placeholders"
              />
              
              <TextField
                fullWidth
                multiline
                rows={3}
                label="System Message (Optional)"
                value={editingPrompt.systemMessage || ''}
                onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, systemMessage: e.target.value } : null)}
                sx={{ mb: 2 }}
                helperText="Additional context or instructions for the AI model"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={editingPrompt.isActive}
                    onChange={(e) => setEditingPrompt(prev => prev ? { ...prev, isActive: e.target.checked } : null)}
                  />
                }
                label="Active"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleSavePrompt}
            variant="contained"
            disabled={saveInProgress}
          >
            {saveInProgress ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Prompt Template</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the prompt template "{promptToDelete?.name}"? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleDeletePrompt} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Test Dialog */}
      <Dialog
        open={testDialogOpen}
        onClose={() => setTestDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Test Prompt: {selectedPrompt?.name}</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Test Text"
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              sx={{ mb: 2 }}
            />
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel>Source Language</InputLabel>
                  <Select
                    value={testSourceLang}
                    label="Source Language"
                    onChange={(e) => setTestSourceLang(e.target.value)}
                  >
                    {supportedLanguages.map((lang) => (
                      <MenuItem key={lang.code} value={lang.code}>
                        {lang.flag} {lang.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel>Target Language</InputLabel>
                  <Select
                    value={testTargetLang}
                    label="Target Language"
                    onChange={(e) => setTestTargetLang(e.target.value)}
                  >
                    {supportedLanguages.map((lang) => (
                      <MenuItem key={lang.code} value={lang.code}>
                        {lang.flag} {lang.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            
            {testResults && !testResults.error && (
              <Paper sx={{ p: 2, backgroundColor: 'primary.50' }}>
                <Typography variant="body1" sx={{ mb: 1 }}>
                  {testResults.translated_text}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip 
                    label={`${(testResults.confidence * 100).toFixed(1)}% confidence`}
                    size="small"
                    color={testResults.confidence > 0.8 ? "success" : "warning"}
                  />
                  <Chip 
                    label={`${testResults.processing_time}ms`}
                    size="small"
                  />
                </Box>
              </Paper>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>
            Close
          </Button>
          <Button 
            onClick={handleTestPrompt}
            variant="contained"
            disabled={!testText.trim() || testInProgress}
          >
            {testInProgress ? 'Testing...' : 'Run Test'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};