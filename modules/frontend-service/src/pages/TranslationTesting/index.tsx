import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tabs,
  Tab,
  Alert,
  Fade,
  Button,
  Card,
  CardContent,
  Chip,
  FormGroup,
  FormControlLabel,
  Checkbox,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Switch,
  AccordionSummary,
  AccordionDetails,
  Accordion,
  CircularProgress,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
  Compare as CompareIcon,
  Assessment as AssessmentIcon,
  Translate as TranslateIcon,
  Settings as SettingsIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useApiClient } from '@/hooks/useApiClient';

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
      id={`translation-testing-tabpanel-${index}`}
      aria-labelledby={`translation-testing-tab-${index}`}
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

interface TranslationTest {
  id: string;
  sourceText: string;
  sourceLanguage: string;
  targetLanguage: string;
  translatedText: string;
  confidence: number;
  processingTime: number;
  timestamp: number;
  promptId?: string;
  modelUsed?: string;
  qualityScore?: number;
}

interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  template: string;
  languagePairs: string[];
  category: 'general' | 'technical' | 'medical' | 'legal' | 'conversational';
  version: string;
  isActive: boolean;
  performanceMetrics?: {
    avgQuality: number;
    avgSpeed: number;
    usageCount: number;
  };
}

const TranslationTesting: React.FC = () => {
  const { isConnected, sendMessage: wsSendMessage } = useWebSocket();
  const { translateText } = useApiClient();
  
  const [tabValue, setTabValue] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [streamingResults, setStreamingResults] = useState<TranslationTest[]>([]);
  
  // Test Configuration
  const [sourceLanguage, setSourceLanguage] = useState('auto');
  const [targetLanguages, setTargetLanguages] = useState<string[]>([...DEFAULT_TARGET_LANGUAGES]);
  const [testText, setTestText] = useState('Hello world! How are you today?');
  const [batchTestTexts, setBatchTestTexts] = useState<string[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<string>('default');
  
  // Translation Results
  const [singleResults, setSingleResults] = useState<TranslationTest[]>([]);
  const [batchResults, setBatchResults] = useState<TranslationTest[]>([]);
  const [comparisonResults, setComparisonResults] = useState<TranslationTest[]>([]);
  const [testHistory, setTestHistory] = useState<TranslationTest[]>([]);
  
  // Processing States
  const [isTesting, setIsTesting] = useState(false);
  const [testProgress, setTestProgress] = useState(0);
  const [batchProgress, setBatchProgress] = useState(0);
  
  // Prompt Management
  const [prompts, setPrompts] = useState<PromptTemplate[]>([
    {
      id: 'default',
      name: 'Default Translation',
      description: 'Standard translation prompt for general use',
      template: 'Translate the following text from {source_language} to {target_language}. Provide only the translation without any explanation:\n\n{text}',
      languagePairs: ['*'],
      category: 'general',
      version: '1.0',
      isActive: true,
      performanceMetrics: {
        avgQuality: 0.85,
        avgSpeed: 450,
        usageCount: 156
      }
    },
    {
      id: 'conversational',
      name: 'Conversational Style',
      description: 'Natural conversational translation with context awareness',
      template: 'Translate this conversational text naturally, maintaining the tone and style from {source_language} to {target_language}:\n\n{text}\n\nKeep the natural flow and cultural context.',
      languagePairs: ['en-es', 'en-fr', 'zh-en'],
      category: 'conversational',
      version: '1.2',
      isActive: true,
      performanceMetrics: {
        avgQuality: 0.92,
        avgSpeed: 380,
        usageCount: 89
      }
    }
  ]);
  
  const [editingPrompt, setEditingPrompt] = useState<PromptTemplate | null>(null);
  const [newPromptText, setNewPromptText] = useState('');

  const supportedLanguages = [
    { code: 'auto', name: 'Auto Detect', flag: '🌐' },
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

  const handleTabChange = useCallback((_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  }, []);

  const handleLanguageToggle = useCallback((language: string) => {
    setTargetLanguages(prev => 
      prev.includes(language) 
        ? prev.filter(lang => lang !== language)
        : [...prev, language]
    );
  }, []);

  const handleSingleTranslationTest = useCallback(async () => {
    if (!testText.trim() || targetLanguages.length === 0) return;
    
    setIsTesting(true);
    setSingleResults([]);
    
    try {
      const startTime = Date.now();
      const results: TranslationTest[] = [];
      
      for (let i = 0; i < targetLanguages.length; i++) {
        const targetLang = targetLanguages[i];
        setTestProgress((i + 1) / targetLanguages.length * 100);
        
        const result = await translateText({
          text: testText,
          source_language: sourceLanguage,
          target_language: targetLang,
          prompt_id: selectedPrompt
        });
        
        // Extract the actual translation data from the API response
        const translationData = result.success ? result.data : null;
        if (!translationData) {
          console.error('Translation failed:', result.error || 'Unknown error');
          continue;
        }
        
        const translationTest: TranslationTest = {
          id: `single-${Date.now()}-${i}`,
          sourceText: testText,
          sourceLanguage: sourceLanguage,
          targetLanguage: targetLang,
          translatedText: translationData.translated_text || '',
          confidence: translationData.confidence || 0,
          processingTime: translationData.processing_time || (Date.now() - startTime),
          timestamp: Date.now(),
          promptId: selectedPrompt,
          modelUsed: translationData.model_used || 'unknown',
          qualityScore: translationData.quality_score || undefined
        };
        
        results.push(translationTest);
      }
      
      setSingleResults(results);
      setTestHistory(prev => [...results, ...prev]);
      
    } catch (error) {
      console.error('Translation test failed:', error);
    } finally {
      setIsTesting(false);
      setTestProgress(0);
    }
  }, [testText, sourceLanguage, targetLanguages, selectedPrompt, translateText]);

  const handleBatchTranslationTest = useCallback(async () => {
    if (batchTestTexts.length === 0 || targetLanguages.length === 0) return;
    
    setIsTesting(true);
    setBatchResults([]);
    setBatchProgress(0);
    
    try {
      const results: TranslationTest[] = [];
      const totalTests = batchTestTexts.length * targetLanguages.length;
      let completedTests = 0;
      
      for (const text of batchTestTexts) {
        for (const targetLang of targetLanguages) {
          const startTime = Date.now();
          
          const result = await translateText({
            text: text,
            source_language: sourceLanguage,
            target_language: targetLang,
            prompt_id: selectedPrompt
          });
          
          // Extract the actual translation data from the API response
          const translationData = result.success ? result.data : null;
          if (!translationData) {
            console.error('Batch translation failed:', result.error || 'Unknown error');
            continue;
          }
          
          const translationTest: TranslationTest = {
            id: `batch-${Date.now()}-${completedTests}`,
            sourceText: text,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLang,
            translatedText: translationData.translated_text || '',
            confidence: translationData.confidence || 0,
            processingTime: translationData.processing_time || (Date.now() - startTime),
            timestamp: Date.now(),
            promptId: selectedPrompt,
            modelUsed: translationData.model_used || 'unknown',
            qualityScore: translationData.quality_score || undefined
          };
          
          results.push(translationTest);
          completedTests++;
          setBatchProgress((completedTests / totalTests) * 100);
        }
      }
      
      setBatchResults(results);
      setTestHistory(prev => [...results, ...prev]);
      
    } catch (error) {
      console.error('Batch translation test failed:', error);
    } finally {
      setIsTesting(false);
      setBatchProgress(0);
    }
  }, [batchTestTexts, sourceLanguage, targetLanguages, selectedPrompt, translateText]);

  const handleStreamingTranslationTest = useCallback(() => {
    if (!isConnected || !streamingText.trim()) return;

    setIsStreaming(true);
    setStreamingResults([]);

    (wsSendMessage as any)('translation:start_streaming', {
      source_language: sourceLanguage,
      target_languages: targetLanguages,
      prompt_id: selectedPrompt
    });

    // Simulate streaming text by sending chunks
    const words = streamingText.split(' ');
    let currentText = '';

    words.forEach((word, index) => {
      setTimeout(() => {
        currentText += (index > 0 ? ' ' : '') + word;
        (wsSendMessage as any)('translation:text_chunk', {
          text_chunk: currentText,
          is_final: index === words.length - 1
        });
      }, index * 500); // Send a word every 500ms
    });

    setTimeout(() => {
      setIsStreaming(false);
    }, words.length * 500 + 1000);
  }, [isConnected, wsSendMessage, streamingText, sourceLanguage, targetLanguages, selectedPrompt]);

  const handlePromptComparison = useCallback(async () => {
    if (!testText.trim() || targetLanguages.length === 0 || prompts.length < 2) return;
    
    setIsTesting(true);
    setComparisonResults([]);
    
    try {
      const results: TranslationTest[] = [];
      
      for (const prompt of prompts.filter(p => p.isActive)) {
        for (const targetLang of targetLanguages) {
          const startTime = Date.now();
          
          const result = await translateText({
            text: testText,
            source_language: sourceLanguage,
            target_language: targetLang,
            prompt_id: prompt.id
          });
          
          // Extract the actual translation data from the API response
          const translationData = result.success ? result.data : null;
          if (!translationData) {
            console.error('Prompt comparison translation failed:', result.error || 'Unknown error');
            continue;
          }
          
          const translationTest: TranslationTest = {
            id: `comparison-${prompt.id}-${targetLang}-${Date.now()}`,
            sourceText: testText,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLang,
            translatedText: translationData.translated_text || '',
            confidence: translationData.confidence || 0,
            processingTime: translationData.processing_time || (Date.now() - startTime),
            timestamp: Date.now(),
            promptId: prompt.id,
            modelUsed: translationData.model_used || 'unknown',
            qualityScore: translationData.quality_score || undefined
          };
          
          results.push(translationTest);
        }
      }
      
      setComparisonResults(results);
      
    } catch (error) {
      console.error('Prompt comparison failed:', error);
    } finally {
      setIsTesting(false);
    }
  }, [testText, sourceLanguage, targetLanguages, prompts, translateText]);

  const handleAddBatchText = useCallback(() => {
    setBatchTestTexts(prev => [...prev, '']);
  }, []);

  const handleUpdateBatchText = useCallback((index: number, text: string) => {
    setBatchTestTexts(prev => prev.map((item, i) => i === index ? text : item));
  }, []);

  const handleRemoveBatchText = useCallback((index: number) => {
    setBatchTestTexts(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleSavePrompt = useCallback(() => {
    if (editingPrompt) {
      setPrompts(prev => prev.map(p => 
        p.id === editingPrompt.id 
          ? { ...editingPrompt, template: newPromptText }
          : p
      ));
      setEditingPrompt(null);
      setNewPromptText('');
    }
  }, [editingPrompt, newPromptText]);

  const getLanguageName = useCallback((code: string) => {
    const lang = supportedLanguages.find(l => l.code === code);
    return lang ? `${lang.flag} ${lang.name}` : code;
  }, []);

  const exportResults = useCallback((results: TranslationTest[], filename: string) => {
    const csv = [
      ['Source Text', 'Source Language', 'Target Language', 'Translation', 'Confidence', 'Processing Time (ms)', 'Quality Score', 'Prompt ID', 'Model Used'].join(','),
      ...results.map(r => [
        `"${r.sourceText.replace(/"/g, '""')}"`,
        r.sourceLanguage,
        r.targetLanguage,
        `"${r.translatedText.replace(/"/g, '""')}"`,
        r.confidence.toFixed(3),
        r.processingTime.toString(),
        (r.qualityScore || 0).toFixed(3),
        r.promptId || '',
        r.modelUsed || ''
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Note: WebSocket message handling is now managed by the useWebSocket hook
  // and Redux store. Streaming results would be handled through Redux actions.

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🌍 Advanced Translation Testing & Prompt Management
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Comprehensive translation testing suite with real-time streaming, batch processing, and advanced prompt management
      </Typography>

      <Alert severity={isConnected ? "success" : "warning"} sx={{ mb: 3 }}>
        <Typography variant="body2">
          WebSocket Status: {isConnected ? "Connected - Real-time features available" : "Disconnected - Using REST API fallback"}
        </Typography>
      </Alert>

      <Paper sx={{ width: '100%', mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="translation testing tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab 
              label="Single Translation" 
              id="translation-testing-tab-0"
              icon={<TranslateIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Batch Testing" 
              id="translation-testing-tab-1"
              icon={<AssessmentIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Streaming Test" 
              id="translation-testing-tab-2"
              icon={<PlayIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Prompt Comparison" 
              id="translation-testing-tab-3"
              icon={<CompareIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Prompt Management" 
              id="translation-testing-tab-4"
              icon={<SettingsIcon />}
              iconPosition="start"
            />
            <Tab 
              label="Analytics" 
              id="translation-testing-tab-5"
              icon={<AnalyticsIcon />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📝 Single Translation Test
                  </Typography>
                  
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    label="Text to Translate"
                    value={testText}
                    onChange={(e) => setTestText(e.target.value)}
                    sx={{ mb: 2 }}
                  />
                  
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Source Language</InputLabel>
                        <Select
                          value={sourceLanguage}
                          label="Source Language"
                          onChange={(e) => setSourceLanguage(e.target.value)}
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
                        <InputLabel>Prompt Template</InputLabel>
                        <Select
                          value={selectedPrompt}
                          label="Prompt Template"
                          onChange={(e) => setSelectedPrompt(e.target.value)}
                        >
                          {prompts.filter(p => p.isActive).map((prompt) => (
                            <MenuItem key={prompt.id} value={prompt.id}>
                              {prompt.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Target Languages:
                  </Typography>
                  <FormGroup row sx={{ mb: 2 }}>
                    {supportedLanguages.filter(lang => lang.code !== 'auto').map((lang) => (
                      <FormControlLabel
                        key={lang.code}
                        control={
                          <Checkbox
                            checked={targetLanguages.includes(lang.code)}
                            onChange={() => handleLanguageToggle(lang.code)}
                            size="small"
                          />
                        }
                        label={`${lang.flag} ${lang.name}`}
                      />
                    ))}
                  </FormGroup>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    {targetLanguages.map((lang) => (
                      <Chip
                        key={lang}
                        label={getLanguageName(lang)}
                        onDelete={() => handleLanguageToggle(lang)}
                        color="primary"
                        size="small"
                      />
                    ))}
                  </Box>
                  
                  <Button
                    variant="contained"
                    onClick={handleSingleTranslationTest}
                    disabled={!testText.trim() || targetLanguages.length === 0 || isTesting}
                    fullWidth
                    startIcon={isTesting ? <CircularProgress size={20} /> : <TranslateIcon />}
                  >
                    {isTesting ? 'Translating...' : 'Test Translation'}
                  </Button>
                  
                  {isTesting && (
                    <LinearProgress 
                      variant="determinate" 
                      value={testProgress} 
                      sx={{ mt: 2 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 Translation Results
                  </Typography>
                  
                  {singleResults.length > 0 ? (
                    <Box>
                      {singleResults.map((result) => (
                        <Paper key={result.id} sx={{ p: 2, mb: 2, backgroundColor: 'primary.50' }}>
                          <Typography variant="body1" sx={{ mb: 1 }}>
                            {result.translatedText}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                            <Chip 
                              label={`${getLanguageName(result.sourceLanguage)} → ${getLanguageName(result.targetLanguage)}`}
                              size="small"
                              color="primary"
                            />
                            <Chip 
                              label={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                              size="small"
                              color={result.confidence > 0.8 ? "success" : "warning"}
                            />
                            <Chip 
                              label={`${result.processingTime}ms`}
                              size="small"
                            />
                            {result.qualityScore && (
                              <Chip 
                                label={`Quality: ${(result.qualityScore * 100).toFixed(1)}%`}
                                size="small"
                                color={result.qualityScore > 0.8 ? "success" : "warning"}
                              />
                            )}
                          </Box>
                        </Paper>
                      ))}
                    </Box>
                  ) : (
                    <Alert severity="info">
                      Run a translation test to see results here.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Batch Translation Testing
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Test multiple texts simultaneously to evaluate translation performance across different inputs
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Test Texts:
                </Typography>
                {batchTestTexts.map((text, index) => (
                  <Box key={index} sx={{ display: 'flex', gap: 1, mb: 1 }}>
                    <TextField
                      fullWidth
                      size="small"
                      placeholder={`Test text ${index + 1}`}
                      value={text}
                      onChange={(e) => handleUpdateBatchText(index, e.target.value)}
                    />
                    <IconButton
                      onClick={() => handleRemoveBatchText(index)}
                      color="error"
                      size="small"
                    >
                      <ClearIcon />
                    </IconButton>
                  </Box>
                ))}
                <Button
                  variant="outlined"
                  onClick={handleAddBatchText}
                  size="small"
                  sx={{ mt: 1 }}
                >
                  Add Text
                </Button>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleBatchTranslationTest}
                  disabled={batchTestTexts.length === 0 || targetLanguages.length === 0 || isTesting}
                  startIcon={isTesting ? <CircularProgress size={20} /> : <AssessmentIcon />}
                >
                  {isTesting ? 'Processing...' : 'Run Batch Test'}
                </Button>
                
                {batchResults.length > 0 && (
                  <Button
                    variant="outlined"
                    onClick={() => exportResults(batchResults, 'batch-translation-results')}
                    startIcon={<DownloadIcon />}
                  >
                    Export Results
                  </Button>
                )}
              </Box>
              
              {isTesting && (
                <LinearProgress 
                  variant="determinate" 
                  value={batchProgress} 
                  sx={{ mb: 2 }}
                />
              )}
              
              {batchResults.length > 0 && (
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Source Text</TableCell>
                        <TableCell>Target Lang</TableCell>
                        <TableCell>Translation</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Time (ms)</TableCell>
                        <TableCell>Quality</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {batchResults.map((result) => (
                        <TableRow key={result.id}>
                          <TableCell sx={{ maxWidth: 200 }}>
                            <Typography variant="body2" noWrap>
                              {result.sourceText}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={getLanguageName(result.targetLanguage)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell sx={{ maxWidth: 300 }}>
                            <Typography variant="body2">
                              {result.translatedText}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={`${(result.confidence * 100).toFixed(1)}%`}
                              size="small"
                              color={result.confidence > 0.8 ? "success" : "warning"}
                            />
                          </TableCell>
                          <TableCell>{result.processingTime}</TableCell>
                          <TableCell>
                            {result.qualityScore && (
                              <Chip 
                                label={`${(result.qualityScore * 100).toFixed(1)}%`}
                                size="small"
                                color={result.qualityScore > 0.8 ? "success" : "warning"}
                              />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎥 Real-time Streaming Translation
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Test real-time translation streaming with live text input
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Streaming Text Input"
                value={streamingText}
                onChange={(e) => setStreamingText(e.target.value)}
                sx={{ mb: 2 }}
                placeholder="Type or paste text that will be sent in chunks to simulate real-time translation..."
              />
              
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleStreamingTranslationTest}
                  disabled={!streamingText.trim() || !isConnected || isStreaming}
                  startIcon={isStreaming ? <CircularProgress size={20} /> : <PlayIcon />}
                >
                  {isStreaming ? 'Streaming...' : 'Start Streaming Test'}
                </Button>

                {isStreaming && (
                  <Button
                    variant="outlined"
                    onClick={() => setIsStreaming(false)}
                    startIcon={<StopIcon />}
                  >
                    Stop Streaming
                  </Button>
                )}
              </Box>

              {!isConnected && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  WebSocket connection required for streaming translation testing.
                  Please check your connection to the translation service.
                </Alert>
              )}
              
              {streamingResults.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Streaming Results:
                  </Typography>
                  {streamingResults.map((result, index) => (
                    <Paper key={index} sx={{ p: 2, mb: 1, backgroundColor: 'success.50' }}>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>{getLanguageName(result.targetLanguage)}:</strong> {result.translatedText}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Confidence: {(result.confidence * 100).toFixed(1)}% | 
                        Time: {result.processingTime}ms
                      </Typography>
                    </Paper>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🔍 Prompt Comparison & A/B Testing
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Compare different prompt templates to find the most effective approach for your use case
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Test Text for Comparison"
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Active Prompts for Comparison:
                </Typography>
                {prompts.filter(p => p.isActive).map((prompt) => (
                  <Chip
                    key={prompt.id}
                    label={`${prompt.name} (v${prompt.version})`}
                    sx={{ mr: 1, mb: 1 }}
                    color="primary"
                  />
                ))}
              </Box>
              
              <Button
                variant="contained"
                onClick={handlePromptComparison}
                disabled={!testText.trim() || targetLanguages.length === 0 || isTesting}
                startIcon={isTesting ? <CircularProgress size={20} /> : <CompareIcon />}
                sx={{ mb: 2 }}
              >
                {isTesting ? 'Comparing...' : 'Run Prompt Comparison'}
              </Button>
              
              {comparisonResults.length > 0 && (
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Prompt</TableCell>
                        <TableCell>Language</TableCell>
                        <TableCell>Translation</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Quality</TableCell>
                        <TableCell>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {comparisonResults.map((result) => {
                        const prompt = prompts.find(p => p.id === result.promptId);
                        return (
                          <TableRow key={result.id}>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">
                                {prompt?.name || result.promptId}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                v{prompt?.version}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={getLanguageName(result.targetLanguage)}
                                size="small"
                              />
                            </TableCell>
                            <TableCell sx={{ maxWidth: 300 }}>
                              <Typography variant="body2">
                                {result.translatedText}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={`${(result.confidence * 100).toFixed(1)}%`}
                                size="small"
                                color={result.confidence > 0.8 ? "success" : "warning"}
                              />
                            </TableCell>
                            <TableCell>
                              {result.qualityScore && (
                                <Chip 
                                  label={`${(result.qualityScore * 100).toFixed(1)}%`}
                                  size="small"
                                  color={result.qualityScore > 0.8 ? "success" : "warning"}
                                />
                              )}
                            </TableCell>
                            <TableCell>{result.processingTime}ms</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                🛠️ Prompt Template Management
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Create, edit, and manage translation prompt templates for different use cases
              </Typography>
            </Grid>
            
            {prompts.map((prompt) => (
              <Grid item xs={12} key={prompt.id}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {prompt.name}
                      </Typography>
                      <Chip 
                        label={prompt.category} 
                        size="small" 
                        color="primary"
                      />
                      <Chip 
                        label={`v${prompt.version}`} 
                        size="small"
                      />
                      <Switch
                        checked={prompt.isActive}
                        onChange={(e) => {
                          setPrompts(prev => prev.map(p => 
                            p.id === prompt.id 
                              ? { ...p, isActive: e.target.checked }
                              : p
                          ));
                        }}
                        size="small"
                      />
                      {prompt.performanceMetrics && (
                        <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
                          <Chip 
                            label={`Quality: ${(prompt.performanceMetrics.avgQuality * 100).toFixed(0)}%`}
                            size="small"
                            color={prompt.performanceMetrics.avgQuality > 0.8 ? "success" : "warning"}
                          />
                          <Chip 
                            label={`${prompt.performanceMetrics.avgSpeed}ms avg`}
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
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {prompt.description}
                    </Typography>
                    
                    <TextField
                      fullWidth
                      multiline
                      rows={6}
                      label="Prompt Template"
                      value={editingPrompt?.id === prompt.id ? newPromptText : prompt.template}
                      onChange={(e) => {
                        if (editingPrompt?.id === prompt.id) {
                          setNewPromptText(e.target.value);
                        }
                      }}
                      disabled={editingPrompt?.id !== prompt.id}
                      sx={{ mb: 2 }}
                      helperText="Use {source_language}, {target_language}, and {text} as placeholders"
                    />
                    
                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                      <Typography variant="caption">Language Pairs:</Typography>
                      {prompt.languagePairs.map((pair, index) => (
                        <Chip key={index} label={pair} size="small" />
                      ))}
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 2 }}>
                      {editingPrompt?.id === prompt.id ? (
                        <>
                          <Button
                            variant="contained"
                            onClick={handleSavePrompt}
                            size="small"
                          >
                            Save Changes
                          </Button>
                          <Button
                            variant="outlined"
                            onClick={() => {
                              setEditingPrompt(null);
                              setNewPromptText('');
                            }}
                            size="small"
                          >
                            Cancel
                          </Button>
                        </>
                      ) : (
                        <Button
                          variant="outlined"
                          onClick={() => {
                            setEditingPrompt(prompt);
                            setNewPromptText(prompt.template);
                          }}
                          size="small"
                        >
                          Edit Template
                        </Button>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Translation Analytics & Performance
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Comprehensive analytics and performance metrics for your translation tests
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      {testHistory.length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Tests Run
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="success.main">
                      {testHistory.length > 0 ? 
                        ((testHistory.reduce((sum, t) => sum + t.confidence, 0) / testHistory.length) * 100).toFixed(1) : 0
                      }%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Average Confidence
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="info.main">
                      {testHistory.length > 0 ? 
                        Math.round(testHistory.reduce((sum, t) => sum + t.processingTime, 0) / testHistory.length) : 0
                      }ms
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Average Processing Time
                    </Typography>
                  </Paper>
                </Grid>
                
                {testHistory.length > 0 && (
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                      <Button
                        variant="outlined"
                        onClick={() => exportResults(testHistory, 'translation-history')}
                        startIcon={<DownloadIcon />}
                      >
                        Export All Results
                      </Button>
                      <Button
                        variant="outlined"
                        onClick={() => setTestHistory([])}
                        startIcon={<ClearIcon />}
                        color="error"
                      >
                        Clear History
                      </Button>
                    </Box>
                    
                    <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                      <Table stickyHeader size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Timestamp</TableCell>
                            <TableCell>Source</TableCell>
                            <TableCell>Target</TableCell>
                            <TableCell>Translation</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Time</TableCell>
                            <TableCell>Prompt</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {testHistory.slice(0, 50).map((result) => (
                            <TableRow key={result.id}>
                              <TableCell>
                                <Typography variant="caption">
                                  {new Date(result.timestamp).toLocaleString()}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={getLanguageName(result.sourceLanguage)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={getLanguageName(result.targetLanguage)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell sx={{ maxWidth: 200 }}>
                                <Typography variant="body2" noWrap>
                                  {result.translatedText}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${(result.confidence * 100).toFixed(1)}%`}
                                  size="small"
                                  color={result.confidence > 0.8 ? "success" : "warning"}
                                />
                              </TableCell>
                              <TableCell>{result.processingTime}ms</TableCell>
                              <TableCell>
                                <Typography variant="caption">
                                  {prompts.find(p => p.id === result.promptId)?.name || result.promptId}
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default TranslationTesting;