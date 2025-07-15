import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  Paper,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  Badge,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
  Translate as TranslateIcon,
  VolumeUp as VolumeIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { ConnectionIndicator } from '@/components/ui/ConnectionIndicator';
import { useAppSelector } from '@/store';

interface WebSocketMessage {
  id: string;
  type: 'send' | 'receive' | 'error' | 'system';
  timestamp: Date;
  data: any;
  service?: 'translation' | 'whisper' | 'orchestration' | 'general';
}

interface TranslationTestMessage {
  text: string;
  source_language: string;
  target_language: string;
  session_id?: string;
  streaming?: boolean;
  quality_threshold?: number;
}

interface StreamingSession {
  id: string;
  service: string;
  isActive: boolean;
  messageCount: number;
  startTime: Date;
}

const WebSocketTest: React.FC = () => {
  const { connection } = useAppSelector(state => state.websocket);
  
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [messageInput, setMessageInput] = useState('');
  const [selectedService, setSelectedService] = useState('translation');
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const [streamingSessions, setStreamingSessions] = useState<StreamingSession[]>([]);
  
  // WebSocket refs
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Translation testing state
  const [translationTest, setTranslationTest] = useState<TranslationTestMessage>({
    text: 'Hello, how are you today?',
    source_language: 'en',
    target_language: 'es',
    streaming: false,
    quality_threshold: 0.8,
  });
  
  // Whisper testing state
  const [whisperTest, setWhisperTest] = useState({
    audio_format: 'wav',
    sample_rate: 16000,
    channels: 1,
    streaming: true,
  });

  // Available languages for testing
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'it', name: 'Italian' },
    { code: 'ru', name: 'Russian' },
  ];

  const services = [
    { id: 'translation', name: 'Translation Service', port: 5003 },
    { id: 'whisper', name: 'Whisper Service', port: 5001 },
    { id: 'orchestration', name: 'Orchestration Service', port: 3000 },
  ];

  useEffect(() => {
    if (isAutoScroll) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isAutoScroll]);

  const connectToService = (service: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const serviceConfig = services.find(s => s.id === service);
    const wsUrl = `ws://localhost:${serviceConfig?.port}/ws`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        addMessage('system', `Connected to ${serviceConfig?.name}`, 'system');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          addMessage('receive', data, service as any);
        } catch (error) {
          addMessage('receive', event.data, service as any);
        }
      };
      
      wsRef.current.onclose = () => {
        addMessage('system', `Disconnected from ${serviceConfig?.name}`, 'system');
      };
      
      wsRef.current.onerror = (error) => {
        addMessage('error', `WebSocket error: ${error}`, 'system');
      };
    } catch (error) {
      addMessage('error', `Failed to connect: ${error}`, 'system');
    }
  };

  const addMessage = (type: WebSocketMessage['type'], data: any, service?: WebSocketMessage['service']) => {
    const message: WebSocketMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      timestamp: new Date(),
      data,
      service,
    };
    setMessages(prev => [...prev, message]);
  };

  const sendMessage = (data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const messageString = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(messageString);
      addMessage('send', data, selectedService as any);
    } else {
      addMessage('error', 'WebSocket not connected', 'system');
    }
  };

  const sendTranslationTest = () => {
    const message = {
      action: 'translate',
      session_id: `test-${Date.now()}`,
      ...translationTest,
    };
    sendMessage(message);
    
    if (translationTest.streaming) {
      const session: StreamingSession = {
        id: message.session_id,
        service: 'translation',
        isActive: true,
        messageCount: 1,
        startTime: new Date(),
      };
      setStreamingSessions(prev => [...prev, session]);
    }
  };

  const sendWhisperTest = () => {
    const message = {
      action: 'start_transcription',
      session_id: `whisper-${Date.now()}`,
      config: whisperTest,
    };
    sendMessage(message);
    
    const session: StreamingSession = {
      id: message.session_id,
      service: 'whisper',
      isActive: true,
      messageCount: 1,
      startTime: new Date(),
    };
    setStreamingSessions(prev => [...prev, session]);
  };

  const sendCustomMessage = () => {
    if (messageInput.trim()) {
      try {
        const data = JSON.parse(messageInput);
        sendMessage(data);
      } catch {
        sendMessage(messageInput);
      }
      setMessageInput('');
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  const downloadMessages = () => {
    const data = JSON.stringify(messages, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `websocket-messages-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const stopSession = (sessionId: string) => {
    const session = streamingSessions.find(s => s.id === sessionId);
    if (session) {
      const stopMessage = {
        action: session.service === 'translation' ? 'stop_translation' : 'stop_transcription',
        session_id: sessionId,
      };
      sendMessage(stopMessage);
      
      setStreamingSessions(prev => 
        prev.map(s => s.id === sessionId ? { ...s, isActive: false } : s)
      );
    }
  };

  const getMessageTypeColor = (type: WebSocketMessage['type']) => {
    switch (type) {
      case 'send': return 'primary';
      case 'receive': return 'success';
      case 'error': return 'error';
      case 'system': return 'default';
      default: return 'default';
    }
  };

  const getServiceIcon = (service?: string) => {
    switch (service) {
      case 'translation': return <TranslateIcon fontSize="small" />;
      case 'whisper': return <VolumeIcon fontSize="small" />;
      case 'orchestration': return <SpeedIcon fontSize="small" />;
      default: return null;
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Typography variant="h4" component="h1">
          WebSocket Testing
        </Typography>
        <ConnectionIndicator 
          isConnected={connection.isConnected}
          reconnectAttempts={connection.reconnectAttempts}
          showLabel
        />
        <Badge badgeContent={streamingSessions.filter(s => s.isActive).length} color="primary">
          <Chip label="Active Sessions" variant="outlined" />
        </Badge>
      </Box>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Real-time WebSocket connection testing with translation and transcription capabilities
      </Typography>

      <Grid container spacing={3}>
        {/* Service Connection */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Service Connection</Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <FormControl sx={{ minWidth: 200 }}>
                  <InputLabel>Service</InputLabel>
                  <Select
                    value={selectedService}
                    onChange={(e) => setSelectedService(e.target.value)}
                    label="Service"
                  >
                    {services.map(service => (
                      <MenuItem key={service.id} value={service.id}>
                        {getServiceIcon(service.id)} {service.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  onClick={() => connectToService(selectedService)}
                  disabled={wsRef.current?.readyState === WebSocket.OPEN}
                >
                  Connect
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => wsRef.current?.close()}
                  disabled={wsRef.current?.readyState !== WebSocket.OPEN}
                >
                  Disconnect
                </Button>
              </Box>
              {streamingSessions.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>Active Sessions:</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {streamingSessions.map(session => (
                      <Chip
                        key={session.id}
                        label={`${session.service} (${session.messageCount})`}
                        color={session.isActive ? 'primary' : 'default'}
                        variant={session.isActive ? 'filled' : 'outlined'}
                        onDelete={session.isActive ? () => stopSession(session.id) : undefined}
                        deleteIcon={<StopIcon />}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Testing Interface */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ pb: 1 }}>
              <Typography variant="h6" gutterBottom>Test Interface</Typography>
              <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
                <Tab label="Translation" />
                <Tab label="Whisper" />
                <Tab label="Custom" />
              </Tabs>
            </CardContent>
            
            <CardContent sx={{ flex: 1, overflow: 'auto' }}>
              {/* Translation Testing */}
              {activeTab === 0 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <TextField
                    label="Text to Translate"
                    multiline
                    rows={3}
                    value={translationTest.text}
                    onChange={(e) => setTranslationTest(prev => ({ ...prev, text: e.target.value }))}
                    fullWidth
                  />
                  
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl sx={{ minWidth: 120 }}>
                      <InputLabel>Source</InputLabel>
                      <Select
                        value={translationTest.source_language}
                        onChange={(e) => setTranslationTest(prev => ({ ...prev, source_language: e.target.value }))}
                        label="Source"
                      >
                        {languages.map(lang => (
                          <MenuItem key={lang.code} value={lang.code}>{lang.name}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    
                    <FormControl sx={{ minWidth: 120 }}>
                      <InputLabel>Target</InputLabel>
                      <Select
                        value={translationTest.target_language}
                        onChange={(e) => setTranslationTest(prev => ({ ...prev, target_language: e.target.value }))}
                        label="Target"
                      >
                        {languages.map(lang => (
                          <MenuItem key={lang.code} value={lang.code}>{lang.name}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                  
                  <TextField
                    label="Quality Threshold"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                    value={translationTest.quality_threshold}
                    onChange={(e) => setTranslationTest(prev => ({ ...prev, quality_threshold: parseFloat(e.target.value) }))}
                    sx={{ width: 150 }}
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={translationTest.streaming}
                        onChange={(e) => setTranslationTest(prev => ({ ...prev, streaming: e.target.checked }))}
                      />
                    }
                    label="Streaming Mode"
                  />
                  
                  <Button
                    variant="contained"
                    startIcon={<TranslateIcon />}
                    onClick={sendTranslationTest}
                    disabled={wsRef.current?.readyState !== WebSocket.OPEN}
                    fullWidth
                  >
                    Test Translation
                  </Button>
                </Box>
              )}

              {/* Whisper Testing */}
              {activeTab === 1 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Alert severity="info">
                    Whisper testing simulates audio transcription sessions
                  </Alert>
                  
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl sx={{ minWidth: 120 }}>
                      <InputLabel>Format</InputLabel>
                      <Select
                        value={whisperTest.audio_format}
                        onChange={(e) => setWhisperTest(prev => ({ ...prev, audio_format: e.target.value }))}
                        label="Format"
                      >
                        <MenuItem value="wav">WAV</MenuItem>
                        <MenuItem value="mp3">MP3</MenuItem>
                        <MenuItem value="webm">WebM</MenuItem>
                      </Select>
                    </FormControl>
                    
                    <FormControl sx={{ minWidth: 120 }}>
                      <InputLabel>Sample Rate</InputLabel>
                      <Select
                        value={whisperTest.sample_rate}
                        onChange={(e) => setWhisperTest(prev => ({ ...prev, sample_rate: Number(e.target.value) }))}
                        label="Sample Rate"
                      >
                        <MenuItem value={16000}>16kHz</MenuItem>
                        <MenuItem value={44100}>44.1kHz</MenuItem>
                        <MenuItem value={48000}>48kHz</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={whisperTest.streaming}
                        onChange={(e) => setWhisperTest(prev => ({ ...prev, streaming: e.target.checked }))}
                      />
                    }
                    label="Streaming Mode"
                  />
                  
                  <Button
                    variant="contained"
                    startIcon={<VolumeIcon />}
                    onClick={sendWhisperTest}
                    disabled={wsRef.current?.readyState !== WebSocket.OPEN}
                    fullWidth
                  >
                    Test Transcription
                  </Button>
                </Box>
              )}

              {/* Custom Message Testing */}
              {activeTab === 2 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Typography variant="subtitle2">Send Custom JSON Message</Typography>
                  <TextField
                    multiline
                    rows={8}
                    value={messageInput}
                    onChange={(e) => setMessageInput(e.target.value)}
                    placeholder='{"action": "test", "data": "your message here"}'
                    fullWidth
                  />
                  <Button
                    variant="contained"
                    startIcon={<SendIcon />}
                    onClick={sendCustomMessage}
                    disabled={wsRef.current?.readyState !== WebSocket.OPEN || !messageInput.trim()}
                    fullWidth
                  >
                    Send Message
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Message Log */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ pb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography variant="h6">Message Log</Typography>
                <Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isAutoScroll}
                        onChange={(e) => setIsAutoScroll(e.target.checked)}
                        size="small"
                      />
                    }
                    label="Auto-scroll"
                  />
                  <Tooltip title="Clear messages">
                    <IconButton onClick={clearMessages} size="small">
                      <ClearIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Download log">
                    <IconButton onClick={downloadMessages} size="small">
                      <DownloadIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {messages.length} messages
              </Typography>
            </CardContent>
            
            <Divider />
            
            <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
              <List dense>
                {messages.map((message) => (
                  <ListItem key={message.id} sx={{ py: 0.5 }}>
                    <Paper 
                      sx={{ 
                        p: 1, 
                        width: '100%',
                        bgcolor: message.type === 'send' ? 'primary.main' : 
                               message.type === 'receive' ? 'success.main' :
                               message.type === 'error' ? 'error.main' : 'grey.100',
                        color: message.type !== 'system' ? 'white' : 'inherit',
                        opacity: message.type === 'system' ? 0.8 : 1,
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Chip 
                          label={message.type} 
                          size="small" 
                          color={getMessageTypeColor(message.type) as any}
                          variant="outlined"
                        />
                        {message.service && getServiceIcon(message.service)}
                        <Typography variant="caption">
                          {message.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Box>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                        {typeof message.data === 'object' 
                          ? JSON.stringify(message.data, null, 2)
                          : message.data
                        }
                      </Typography>
                    </Paper>
                  </ListItem>
                ))}
                <div ref={messagesEndRef} />
              </List>
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default WebSocketTest;