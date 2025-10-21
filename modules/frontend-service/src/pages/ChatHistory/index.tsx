import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Chip,
  Divider,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  InputAdornment,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Stack,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Search as SearchIcon,
  Chat as ChatIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Person as PersonIcon,
  Assistant as AssistantIcon,
  Today as TodayIcon,
  InsertChart as StatsIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import {
  useListSessionsQuery,
  useGetMessagesQuery,
  useDeleteSessionMutation,
  useSearchMessagesQuery,
  useGetSessionStatisticsQuery,
  useExportSessionQuery,
} from '@/store/slices/apiSlice';

// For demo purposes - in production this would come from auth
const DEMO_USER_ID = 'demo_user_123';

const ChatHistory: React.FC = () => {
  const theme = useTheme();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sessionType, setSessionType] = useState('all');
  const [showStats, setShowStats] = useState(false);

  // API hooks
  const {
    data: sessions = [],
    isLoading: sessionsLoading,
    refetch: refetchSessions,
  } = useListSessionsQuery({
    user_id: DEMO_USER_ID,
    session_type: sessionType === 'all' ? undefined : sessionType,
  });

  const {
    data: messages = [],
    isLoading: messagesLoading,
  } = useGetMessagesQuery(
    { session_id: selectedSessionId! },
    { skip: !selectedSessionId }
  );

  const {
    data: statistics,
    isLoading: statsLoading,
  } = useGetSessionStatisticsQuery(selectedSessionId!, {
    skip: !selectedSessionId || !showStats,
  });

  const [deleteSession] = useDeleteSessionMutation();

  // Handlers
  const handleDeleteSession = async (sessionId: string) => {
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      try {
        await deleteSession(sessionId).unwrap();
        if (selectedSessionId === sessionId) {
          setSelectedSessionId(null);
        }
        refetchSessions();
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const handleExport = (sessionId: string, format: 'json' | 'txt') => {
    // Trigger download
    const url = `/api/chat/export/${sessionId}?format=${format}`;
    window.open(url, '_blank');
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChatIcon sx={{ fontSize: 32 }} />
          Chat History
        </Typography>
        <Typography variant="body2" color="text.secondary">
          View and manage your conversation history
        </Typography>
      </Box>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Filter by Type</InputLabel>
            <Select
              value={sessionType}
              label="Filter by Type"
              onChange={(e) => setSessionType(e.target.value)}
            >
              <MenuItem value="all">All Types</MenuItem>
              <MenuItem value="user_chat">User Chat</MenuItem>
              <MenuItem value="bot_meeting">Bot Meeting</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={3}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetchSessions()}
            sx={{ height: '56px' }}
          >
            Refresh
          </Button>
        </Grid>
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Session List */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Conversations
              </Typography>
              {sessionsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : sessions.length === 0 ? (
                <Alert severity="info">No conversations found</Alert>
              ) : (
                <List sx={{ maxHeight: 600, overflow: 'auto' }}>
                  {sessions.map((session: any, index: number) => (
                    <React.Fragment key={session.session_id}>
                      {index > 0 && <Divider />}
                      <ListItemButton
                        selected={selectedSessionId === session.session_id}
                        onClick={() => setSelectedSessionId(session.session_id)}
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                              <Typography variant="subtitle2">
                                {session.session_title || `Session ${session.session_id.slice(0, 8)}`}
                              </Typography>
                              <Chip
                                label={session.message_count}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            </Box>
                          }
                          secondary={
                            <>
                              <Typography variant="caption" display="block">
                                {formatDate(session.started_at)}
                              </Typography>
                              <Chip
                                label={session.session_type}
                                size="small"
                                sx={{ mt: 0.5 }}
                              />
                            </>
                          }
                        />
                        <Box sx={{ ml: 1 }}>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteSession(session.session_id);
                              }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </ListItemButton>
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Messages Display */}
        <Grid item xs={12} md={8}>
          {!selectedSessionId ? (
            <Card sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
              <CardContent>
                <Typography variant="h6" color="text.secondary" align="center">
                  Select a conversation to view messages
                </Typography>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Messages
                  </Typography>
                  <Stack direction="row" spacing={1}>
                    <Tooltip title="Statistics">
                      <IconButton onClick={() => setShowStats(!showStats)} color={showStats ? 'primary' : 'default'}>
                        <StatsIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Export as JSON">
                      <IconButton onClick={() => handleExport(selectedSessionId, 'json')}>
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Export as TXT">
                      <IconButton onClick={() => handleExport(selectedSessionId, 'txt')}>
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </Box>

                {/* Statistics Panel */}
                {showStats && statistics && (
                  <Alert severity="info" sx={{ mb: 2 }} icon={<StatsIcon />}>
                    <Typography variant="subtitle2" gutterBottom>
                      Session Statistics
                    </Typography>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="caption">
                          Total Messages: {statistics.total_messages}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">
                          User Messages: {statistics.user_messages}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">
                          Assistant Messages: {statistics.assistant_messages}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">
                          Total Words: {statistics.total_words}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Alert>
                )}

                {/* Messages List */}
                {messagesLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                    <CircularProgress />
                  </Box>
                ) : messages.length === 0 ? (
                  <Alert severity="info">No messages in this conversation</Alert>
                ) : (
                  <Box sx={{ maxHeight: 500, overflow: 'auto' }}>
                    {messages.map((message: any) => (
                      <Paper
                        key={message.message_id}
                        elevation={1}
                        sx={{
                          p: 2,
                          mb: 2,
                          backgroundColor:
                            message.role === 'user'
                              ? alpha(theme.palette.primary.main, 0.05)
                              : alpha(theme.palette.secondary.main, 0.05),
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                          {message.role === 'user' ? (
                            <PersonIcon color="primary" />
                          ) : (
                            <AssistantIcon color="secondary" />
                          )}
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="subtitle2" color="text.secondary">
                              {message.role.toUpperCase()} #{message.sequence_number}
                            </Typography>
                            <Typography variant="body1" sx={{ mt: 1 }}>
                              {message.content}
                            </Typography>
                            {message.translated_content && (
                              <Box sx={{ mt: 1, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                                <Typography variant="caption" color="text.secondary">
                                  Translations:
                                </Typography>
                                {Object.entries(message.translated_content).map(([lang, text]: [string, any]) => (
                                  <Typography key={lang} variant="body2" sx={{ mt: 0.5 }}>
                                    <strong>{lang}:</strong> {text}
                                  </Typography>
                                ))}
                              </Box>
                            )}
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              {formatDate(message.timestamp)}
                            </Typography>
                          </Box>
                        </Box>
                      </Paper>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ChatHistory;
