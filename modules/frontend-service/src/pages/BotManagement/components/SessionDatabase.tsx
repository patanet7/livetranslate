import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  InputAdornment,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Search as SearchIcon,
  Visibility as VisibilityIcon,
  Download as DownloadIcon,
  FilterList as FilterListIcon,
  ExpandMore as ExpandMoreIcon,
  Storage as StorageIcon,
  Translate as TranslateIcon,
  Person as PersonIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { DatePicker as MuiDatePicker } from '@mui/x-date-pickers/DatePicker';
import { format, parseISO } from 'date-fns';
import { TabPanel } from '@/components/ui';

interface SessionDatabaseProps {
  onError: (error: string) => void;
}

interface BotSession {
  id: string;
  botId: string;
  meetingId: string;
  meetingTitle: string;
  organizerEmail: string;
  startTime: string;
  endTime: string | null;
  status: 'active' | 'completed' | 'error' | 'terminated';
  participantCount: number;
  totalAudioChunks: number;
  totalCaptions: number;
  totalTranslations: number;
  averageLatency: number;
  qualityScore: number;
  errorCount: number;
  createdAt: string;
  updatedAt: string;
}

interface Translation {
  id: string;
  sessionId: string;
  botId: string;
  originalText: string;
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
  speakerName: string;
  confidence: number;
  timestamp: string;
  processingTime: number;
}

interface SpeakerActivity {
  id: string;
  sessionId: string;
  speakerId: string;
  speakerName: string;
  eventType: 'join' | 'leave' | 'speaking_start' | 'speaking_end';
  timestamp: string;
  confidence: number;
  duration?: number;
}

export const SessionDatabase: React.FC<SessionDatabaseProps> = ({ onError }) => {
  const [tabValue, setTabValue] = useState(0);
  const [sessions, setSessions] = useState<BotSession[]>([]);
  const [translations, setTranslations] = useState<Translation[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerActivity[]>([]);
  const [selectedSession, setSelectedSession] = useState<BotSession | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Helper function to safely format dates
  const safeFormatDate = (dateString: string, formatString: string): string => {
    try {
      const date = parseISO(dateString);
      if (isNaN(date.getTime())) {
        return 'Invalid date';
      }
      return format(date, formatString);
    } catch (error) {
      return 'Invalid date';
    }
  };
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [translationPage, setTranslationPage] = useState(0);
  const [speakerPage, setSpeakerPage] = useState(0);
  
  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateFrom, setDateFrom] = useState<Date | null>(null);
  const [dateTo, setDateTo] = useState<Date | null>(null);
  const [languageFilter, setLanguageFilter] = useState('all');

  useEffect(() => {
    loadSessions();
    loadTranslations();
    loadSpeakerActivity();
  }, []);

  const loadSessions = async () => {
    try {
      const response = await fetch('/api/bot/sessions');
      if (!response.ok) {
        console.warn(`Sessions API failed with status ${response.status}, using empty array`);
        setSessions([]);
        return;
      }
      const data = await response.json();
      setSessions(data);
    } catch (error) {
      console.error('Error loading sessions:', error);
      setSessions([]);
      onError('Failed to load session data');
    }
  };

  const loadTranslations = async () => {
    try {
      const response = await fetch('/api/bot/translations');
      if (!response.ok) {
        console.warn(`Translations API failed with status ${response.status}, using empty array`);
        setTranslations([]);
        return;
      }
      const data = await response.json();
      setTranslations(data);
    } catch (error) {
      console.error('Error loading translations:', error);
      setTranslations([]);
      onError('Failed to load translations');
    }
  };

  const loadSpeakerActivity = async () => {
    try {
      const response = await fetch('/api/bot/speaker-activity');
      if (!response.ok) {
        console.warn(`Speaker activity API failed with status ${response.status}, using empty array`);
        setSpeakers([]);
        return;
      }
      const data = await response.json();
      setSpeakers(data);
    } catch (error) {
      console.error('Error loading speaker activity:', error);
      setSpeakers([]);
      onError('Failed to load speaker activity');
    }
  };

  const handleViewDetails = (session: BotSession) => {
    setSelectedSession(session);
    setDetailsOpen(true);
  };

  const handleExportSession = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/bot/sessions/${sessionId}/export`);
      if (!response.ok) throw new Error('Failed to export session');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${sessionId}_export.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      onError('Failed to export session');
    }
  };

  const filteredSessions = sessions.filter(session => {
    const matchesSearch = !searchTerm || 
      session.meetingTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.meetingId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.organizerEmail.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || session.status === statusFilter;
    
    const matchesDate = (!dateFrom || new Date(session.startTime) >= dateFrom) &&
                       (!dateTo || new Date(session.startTime) <= dateTo);
    
    return matchesSearch && matchesStatus && matchesDate;
  });

  const filteredTranslations = translations.filter(translation => {
    const matchesSearch = !searchTerm ||
      translation.originalText.toLowerCase().includes(searchTerm.toLowerCase()) ||
      translation.translatedText.toLowerCase().includes(searchTerm.toLowerCase()) ||
      translation.speakerName.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesLanguage = languageFilter === 'all' || 
      translation.sourceLanguage === languageFilter ||
      translation.targetLanguage === languageFilter;
    
    return matchesSearch && matchesLanguage;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'completed': return 'info';
      case 'error': return 'error';
      case 'terminated': return 'warning';
      default: return 'default';
    }
  };

  const formatDuration = (startTime: string, endTime?: string | null) => {
    if (!endTime) return 'Ongoing';
    
    const start = new Date(startTime);
    const end = new Date(endTime);
    const diffMs = end.getTime() - start.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    
    if (diffHours > 0) {
      return `${diffHours}h ${diffMins % 60}m`;
    }
    return `${diffMins}m`;
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6">
            <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Session Database
          </Typography>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={() => {
              // Export all data functionality
              const exportData = {
                sessions: filteredSessions,
                translations: filteredTranslations,
                speakers: speakers,
                exportDate: new Date().toISOString()
              };
              
              const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
              });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `livetranslate_export_${format(new Date(), 'yyyy-MM-dd_HH-mm-ss')}.json`;
              document.body.appendChild(a);
              a.click();
              URL.revokeObjectURL(url);
              document.body.removeChild(a);
            }}
          >
            Export All Data
          </Button>
        </Box>

        {/* Filters */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">
                  <FilterListIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Filters & Search
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="Search"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <SearchIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <FormControl fullWidth>
                      <InputLabel>Status</InputLabel>
                      <Select
                        value={statusFilter}
                        label="Status"
                        onChange={(e) => setStatusFilter(e.target.value)}
                      >
                        <MenuItem value="all">All</MenuItem>
                        <MenuItem value="active">Active</MenuItem>
                        <MenuItem value="completed">Completed</MenuItem>
                        <MenuItem value="error">Error</MenuItem>
                        <MenuItem value="terminated">Terminated</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <FormControl fullWidth>
                      <InputLabel>Language</InputLabel>
                      <Select
                        value={languageFilter}
                        label="Language"
                        onChange={(e) => setLanguageFilter(e.target.value)}
                      >
                        <MenuItem value="all">All</MenuItem>
                        <MenuItem value="en">English</MenuItem>
                        <MenuItem value="es">Spanish</MenuItem>
                        <MenuItem value="fr">French</MenuItem>
                        <MenuItem value="de">German</MenuItem>
                        <MenuItem value="it">Italian</MenuItem>
                        <MenuItem value="pt">Portuguese</MenuItem>
                        <MenuItem value="ja">Japanese</MenuItem>
                        <MenuItem value="ko">Korean</MenuItem>
                        <MenuItem value="zh">Chinese</MenuItem>
                        <MenuItem value="ru">Russian</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <MuiDatePicker
                      label="Date From"
                      value={dateFrom}
                      onChange={(date) => setDateFrom(date)}
                      slotProps={{ textField: { fullWidth: true } }}
                    />
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <MuiDatePicker
                      label="Date To"
                      value={dateTo}
                      onChange={(date) => setDateTo(date)}
                      slotProps={{ textField: { fullWidth: true } }}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Paper sx={{ width: '100%' }}>
          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
            <Tab
              icon={<AssessmentIcon />}
              label={`Sessions (${filteredSessions.length})`}
            />
            <Tab
              icon={<TranslateIcon />}
              label={`Translations (${filteredTranslations.length})`}
            />
            <Tab
              icon={<PersonIcon />}
              label={`Speaker Activity (${speakers.length})`}
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {/* Sessions Table */}
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Meeting</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Participants</TableCell>
                    <TableCell>Translations</TableCell>
                    <TableCell>Quality</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredSessions
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((session) => (
                      <TableRow key={session.id}>
                        <TableCell>
                          <Box>
                            <Typography variant="body2" fontWeight="bold">
                              {session.meetingTitle || 'Untitled Meeting'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {session.meetingId} • {safeFormatDate(session.startTime, 'MMM dd, yyyy HH:mm')}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={session.status}
                            color={getStatusColor(session.status)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {formatDuration(session.startTime, session.endTime)}
                        </TableCell>
                        <TableCell>{session.participantCount}</TableCell>
                        <TableCell>{session.totalTranslations.toLocaleString()}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ mr: 1 }}>
                              {Math.round(session.qualityScore * 100)}%
                            </Typography>
                            <Chip
                              label={session.qualityScore > 0.8 ? 'Good' : session.qualityScore > 0.6 ? 'Fair' : 'Poor'}
                              color={session.qualityScore > 0.8 ? 'success' : session.qualityScore > 0.6 ? 'warning' : 'error'}
                              size="small"
                              variant="outlined"
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <IconButton
                            onClick={() => handleViewDetails(session)}
                            size="small"
                            title="View Details"
                          >
                            <VisibilityIcon />
                          </IconButton>
                          <IconButton
                            onClick={() => handleExportSession(session.id)}
                            size="small"
                            title="Export Session"
                          >
                            <DownloadIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              rowsPerPageOptions={[10, 25, 50]}
              component="div"
              count={filteredSessions.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={(_, newPage) => setPage(newPage)}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
            />
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {/* Translations Table */}
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Original Text</TableCell>
                    <TableCell>Translation</TableCell>
                    <TableCell>Speaker</TableCell>
                    <TableCell>Languages</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredTranslations
                    .slice(translationPage * rowsPerPage, translationPage * rowsPerPage + rowsPerPage)
                    .map((translation) => (
                      <TableRow key={translation.id}>
                        <TableCell sx={{ maxWidth: 200 }}>
                          <Typography variant="body2" noWrap>
                            {translation.originalText}
                          </Typography>
                        </TableCell>
                        <TableCell sx={{ maxWidth: 200 }}>
                          <Typography variant="body2" noWrap>
                            {translation.translatedText}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={translation.speakerName}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={1}>
                            <Chip label={translation.sourceLanguage.toUpperCase()} size="small" />
                            <Typography variant="body2">→</Typography>
                            <Chip label={translation.targetLanguage.toUpperCase()} size="small" />
                          </Stack>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2">
                              {Math.round(translation.confidence * 100)}%
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption">
                            {safeFormatDate(translation.timestamp, 'HH:mm:ss')}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              rowsPerPageOptions={[10, 25, 50]}
              component="div"
              count={filteredTranslations.length}
              rowsPerPage={rowsPerPage}
              page={translationPage}
              onPageChange={(_, newPage) => setTranslationPage(newPage)}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setTranslationPage(0);
              }}
            />
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {/* Speaker Activity Table */}
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Speaker</TableCell>
                    <TableCell>Event</TableCell>
                    <TableCell>Session</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {speakers
                    .slice(speakerPage * rowsPerPage, speakerPage * rowsPerPage + rowsPerPage)
                    .map((speaker) => (
                      <TableRow key={speaker.id}>
                        <TableCell>
                          <Chip
                            label={speaker.speakerName}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={speaker.eventType.replace('_', ' ')}
                            size="small"
                            color={speaker.eventType.includes('join') || speaker.eventType.includes('start') ? 'success' : 'warning'}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption">
                            {speaker.sessionId.slice(0, 8)}...
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {Math.round(speaker.confidence * 100)}%
                        </TableCell>
                        <TableCell>
                          {speaker.duration ? `${speaker.duration}s` : 'N/A'}
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption">
                            {safeFormatDate(speaker.timestamp, 'MMM dd, HH:mm:ss')}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              rowsPerPageOptions={[10, 25, 50]}
              component="div"
              count={speakers.length}
              rowsPerPage={rowsPerPage}
              page={speakerPage}
              onPageChange={(_, newPage) => setSpeakerPage(newPage)}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setSpeakerPage(0);
              }}
            />
          </TabPanel>
        </Paper>

        {/* Session Details Modal */}
        <Dialog
          open={detailsOpen}
          onClose={() => setDetailsOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            Session Details: {selectedSession?.meetingTitle || 'Untitled Meeting'}
          </DialogTitle>
          <DialogContent>
            {selectedSession && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Basic Information
                  </Typography>
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Bot ID:</Typography>
                      <Typography variant="body2">{selectedSession.botId}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Meeting ID:</Typography>
                      <Typography variant="body2">{selectedSession.meetingId}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Organizer:</Typography>
                      <Typography variant="body2">{selectedSession.organizerEmail}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Status:</Typography>
                      <Chip
                        label={selectedSession.status}
                        color={getStatusColor(selectedSession.status)}
                        size="small"
                      />
                    </Box>
                  </Stack>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Total Audio Chunks:</Typography>
                      <Typography variant="body2">{selectedSession.totalAudioChunks.toLocaleString()}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Total Captions:</Typography>
                      <Typography variant="body2">{selectedSession.totalCaptions.toLocaleString()}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Total Translations:</Typography>
                      <Typography variant="body2">{selectedSession.totalTranslations.toLocaleString()}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Average Latency:</Typography>
                      <Typography variant="body2">{selectedSession.averageLatency}ms</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Quality Score:</Typography>
                      <Typography variant="body2">{Math.round(selectedSession.qualityScore * 100)}%</Typography>
                    </Box>
                  </Stack>
                </Grid>
              </Grid>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailsOpen(false)}>Close</Button>
            {selectedSession && (
              <Button
                onClick={() => handleExportSession(selectedSession.id)}
                variant="contained"
                startIcon={<DownloadIcon />}
              >
                Export
              </Button>
            )}
          </DialogActions>
        </Dialog>
      </Box>
    </LocalizationProvider>
  );
};