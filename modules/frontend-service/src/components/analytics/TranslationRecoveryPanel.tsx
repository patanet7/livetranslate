import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import {
  Alert,
  alpha,
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
  useTheme,
} from '@mui/material';
import { ArrowForward, DoneAll, Refresh, Storage, Sync, WarningAmber } from '@mui/icons-material';

import {
  useGetMeetingTranslationBacklogQuery,
  useGetMeetingTranslationStatusQuery,
  useRecoverMeetingTranslationsMutation,
} from '@/store/slices/apiSlice';

type TranslationRecoveryPanelProps = {
  compact?: boolean;
};

const formatDateValue = (value: string | number | null | undefined): string => {
  if (!value) {
    return 'Never';
  }

  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'Unknown';
  }

  return date.toLocaleString();
};

const getBacklogColor = (count: number): 'success' | 'warning' | 'error' => {
  if (count === 0) {
    return 'success';
  }
  if (count < 25) {
    return 'warning';
  }
  return 'error';
};

const TranslationRecoveryPanel: React.FC<TranslationRecoveryPanelProps> = ({ compact = false }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [selectedMeetingId, setSelectedMeetingId] = React.useState<string | null>(null);
  const [recoveringMeetingId, setRecoveringMeetingId] = React.useState<string | null>(null);

  const {
    data: backlogData,
    isLoading,
    isFetching,
    error,
    refetch,
  } = useGetMeetingTranslationBacklogQuery({
    limit: compact ? 5 : 25,
    offset: 0,
    onlyPending: true,
  });
  const meetings = backlogData?.meetings || [];
  const summary = backlogData?.summary;
  const recoveryCounters = backlogData?.recovery_counters;

  React.useEffect(() => {
    if (meetings.length === 0) {
      setSelectedMeetingId(null);
      return;
    }

    if (!selectedMeetingId || !meetings.some((meeting) => meeting.id === selectedMeetingId)) {
      setSelectedMeetingId(meetings[0].id);
    }
  }, [meetings, selectedMeetingId]);

  const { data: selectedStatusData, isFetching: selectedStatusLoading } =
    useGetMeetingTranslationStatusQuery(selectedMeetingId || '', {
      skip: !selectedMeetingId || compact,
    });

  const [recoverMeetingTranslations] = useRecoverMeetingTranslationsMutation();

  const handleRecover = async (meetingId: string) => {
    try {
      setRecoveringMeetingId(meetingId);
      const result = await recoverMeetingTranslations({ meetingId, limit: 500 }).unwrap();
      enqueueSnackbar(
        `Recovery completed for ${result.meeting_id}: ${result.recovery.recovered} recovered, ${result.recovery.failed} failed.`,
        { variant: result.recovery.failed > 0 ? 'warning' : 'success' }
      );
      await refetch();
    } catch (recoverError: any) {
      const detail =
        recoverError?.data?.detail || recoverError?.error || 'Translation recovery request failed';
      enqueueSnackbar(detail, { variant: 'error' });
    } finally {
      setRecoveringMeetingId(null);
    }
  };

  const selectedStatus = selectedStatusData?.translation_status;
  const pendingTotal = summary?.pending_translation_count || 0;
  const pendingMeetings = backlogData?.total || 0;

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              mb: 2,
            }}
          >
            <Box>
              <Typography variant="h6" component="h2">
                Translation Recovery
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Async backlog and replay controls for loopback and Fireflies meetings
              </Typography>
            </Box>
            <Tooltip title="Refresh backlog">
              <span>
                <IconButton onClick={() => refetch()} disabled={isFetching}>
                  <Refresh />
                </IconButton>
              </span>
            </Tooltip>
          </Box>

          {isLoading ? (
            <Box sx={{ py: 2 }}>
              <LinearProgress />
            </Box>
          ) : error ? (
            <Alert severity="error">
              Unable to load translation backlog. Check orchestration service health.
            </Alert>
          ) : (
            <Stack spacing={2}>
              <Grid container spacing={1.5}>
                <Grid item xs={6}>
                  <Paper variant="outlined" sx={{ p: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      Pending Units
                    </Typography>
                    <Typography variant="h5" color={`${getBacklogColor(pendingTotal)}.main`}>
                      {pendingTotal.toLocaleString()}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6}>
                  <Paper variant="outlined" sx={{ p: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      Meetings Queued
                    </Typography>
                    <Typography variant="h5">{pendingMeetings.toLocaleString()}</Typography>
                  </Paper>
                </Grid>
              </Grid>

              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: alpha(theme.palette.info.main, 0.08),
                  border: `1px solid ${alpha(theme.palette.info.main, 0.18)}`,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Last recovery run
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  {formatDateValue(recoveryCounters?.last_run_completed_at)} • recovered{' '}
                  {recoveryCounters?.last_run_recovered || 0} / pending{' '}
                  {recoveryCounters?.last_run_pending || 0}
                </Typography>
              </Box>

              {meetings.length === 0 ? (
                <Alert severity="success" icon={<DoneAll />}>
                  No pending translation backlog.
                </Alert>
              ) : (
                <Stack spacing={1}>
                  {meetings.slice(0, 3).map((meeting) => (
                    <Paper
                      key={meeting.id}
                      variant="outlined"
                      sx={{
                        p: 1.5,
                        display: 'flex',
                        justifyContent: 'space-between',
                        gap: 2,
                        alignItems: 'center',
                      }}
                    >
                      <Box sx={{ minWidth: 0, flex: 1 }}>
                        <Typography variant="subtitle2" noWrap>
                          {meeting.title || meeting.id}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {meeting.source} • {meeting.pending_translation_count} pending
                        </Typography>
                      </Box>
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={
                          recoveringMeetingId === meeting.id ? (
                            <CircularProgress size={14} />
                          ) : (
                            <Sync />
                          )
                        }
                        onClick={() => void handleRecover(meeting.id)}
                        disabled={recoveringMeetingId === meeting.id}
                      >
                        Recover
                      </Button>
                    </Paper>
                  ))}
                </Stack>
              )}
            </Stack>
          )}
        </CardContent>
        <CardActions>
          <Button
            size="small"
            fullWidth
            startIcon={<Storage />}
            endIcon={<ArrowForward />}
            onClick={() => navigate('/system-analytics')}
          >
            Open Recovery Console
          </Button>
        </CardActions>
      </Card>
    );
  }

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 0.5 }}>
            Translation Recovery Console
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Monitor pending chunk and sentence translations, including Fireflies imports that land
            as sentence rows before asynchronous translation catches up.
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetch()}
          disabled={isFetching}
        >
          Refresh Backlog
        </Button>
      </Box>

      {isLoading ? (
        <LinearProgress />
      ) : error ? (
        <Alert severity="error">
          Unable to load backlog telemetry. Validate database connectivity and orchestration health
          before retrying.
        </Alert>
      ) : (
        <Stack spacing={3}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Pending Translation Units
                </Typography>
                <Typography variant="h4" color={`${getBacklogColor(pendingTotal)}.main`}>
                  {pendingTotal.toLocaleString()}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Pending Meetings
                </Typography>
                <Typography variant="h4">{pendingMeetings.toLocaleString()}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Final Chunk Backlog
                </Typography>
                <Typography variant="h4">
                  {(summary?.pending_chunk_translation_count || 0).toLocaleString()}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={3}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Sentence Backlog
                </Typography>
                <Typography variant="h4">
                  {(summary?.pending_sentence_translation_count || 0).toLocaleString()}
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          <Grid container spacing={2}>
            <Grid item xs={12} lg={4}>
              <Card
                sx={{
                  height: '100%',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
              >
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recovery Counters
                  </Typography>
                  <Stack spacing={1.5}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Last run completed
                      </Typography>
                      <Typography variant="body2">
                        {formatDateValue(recoveryCounters?.last_run_completed_at)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Last run scope
                      </Typography>
                      <Typography variant="body2">
                        {recoveryCounters?.last_run_scope || 'all'}
                      </Typography>
                    </Box>
                    <Divider />
                    <Grid container spacing={1.5}>
                      <Grid item xs={4}>
                        <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center' }}>
                          <Typography variant="caption" color="text.secondary">
                            Runs
                          </Typography>
                          <Typography variant="h6">{recoveryCounters?.runs || 0}</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center' }}>
                          <Typography variant="caption" color="text.secondary">
                            Recovered
                          </Typography>
                          <Typography variant="h6">
                            {recoveryCounters?.total_recovered || 0}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center' }}>
                          <Typography variant="caption" color="text.secondary">
                            Failed
                          </Typography>
                          <Typography
                            variant="h6"
                            color={
                              (recoveryCounters?.total_failed || 0) > 0
                                ? 'error.main'
                                : 'text.primary'
                            }
                          >
                            {recoveryCounters?.total_failed || 0}
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={8}>
              <Card
                sx={{
                  height: '100%',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
              >
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Selected Meeting
                  </Typography>
                  {!selectedStatus ? (
                    <Alert severity="info" icon={<Storage />}>
                      Select a meeting with pending work to inspect its recovery posture.
                    </Alert>
                  ) : (
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {selectedStatus.title || selectedStatus.id}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {selectedStatus.source} • created{' '}
                          {formatDateValue(selectedStatus.created_at || null)}
                        </Typography>
                      </Box>
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        <Chip
                          label={`Status: ${selectedStatus.status}`}
                          size="small"
                          color="default"
                        />
                        <Chip
                          label={`Pending: ${selectedStatus.pending_translation_count}`}
                          size="small"
                          color={getBacklogColor(selectedStatus.pending_translation_count)}
                        />
                        <Chip
                          label={`Stored translations: ${selectedStatus.translation_count}`}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                        {(selectedStatus.target_languages || []).map((language) => (
                          <Chip key={language} label={language} size="small" variant="outlined" />
                        ))}
                      </Stack>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                              Final chunk backlog
                            </Typography>
                            <Typography variant="h5">
                              {selectedStatus.pending_chunk_translation_count}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                              Sentence backlog
                            </Typography>
                            <Typography variant="h5">
                              {selectedStatus.pending_sentence_translation_count}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          {selectedStatusLoading
                            ? 'Refreshing meeting detail...'
                            : 'Per-meeting backlog detail'}
                        </Typography>
                        <Button
                          variant="contained"
                          startIcon={
                            recoveringMeetingId === selectedStatus.id ? (
                              <CircularProgress size={16} color="inherit" />
                            ) : (
                              <Sync />
                            )
                          }
                          onClick={() => void handleRecover(selectedStatus.id)}
                          disabled={recoveringMeetingId === selectedStatus.id}
                        >
                          Recover Selected Meeting
                        </Button>
                      </Box>
                    </Stack>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {meetings.length === 0 ? (
            <Alert severity="success" icon={<DoneAll />}>
              Translation backlog is clear. No loopback chunks or Fireflies sentences are waiting
              for recovery.
            </Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Meeting</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Pending Chunks</TableCell>
                    <TableCell align="right">Pending Sentences</TableCell>
                    <TableCell align="right">Pending Total</TableCell>
                    <TableCell align="right">Stored Translations</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {meetings.map((meeting) => {
                    const isSelected = meeting.id === selectedMeetingId;
                    return (
                      <TableRow
                        key={meeting.id}
                        hover
                        selected={isSelected}
                        onClick={() => setSelectedMeetingId(meeting.id)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {meeting.title || meeting.id}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {meeting.id}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip label={meeting.source} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell>
                          <Chip label={meeting.status} size="small" color="default" />
                        </TableCell>
                        <TableCell align="right">
                          {meeting.pending_chunk_translation_count}
                        </TableCell>
                        <TableCell align="right">
                          {meeting.pending_sentence_translation_count}
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            color={`${getBacklogColor(meeting.pending_translation_count)}.main`}
                            sx={{ fontWeight: 600 }}
                          >
                            {meeting.pending_translation_count}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">{meeting.translation_count}</TableCell>
                        <TableCell>{formatDateValue(meeting.created_at || null)}</TableCell>
                        <TableCell align="right">
                          <Button
                            size="small"
                            variant={isSelected ? 'contained' : 'outlined'}
                            color={meeting.pending_translation_count > 0 ? 'warning' : 'primary'}
                            startIcon={
                              recoveringMeetingId === meeting.id ? (
                                <CircularProgress size={14} color="inherit" />
                              ) : (
                                <Sync />
                              )
                            }
                            onClick={(event) => {
                              event.stopPropagation();
                              void handleRecover(meeting.id);
                            }}
                            disabled={recoveringMeetingId === meeting.id}
                          >
                            Recover
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          <Alert severity={pendingTotal > 0 ? 'warning' : 'success'} icon={<WarningAmber />}>
            Fireflies imports stay fast by persisting transcript sentences first and translating
            asynchronously. This console shows the same shared recovery path used for loopback
            chunks and Fireflies sentence-backed transcripts.
          </Alert>
        </Stack>
      )}
    </Box>
  );
};

export default TranslationRecoveryPanel;
