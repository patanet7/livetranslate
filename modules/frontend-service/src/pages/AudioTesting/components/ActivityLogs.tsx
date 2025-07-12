import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Info as InfoIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store';
import { clearProcessingLogs } from '@/store/slices/audioSlice';
import { ProcessingLog } from '@/types/audio';

const getLogIcon = (level: string) => {
  switch (level) {
    case 'INFO':
      return <InfoIcon color="info" />;
    case 'SUCCESS':
      return <SuccessIcon color="success" />;
    case 'ERROR':
      return <ErrorIcon color="error" />;
    case 'WARNING':
      return <WarningIcon color="warning" />;
    default:
      return <InfoIcon color="disabled" />;
  }
};

const getLogColor = (level: string) => {
  switch (level) {
    case 'INFO':
      return 'info';
    case 'SUCCESS':
      return 'success';
    case 'ERROR':
      return 'error';
    case 'WARNING':
      return 'warning';
    default:
      return 'default';
  }
};

export const ActivityLogs: React.FC = () => {
  const dispatch = useAppDispatch();
  const { logs } = useAppSelector(state => state.audio.processing);
  const [filterOpen, setFilterOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState<string>('all');

  const filteredLogs = logs.filter(log => {
    const matchesSearch = log.message.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = levelFilter === 'all' || log.level === levelFilter;
    return matchesSearch && matchesLevel;
  });

  const handleClearLogs = () => {
    dispatch(clearProcessingLogs());
  };

  const handleExportLogs = () => {
    const logsData = {
      exportDate: new Date().toISOString(),
      totalLogs: logs.length,
      logs: logs.map(log => ({
        timestamp: new Date(log.timestamp).toISOString(),
        level: log.level,
        message: log.message,
      })),
    };

    const dataStr = JSON.stringify(logsData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `audio-test-logs-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getLogStats = () => {
    const stats = logs.reduce((acc, log) => {
      acc[log.level] = (acc[log.level] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      total: logs.length,
      info: stats.INFO || 0,
      success: stats.SUCCESS || 0,
      warning: stats.WARNING || 0,
      error: stats.ERROR || 0,
    };
  };

  const stats = getLogStats();

  return (
    <Box>
      {/* Header with controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          📋 Activity Logs
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<FilterIcon />}
            onClick={() => setFilterOpen(true)}
            size="small"
          >
            Filter
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportLogs}
            disabled={logs.length === 0}
            size="small"
          >
            Export
          </Button>
          <Button
            variant="outlined"
            startIcon={<ClearIcon />}
            onClick={handleClearLogs}
            disabled={logs.length === 0}
            size="small"
            color="warning"
          >
            Clear
          </Button>
        </Stack>
      </Box>

      {/* Log statistics */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Log Statistics
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          <Chip
            label={`Total: ${stats.total}`}
            variant="outlined"
          />
          <Chip
            label={`Info: ${stats.info}`}
            color="info"
            variant="outlined"
          />
          <Chip
            label={`Success: ${stats.success}`}
            color="success"
            variant="outlined"
          />
          <Chip
            label={`Warning: ${stats.warning}`}
            color="warning"
            variant="outlined"
          />
          <Chip
            label={`Error: ${stats.error}`}
            color="error"
            variant="outlined"
          />
        </Stack>
      </Paper>

      {/* Search bar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <TextField
          fullWidth
          placeholder="Search logs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
          }}
          size="small"
        />
      </Paper>

      {/* Logs list */}
      <Paper sx={{ maxHeight: 400, overflow: 'auto' }}>
        {filteredLogs.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="text.secondary">
              {logs.length === 0 ? 'No logs available' : 'No logs match the current filter'}
            </Typography>
          </Box>
        ) : (
          <List>
            {filteredLogs.map((log, index) => (
              <ListItem key={index} sx={{ py: 1 }}>
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {getLogIcon(log.level)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography
                        variant="body2"
                        fontFamily="monospace"
                        color="text.secondary"
                        sx={{ minWidth: 80 }}
                      >
                        {formatTimestamp(log.timestamp)}
                      </Typography>
                      <Chip
                        label={log.level}
                        size="small"
                        color={getLogColor(log.level) as any}
                        sx={{ minWidth: 70 }}
                      />
                      <Typography variant="body2" sx={{ flexGrow: 1 }}>
                        {log.message}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* Filter Dialog */}
      <Dialog open={filterOpen} onClose={() => setFilterOpen(false)}>
        <DialogTitle>Filter Logs</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 1, minWidth: 300 }}>
            <TextField
              label="Search"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              fullWidth
              placeholder="Search in messages..."
            />
            
            <FormControl fullWidth>
              <InputLabel>Log Level</InputLabel>
              <Select
                value={levelFilter}
                label="Log Level"
                onChange={(e) => setLevelFilter(e.target.value)}
              >
                <MenuItem value="all">All Levels</MenuItem>
                <MenuItem value="INFO">Info</MenuItem>
                <MenuItem value="SUCCESS">Success</MenuItem>
                <MenuItem value="WARNING">Warning</MenuItem>
                <MenuItem value="ERROR">Error</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFilterOpen(false)}>
            Close
          </Button>
          <Button
            onClick={() => {
              setSearchTerm('');
              setLevelFilter('all');
            }}
            variant="outlined"
          >
            Clear Filters
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};