import React from 'react';
import {
  Box,
  Button,
  Typography,
  Stack,
  Chip,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  Mic as MicIcon,
  Stop as StopIcon,
  PlayArrow as PlayIcon,
  Download as DownloadIcon,
  Clear as ClearIcon,
  Pause as PauseIcon,
} from '@mui/icons-material';

interface RecordingControlsProps {
  onStartRecording: () => void;
  onStopRecording: () => void;
  onPlayRecording: () => void;
  onDownloadRecording: () => void;
  onClearRecording: () => void;
  isRecording: boolean;
  isPlaying: boolean;
  hasRecording: boolean;
  recordingDuration: number;
}

export const RecordingControls: React.FC<RecordingControlsProps> = ({
  onStartRecording,
  onStopRecording,
  onPlayRecording,
  onDownloadRecording,
  onClearRecording,
  isRecording,
  isPlaying,
  hasRecording,
  recordingDuration,
}) => {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getRecordingStatus = () => {
    if (isRecording) {
      return {
        text: 'Recording in progress...',
        color: 'error' as const,
        variant: 'filled' as const,
      };
    }
    if (hasRecording) {
      return {
        text: 'Recording completed',
        color: 'success' as const,
        variant: 'filled' as const,
      };
    }
    return {
      text: 'Ready to record',
      color: 'primary' as const,
      variant: 'outlined' as const,
    };
  };

  const status = getRecordingStatus();

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Stack direction="row" spacing={2} alignItems="center">
            <Button
              variant={isRecording ? "outlined" : "contained"}
              color={isRecording ? "error" : "primary"}
              startIcon={isRecording ? <StopIcon /> : <MicIcon />}
              onClick={isRecording ? onStopRecording : onStartRecording}
              size="large"
              sx={{ minWidth: 160 }}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>
            
            <Button
              variant="outlined"
              startIcon={isPlaying ? <PauseIcon /> : <PlayIcon />}
              onClick={onPlayRecording}
              disabled={!hasRecording}
              size="large"
            >
              {isPlaying ? 'Stop Playing' : 'Play Recording'}
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={onDownloadRecording}
              disabled={!hasRecording}
            >
              Download
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<ClearIcon />}
              onClick={onClearRecording}
              color="warning"
            >
              Clear
            </Button>
          </Stack>
          
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h4" component="div" fontFamily="monospace" color="primary.main">
              {formatDuration(recordingDuration)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Duration
            </Typography>
          </Box>
        </Box>
        
        <Chip
          label={status.text}
          color={status.color}
          variant={status.variant}
          sx={{ mb: 1 }}
        />
        
        {isRecording && (
          <LinearProgress 
            sx={{ mt: 2, height: 6, borderRadius: 3 }}
            color="error"
          />
        )}
      </Box>
    </Paper>
  );
};