import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Slider,
  LinearProgress,
  Chip,
  Tooltip,
  Alert,
  Menu,
  MenuItem,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  VolumeUp,
  VolumeOff,
  Download,
  Compare,
  Waveform,
  MoreVert,
  Loop,
} from '@mui/icons-material';

interface StagePlayerProps {
  stageName: string;
  stageDisplayName: string;
  audioData?: {
    original: string; // base64 audio data
    processed: string; // base64 audio data  
    metadata: {
      duration: number;
      sampleRate: number;
      channels: number;
      format: string;
      processing_time_ms: number;
      quality_metrics: {
        input_rms: number;
        output_rms: number;
        level_change_db: number;
        estimated_snr_db: number;
      };
    };
  };
  onCompare?: (stageName: string) => void;
  showComparison?: boolean;
}

type PlayMode = 'original' | 'processed' | 'comparison';

export const StagePlayer: React.FC<StagePlayerProps> = ({
  stageName,
  stageDisplayName,
  audioData,
  onCompare,
  showComparison = true,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [playMode, setPlayMode] = useState<PlayMode>('processed');
  const [isLooping, setIsLooping] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => {
      setCurrentTime(audio.currentTime);
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(updateTime);
      }
    };

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      if (isLooping) {
        audio.currentTime = 0;
        audio.play();
        setIsPlaying(true);
      } else {
        setCurrentTime(0);
      }
    };

    const handleError = (e: any) => {
      console.error('Audio playback error:', e);
      setIsPlaying(false);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);

    if (isPlaying) {
      updateTime();
    }

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, isLooping]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  useEffect(() => {
    // Update audio source when play mode changes
    if (audioRef.current && audioData) {
      const audioBlob = playMode === 'original' ? audioData.original : audioData.processed;
      const audioUrl = `data:audio/wav;base64,${audioBlob}`;
      audioRef.current.src = audioUrl;
      audioRef.current.load();
    }
  }, [playMode, audioData]);

  const handlePlay = async () => {
    const audio = audioRef.current;
    if (!audio || !audioData) return;

    try {
      if (isPlaying) {
        audio.pause();
        setIsPlaying(false);
      } else {
        await audio.play();
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlaying(false);
    }
  };

  const handleStop = () => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.pause();
    audio.currentTime = 0;
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleSeek = (value: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = value;
    setCurrentTime(value);
  };

  const handleVolumeChange = (value: number) => {
    setVolume(value);
    setIsMuted(false);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleDownload = () => {
    if (!audioData) return;

    const audioBlob = playMode === 'original' ? audioData.original : audioData.processed;
    const blob = new Blob([Uint8Array.from(atob(audioBlob), c => c.charCodeAt(0))], 
      { type: 'audio/wav' });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${stageName}_${playMode}_audio.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getProgressValue = () => {
    if (duration === 0) return 0;
    return (currentTime / duration) * 100;
  };

  if (!audioData) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
            <Typography variant="h6" component="h3">
              {stageDisplayName} Player
            </Typography>
            <Chip label="No Audio" size="small" color="default" />
          </Box>
          
          <Alert severity="info">
            No processed audio available. Upload and process audio through this stage to enable playback.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h3">
            {stageDisplayName} Player
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={playMode === 'original' ? 'Original' : 'Processed'} 
              size="small" 
              color={playMode === 'processed' ? 'primary' : 'default'}
              variant="outlined"
            />
            <IconButton 
              size="small" 
              onClick={(e) => setMenuAnchor(e.currentTarget)}
            >
              <MoreVert />
            </IconButton>
          </Box>
        </Box>

        {/* Audio Element */}
        <audio ref={audioRef} preload="metadata" />

        {/* Play Mode Selector */}
        <Box mb={2}>
          <Typography variant="body2" gutterBottom>
            Playback Mode:
          </Typography>
          <Box display="flex" gap={1}>
            <Chip
              label="Original"
              size="small"
              clickable
              color={playMode === 'original' ? 'primary' : 'default'}
              variant={playMode === 'original' ? 'filled' : 'outlined'}
              onClick={() => setPlayMode('original')}
            />
            <Chip
              label="Processed"
              size="small"
              clickable
              color={playMode === 'processed' ? 'primary' : 'default'}
              variant={playMode === 'processed' ? 'filled' : 'outlined'}
              onClick={() => setPlayMode('processed')}
            />
            {showComparison && (
              <Chip
                label="A/B Compare"
                size="small"
                clickable
                color={playMode === 'comparison' ? 'primary' : 'default'}
                variant={playMode === 'comparison' ? 'filled' : 'outlined'}
                onClick={() => setPlayMode('comparison')}
                icon={<Compare />}
              />
            )}
          </Box>
        </Box>

        {/* Waveform Placeholder */}
        <Box 
          height={60} 
          bgcolor="background.default" 
          borderRadius={1} 
          display="flex" 
          alignItems="center" 
          justifyContent="center"
          mb={2}
          position="relative"
        >
          <Waveform color="disabled" />
          <Typography variant="caption" color="text.secondary" ml={1}>
            Waveform visualization
          </Typography>
          
          {/* Progress overlay */}
          <Box
            position="absolute"
            left={0}
            top={0}
            height="100%"
            width={`${getProgressValue()}%`}
            bgcolor="primary.main"
            borderRadius={1}
            sx={{ opacity: 0.3 }}
          />
        </Box>

        {/* Progress and Time */}
        <Box mb={2}>
          <Slider
            value={currentTime}
            max={duration}
            onChange={(_, value) => handleSeek(value as number)}
            size="small"
            sx={{ mb: 1 }}
          />
          <Box display="flex" justifyContent="space-between">
            <Typography variant="caption" color="text.secondary">
              {formatTime(currentTime)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatTime(duration)}
            </Typography>
          </Box>
        </Box>

        {/* Controls */}
        <Box display="flex" alignItems="center" justifyContent="center" gap={1} mb={2}>
          <IconButton onClick={handleStop} disabled={!audioData}>
            <Stop />
          </IconButton>
          
          <IconButton 
            onClick={handlePlay} 
            disabled={!audioData}
            size="large"
            color="primary"
          >
            {isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>

          <IconButton 
            onClick={() => setIsLooping(!isLooping)}
            color={isLooping ? "primary" : "default"}
          >
            <Loop />
          </IconButton>

          <Box display="flex" alignItems="center" gap={1} ml={2}>
            <IconButton onClick={toggleMute} size="small">
              {isMuted ? <VolumeOff /> : <VolumeUp />}
            </IconButton>
            
            <Slider
              value={volume}
              max={1}
              step={0.1}
              onChange={(_, value) => handleVolumeChange(value as number)}
              size="small"
              sx={{ width: 80 }}
            />
          </Box>
        </Box>

        {/* Audio Info */}
        <Box p={2} bgcolor="background.default" borderRadius={1}>
          <Typography variant="caption" color="text.secondary" display="block" mb={1}>
            Audio Information:
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={1}>
            <Chip 
              label={`${audioData.metadata.duration.toFixed(1)}s`} 
              size="small" 
              variant="outlined" 
            />
            <Chip 
              label={`${audioData.metadata.sampleRate}Hz`} 
              size="small" 
              variant="outlined" 
            />
            <Chip 
              label={`${audioData.metadata.channels}ch`} 
              size="small" 
              variant="outlined" 
            />
            <Chip 
              label={`${audioData.metadata.processing_time_ms.toFixed(1)}ms`} 
              size="small" 
              variant="outlined"
              color="primary"
            />
          </Box>
          
          {/* Quality Metrics */}
          <Box mt={1}>
            <Typography variant="caption" color="text.secondary" display="block">
              Quality Metrics:
            </Typography>
            <Typography variant="caption" component="div">
              Level Change: {audioData.metadata.quality_metrics.level_change_db > 0 ? '+' : ''}
              {audioData.metadata.quality_metrics.level_change_db.toFixed(1)}dB | 
              SNR: {audioData.metadata.quality_metrics.estimated_snr_db.toFixed(1)}dB
            </Typography>
          </Box>
        </Box>

        {/* Options Menu */}
        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={handleDownload}>
            <Download sx={{ mr: 1 }} />
            Download {playMode} audio
          </MenuItem>
          
          {onCompare && (
            <MenuItem onClick={() => {
              onCompare(stageName);
              setMenuAnchor(null);
            }}>
              <Compare sx={{ mr: 1 }} />
              Compare with other stages
            </MenuItem>
          )}
          
          <Divider />
          
          <MenuItem onClick={() => {
            setIsLooping(!isLooping);
            setMenuAnchor(null);
          }}>
            <Loop sx={{ mr: 1 }} />
            {isLooping ? 'Disable' : 'Enable'} loop
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};