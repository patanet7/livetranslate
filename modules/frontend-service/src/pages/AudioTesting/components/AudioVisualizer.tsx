import React, { useRef, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Stack,
  Chip,
} from '@mui/material';

interface AudioVisualizerProps {
  frequencyData: number[];
  timeData: number[];
  audioLevel: number;
  isRecording: boolean;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  frequencyData,
  timeData,
  audioLevel,
  isRecording,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  const drawVisualization = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frequencyData || frequencyData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Draw frequency spectrum
    const barWidth = (width / frequencyData.length) * 2.5;
    let x = 0;

    for (let i = 0; i < frequencyData.length; i++) {
      const barHeight = (frequencyData[i] / 255) * height;
      
      // Color gradient based on frequency
      const hue = (i / frequencyData.length) * 120;
      ctx.fillStyle = `hsl(${120 - hue}, 70%, 50%)`;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }

    // Draw waveform overlay
    if (timeData && timeData.length > 0) {
      ctx.strokeStyle = isRecording ? '#ff4757' : '#4ECDC4';
      ctx.lineWidth = 2;
      ctx.beginPath();

      const sliceWidth = width / timeData.length;
      let x = 0;

      for (let i = 0; i < timeData.length; i++) {
        const v = timeData[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.stroke();
    }
  }, [frequencyData, timeData, isRecording]);

  useEffect(() => {
    const animate = () => {
      drawVisualization();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawVisualization]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  const getLevelColor = (level: number) => {
    if (level < 20) return 'success';
    if (level < 70) return 'warning';
    return 'error';
  };

  const getLevelLabel = (level: number) => {
    if (level < 10) return 'Very Low';
    if (level < 30) return 'Low';
    if (level < 60) return 'Normal';
    if (level < 80) return 'High';
    return 'Very High';
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Audio Visualization
        </Typography>
        
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Typography variant="body2" minWidth={50}>
            Level:
          </Typography>
          <Box sx={{ flexGrow: 1 }}>
            <LinearProgress
              variant="determinate"
              value={audioLevel}
              color={getLevelColor(audioLevel)}
              sx={{ 
                height: 12, 
                borderRadius: 6,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              }}
            />
          </Box>
          <Typography variant="body2" fontWeight="bold" minWidth={40}>
            {Math.round(audioLevel)}%
          </Typography>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
          <Chip
            label={getLevelLabel(audioLevel)}
            color={getLevelColor(audioLevel)}
            size="small"
            variant="outlined"
          />
          {isRecording && (
            <Chip
              label="Recording"
              color="error"
              size="small"
              sx={{ 
                animation: 'pulse 1s infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                  '100%': { opacity: 1 },
                },
              }}
            />
          )}
        </Stack>
      </Box>

      <Box
        sx={{
          position: 'relative',
          width: '100%',
          height: 200,
          backgroundColor: '#0a0a0a',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
          }}
        />
        
        {frequencyData.length === 0 && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: 'text.secondary',
              textAlign: 'center',
            }}
          >
            <Typography variant="body2">
              Audio visualization will appear here
            </Typography>
            <Typography variant="caption">
              Grant microphone access to see real-time audio
            </Typography>
          </Box>
        )}
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        Real-time frequency spectrum and waveform visualization
      </Typography>
    </Paper>
  );
};