import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Chip,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  CloudUpload,
  AudioFile,
  PlayArrow,
  Info,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface StageUploadProps {
  stageName: string;
  stageDisplayName: string;
  onUpload: (file: File, stageName: string) => Promise<void>;
  isProcessing?: boolean;
  supportedFormats?: string[];
  maxFileSize?: number;
  showStageInfo?: boolean;
}

interface StageInfo {
  description: string;
  processingTime: string;
  qualityImpact: string;
  parameters: string[];
}

const STAGE_INFO: Record<string, StageInfo> = {
  vad: {
    description: 'Voice Activity Detection - Identifies speech segments and filters silence',
    processingTime: '< 5ms',
    qualityImpact: 'Removes silence, preserves speech content',
    parameters: ['Voice threshold', 'Sensitivity', 'Frame duration'],
  },
  voice_filter: {
    description: 'Voice Frequency Filtering - Enhances speech frequencies while preserving natural voice',
    processingTime: '< 8ms',
    qualityImpact: 'Enhances fundamental and formant frequencies',
    parameters: ['Fundamental range', 'Formant preservation', 'High-freq rolloff'],
  },
  noise_reduction: {
    description: 'Spectral Noise Reduction - Removes background noise using frequency domain analysis',
    processingTime: '< 15ms',
    qualityImpact: 'Reduces noise while protecting speech frequencies',
    parameters: ['Strength', 'Voice protection', 'Adaptation rate'],
  },
  voice_enhancement: {
    description: 'Voice Enhancement - Improves speech clarity, presence, and intelligibility',
    processingTime: '< 10ms',
    qualityImpact: 'Enhances clarity, presence, and warmth',
    parameters: ['Clarity', 'Presence boost', 'Warmth/brightness'],
  },
  equalizer: {
    description: 'Parametric Equalizer - Multi-band frequency response shaping with professional presets',
    processingTime: '< 12ms',
    qualityImpact: 'Professional tonal shaping and frequency balance',
    parameters: ['5-band EQ', 'Professional presets', 'Filter types'],
  },
  spectral_denoising: {
    description: 'Advanced Spectral Denoising - FFT-based noise reduction with multiple algorithms',
    processingTime: '< 20ms',
    qualityImpact: 'Advanced noise reduction with artifact minimization',
    parameters: ['Algorithm mode', 'Spectral floor', 'Overlap factor'],
  },
  conventional_denoising: {
    description: 'Conventional Denoising - Time-domain filtering with traditional methods',
    processingTime: '< 8ms',
    qualityImpact: 'Edge-preserving noise reduction',
    parameters: ['Filter type', 'Strength', 'Transient preservation'],
  },
  lufs_normalization: {
    description: 'LUFS Normalization - Professional loudness normalization (ITU-R BS.1770-4)',
    processingTime: '< 18ms',
    qualityImpact: 'Broadcast-compliant loudness standardization',
    parameters: ['Target LUFS', 'Gating threshold', 'Max adjustment'],
  },
  agc: {
    description: 'Auto Gain Control - Adaptive level control with lookahead processing',
    processingTime: '< 12ms',
    qualityImpact: 'Consistent levels with natural dynamics',
    parameters: ['Target level', 'Attack/release', 'Lookahead time'],
  },
  compression: {
    description: 'Dynamic Range Compression - Professional dynamics control for consistent levels',
    processingTime: '< 8ms',
    qualityImpact: 'Controlled dynamics with professional sound',
    parameters: ['Threshold', 'Ratio', 'Knee width', 'Makeup gain'],
  },
  limiter: {
    description: 'Peak Limiting - Transparent brick-wall limiting to prevent clipping',
    processingTime: '< 6ms',
    qualityImpact: 'Peak control with transparent limiting',
    parameters: ['Threshold', 'Release time', 'Lookahead', 'Soft clipping'],
  },
};

export const StageUpload: React.FC<StageUploadProps> = ({
  stageName,
  stageDisplayName,
  onUpload,
  isProcessing = false,
  supportedFormats = ['audio/wav', 'audio/mp3', 'audio/webm', 'audio/ogg'],
  maxFileSize = 50 * 1024 * 1024, // 50MB
  showStageInfo = true,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  const stageInfo = STAGE_INFO[stageName];

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null);
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Validate file size
      if (file.size > maxFileSize) {
        setError(`File size exceeds ${Math.round(maxFileSize / 1024 / 1024)}MB limit`);
        return;
      }
      
      // Validate file type
      if (!supportedFormats.includes(file.type)) {
        setError(`Unsupported file format. Supported: ${supportedFormats.map(f => f.split('/')[1]).join(', ')}`);
        return;
      }
      
      setSelectedFile(file);
    }
  }, [maxFileSize, supportedFormats]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': supportedFormats.map(format => `.${format.split('/')[1]}`),
    },
    multiple: false,
  });

  const handleUpload = async () => {
    if (!selectedFile) return;
    
    try {
      setError(null);
      setUploadProgress(0);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);

      await onUpload(selectedFile, stageName);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Reset after successful upload
      setTimeout(() => {
        setSelectedFile(null);
        setUploadProgress(0);
      }, 1500);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h3">
            {stageDisplayName} Stage Upload
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={stageInfo?.processingTime || '< 10ms'} 
              size="small" 
              color="primary" 
              variant="outlined"
            />
            {showStageInfo && (
              <Tooltip title="Stage Information">
                <IconButton 
                  size="small" 
                  onClick={() => setShowInfo(!showInfo)}
                  color={showInfo ? "primary" : "default"}
                >
                  <Info />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {/* Stage Information Panel */}
        {showInfo && stageInfo && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              {stageInfo.description}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Quality Impact:</strong> {stageInfo.qualityImpact}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Key Parameters:</strong> {stageInfo.parameters.join(', ')}
            </Typography>
          </Alert>
        )}

        {/* Upload Area */}
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 3,
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: isDragActive ? 'action.hover' : 'background.default',
            transition: 'all 0.2s ease-in-out',
            mb: 2,
          }}
        >
          <input {...getInputProps()} />
          <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          
          {selectedFile ? (
            <Box>
              <Typography variant="body1" gutterBottom>
                <AudioFile sx={{ verticalAlign: 'middle', mr: 1 }} />
                {selectedFile.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatFileSize(selectedFile.size)} • {selectedFile.type}
              </Typography>
            </Box>
          ) : (
            <Box>
              <Typography variant="body1" gutterBottom>
                {isDragActive ? 'Drop audio file here' : 'Drag & drop audio file or click to select'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Supported formats: {supportedFormats.map(f => f.split('/')[1].toUpperCase()).join(', ')}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Max size: {Math.round(maxFileSize / 1024 / 1024)}MB
              </Typography>
            </Box>
          )}
        </Box>

        {/* Upload Progress */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <Box mb={2}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="body2" color="text.secondary" textAlign="center" mt={1}>
              Processing through {stageDisplayName} stage... {uploadProgress}%
            </Typography>
          </Box>
        )}

        {/* Success Message */}
        {uploadProgress === 100 && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Audio successfully processed through {stageDisplayName} stage!
          </Alert>
        )}

        {/* Error Message */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Upload Button */}
        <Button
          fullWidth
          variant="contained"
          startIcon={<PlayArrow />}
          onClick={handleUpload}
          disabled={!selectedFile || isProcessing || uploadProgress > 0}
          size="large"
        >
          {isProcessing ? 'Processing...' : `Process through ${stageDisplayName}`}
        </Button>
      </CardContent>
    </Card>
  );
};