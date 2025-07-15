import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Tooltip,
  IconButton,
  TextField,
  InputAdornment,
  Collapse,
  Divider,
} from '@mui/material';
import {
  Search,
  ExpandMore,
  ExpandLess,
  Info,
  // Input Components
  FileUpload,
  Mic,
  VolumeUp,
  // Processing Components  
  FilterList,
  Tune,
  Equalizer,
  VolumeDown,
  Compress,
  Speed,
  AutoFixHigh,
  GraphicEq,
  AudioFile,
  VolumeOff,
  Settings,
  // Output Components
  Speaker,
  Download,
  CloudUpload,
} from '@mui/icons-material';

interface AudioComponent {
  id: string;
  type: 'input' | 'processing' | 'output';
  name: string;
  label: string;
  description: string;
  icon: React.ComponentType;
  category: string;
  complexity: 'basic' | 'intermediate' | 'advanced';
  processingTime: { target: number; max: number }; // milliseconds
  defaultConfig: Record<string, any>;
  parameters: ComponentParameter[];
  tags: string[];
}

interface ComponentParameter {
  name: string;
  displayName: string;
  type: 'slider' | 'select' | 'toggle' | 'input';
  defaultValue: any;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: { value: any; label: string }[];
  description: string;
}

interface ComponentLibraryProps {
  onComponentSelect: (component: AudioComponent) => void;
  onDragStart: (event: React.DragEvent, component: AudioComponent) => void;
  searchQuery?: string;
  filterByType?: 'input' | 'processing' | 'output' | 'all';
}

// Define the comprehensive audio component library
const AUDIO_COMPONENT_LIBRARY: AudioComponent[] = [
  // INPUT COMPONENTS
  {
    id: 'file_input',
    type: 'input',
    name: 'file_input',
    label: 'File Input',
    description: 'Upload and process audio files (WAV, MP3, FLAC, OGG)',
    icon: FileUpload,
    category: 'Input Sources',
    complexity: 'basic',
    processingTime: { target: 0, max: 0 },
    defaultConfig: {
      supportedFormats: ['wav', 'mp3', 'flac', 'ogg'],
      maxFileSize: 100, // MB
      autoPlay: false,
    },
    parameters: [
      {
        name: 'maxFileSize',
        displayName: 'Max File Size',
        type: 'slider',
        defaultValue: 100,
        min: 10,
        max: 500,
        step: 10,
        unit: 'MB',
        description: 'Maximum allowed file size for upload'
      },
      {
        name: 'autoPlay',
        displayName: 'Auto Play',
        type: 'toggle',
        defaultValue: false,
        description: 'Automatically start processing when file is loaded'
      }
    ],
    tags: ['upload', 'file', 'source', 'basic']
  },
  {
    id: 'microphone_input',
    type: 'input', 
    name: 'microphone_input',
    label: 'Microphone Input',
    description: 'Real-time audio recording from microphone',
    icon: Mic,
    category: 'Input Sources',
    complexity: 'basic',
    processingTime: { target: 0, max: 0 },
    defaultConfig: {
      sampleRate: 16000,
      channels: 1,
      bitDepth: 16,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
    },
    parameters: [
      {
        name: 'sampleRate',
        displayName: 'Sample Rate',
        type: 'select',
        defaultValue: 16000,
        options: [
          { value: 8000, label: '8 kHz' },
          { value: 16000, label: '16 kHz' },
          { value: 44100, label: '44.1 kHz' },
          { value: 48000, label: '48 kHz' },
        ],
        unit: 'Hz',
        description: 'Audio sampling frequency'
      },
      {
        name: 'echoCancellation',
        displayName: 'Echo Cancellation',
        type: 'toggle',
        defaultValue: true,
        description: 'Enable browser echo cancellation'
      },
      {
        name: 'noiseSuppression',
        displayName: 'Noise Suppression',
        type: 'toggle',
        defaultValue: true,
        description: 'Enable browser noise suppression'
      },
      {
        name: 'autoGainControl',
        displayName: 'Auto Gain Control',
        type: 'toggle',
        defaultValue: false,
        description: 'Enable browser automatic gain control'
      }
    ],
    tags: ['microphone', 'recording', 'realtime', 'live', 'basic']
  },

  // PROCESSING COMPONENTS - 11 Audio Pipeline Stages
  {
    id: 'vad_stage',
    type: 'processing',
    name: 'vad_stage',
    label: 'Voice Activity Detection',
    description: 'Detect voice vs silence with configurable sensitivity',
    icon: VolumeOff,
    category: 'Detection',
    complexity: 'basic',
    processingTime: { target: 5.0, max: 10.0 },
    defaultConfig: {
      aggressiveness: 2,
      energyThreshold: 0.01,
      voiceFreqMin: 85,
      voiceFreqMax: 300,
    },
    parameters: [
      {
        name: 'aggressiveness',
        displayName: 'Aggressiveness',
        type: 'slider',
        defaultValue: 2,
        min: 0,
        max: 3,
        step: 1,
        description: 'VAD sensitivity level (0=least, 3=most aggressive)'
      },
      {
        name: 'energyThreshold',
        displayName: 'Energy Threshold',
        type: 'slider',
        defaultValue: 0.01,
        min: 0.001,
        max: 0.1,
        step: 0.001,
        description: 'Minimum energy level to consider as voice'
      },
      {
        name: 'voiceFreqMin',
        displayName: 'Voice Freq Min',
        type: 'slider',
        defaultValue: 85,
        min: 50,
        max: 150,
        step: 5,
        unit: 'Hz',
        description: 'Minimum fundamental frequency for voice'
      },
      {
        name: 'voiceFreqMax',
        displayName: 'Voice Freq Max',
        type: 'slider',
        defaultValue: 300,
        min: 200,
        max: 500,
        step: 10,
        unit: 'Hz',
        description: 'Maximum fundamental frequency for voice'
      }
    ],
    tags: ['vad', 'detection', 'voice', 'silence', 'basic']
  },
  {
    id: 'voice_filter_stage',
    type: 'processing',
    name: 'voice_filter_stage', 
    label: 'Voice Filter',
    description: 'Isolate and enhance voice frequency ranges',
    icon: FilterList,
    category: 'Filtering',
    complexity: 'intermediate',
    processingTime: { target: 8.0, max: 15.0 },
    defaultConfig: {
      fundamentalMin: 85,
      fundamentalMax: 300,
      voiceBandGain: 1.2,
      preserveFormants: true,
    },
    parameters: [
      {
        name: 'fundamentalMin',
        displayName: 'Fundamental Min',
        type: 'slider',
        defaultValue: 85,
        min: 50,
        max: 150,
        step: 5,
        unit: 'Hz',
        description: 'Lower bound of fundamental frequency range'
      },
      {
        name: 'fundamentalMax',
        displayName: 'Fundamental Max',
        type: 'slider',
        defaultValue: 300,
        min: 200,
        max: 500,
        step: 10,
        unit: 'Hz',
        description: 'Upper bound of fundamental frequency range'
      },
      {
        name: 'voiceBandGain',
        displayName: 'Voice Band Gain',
        type: 'slider',
        defaultValue: 1.2,
        min: 0.1,
        max: 3.0,
        step: 0.1,
        description: 'Amplification factor for voice frequencies'
      },
      {
        name: 'preserveFormants',
        displayName: 'Preserve Formants',
        type: 'toggle',
        defaultValue: true,
        description: 'Maintain formant structure for natural voice'
      }
    ],
    tags: ['filter', 'voice', 'frequency', 'enhancement', 'intermediate']
  },
  {
    id: 'noise_reduction_stage',
    type: 'processing',
    name: 'noise_reduction_stage',
    label: 'Noise Reduction',
    description: 'Advanced noise suppression with voice preservation',
    icon: VolumeDown,
    category: 'Noise Control',
    complexity: 'advanced',
    processingTime: { target: 12.0, max: 25.0 },
    defaultConfig: {
      strength: 0.7,
      voiceProtection: true,
      adaptationRate: 0.1,
      noiseFloorDb: -45,
    },
    parameters: [
      {
        name: 'strength',
        displayName: 'Reduction Strength',
        type: 'slider',
        defaultValue: 0.7,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        description: 'Intensity of noise reduction processing'
      },
      {
        name: 'voiceProtection',
        displayName: 'Voice Protection',
        type: 'toggle',
        defaultValue: true,
        description: 'Preserve voice quality during noise reduction'
      },
      {
        name: 'adaptationRate',
        displayName: 'Adaptation Rate',
        type: 'slider',
        defaultValue: 0.1,
        min: 0.01,
        max: 1.0,
        step: 0.01,
        description: 'Speed of noise profile adaptation'
      },
      {
        name: 'noiseFloorDb',
        displayName: 'Noise Floor',
        type: 'slider',
        defaultValue: -45,
        min: -60,
        max: -20,
        step: 1,
        unit: 'dB',
        description: 'Target noise floor level'
      }
    ],
    tags: ['noise', 'reduction', 'suppression', 'advanced', 'adaptive']
  },
  {
    id: 'voice_enhancement_stage',
    type: 'processing',
    name: 'voice_enhancement_stage',
    label: 'Voice Enhancement',
    description: 'Professional voice clarity and presence enhancement',
    icon: AutoFixHigh,
    category: 'Enhancement',
    complexity: 'advanced',
    processingTime: { target: 10.0, max: 20.0 },
    defaultConfig: {
      clarityEnhancement: 0.6,
      presenceBoost: 0.5,
      warmthAdjustment: 0.2,
      brightnessAdjustment: 0.3,
    },
    parameters: [
      {
        name: 'clarityEnhancement',
        displayName: 'Clarity Enhancement',
        type: 'slider',
        defaultValue: 0.6,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        description: 'Improve voice intelligibility and definition'
      },
      {
        name: 'presenceBoost',
        displayName: 'Presence Boost',
        type: 'slider',
        defaultValue: 0.5,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        description: 'Enhance voice presence and forward sound'
      },
      {
        name: 'warmthAdjustment',
        displayName: 'Warmth',
        type: 'slider',
        defaultValue: 0.2,
        min: -1.0,
        max: 1.0,
        step: 0.1,
        description: 'Adjust voice warmth (negative=cooler, positive=warmer)'
      },
      {
        name: 'brightnessAdjustment',
        displayName: 'Brightness',
        type: 'slider',
        defaultValue: 0.3,
        min: -1.0,
        max: 1.0,
        step: 0.1,
        description: 'Adjust voice brightness (negative=darker, positive=brighter)'
      }
    ],
    tags: ['enhancement', 'voice', 'clarity', 'presence', 'professional', 'advanced']
  },
  {
    id: 'equalizer_stage',
    type: 'processing',
    name: 'equalizer_stage',
    label: 'Equalizer',
    description: 'Multi-band frequency shaping with presets',
    icon: Equalizer,
    category: 'Frequency',
    complexity: 'intermediate',
    processingTime: { target: 6.0, max: 12.0 },
    defaultConfig: {
      presetName: 'voice_enhance',
      customBands: [],
    },
    parameters: [
      {
        name: 'presetName',
        displayName: 'EQ Preset',
        type: 'select',
        defaultValue: 'voice_enhance',
        options: [
          { value: 'flat', label: 'Flat Response' },
          { value: 'voice_enhance', label: 'Voice Enhancement' },
          { value: 'broadcast', label: 'Broadcast Standard' },
          { value: 'bass_boost', label: 'Bass Boost' },
          { value: 'treble_boost', label: 'Treble Boost' },
          { value: 'custom', label: 'Custom Bands' },
        ],
        description: 'Pre-configured EQ curve or custom settings'
      }
    ],
    tags: ['equalizer', 'eq', 'frequency', 'tone', 'preset', 'intermediate']
  },
  {
    id: 'spectral_denoising_stage',
    type: 'processing',
    name: 'spectral_denoising_stage',
    label: 'Spectral Denoising',
    description: 'Advanced frequency-domain noise reduction',
    icon: GraphicEq,
    category: 'Noise Control',
    complexity: 'advanced',
    processingTime: { target: 15.0, max: 30.0 },
    defaultConfig: {
      mode: 'spectral_subtraction',
      noiseReductionFactor: 0.8,
      spectralFloor: 0.1,
    },
    parameters: [
      {
        name: 'mode',
        displayName: 'Denoising Mode',
        type: 'select',
        defaultValue: 'spectral_subtraction',
        options: [
          { value: 'minimal', label: 'Minimal Processing' },
          { value: 'spectral_subtraction', label: 'Spectral Subtraction' },
          { value: 'wiener_filter', label: 'Wiener Filter' },
          { value: 'adaptive', label: 'Adaptive Filtering' },
        ],
        description: 'Spectral denoising algorithm to use'
      },
      {
        name: 'noiseReductionFactor',
        displayName: 'Reduction Factor',
        type: 'slider',
        defaultValue: 0.8,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        description: 'Strength of spectral noise reduction'
      },
      {
        name: 'spectralFloor',
        displayName: 'Spectral Floor',
        type: 'slider',
        defaultValue: 0.1,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        description: 'Minimum spectral magnitude to preserve'
      }
    ],
    tags: ['spectral', 'denoising', 'frequency', 'advanced', 'filtering']
  },
  {
    id: 'conventional_denoising_stage',
    type: 'processing',
    name: 'conventional_denoising_stage',
    label: 'Conventional Denoising',
    description: 'Time-domain denoising with transient preservation',
    icon: Tune,
    category: 'Noise Control',
    complexity: 'intermediate',
    processingTime: { target: 8.0, max: 16.0 },
    defaultConfig: {
      mode: 'bilateral_filter',
      filterStrength: 0.6,
      preserveTransients: true,
    },
    parameters: [
      {
        name: 'mode',
        displayName: 'Filter Mode',
        type: 'select',
        defaultValue: 'bilateral_filter',
        options: [
          { value: 'median_filter', label: 'Median Filter' },
          { value: 'gaussian_filter', label: 'Gaussian Filter' },
          { value: 'bilateral_filter', label: 'Bilateral Filter' },
          { value: 'wavelet_denoising', label: 'Wavelet Denoising' },
        ],
        description: 'Type of conventional denoising filter'
      },
      {
        name: 'filterStrength',
        displayName: 'Filter Strength',
        type: 'slider',
        defaultValue: 0.6,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        description: 'Intensity of denoising filter application'
      },
      {
        name: 'preserveTransients',
        displayName: 'Preserve Transients',
        type: 'toggle',
        defaultValue: true,
        description: 'Maintain sharp transients and attack characteristics'
      }
    ],
    tags: ['conventional', 'denoising', 'time-domain', 'filter', 'intermediate']
  },
  {
    id: 'lufs_normalization_stage',
    type: 'processing',
    name: 'lufs_normalization_stage',
    label: 'LUFS Normalization',
    description: 'Broadcast-compliant loudness normalization',
    icon: VolumeUp,
    category: 'Loudness',
    complexity: 'advanced',
    processingTime: { target: 10.0, max: 20.0 },
    defaultConfig: {
      targetLufs: -23,
      maxAdjustment: 12,
      gatingThreshold: -70,
    },
    parameters: [
      {
        name: 'targetLufs',
        displayName: 'Target LUFS',
        type: 'slider',
        defaultValue: -23,
        min: -30,
        max: -10,
        step: 1,
        unit: 'LUFS',
        description: 'Target loudness level (EBU R128: -23 LUFS)'
      },
      {
        name: 'maxAdjustment',
        displayName: 'Max Adjustment',
        type: 'slider',
        defaultValue: 12,
        min: 3,
        max: 20,
        step: 1,
        unit: 'dB',
        description: 'Maximum allowed gain adjustment'
      },
      {
        name: 'gatingThreshold',
        displayName: 'Gating Threshold',
        type: 'slider',
        defaultValue: -70,
        min: -80,
        max: -60,
        step: 1,
        unit: 'dB',
        description: 'Loudness measurement gating threshold'
      }
    ],
    tags: ['lufs', 'loudness', 'normalization', 'broadcast', 'ebu', 'advanced']
  },
  {
    id: 'agc_stage',
    type: 'processing',
    name: 'agc_stage',
    label: 'Auto Gain Control',
    description: 'Automatic level control with configurable response',
    icon: Settings,
    category: 'Dynamics',
    complexity: 'intermediate',
    processingTime: { target: 5.0, max: 10.0 },
    defaultConfig: {
      targetLevel: -18,
      maxGain: 12,
      attackTime: 10,
      releaseTime: 100,
    },
    parameters: [
      {
        name: 'targetLevel',
        displayName: 'Target Level',
        type: 'slider',
        defaultValue: -18,
        min: -30,
        max: -6,
        step: 1,
        unit: 'dB',
        description: 'Desired output level'
      },
      {
        name: 'maxGain',
        displayName: 'Max Gain',
        type: 'slider',
        defaultValue: 12,
        min: 6,
        max: 20,
        step: 1,
        unit: 'dB',
        description: 'Maximum gain that can be applied'
      },
      {
        name: 'attackTime',
        displayName: 'Attack Time',
        type: 'slider',
        defaultValue: 10,
        min: 1,
        max: 50,
        step: 1,
        unit: 'ms',
        description: 'Speed of gain increase'
      },
      {
        name: 'releaseTime',
        displayName: 'Release Time',
        type: 'slider',
        defaultValue: 100,
        min: 50,
        max: 500,
        step: 10,
        unit: 'ms',
        description: 'Speed of gain decrease'
      }
    ],
    tags: ['agc', 'auto', 'gain', 'level', 'control', 'intermediate']
  },
  {
    id: 'compression_stage',
    type: 'processing',
    name: 'compression_stage',
    label: 'Compression',
    description: 'Dynamic range compression with professional controls',
    icon: Compress,
    category: 'Dynamics',
    complexity: 'advanced',
    processingTime: { target: 7.0, max: 14.0 },
    defaultConfig: {
      threshold: -20,
      ratio: 4,
      knee: 2,
      attackTime: 5,
      releaseTime: 50,
    },
    parameters: [
      {
        name: 'threshold',
        displayName: 'Threshold',
        type: 'slider',
        defaultValue: -20,
        min: -40,
        max: -5,
        step: 1,
        unit: 'dB',
        description: 'Level above which compression is applied'
      },
      {
        name: 'ratio',
        displayName: 'Ratio',
        type: 'slider',
        defaultValue: 4,
        min: 1,
        max: 20,
        step: 0.5,
        unit: ':1',
        description: 'Compression ratio (input:output)'
      },
      {
        name: 'knee',
        displayName: 'Knee',
        type: 'slider',
        defaultValue: 2,
        min: 0,
        max: 10,
        step: 0.5,
        unit: 'dB',
        description: 'Softness of compression transition'
      },
      {
        name: 'attackTime',
        displayName: 'Attack Time',
        type: 'slider',
        defaultValue: 5,
        min: 0.1,
        max: 20,
        step: 0.1,
        unit: 'ms',
        description: 'Speed of compression engagement'
      },
      {
        name: 'releaseTime',
        displayName: 'Release Time',
        type: 'slider',
        defaultValue: 50,
        min: 10,
        max: 500,
        step: 5,
        unit: 'ms',
        description: 'Speed of compression release'
      }
    ],
    tags: ['compression', 'dynamics', 'compressor', 'professional', 'advanced']
  },
  {
    id: 'limiter_stage',
    type: 'processing',
    name: 'limiter_stage',
    label: 'Limiter',
    description: 'Peak limiting with lookahead and soft clipping',
    icon: Speed,
    category: 'Dynamics',
    complexity: 'advanced',
    processingTime: { target: 5.0, max: 10.0 },
    defaultConfig: {
      threshold: -1,
      releaseTime: 50,
      lookahead: 5,
      softClip: true,
    },
    parameters: [
      {
        name: 'threshold',
        displayName: 'Threshold',
        type: 'slider',
        defaultValue: -1,
        min: -10,
        max: 0,
        step: 0.1,
        unit: 'dB',
        description: 'Maximum output level'
      },
      {
        name: 'releaseTime',
        displayName: 'Release Time',
        type: 'slider',
        defaultValue: 50,
        min: 10,
        max: 200,
        step: 5,
        unit: 'ms',
        description: 'Speed of limiter release'
      },
      {
        name: 'lookahead',
        displayName: 'Lookahead',
        type: 'slider',
        defaultValue: 5,
        min: 1,
        max: 20,
        step: 1,
        unit: 'ms',
        description: 'Preview time for peak detection'
      },
      {
        name: 'softClip',
        displayName: 'Soft Clipping',
        type: 'toggle',
        defaultValue: true,
        description: 'Use soft clipping instead of hard limiting'
      }
    ],
    tags: ['limiter', 'peak', 'limiting', 'ceiling', 'advanced']
  },

  // OUTPUT COMPONENTS
  {
    id: 'speaker_output',
    type: 'output',
    name: 'speaker_output',
    label: 'Speaker Output',
    description: 'Real-time audio playback with monitoring',
    icon: Speaker,
    category: 'Output Destinations',
    complexity: 'basic',
    processingTime: { target: 0, max: 0 },
    defaultConfig: {
      volume: 0.8,
      muted: false,
      outputDevice: 'default',
    },
    parameters: [
      {
        name: 'volume',
        displayName: 'Volume',
        type: 'slider',
        defaultValue: 0.8,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        description: 'Output volume level'
      },
      {
        name: 'muted',
        displayName: 'Muted',
        type: 'toggle',
        defaultValue: false,
        description: 'Mute audio output'
      }
    ],
    tags: ['speaker', 'playback', 'output', 'realtime', 'basic']
  },
  {
    id: 'file_output',
    type: 'output',
    name: 'file_output',
    label: 'File Output',
    description: 'Export processed audio to various formats',
    icon: Download,
    category: 'Output Destinations',
    complexity: 'basic',
    processingTime: { target: 0, max: 0 },
    defaultConfig: {
      format: 'wav',
      quality: 'high',
      filename: 'processed_audio',
    },
    parameters: [
      {
        name: 'format',
        displayName: 'Output Format',
        type: 'select',
        defaultValue: 'wav',
        options: [
          { value: 'wav', label: 'WAV (Uncompressed)' },
          { value: 'mp3', label: 'MP3 (Compressed)' },
          { value: 'flac', label: 'FLAC (Lossless)' },
          { value: 'ogg', label: 'OGG Vorbis' },
        ],
        description: 'Audio file format for export'
      },
      {
        name: 'quality',
        displayName: 'Quality',
        type: 'select',
        defaultValue: 'high',
        options: [
          { value: 'low', label: 'Low (64 kbps)' },
          { value: 'medium', label: 'Medium (128 kbps)' },
          { value: 'high', label: 'High (256 kbps)' },
          { value: 'lossless', label: 'Lossless' },
        ],
        description: 'Audio quality/bitrate setting'
      },
      {
        name: 'filename',
        displayName: 'Filename',
        type: 'input',
        defaultValue: 'processed_audio',
        description: 'Output filename (without extension)'
      }
    ],
    tags: ['file', 'export', 'download', 'save', 'basic']
  },
];

const ComponentLibrary: React.FC<ComponentLibraryProps> = ({
  onComponentSelect,
  onDragStart,
  searchQuery = '',
  filterByType = 'all',
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [searchTerm, setSearchTerm] = useState(searchQuery);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    'Input Sources': true,
    'Detection': false,
    'Filtering': false,
    'Noise Control': false,
    'Enhancement': false,
    'Frequency': false,
    'Loudness': false,
    'Dynamics': false,
    'Output Destinations': true,
  });

  // Filter components based on search and type
  const filteredComponents = AUDIO_COMPONENT_LIBRARY.filter(component => {
    const matchesSearch = searchTerm === '' || 
      component.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      component.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
      component.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      component.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesType = filterByType === 'all' || component.type === filterByType;
    
    return matchesSearch && matchesType;
  });

  // Group components by category
  const componentsByCategory = filteredComponents.reduce((acc, component) => {
    if (!acc[component.category]) {
      acc[component.category] = [];
    }
    acc[component.category].push(component);
    return acc;
  }, {} as Record<string, AudioComponent[]>);

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'basic': return '#4caf50';
      case 'intermediate': return '#ff9800';
      case 'advanced': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const getProcessingTimeColor = (target: number) => {
    if (target === 0) return '#9e9e9e';
    if (target <= 10) return '#4caf50';
    if (target <= 20) return '#ff9800';
    return '#f44336';
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const handleComponentClick = (component: AudioComponent) => {
    onComponentSelect(component);
  };

  const handleDragStart = (event: React.DragEvent, component: AudioComponent) => {
    onDragStart(event, component);
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Typography variant="h6" gutterBottom>
          Audio Component Library
        </Typography>

        {/* Search */}
        <TextField
          size="small"
          placeholder="Search components..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ fontSize: 18 }} />
              </InputAdornment>
            ),
          }}
        />

        {/* Type Filter Tabs */}
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
          sx={{ mb: 2, minHeight: 36 }}
        >
          <Tab label={`All (${AUDIO_COMPONENT_LIBRARY.length})`} sx={{ minHeight: 36, py: 1 }} />
          <Tab label={`Input (${AUDIO_COMPONENT_LIBRARY.filter(c => c.type === 'input').length})`} sx={{ minHeight: 36, py: 1 }} />
          <Tab label={`Processing (${AUDIO_COMPONENT_LIBRARY.filter(c => c.type === 'processing').length})`} sx={{ minHeight: 36, py: 1 }} />
          <Tab label={`Output (${AUDIO_COMPONENT_LIBRARY.filter(c => c.type === 'output').length})`} sx={{ minHeight: 36, py: 1 }} />
        </Tabs>

        {/* Component List */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          {Object.entries(componentsByCategory).map(([category, components]) => (
            <Box key={category} mb={2}>
              <Box
                display="flex"
                alignItems="center"
                justifyContent="space-between"
                sx={{ cursor: 'pointer', p: 1, borderRadius: 1, '&:hover': { bgcolor: 'action.hover' } }}
                onClick={() => toggleCategory(category)}
              >
                <Typography variant="subtitle2" fontWeight="bold">
                  {category} ({components.length})
                </Typography>
                {expandedCategories[category] ? <ExpandLess /> : <ExpandMore />}
              </Box>
              
              <Collapse in={expandedCategories[category]}>
                <List dense sx={{ pl: 1 }}>
                  {components.map((component) => (
                    <ListItem
                      key={component.id}
                      draggable
                      onDragStart={(e) => handleDragStart(e, component)}
                      onClick={() => handleComponentClick(component)}
                      sx={{
                        cursor: 'grab',
                        borderRadius: 1,
                        mb: 0.5,
                        border: '1px solid transparent',
                        transition: 'all 0.2s ease-in-out',
                        '&:hover': {
                          bgcolor: 'action.hover',
                          border: '1px solid',
                          borderColor: 'primary.main',
                          transform: 'translateX(4px)',
                        },
                        '&:active': {
                          cursor: 'grabbing',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        <component.icon 
                          sx={{ 
                            fontSize: 20,
                            color: component.type === 'input' ? '#2196f3' :
                                   component.type === 'processing' ? '#4caf50' :
                                   component.type === 'output' ? '#ff9800' : '#9e9e9e'
                          }} 
                        />
                      </ListItemIcon>
                      
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="body2" fontWeight="medium">
                              {component.label}
                            </Typography>
                            <Chip
                              size="small"
                              label={component.complexity}
                              sx={{
                                height: 18,
                                fontSize: '0.65rem',
                                backgroundColor: getComplexityColor(component.complexity),
                                color: 'white',
                              }}
                            />
                            {component.processingTime.target > 0 && (
                              <Chip
                                size="small"
                                label={`${component.processingTime.target}ms`}
                                sx={{
                                  height: 18,
                                  fontSize: '0.65rem',
                                  backgroundColor: getProcessingTimeColor(component.processingTime.target),
                                  color: 'white',
                                }}
                              />
                            )}
                          </Box>
                        }
                        secondary={
                          <Typography variant="caption" color="text.secondary">
                            {component.description}
                          </Typography>
                        }
                        sx={{ m: 0 }}
                      />
                      
                      <Tooltip title="Component Information" arrow>
                        <IconButton size="small" sx={{ p: 0.5 }}>
                          <Info sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    </ListItem>
                  ))}
                </List>
              </Collapse>
              
              {Object.keys(componentsByCategory).indexOf(category) < Object.keys(componentsByCategory).length - 1 && (
                <Divider sx={{ mt: 1 }} />
              )}
            </Box>
          ))}
          
          {filteredComponents.length === 0 && (
            <Box textAlign="center" py={4}>
              <Typography variant="body2" color="text.secondary">
                No components found matching your search.
              </Typography>
            </Box>
          )}
        </Box>

        {/* Component Count Summary */}
        <Box mt={1} p={1} bgcolor="action.hover" borderRadius={1}>
          <Typography variant="caption" color="text.secondary">
            Showing {filteredComponents.length} of {AUDIO_COMPONENT_LIBRARY.length} components
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ComponentLibrary;
export { AUDIO_COMPONENT_LIBRARY };
export type { AudioComponent, ComponentParameter };