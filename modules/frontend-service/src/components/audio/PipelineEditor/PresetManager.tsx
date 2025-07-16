import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Avatar,
  Menu,
  MenuItem,
  Alert,
  Tooltip,
  Grid,
  Divider,
  FormControl,
  InputLabel,
  Select,
  Tab,
  Tabs,
} from '@mui/material';
import {
  Save,
  SaveAs,
  FolderOpen,
  Delete,
  Edit,
  ContentCopy,
  FileDownload,
  FileUpload,
  Star,
  StarBorder,
  Category,
  Speed,
  VolumeUp,
  FilterList,
  AutoFixHigh,
  GraphicEq,
  Mic,
  Radio,
  RecordVoiceOver,
  Podcasts,
  MusicNote,
  MoreVert,
  Add,
  Check,
} from '@mui/icons-material';

interface PresetManagerProps {
  currentPipeline?: PipelinePreset;
  onLoadPreset: (preset: PipelinePreset) => void;
  onSavePreset: (preset: PipelinePreset) => void;
  onDeletePreset: (presetId: string) => void;
  onExportPreset: (preset: PipelinePreset) => void;
  onImportPreset: (file: File) => void;
}

interface PipelinePreset {
  id: string;
  name: string;
  description: string;
  category: PresetCategory;
  tags: string[];
  isFavorite: boolean;
  isBuiltIn: boolean;
  author: string;
  created: Date;
  modified: Date;
  pipeline: {
    nodes: any[];
    edges: any[];
  };
  metadata: {
    totalLatency: number;
    complexity: 'simple' | 'moderate' | 'complex';
    stageCount: number;
    audioFormat?: {
      sampleRate: number;
      channels: number;
      bitDepth: number;
    };
    targetUseCase: string[];
  };
  thumbnail?: string;
}

type PresetCategory = 
  | 'voice-enhancement'
  | 'noise-reduction'
  | 'broadcast'
  | 'podcast'
  | 'music'
  | 'streaming'
  | 'recording'
  | 'custom';

const CATEGORY_INFO: Record<PresetCategory, { label: string; icon: React.ElementType; color: string }> = {
  'voice-enhancement': { label: 'Voice Enhancement', icon: RecordVoiceOver, color: '#2196f3' },
  'noise-reduction': { label: 'Noise Reduction', icon: VolumeUp, color: '#4caf50' },
  'broadcast': { label: 'Broadcast', icon: Radio, color: '#ff9800' },
  'podcast': { label: 'Podcast', icon: Podcasts, color: '#9c27b0' },
  'music': { label: 'Music', icon: MusicNote, color: '#f44336' },
  'streaming': { label: 'Streaming', icon: Speed, color: '#00bcd4' },
  'recording': { label: 'Recording', icon: Mic, color: '#795548' },
  'custom': { label: 'Custom', icon: Category, color: '#607d8b' },
};

// Built-in presets
const BUILT_IN_PRESETS: PipelinePreset[] = [
  {
    id: 'preset_voice_clarity',
    name: 'Voice Clarity Pro',
    description: 'Professional voice enhancement with noise reduction and clarity optimization',
    category: 'voice-enhancement',
    tags: ['voice', 'clarity', 'professional', 'realtime'],
    isFavorite: false,
    isBuiltIn: true,
    author: 'LiveTranslate Team',
    created: new Date('2024-01-01'),
    modified: new Date('2024-01-01'),
    pipeline: {
      nodes: [], // Would contain actual pipeline configuration
      edges: [],
    },
    metadata: {
      totalLatency: 45.5,
      complexity: 'moderate',
      stageCount: 6,
      audioFormat: {
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
      },
      targetUseCase: ['meeting', 'conference', 'presentation'],
    },
  },
  {
    id: 'preset_broadcast_standard',
    name: 'Broadcast Standard EBU R128',
    description: 'EBU R128 compliant loudness normalization for broadcast',
    category: 'broadcast',
    tags: ['broadcast', 'ebu', 'loudness', 'compliance'],
    isFavorite: false,
    isBuiltIn: true,
    author: 'LiveTranslate Team',
    created: new Date('2024-01-01'),
    modified: new Date('2024-01-01'),
    pipeline: {
      nodes: [],
      edges: [],
    },
    metadata: {
      totalLatency: 68.2,
      complexity: 'complex',
      stageCount: 8,
      audioFormat: {
        sampleRate: 48000,
        channels: 2,
        bitDepth: 24,
      },
      targetUseCase: ['broadcast', 'television', 'radio'],
    },
  },
  {
    id: 'preset_podcast_master',
    name: 'Podcast Master',
    description: 'Complete podcast audio processing chain with voice optimization',
    category: 'podcast',
    tags: ['podcast', 'voice', 'master', 'production'],
    isFavorite: false,
    isBuiltIn: true,
    author: 'LiveTranslate Team',
    created: new Date('2024-01-01'),
    modified: new Date('2024-01-01'),
    pipeline: {
      nodes: [],
      edges: [],
    },
    metadata: {
      totalLatency: 82.7,
      complexity: 'complex',
      stageCount: 10,
      audioFormat: {
        sampleRate: 44100,
        channels: 1,
        bitDepth: 16,
      },
      targetUseCase: ['podcast', 'audiobook', 'narration'],
    },
  },
];

const PresetManager: React.FC<PresetManagerProps> = ({
  currentPipeline,
  onLoadPreset,
  onSavePreset,
  onDeletePreset,
  onExportPreset,
  onImportPreset,
}) => {
  const [presets, setPresets] = useState<PipelinePreset[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<PresetCategory | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState<PipelinePreset | null>(null);
  const [presetMenuAnchor, setPresetMenuAnchor] = useState<{ element: HTMLElement; preset: PipelinePreset } | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  // Form state for save dialog
  const [presetForm, setPresetForm] = useState({
    name: '',
    description: '',
    category: 'custom' as PresetCategory,
    tags: '',
  });

  useEffect(() => {
    // Load presets from localStorage and combine with built-in presets
    const savedPresets = localStorage.getItem('audioPipelinePresets');
    if (savedPresets) {
      const userPresets = JSON.parse(savedPresets);
      setPresets([...BUILT_IN_PRESETS, ...userPresets]);
    } else {
      setPresets(BUILT_IN_PRESETS);
    }
  }, []);

  const savePresetsToStorage = (updatedPresets: PipelinePreset[]) => {
    const userPresets = updatedPresets.filter(p => !p.isBuiltIn);
    localStorage.setItem('audioPipelinePresets', JSON.stringify(userPresets));
    setPresets(updatedPresets);
  };

  const filteredPresets = presets.filter(preset => {
    const matchesCategory = selectedCategory === 'all' || preset.category === selectedCategory;
    const matchesSearch = searchQuery === '' ||
      preset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      preset.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      preset.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesCategory && matchesSearch;
  });

  const favoritePresets = filteredPresets.filter(p => p.isFavorite);
  const regularPresets = filteredPresets.filter(p => !p.isFavorite);

  const handleSavePreset = () => {
    if (!currentPipeline || !presetForm.name) return;

    const newPreset: PipelinePreset = {
      id: editingPreset?.id || `preset_${Date.now()}`,
      name: presetForm.name,
      description: presetForm.description,
      category: presetForm.category,
      tags: presetForm.tags.split(',').map(t => t.trim()).filter(t => t),
      isFavorite: editingPreset?.isFavorite || false,
      isBuiltIn: false,
      author: 'Current User',
      created: editingPreset?.created || new Date(),
      modified: new Date(),
      pipeline: currentPipeline.pipeline,
      metadata: currentPipeline.metadata,
    };

    if (editingPreset) {
      const updatedPresets = presets.map(p => p.id === editingPreset.id ? newPreset : p);
      savePresetsToStorage(updatedPresets);
    } else {
      savePresetsToStorage([...presets, newPreset]);
    }

    onSavePreset(newPreset);
    setSaveDialogOpen(false);
    setEditingPreset(null);
    resetForm();
  };

  const handleDeletePreset = (preset: PipelinePreset) => {
    if (preset.isBuiltIn) return;
    
    const updatedPresets = presets.filter(p => p.id !== preset.id);
    savePresetsToStorage(updatedPresets);
    onDeletePreset(preset.id);
  };

  const handleToggleFavorite = (preset: PipelinePreset) => {
    const updatedPresets = presets.map(p => 
      p.id === preset.id ? { ...p, isFavorite: !p.isFavorite } : p
    );
    savePresetsToStorage(updatedPresets);
  };

  const handleDuplicatePreset = (preset: PipelinePreset) => {
    const duplicatedPreset: PipelinePreset = {
      ...preset,
      id: `preset_${Date.now()}`,
      name: `${preset.name} (Copy)`,
      isBuiltIn: false,
      isFavorite: false,
      author: 'Current User',
      created: new Date(),
      modified: new Date(),
    };
    
    savePresetsToStorage([...presets, duplicatedPreset]);
  };

  const handleImportPreset = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onImportPreset(file);
      // Handle file reading and preset import
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedPreset = JSON.parse(e.target?.result as string);
          importedPreset.id = `preset_${Date.now()}`;
          importedPreset.isBuiltIn = false;
          importedPreset.author = 'Imported';
          importedPreset.created = new Date();
          importedPreset.modified = new Date();
          
          savePresetsToStorage([...presets, importedPreset]);
        } catch (error) {
          console.error('Failed to import preset:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  const resetForm = () => {
    setPresetForm({
      name: '',
      description: '',
      category: 'custom',
      tags: '',
    });
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'simple': return '#4caf50';
      case 'moderate': return '#ff9800';
      case 'complex': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const formatLatency = (ms: number) => {
    return `${ms.toFixed(1)}ms`;
  };

  const PresetCard: React.FC<{ preset: PipelinePreset }> = ({ preset }) => {
    const CategoryIcon = CATEGORY_INFO[preset.category].icon;
    
    return (
      <Card
        sx={{
          height: '100%',
          cursor: 'pointer',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: 4,
          },
        }}
        onClick={() => onLoadPreset(preset)}
      >
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
            <Box display="flex" alignItems="center" gap={1}>
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  bgcolor: CATEGORY_INFO[preset.category].color,
                }}
              >
                <CategoryIcon sx={{ fontSize: 18 }} />
              </Avatar>
              <Box>
                <Typography variant="subtitle2" fontWeight="bold">
                  {preset.name}
                </Typography>
                {preset.isBuiltIn && (
                  <Chip label="Built-in" size="small" sx={{ height: 16, fontSize: '0.7rem' }} />
                )}
              </Box>
            </Box>
            
            <Box display="flex" alignItems="center" gap={0.5}>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  handleToggleFavorite(preset);
                }}
              >
                {preset.isFavorite ? <Star sx={{ fontSize: 18 }} /> : <StarBorder sx={{ fontSize: 18 }} />}
              </IconButton>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setPresetMenuAnchor({ element: e.currentTarget, preset });
                }}
              >
                <MoreVert sx={{ fontSize: 18 }} />
              </IconButton>
            </Box>
          </Box>

          <Typography variant="caption" color="text.secondary" paragraph>
            {preset.description}
          </Typography>

          <Box display="flex" gap={1} flexWrap="wrap" mb={1}>
            <Chip
              size="small"
              label={`${preset.metadata.stageCount} stages`}
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
            <Chip
              size="small"
              label={formatLatency(preset.metadata.totalLatency)}
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
            <Chip
              size="small"
              label={preset.metadata.complexity}
              sx={{
                height: 20,
                fontSize: '0.7rem',
                backgroundColor: getComplexityColor(preset.metadata.complexity),
                color: 'white',
              }}
            />
          </Box>

          <Box display="flex" gap={0.5} flexWrap="wrap">
            {preset.tags.slice(0, 3).map((tag, index) => (
              <Chip
                key={index}
                label={tag}
                size="small"
                variant="outlined"
                sx={{ height: 18, fontSize: '0.65rem' }}
              />
            ))}
            {preset.tags.length > 3 && (
              <Typography variant="caption" color="text.secondary">
                +{preset.tags.length - 3} more
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box mb={2}>
        <Typography variant="h6" gutterBottom>
          Pipeline Presets
        </Typography>
        
        <Box display="flex" gap={2} mb={2}>
          <TextField
            size="small"
            placeholder="Search presets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{ flexGrow: 1 }}
          />
          <Button
            variant="contained"
            startIcon={<SaveAs />}
            onClick={() => setSaveDialogOpen(true)}
            disabled={!currentPipeline}
          >
            Save Current
          </Button>
          <Button
            variant="outlined"
            component="label"
            startIcon={<FileUpload />}
          >
            Import
            <input
              type="file"
              hidden
              accept=".json"
              onChange={handleImportPreset}
            />
          </Button>
        </Box>

        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label={`All (${presets.length})`} />
          {Object.entries(CATEGORY_INFO).map(([category, info]) => {
            const count = presets.filter(p => p.category === category).length;
            return (
              <Tab
                key={category}
                label={`${info.label} (${count})`}
                icon={React.createElement(info.icon, { sx: { fontSize: 16 } })}
                iconPosition="start"
              />
            );
          })}
        </Tabs>
      </Box>

      <Box flexGrow={1} overflow="auto">
        {favoritePresets.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Star sx={{ fontSize: 18 }} /> Favorites
            </Typography>
            <Grid container spacing={2}>
              {favoritePresets.map((preset) => (
                <Grid item xs={12} sm={6} md={4} key={preset.id}>
                  <PresetCard preset={preset} />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {regularPresets.length > 0 && (
          <Box>
            {favoritePresets.length > 0 && (
              <Typography variant="subtitle2" gutterBottom>
                All Presets
              </Typography>
            )}
            <Grid container spacing={2}>
              {regularPresets.map((preset) => (
                <Grid item xs={12} sm={6} md={4} key={preset.id}>
                  <PresetCard preset={preset} />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {filteredPresets.length === 0 && (
          <Box textAlign="center" py={4}>
            <Typography variant="body2" color="text.secondary">
              No presets found matching your search.
            </Typography>
          </Box>
        )}
      </Box>

      {/* Save Preset Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingPreset ? 'Edit Preset' : 'Save Pipeline as Preset'}
        </DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} mt={1}>
            <TextField
              label="Preset Name"
              value={presetForm.name}
              onChange={(e) => setPresetForm({ ...presetForm, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={presetForm.description}
              onChange={(e) => setPresetForm({ ...presetForm, description: e.target.value })}
              fullWidth
              multiline
              rows={3}
            />
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={presetForm.category}
                label="Category"
                onChange={(e) => setPresetForm({ ...presetForm, category: e.target.value as PresetCategory })}
              >
                {Object.entries(CATEGORY_INFO).map(([category, info]) => (
                  <MenuItem key={category} value={category}>
                    <Box display="flex" alignItems="center" gap={1}>
                      {React.createElement(info.icon, { sx: { fontSize: 18, color: info.color } })}
                      {info.label}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Tags (comma-separated)"
              value={presetForm.tags}
              onChange={(e) => setPresetForm({ ...presetForm, tags: e.target.value })}
              fullWidth
              placeholder="voice, enhancement, realtime"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setSaveDialogOpen(false);
            setEditingPreset(null);
            resetForm();
          }}>
            Cancel
          </Button>
          <Button
            onClick={handleSavePreset}
            variant="contained"
            disabled={!presetForm.name}
          >
            {editingPreset ? 'Update' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preset Menu */}
      <Menu
        open={Boolean(presetMenuAnchor)}
        onClose={() => setPresetMenuAnchor(null)}
        anchorEl={presetMenuAnchor?.element}
      >
        {presetMenuAnchor && !presetMenuAnchor.preset.isBuiltIn && (
          <>
            <MenuItem onClick={() => {
              setEditingPreset(presetMenuAnchor.preset);
              setPresetForm({
                name: presetMenuAnchor.preset.name,
                description: presetMenuAnchor.preset.description,
                category: presetMenuAnchor.preset.category,
                tags: presetMenuAnchor.preset.tags.join(', '),
              });
              setSaveDialogOpen(true);
              setPresetMenuAnchor(null);
            }}>
              <ListItemIcon><Edit fontSize="small" /></ListItemIcon>
              <ListItemText>Edit</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => {
              handleDeletePreset(presetMenuAnchor.preset);
              setPresetMenuAnchor(null);
            }}>
              <ListItemIcon><Delete fontSize="small" /></ListItemIcon>
              <ListItemText>Delete</ListItemText>
            </MenuItem>
            <Divider />
          </>
        )}
        <MenuItem onClick={() => {
          if (presetMenuAnchor) {
            handleDuplicatePreset(presetMenuAnchor.preset);
          }
          setPresetMenuAnchor(null);
        }}>
          <ListItemIcon><ContentCopy fontSize="small" /></ListItemIcon>
          <ListItemText>Duplicate</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => {
          if (presetMenuAnchor) {
            onExportPreset(presetMenuAnchor.preset);
          }
          setPresetMenuAnchor(null);
        }}>
          <ListItemIcon><FileDownload fontSize="small" /></ListItemIcon>
          <ListItemText>Export</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default PresetManager;
export type { PipelinePreset, PresetCategory };