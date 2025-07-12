import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Box,
  Chip,
  Typography,
  Stack,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  CircularProgress,
  Divider,
} from '@mui/material';
import {
  VideoCall as VideoCallIcon,
  Language as LanguageIcon,
  Settings as SettingsIcon,
  Check as CheckIcon,
  ArrowBack as ArrowBackIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';
import { MeetingRequest } from '@/types';

interface CreateBotModalProps {
  open: boolean;
  onClose: () => void;
  onBotCreated: (botId: string) => void;
  onError: (error: string) => void;
}

const availableLanguages = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ru', name: 'Russian' },
];

const steps = [
  {
    label: 'Meeting Information',
    description: 'Enter the Google Meet details',
    icon: <VideoCallIcon />,
  },
  {
    label: 'Translation Settings',
    description: 'Configure language and translation options',
    icon: <LanguageIcon />,
  },
  {
    label: 'Advanced Settings',
    description: 'Set priority and additional options',
    icon: <SettingsIcon />,
  },
  {
    label: 'Review & Create',
    description: 'Review configuration and create bot',
    icon: <CheckIcon />,
  },
];

export const CreateBotModal: React.FC<CreateBotModalProps> = ({
  open,
  onClose,
  onBotCreated,
  onError,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState<MeetingRequest>({
    meetingId: '',
    meetingTitle: '',
    organizerEmail: '',
    targetLanguages: ['en', 'es'],
    autoTranslation: true,
    priority: 'medium',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleNext = () => {
    if (validateStep(activeStep)) {
      setActiveStep(prev => prev + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setFormData({
      meetingId: '',
      meetingTitle: '',
      organizerEmail: '',
      targetLanguages: ['en', 'es'],
      autoTranslation: true,
      priority: 'medium',
    });
    setErrors({});
  };

  const validateStep = (step: number) => {
    const newErrors: Record<string, string> = {};
    
    switch (step) {
      case 0:
        if (!formData.meetingId.trim()) {
          newErrors.meetingId = 'Meeting ID is required';
        } else if (!/^[a-zA-Z0-9-_]+$/.test(formData.meetingId.replace(/https?:\/\/meet\.google\.com\//, ''))) {
          newErrors.meetingId = 'Invalid meeting ID format';
        }
        break;
      case 1:
        if (formData.targetLanguages.length === 0) {
          newErrors.targetLanguages = 'At least one target language must be selected';
        }
        break;
      case 2:
        // Advanced settings validation if needed
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (field: keyof MeetingRequest, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleLanguageToggle = (languageCode: string) => {
    setFormData(prev => ({
      ...prev,
      targetLanguages: prev.targetLanguages.includes(languageCode)
        ? prev.targetLanguages.filter(lang => lang !== languageCode)
        : [...prev.targetLanguages, languageCode]
    }));
  };

  const handleSubmit = async () => {
    if (!validateStep(activeStep)) return;

    setLoading(true);
    try {
      const response = await fetch('/api/bot/spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        onBotCreated(data.botId);
        handleReset();
        onClose();
      } else {
        const errorData = await response.json();
        onError(errorData.message || 'Failed to create bot');
      }
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Failed to create bot');
    } finally {
      setLoading(false);
    }
  };

  const getLanguageName = (code: string) => {
    return availableLanguages.find(lang => lang.code === code)?.name || code;
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Stack spacing={3}>
            <TextField
              fullWidth
              label="Google Meet ID or URL"
              value={formData.meetingId}
              onChange={(e) => handleInputChange('meetingId', e.target.value)}
              error={!!errors.meetingId}
              helperText={errors.meetingId || 'Enter meet ID (e.g., abc-defg-hij) or full URL'}
              placeholder="abc-defg-hij or https://meet.google.com/abc-defg-hij"
              InputProps={{
                startAdornment: <VideoCallIcon sx={{ mr: 1, color: 'action.active' }} />,
              }}
            />
            
            <TextField
              fullWidth
              label="Meeting Title (Optional)"
              value={formData.meetingTitle}
              onChange={(e) => handleInputChange('meetingTitle', e.target.value)}
              placeholder="Weekly Team Meeting"
              helperText="Optional: Provide a descriptive title for this meeting"
            />
            
            <TextField
              fullWidth
              label="Organizer Email (Optional)"
              value={formData.organizerEmail}
              onChange={(e) => handleInputChange('organizerEmail', e.target.value)}
              type="email"
              placeholder="organizer@company.com"
              helperText="Optional: Email of the meeting organizer"
            />
          </Stack>
        );

      case 1:
        return (
          <Stack spacing={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={formData.autoTranslation}
                  onChange={(e) => handleInputChange('autoTranslation', e.target.checked)}
                  color="primary"
                />
              }
              label="Enable Auto-Translation"
            />

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Target Languages
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Select languages for translation output
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                {availableLanguages.map((language) => (
                  <Chip
                    key={language.code}
                    label={language.name}
                    clickable
                    color={formData.targetLanguages.includes(language.code) ? 'primary' : 'default'}
                    variant={formData.targetLanguages.includes(language.code) ? 'filled' : 'outlined'}
                    onClick={() => handleLanguageToggle(language.code)}
                    icon={<LanguageIcon />}
                  />
                ))}
              </Box>
              {errors.targetLanguages && (
                <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
                  {errors.targetLanguages}
                </Typography>
              )}
            </Box>

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Selected Languages ({formData.targetLanguages.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {formData.targetLanguages.map((code) => (
                  <Chip
                    key={code}
                    label={getLanguageName(code)}
                    size="small"
                    color="primary"
                    variant="filled"
                  />
                ))}
              </Box>
            </Box>
          </Stack>
        );

      case 2:
        return (
          <Stack spacing={3}>
            <FormControl fullWidth>
              <InputLabel>Bot Priority</InputLabel>
              <Select
                value={formData.priority}
                label="Bot Priority"
                onChange={(e) => handleInputChange('priority', e.target.value)}
              >
                <MenuItem value="low">
                  <Box>
                    <Typography variant="body2">Low Priority</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Background processing, lower resource usage
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value="medium">
                  <Box>
                    <Typography variant="body2">Medium Priority</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Standard processing, balanced performance
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value="high">
                  <Box>
                    <Typography variant="body2">High Priority</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Real-time processing, maximum performance
                    </Typography>
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info">
              <Typography variant="body2">
                High priority bots will use more system resources but provide better real-time performance.
                Medium priority is recommended for most use cases.
              </Typography>
            </Alert>
          </Stack>
        );

      case 3:
        return (
          <Stack spacing={3}>
            <Alert severity="success">
              <Typography variant="h6" gutterBottom>
                Ready to Create Bot
              </Typography>
              <Typography variant="body2">
                Please review the configuration below and click "Create Bot" to proceed.
              </Typography>
            </Alert>

            <Box>
              <Typography variant="h6" gutterBottom>
                Meeting Information
              </Typography>
              <Stack spacing={1}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Meeting ID:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {formData.meetingId}
                  </Typography>
                </Box>
                {formData.meetingTitle && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Title:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formData.meetingTitle}
                    </Typography>
                  </Box>
                )}
                {formData.organizerEmail && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Organizer:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formData.organizerEmail}
                    </Typography>
                  </Box>
                )}
              </Stack>
            </Box>

            <Divider />

            <Box>
              <Typography variant="h6" gutterBottom>
                Translation Settings
              </Typography>
              <Stack spacing={1}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Auto-Translation:</Typography>
                  <Chip 
                    label={formData.autoTranslation ? 'Enabled' : 'Disabled'}
                    color={formData.autoTranslation ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Target Languages:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {formData.targetLanguages.map((code) => (
                      <Chip
                        key={code}
                        label={getLanguageName(code)}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </Stack>
            </Box>

            <Divider />

            <Box>
              <Typography variant="h6" gutterBottom>
                Advanced Settings
              </Typography>
              <Stack spacing={1}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Priority:</Typography>
                  <Chip 
                    label={formData.priority.charAt(0).toUpperCase() + formData.priority.slice(1)}
                    color={
                      formData.priority === 'high' ? 'error' :
                      formData.priority === 'medium' ? 'warning' : 'success'
                    }
                    size="small"
                  />
                </Box>
              </Stack>
            </Box>
          </Stack>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <VideoCallIcon sx={{ mr: 1 }} />
          Create New Bot
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel
                optional={
                  index === 3 ? (
                    <Typography variant="caption">Last step</Typography>
                  ) : null
                }
                icon={step.icon}
              >
                {step.label}
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {step.description}
                </Typography>
                {renderStepContent(index)}
                <Box sx={{ mb: 2 }}>
                  <div>
                    {index === steps.length - 1 ? (
                      <Button
                        variant="contained"
                        onClick={handleSubmit}
                        disabled={loading}
                        sx={{ mt: 1, mr: 1 }}
                        startIcon={loading ? <CircularProgress size={20} /> : <CheckIcon />}
                      >
                        {loading ? 'Creating Bot...' : 'Create Bot'}
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        onClick={handleNext}
                        sx={{ mt: 1, mr: 1 }}
                        endIcon={<ArrowForwardIcon />}
                      >
                        Next
                      </Button>
                    )}
                    <Button
                      disabled={index === 0}
                      onClick={handleBack}
                      sx={{ mt: 1, mr: 1 }}
                      startIcon={<ArrowBackIcon />}
                    >
                      Back
                    </Button>
                  </div>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleReset} variant="outlined">
          Reset
        </Button>
      </DialogActions>
    </Dialog>
  );
};