import React, { useState } from "react";
import {
  Card,
  CardContent,
  Typography,
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
  Stack,
  Alert,
  CircularProgress,
  Divider,
} from "@mui/material";
import {
  Add as AddIcon,
  SmartToy as BotIcon,
  VideoCall as VideoCallIcon,
  Language as LanguageIcon,
} from "@mui/icons-material";
import { MeetingRequest } from "@/types";
import { useBotManager } from "@/hooks/useBotManager";
import { SUPPORTED_LANGUAGES } from "@/constants/languages";

interface BotSpawnerProps {
  onBotSpawned?: (botId: string) => void;
  onError?: (error: string) => void;
}

export const BotSpawner: React.FC<BotSpawnerProps> = ({
  onBotSpawned,
  onError,
}) => {
  const { spawnBot, isLoading } = useBotManager();
  const [formData, setFormData] = useState<MeetingRequest>({
    meetingId: "",
    meetingTitle: "",
    organizerEmail: "",
    targetLanguages: ["en", "es"],
    autoTranslation: true,
    priority: "medium",
  });

  const availableLanguages = SUPPORTED_LANGUAGES;

  const handleInputChange = (field: keyof MeetingRequest, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleLanguageToggle = (languageCode: string) => {
    setFormData((prev) => ({
      ...prev,
      targetLanguages: prev.targetLanguages.includes(languageCode)
        ? prev.targetLanguages.filter((lang) => lang !== languageCode)
        : [...prev.targetLanguages, languageCode],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.meetingId.trim()) {
      onError?.("Meeting ID is required");
      return;
    }

    // Only require languages when translation is enabled
    if (formData.autoTranslation && formData.targetLanguages.length === 0) {
      onError?.(
        "At least one target language must be selected for translation",
      );
      return;
    }

    try {
      const botId = await spawnBot(formData);
      onBotSpawned?.(botId);

      // Reset form
      setFormData({
        meetingId: "",
        meetingTitle: "",
        organizerEmail: "",
        targetLanguages: ["en", "es"],
        autoTranslation: true,
        priority: "medium",
      });
    } catch (error) {
      onError?.(error instanceof Error ? error.message : "Failed to spawn bot");
    }
  };

  const getLanguageName = (code: string) => {
    return availableLanguages.find((lang) => lang.code === code)?.name || code;
  };

  return (
    <Card sx={{ height: "fit-content" }}>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
          <BotIcon sx={{ mr: 1, color: "primary.main" }} />
          <Typography variant="h6" component="h2">
            Spawn New Bot
          </Typography>
        </Box>

        <form onSubmit={handleSubmit}>
          <Stack spacing={3}>
            {/* Meeting Information */}
            <Box>
              <Typography
                variant="subtitle2"
                gutterBottom
                color="text.secondary"
              >
                Meeting Information
              </Typography>
              <Stack spacing={2}>
                <TextField
                  fullWidth
                  label="Google Meet ID"
                  value={formData.meetingId}
                  onChange={(e) =>
                    handleInputChange("meetingId", e.target.value)
                  }
                  placeholder="abc-defg-hij or full meet.google.com URL"
                  required
                  InputProps={{
                    startAdornment: (
                      <VideoCallIcon sx={{ mr: 1, color: "action.active" }} />
                    ),
                  }}
                />

                <TextField
                  fullWidth
                  label="Meeting Title (Optional)"
                  value={formData.meetingTitle}
                  onChange={(e) =>
                    handleInputChange("meetingTitle", e.target.value)
                  }
                  placeholder="Weekly Team Meeting"
                />

                <TextField
                  fullWidth
                  label="Organizer Email (Optional)"
                  value={formData.organizerEmail}
                  onChange={(e) =>
                    handleInputChange("organizerEmail", e.target.value)
                  }
                  type="email"
                  placeholder="organizer@company.com"
                />
              </Stack>
            </Box>

            <Divider />

            {/* Translation Settings */}
            <Box>
              <Typography
                variant="subtitle2"
                gutterBottom
                color="text.secondary"
              >
                Translation Settings
              </Typography>

              <FormControlLabel
                control={
                  <Switch
                    checked={formData.autoTranslation}
                    onChange={(e) =>
                      handleInputChange("autoTranslation", e.target.checked)
                    }
                    color="primary"
                  />
                }
                label="Enable Auto-Translation"
                sx={{ mb: 2 }}
              />

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Target Languages
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                  {availableLanguages.map((language) => (
                    <Chip
                      key={language.code}
                      label={language.name}
                      clickable
                      color={
                        formData.targetLanguages.includes(language.code)
                          ? "primary"
                          : "default"
                      }
                      variant={
                        formData.targetLanguages.includes(language.code)
                          ? "filled"
                          : "outlined"
                      }
                      onClick={() => handleLanguageToggle(language.code)}
                      icon={<LanguageIcon />}
                      size="small"
                    />
                  ))}
                </Box>
              </Box>

              <Typography variant="body2" color="text.secondary" gutterBottom>
                Selected Languages:
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
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

            <Divider />

            {/* Bot Priority */}
            <FormControl fullWidth>
              <InputLabel>Bot Priority</InputLabel>
              <Select
                value={formData.priority}
                label="Bot Priority"
                onChange={(e) => handleInputChange("priority", e.target.value)}
              >
                <MenuItem value="low">Low - Background processing</MenuItem>
                <MenuItem value="medium">Medium - Standard priority</MenuItem>
                <MenuItem value="high">High - Real-time processing</MenuItem>
              </Select>
            </FormControl>

            {/* Submit Button */}
            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={isLoading}
              startIcon={
                isLoading ? <CircularProgress size={20} /> : <AddIcon />
              }
              sx={{ mt: 2 }}
            >
              {isLoading ? "Spawning Bot..." : "Spawn Bot"}
            </Button>
          </Stack>
        </form>

        {/* Quick Actions */}
        <Box
          sx={{ mt: 3, pt: 2, borderTop: "1px solid", borderColor: "divider" }}
        >
          <Typography variant="subtitle2" gutterBottom color="text.secondary">
            Quick Actions
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Button
              size="small"
              variant="outlined"
              onClick={() => handleInputChange("meetingId", "demo-meeting-123")}
            >
              Demo Meeting
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() =>
                setFormData((prev) => ({
                  ...prev,
                  targetLanguages: ["en", "es", "fr"],
                }))
              }
            >
              Multi-Language
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() => handleInputChange("priority", "high")}
            >
              High Priority
            </Button>
          </Stack>
        </Box>

        {/* Info Alert */}
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            Bot will join the meeting, capture audio, process captions, and
            generate real-time translations. Virtual webcam will display
            translations overlaid on the video feed.
          </Typography>
        </Alert>
      </CardContent>
    </Card>
  );
};
