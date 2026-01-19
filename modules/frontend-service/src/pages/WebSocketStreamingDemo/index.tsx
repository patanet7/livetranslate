/**
 * WebSocket Audio Streaming Demo
 *
 * Simple demo page to test real-time mic → WebSocket → Whisper → transcription
 * Follows the same pattern as bot containers.
 */

import React from "react";
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  Chip,
  Stack,
  Alert,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Switch,
  Divider,
} from "@mui/material";
import {
  Mic as MicIcon,
  MicOff as MicOffIcon,
  Link as LinkIcon,
  LinkOff as LinkOffIcon,
  Clear as ClearIcon,
} from "@mui/icons-material";

import { useAudioStreaming } from "@/hooks/useAudioStreaming";

const WebSocketStreamingDemo: React.FC = () => {
  // WebSocket streaming hook
  const {
    isConnected,
    isStreaming,
    segments,
    translations,
    error,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    clearSegments,
    clearTranslations,
    sessionId,
  } = useAudioStreaming({
    model: "whisper-base",
    language: "en",
    enableVAD: true,
    enableDiarization: true,
    enableCIF: true,
    enableRollingContext: true,
    orchestrationUrl: "ws://localhost:3000", // Orchestration WebSocket
  });

  // Configuration state
  const [model, setModel] = React.useState("whisper-base");
  const [language, setLanguage] = React.useState("en");
  const [enableVAD, setEnableVAD] = React.useState(true);
  const [enableDiarization, setEnableDiarization] = React.useState(true);

  const availableModels = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3",
  ];

  const supportedLanguages = [
    { code: "auto", name: "Auto Detect" },
    { code: "en", name: "English" },
    { code: "es", name: "Spanish" },
    { code: "fr", name: "French" },
    { code: "de", name: "German" },
    { code: "it", name: "Italian" },
    { code: "pt", name: "Portuguese" },
    { code: "ru", name: "Russian" },
    { code: "ja", name: "Japanese" },
    { code: "ko", name: "Korean" },
    { code: "zh", name: "Chinese" },
  ];

  return (
    <Box sx={{ p: 3, maxWidth: 1400, margin: "0 auto" }}>
      <Typography variant="h4" gutterBottom>
        🎙️ WebSocket Audio Streaming Demo
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Real-time microphone → WebSocket → Whisper → transcription (same pattern
        as bots)
      </Typography>

      {/* Connection Status */}
      <Alert severity={isConnected ? "success" : "info"} sx={{ mb: 3 }}>
        <strong>WebSocket Status:</strong>{" "}
        {isConnected ? (
          <>
            ✅ Connected to ws://localhost:3000/api/audio/stream
            {isStreaming && " | 🎙️ Streaming Audio"}
          </>
        ) : (
          "🔴 Disconnected"
        )}
        <br />
        <strong>Session ID:</strong> {sessionId}
      </Alert>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={clearSegments}>
          <strong>Error:</strong> {error}
        </Alert>
      )}

      <Stack spacing={3}>
        {/* Connection Controls */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🔌 Connection
            </Typography>
            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                startIcon={<LinkIcon />}
                onClick={connect}
                disabled={isConnected}
                color="primary"
              >
                Connect
              </Button>
              <Button
                variant="outlined"
                startIcon={<LinkOffIcon />}
                onClick={disconnect}
                disabled={!isConnected}
                color="error"
              >
                Disconnect
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {/* Streaming Controls */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🎤 Audio Streaming
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Start streaming to send microphone audio via WebSocket (100ms
              chunks)
            </Typography>
            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Button
                variant="contained"
                startIcon={<MicIcon />}
                onClick={startStreaming}
                disabled={!isConnected || isStreaming}
                color="success"
                size="large"
              >
                Start Streaming
              </Button>
              <Button
                variant="outlined"
                startIcon={<MicOffIcon />}
                onClick={stopStreaming}
                disabled={!isStreaming}
                color="error"
                size="large"
              >
                Stop Streaming
              </Button>
            </Stack>

            {isStreaming && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  🎵 Streaming audio chunks every 100ms...
                </Typography>
                <LinearProgress sx={{ mt: 1 }} />
              </Box>
            )}
          </CardContent>
        </Card>

        {/* Configuration */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              ⚙️ Configuration
            </Typography>

            <Stack spacing={2}>
              <FormControl fullWidth>
                <InputLabel>Whisper Model</InputLabel>
                <Select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  label="Whisper Model"
                  disabled={isStreaming}
                >
                  {availableModels.map((m) => (
                    <MenuItem key={m} value={m}>
                      {m}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  label="Language"
                  disabled={isStreaming}
                >
                  {supportedLanguages.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={enableVAD}
                      onChange={(e) => setEnableVAD(e.target.checked)}
                      disabled={isStreaming}
                    />
                  }
                  label="Enable Voice Activity Detection (VAD)"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={enableDiarization}
                      onChange={(e) => setEnableDiarization(e.target.checked)}
                      disabled={isStreaming}
                    />
                  }
                  label="Enable Speaker Diarization"
                />
              </FormGroup>
            </Stack>
          </CardContent>
        </Card>

        {/* Real-time Transcription Display */}
        <Card>
          <CardContent>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 2,
              }}
            >
              <Typography variant="h6">
                📝 Real-time Transcription ({segments.length} segments)
              </Typography>
              <Button
                startIcon={<ClearIcon />}
                onClick={() => {
                  clearSegments();
                  clearTranslations();
                }}
                disabled={segments.length === 0}
                size="small"
              >
                Clear
              </Button>
            </Box>

            <Divider sx={{ mb: 2 }} />

            {segments.length === 0 ? (
              <Alert severity="info">
                {isStreaming
                  ? "Listening for speech... Start speaking to see real-time transcription."
                  : 'Click "Start Streaming" to begin real-time transcription.'}
              </Alert>
            ) : (
              <Stack spacing={2} sx={{ maxHeight: 500, overflow: "auto" }}>
                {segments.map((segment, index) => (
                  <Paper
                    key={index}
                    sx={{
                      p: 2,
                      backgroundColor: segment.is_final
                        ? "success.50"
                        : "grey.50",
                      borderLeft: segment.is_final ? "4px solid" : "4px dashed",
                      borderColor: segment.is_final
                        ? "success.main"
                        : "grey.400",
                    }}
                  >
                    <Stack spacing={1}>
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "flex-start",
                        }}
                      >
                        <Typography
                          variant="body1"
                          sx={{
                            flexGrow: 1,
                            fontWeight: segment.is_final ? 600 : 400,
                          }}
                        >
                          {segment.text}
                        </Typography>
                        <Chip
                          label={segment.is_final ? "✅ Final" : "⏳ Partial"}
                          size="small"
                          color={segment.is_final ? "success" : "default"}
                          sx={{ ml: 2 }}
                        />
                      </Box>

                      <Stack
                        direction="row"
                        spacing={1}
                        flexWrap="wrap"
                        useFlexGap
                      >
                        {segment.speaker && (
                          <Chip
                            label={`👤 ${segment.speaker}`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                        {segment.language && (
                          <Chip
                            label={`🌐 ${segment.language.toUpperCase()}`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                        <Chip
                          label={`📊 ${(segment.confidence * 100).toFixed(1)}%`}
                          size="small"
                          variant="outlined"
                          color={
                            segment.confidence > 0.8 ? "success" : "warning"
                          }
                        />
                        <Chip
                          label={`⏱️ ${new Date(segment.absolute_start_time).toLocaleTimeString()}`}
                          size="small"
                          variant="outlined"
                        />
                      </Stack>
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            )}
          </CardContent>
        </Card>

        {/* Translations (if any) */}
        {translations.length > 0 && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🌐 Translations ({translations.length})
              </Typography>

              <Divider sx={{ mb: 2 }} />

              <Stack spacing={2}>
                {translations.map((translation, index) => (
                  <Paper key={index} sx={{ p: 2, backgroundColor: "info.50" }}>
                    <Typography variant="body1">{translation.text}</Typography>
                    <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                      <Chip
                        label={`${translation.source_lang} → ${translation.target_lang}`}
                        size="small"
                        color="info"
                      />
                      <Chip
                        label={`${(translation.confidence * 100).toFixed(1)}%`}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </Paper>
                ))}
              </Stack>
            </CardContent>
          </Card>
        )}

        {/* Architecture Info */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🏗️ Architecture
            </Typography>
            <Typography
              variant="body2"
              component="div"
              sx={{ fontFamily: "monospace", whiteSpace: "pre-line" }}
            >
              {`Browser Microphone
  ↓ MediaRecorder (100ms chunks)
  ↓ base64-encoded WebM/Opus
WebSocket (ws://localhost:3000/api/audio/stream)
  ↓ audio_chunk messages
Orchestration Service
  ↓ forward to Whisper WebSocket
Whisper Service (NPU/GPU processing)
  ↓ VAD, CIF, AlignAtt, Rolling Context
  ↓ transcription segments
Orchestration Service
  ↓ deduplication, speaker grouping
  ↓ WebSocket segments back
Browser (Real-time display)`}
            </Typography>

            <Alert severity="success" sx={{ mt: 2 }}>
              <strong>✅ Same pattern as bot containers!</strong>
              <br />
              Bot: Container → WebSocket → Orchestration → Whisper
              <br />
              Frontend: Browser → WebSocket → Orchestration → Whisper
            </Alert>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );
};

export default WebSocketStreamingDemo;
