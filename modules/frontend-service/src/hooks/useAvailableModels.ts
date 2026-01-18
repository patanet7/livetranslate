import { useState, useEffect } from "react";

export interface ModelInfo {
  name: string;
  displayName: string;
  description: string;
}

export interface DeviceInfo {
  device: string;
  device_type?: string;
  status: string;
  details?: Record<string, any>;
  acceleration?: string;
  error?: string;
}

export interface ModelsResponse {
  available_models: string[];
  models: string[];
  status: "success" | "fallback" | "error";
  service: string;
  total_models: number;
  error?: string;
  message?: string;
  device_info?: DeviceInfo; // For /models/transcription endpoint (single service)
  // For backwards compatibility with /models endpoint (all models)
  _legacy_device_info?: {
    audio_service: DeviceInfo;
    translation_service: DeviceInfo;
  };
}

const MODEL_DESCRIPTIONS: Record<
  string,
  { displayName: string; description: string }
> = {
  "whisper-tiny": {
    displayName: "Tiny (fastest)",
    description: "Fastest model, good for real-time processing",
  },
  "whisper-base": {
    displayName: "Base (recommended)",
    description: "Balanced speed and accuracy",
  },
  "whisper-small": {
    displayName: "Small",
    description: "Better accuracy than base",
  },
  "whisper-medium": {
    displayName: "Medium",
    description: "Good accuracy, slower processing",
  },
  "whisper-large": {
    displayName: "Large (highest quality)",
    description: "Best accuracy, slowest processing",
  },
  "whisper-large-v2": {
    displayName: "Large v2",
    description: "Enhanced large model",
  },
  "whisper-large-v3": {
    displayName: "Large v3",
    description: "Latest large model",
  },
  // Legacy format support (without "whisper-" prefix)
  tiny: {
    displayName: "Tiny (fastest)",
    description: "Fastest model, good for real-time processing",
  },
  base: {
    displayName: "Base (recommended)",
    description: "Balanced speed and accuracy",
  },
  small: { displayName: "Small", description: "Better accuracy than base" },
  medium: {
    displayName: "Medium",
    description: "Good accuracy, slower processing",
  },
  large: {
    displayName: "Large (highest quality)",
    description: "Best accuracy, slowest processing",
  },
  "large-v2": { displayName: "Large v2", description: "Enhanced large model" },
  "large-v3": { displayName: "Large v3", description: "Latest large model" },
};

export const useAvailableModels = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<"success" | "fallback" | "error">(
    "success",
  );
  const [serviceMessage, setServiceMessage] = useState<string | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<{
    audio_service: DeviceInfo;
    translation_service: DeviceInfo;
  } | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoading(true);
        setError(null);

        // Use specific transcription models endpoint
        const response = await fetch("/api/audio/models/transcription");

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: ModelsResponse = await response.json();

        // Use available_models first, fallback to models for compatibility
        const modelNames = data.available_models || data.models || [];

        // Transform model names to ModelInfo objects
        const modelInfos: ModelInfo[] = modelNames.map((name) => ({
          name,
          displayName:
            MODEL_DESCRIPTIONS[name]?.displayName ||
            `${name.charAt(0).toUpperCase()}${name.slice(1)}`,
          description:
            MODEL_DESCRIPTIONS[name]?.description || `Whisper ${name} model`,
        }));

        setModels(modelInfos);
        setStatus(data.status);

        // Handle device info - transcription endpoint returns single DeviceInfo
        if (data.device_info) {
          // Normalize to expected structure for consistency
          setDeviceInfo({
            audio_service: data.device_info as DeviceInfo,
            translation_service: { device: "unknown", status: "not_requested" },
          });
        } else {
          setDeviceInfo(null);
        }

        if (data.status === "fallback") {
          setServiceMessage(
            data.message ||
              "Using fallback models - audio service may be offline",
          );
        } else {
          setServiceMessage(null);
        }

        console.log(
          `Loaded ${modelInfos.length} transcription models from Whisper service (status: ${data.status})`,
        );

        if (data.device_info) {
          console.log("Whisper device info:", data.device_info);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(`Failed to load models: ${errorMessage}`);
        setStatus("error");

        // Provide emergency fallback models
        const fallbackModels: ModelInfo[] = [
          {
            name: "base",
            displayName: "Base (fallback)",
            description: "Fallback model when service unavailable",
          },
        ];
        setModels(fallbackModels);

        console.error("Failed to load models from orchestration service:", err);
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, []);

  const refetch = () => {
    setLoading(true);
    setError(null);
    // Re-trigger the effect by forcing a re-mount (simple approach)
    window.location.reload();
  };

  return {
    models,
    loading,
    error,
    status,
    serviceMessage,
    deviceInfo,
    refetch,
  };
};
