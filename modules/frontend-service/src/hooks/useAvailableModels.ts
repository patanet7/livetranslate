import { useState, useEffect } from 'react';

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
  status: 'success' | 'fallback' | 'error';
  service: string;
  total_models: number;
  error?: string;
  message?: string;
  device_info?: {
    audio_service: DeviceInfo;
    translation_service: DeviceInfo;
  };
}

const MODEL_DESCRIPTIONS: Record<string, { displayName: string; description: string }> = {
  tiny: { displayName: 'Tiny (fastest)', description: 'Fastest model, good for real-time processing' },
  base: { displayName: 'Base (recommended)', description: 'Balanced speed and accuracy' },
  small: { displayName: 'Small', description: 'Better accuracy than base' },
  medium: { displayName: 'Medium', description: 'Good accuracy, slower processing' },
  large: { displayName: 'Large (highest quality)', description: 'Best accuracy, slowest processing' },
  'large-v2': { displayName: 'Large v2', description: 'Enhanced large model' },
  'large-v3': { displayName: 'Large v3', description: 'Latest large model' },
};

export const useAvailableModels = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<'success' | 'fallback' | 'error'>('success');
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

        const response = await fetch('/api/audio/models');
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: ModelsResponse = await response.json();
        
        // Use available_models first, fallback to models for compatibility
        const modelNames = data.available_models || data.models || [];
        
        // Transform model names to ModelInfo objects
        const modelInfos: ModelInfo[] = modelNames.map(name => ({
          name,
          displayName: MODEL_DESCRIPTIONS[name]?.displayName || `${name.charAt(0).toUpperCase()}${name.slice(1)}`,
          description: MODEL_DESCRIPTIONS[name]?.description || `Whisper ${name} model`,
        }));

        setModels(modelInfos);
        setStatus(data.status);
        setDeviceInfo(data.device_info || null);
        
        if (data.status === 'fallback') {
          setServiceMessage(data.message || 'Using fallback models - audio service may be offline');
        } else {
          setServiceMessage(null);
        }

        console.log(`Loaded ${modelInfos.length} models from orchestration service (status: ${data.status})`);
        
        if (data.device_info) {
          console.log('Device info:', {
            audio: data.device_info.audio_service.device,
            translation: data.device_info.translation_service.device
          });
        }
        
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(`Failed to load models: ${errorMessage}`);
        setStatus('error');
        
        // Provide emergency fallback models
        const fallbackModels: ModelInfo[] = [
          { name: 'base', displayName: 'Base (fallback)', description: 'Fallback model when service unavailable' },
        ];
        setModels(fallbackModels);
        
        console.error('Failed to load models from orchestration service:', err);
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