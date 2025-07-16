/**
 * Pipeline Studio API Client
 * 
 * Provides real-time audio processing integration for the visual pipeline editor
 */

import { AudioComponent } from '@/components/audio/PipelineEditor/ComponentLibrary';

export interface PipelineProcessingRequest {
  pipelineConfig: {
    id: string;
    name: string;
    stages: PipelineStage[];
    connections: PipelineConnection[];
  };
  audioData?: Blob | string; // Base64 encoded audio or Blob
  processingMode: 'realtime' | 'batch' | 'preview';
  outputFormat?: 'wav' | 'mp3' | 'base64';
  metadata?: {
    sampleRate?: number;
    channels?: number;
    bitDepth?: number;
  };
}

export interface PipelineStage {
  id: string;
  type: string; // e.g., 'vad', 'noise_reduction', 'voice_enhancement'
  enabled: boolean;
  gainIn: number;    // -20 to +20 dB
  gainOut: number;   // -20 to +20 dB
  parameters: Record<string, any>;
  position: { x: number; y: number };
}

export interface PipelineConnection {
  id: string;
  sourceStageId: string;
  targetStageId: string;
}

export interface PipelineProcessingResponse {
  success: boolean;
  pipelineId: string;
  processedAudio?: string; // Base64 encoded
  metrics: {
    totalLatency: number;
    stageLatencies: Record<string, number>;
    qualityMetrics: {
      snr: number;
      thd: number;
      lufs: number;
      rms: number;
    };
    cpuUsage: number;
  };
  stageOutputs?: Record<string, string>; // Base64 encoded audio for each stage
  errors?: string[];
  warnings?: string[];
}

export interface RealTimeProcessingSession {
  sessionId: string;
  pipelineId: string;
  status: 'initializing' | 'running' | 'paused' | 'stopped' | 'error';
  metrics: {
    chunksProcessed: number;
    averageLatency: number;
    qualityScore: number;
  };
}

class PipelineApiClient {
  private baseUrl: string;
  private wsUrl: string;
  private websocket: WebSocket | null = null;
  private messageCallbacks: Map<string, (data: any) => void> = new Map();

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';
    this.wsUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:3000';
  }

  /**
   * Process audio through a complete pipeline (batch mode)
   */
  async processPipeline(request: PipelineProcessingRequest): Promise<PipelineProcessingResponse> {
    const formData = new FormData();
    
    // Convert pipeline config to backend format
    const backendConfig = this.convertToBackendFormat(request.pipelineConfig);
    formData.append('pipeline_config', JSON.stringify(backendConfig));
    
    if (request.audioData instanceof Blob) {
      formData.append('audio_file', request.audioData, 'audio.wav');
    } else if (request.audioData) {
      formData.append('audio_data', request.audioData);
    }
    
    formData.append('processing_mode', request.processingMode);
    formData.append('output_format', request.outputFormat || 'wav');
    
    if (request.metadata) {
      formData.append('metadata', JSON.stringify(request.metadata));
    }

    const response = await fetch(`${this.baseUrl}/api/audio/pipeline/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Pipeline processing failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Process audio through a single stage for testing/debugging
   */
  async processSingleStage(
    stageType: string,
    audioData: Blob | string,
    stageConfig: Record<string, any>
  ): Promise<{
    processedAudio: string;
    metrics: {
      latency: number;
      qualityImprovement: number;
      inputLevel: number;
      outputLevel: number;
    };
  }> {
    const formData = new FormData();
    
    if (audioData instanceof Blob) {
      formData.append('audio_file', audioData, 'audio.wav');
    } else {
      formData.append('audio_data', audioData);
    }
    
    formData.append('stage_config', JSON.stringify(stageConfig));

    const response = await fetch(`${this.baseUrl}/api/audio/process/stage/${stageType}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Stage processing failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get real-time FFT analysis of audio
   */
  async getFFTAnalysis(audioData: Blob | string): Promise<{
    frequencies: number[];
    magnitudes: number[];
    sampleRate: number;
    spectralFeatures: {
      centroid: number;
      rolloff: number;
      bandwidth: number;
      flatness: number;
    };
    voiceCharacteristics: {
      fundamentalFreq: number;
      formants: number[];
      voiceConfidence: number;
    };
  }> {
    const formData = new FormData();
    
    if (audioData instanceof Blob) {
      formData.append('audio_file', audioData, 'audio.wav');
    } else {
      formData.append('audio_data', audioData);
    }

    const response = await fetch(`${this.baseUrl}/api/audio/analyze/fft`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`FFT analysis failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get LUFS analysis for broadcast compliance
   */
  async getLUFSAnalysis(audioData: Blob | string): Promise<{
    integratedLoudness: number;
    loudnessRange: number;
    truePeak: number;
    momentaryLoudness: number[];
    shortTermLoudness: number[];
    complianceCheck: {
      standard: string;
      compliant: boolean;
      recommendations: string[];
    };
  }> {
    const formData = new FormData();
    
    if (audioData instanceof Blob) {
      formData.append('audio_file', audioData, 'audio.wav');
    } else {
      formData.append('audio_data', audioData);
    }

    const response = await fetch(`${this.baseUrl}/api/audio/analyze/lufs`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`LUFS analysis failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Start real-time processing session
   */
  async startRealtimeSession(pipelineConfig: PipelineProcessingRequest['pipelineConfig']): Promise<RealTimeProcessingSession> {
    const backendConfig = this.convertToBackendFormat(pipelineConfig);
    
    const response = await fetch(`${this.baseUrl}/api/audio/pipeline/realtime/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pipeline_config: backendConfig,
        session_config: {
          chunkSize: 1024,
          sampleRate: 16000,
          channels: 1,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start realtime session: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Connect to real-time processing WebSocket
   */
  connectRealtimeWebSocket(sessionId: string, callbacks: {
    onMetrics?: (metrics: any) => void;
    onProcessedAudio?: (audio: string) => void;
    onError?: (error: string) => void;
  }): WebSocket {
    if (this.websocket) {
      this.websocket.close();
    }

    const wsUrl = `${this.wsUrl}/ws/audio/pipeline/${sessionId}`;
    this.websocket = new WebSocket(wsUrl);

    this.websocket.onopen = () => {
      console.log('Connected to pipeline WebSocket');
    };

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'metrics':
            callbacks.onMetrics?.(data.metrics);
            break;
          case 'processed_audio':
            callbacks.onProcessedAudio?.(data.audio);
            break;
          case 'error':
            callbacks.onError?.(data.error);
            break;
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.websocket.onerror = (error) => {
      console.error('Pipeline WebSocket error:', error);
      callbacks.onError?.('WebSocket connection error');
    };

    this.websocket.onclose = () => {
      console.log('Pipeline WebSocket closed');
    };

    return this.websocket;
  }

  /**
   * Send audio chunk for real-time processing
   */
  sendAudioChunk(audioChunk: Blob | ArrayBuffer): void {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    if (audioChunk instanceof Blob) {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64 = this.arrayBufferToBase64(arrayBuffer);
        this.websocket?.send(JSON.stringify({
          type: 'audio_chunk',
          data: base64,
        }));
      };
      reader.readAsArrayBuffer(audioChunk);
    } else {
      const base64 = this.arrayBufferToBase64(audioChunk);
      this.websocket.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64,
      }));
    }
  }

  /**
   * Update pipeline configuration in real-time
   */
  updatePipelineConfig(stageId: string, parameters: Record<string, any>): void {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.websocket.send(JSON.stringify({
      type: 'update_stage',
      stage_id: stageId,
      parameters: parameters,
    }));
  }

  /**
   * Get available presets
   */
  async getPresets(): Promise<Array<{
    name: string;
    description: string;
    category: string;
    stages: PipelineStage[];
    metadata: {
      totalLatency: number;
      complexity: string;
      targetUseCase: string[];
    };
  }>> {
    const response = await fetch(`${this.baseUrl}/api/audio/presets`);
    
    if (!response.ok) {
      throw new Error(`Failed to get presets: ${response.statusText}`);
    }

    const data = await response.json();
    return this.convertPresetsFromBackend(data);
  }

  /**
   * Save custom preset
   */
  async savePreset(
    name: string,
    pipelineConfig: PipelineProcessingRequest['pipelineConfig'],
    metadata: {
      description: string;
      category: string;
      tags: string[];
    }
  ): Promise<void> {
    const backendConfig = this.convertToBackendFormat(pipelineConfig);
    
    const response = await fetch(`${this.baseUrl}/api/audio/presets/save`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name,
        pipeline_config: backendConfig,
        metadata,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to save preset: ${response.statusText}`);
    }
  }

  /**
   * Convert frontend pipeline format to backend format
   */
  private convertToBackendFormat(pipelineConfig: PipelineProcessingRequest['pipelineConfig']): any {
    const enabledStages: Record<string, any> = {};
    
    // Build stage configuration
    pipelineConfig.stages.forEach(stage => {
      if (stage.enabled) {
        enabledStages[stage.type] = {
          enabled: true,
          gain_in: stage.gainIn,
          gain_out: stage.gainOut,
          parameters: stage.parameters,
        };
      }
    });

    return {
      pipeline_id: pipelineConfig.id,
      name: pipelineConfig.name,
      stages: enabledStages,
      connections: pipelineConfig.connections,
    };
  }

  /**
   * Convert backend presets to frontend format
   */
  private convertPresetsFromBackend(backendPresets: any): any[] {
    return Object.entries(backendPresets).map(([name, preset]: [string, any]) => ({
      name,
      description: preset.description || '',
      category: preset.category || 'custom',
      stages: this.convertStagesFromBackend(preset.stages || {}),
      metadata: {
        totalLatency: preset.metadata?.total_latency || 0,
        complexity: preset.metadata?.complexity || 'simple',
        targetUseCase: preset.metadata?.target_use_case || [],
      },
    }));
  }

  /**
   * Convert backend stages to frontend format
   */
  private convertStagesFromBackend(backendStages: Record<string, any>): PipelineStage[] {
    return Object.entries(backendStages).map(([stageName, stageConfig], index) => ({
      id: `${stageName}_${Date.now()}_${index}`,
      type: stageName,
      enabled: stageConfig.enabled || false,
      gainIn: stageConfig.gain_in || 0,
      gainOut: stageConfig.gain_out || 0,
      parameters: stageConfig.parameters || {},
      position: { x: index * 200 + 100, y: 100 },
    }));
  }

  /**
   * Convert ArrayBuffer to Base64 string
   */
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  /**
   * Clean up WebSocket connection
   */
  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }
}

export const pipelineApiClient = new PipelineApiClient();
export default pipelineApiClient;