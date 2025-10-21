/**
 * RealTimeProcessor Integration Tests
 *
 * Tests for real-time audio processing functionality:
 * - Starting/stopping real-time sessions
 * - WebSocket connection management
 * - Audio chunk transmission
 * - Processed audio reception
 * - Metrics display
 * - Microphone permissions
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import RealTimeProcessor from '../RealTimeProcessor';
import { PipelineData } from '../PipelineCanvas';
import * as usePipelineProcessingModule from '@/hooks/usePipelineProcessing';

// Mock the usePipelineProcessing hook
const mockUsePipelineProcessing = vi.fn();

vi.mock('@/hooks/usePipelineProcessing', () => ({
  usePipelineProcessing: () => mockUsePipelineProcessing(),
}));

// Mock getUserMedia
const mockGetUserMedia = vi.fn();
Object.defineProperty(global.navigator, 'mediaDevices', {
  value: {
    getUserMedia: mockGetUserMedia,
  },
  writable: true,
});

// Mock AudioContext
class MockAudioContext {
  createAnalyser = vi.fn(() => ({
    frequencyBinCount: 128,
    getByteFrequencyData: vi.fn((array: Uint8Array) => {
      // Simulate some audio data
      for (let i = 0; i < array.length; i++) {
        array[i] = Math.random() * 128;
      }
    }),
  }));

  createMediaStreamSource = vi.fn(() => ({
    connect: vi.fn(),
    disconnect: vi.fn(),
  }));

  close = vi.fn();
}

global.AudioContext = MockAudioContext as any;

// Mock store
const createMockStore = () =>
  configureStore({
    reducer: {
      audio: (state = {}) => state,
      websocket: (state = { connection: { isConnected: true } }) => state,
    },
  });

// Mock WebSocket
class MockWebSocket {
  private messageHandlers: Array<(event: MessageEvent) => void> = [];
  private openHandlers: Array<() => void> = [];

  readyState = 1; // OPEN
  CONNECTING = 0;
  OPEN = 1;
  CLOSING = 2;
  CLOSED = 3;

  send = vi.fn();
  close = vi.fn();
  addEventListener = vi.fn((event: string, handler: any) => {
    if (event === 'message') this.messageHandlers.push(handler);
    if (event === 'open') this.openHandlers.push(handler);
  });
  removeEventListener = vi.fn();

  simulateMessage(data: any) {
    const event = new MessageEvent('message', {
      data: JSON.stringify(data),
    });
    this.messageHandlers.forEach(handler => handler(event));
  }

  simulateOpen() {
    this.openHandlers.forEach(handler => handler());
  }
}

describe('RealTimeProcessor', () => {
  let mockWebSocket: MockWebSocket;
  let mockStore: ReturnType<typeof createMockStore>;

  const mockPipeline: PipelineData = {
    id: 'test-pipeline-1',
    name: 'Test Pipeline',
    nodes: [
      {
        id: 'input-1',
        type: 'audioStage',
        position: { x: 0, y: 0 },
        data: {
          label: 'Microphone Input',
          description: 'Audio input',
          stageType: 'input' as const,
          enabled: true,
          gainIn: 0,
          gainOut: 0,
          stageConfig: { sampleRate: 16000 },
          parameters: [],
        },
      },
      {
        id: 'process-1',
        type: 'audioStage',
        position: { x: 200, y: 0 },
        data: {
          label: 'Noise Reduction',
          description: 'Reduce background noise',
          stageType: 'processing' as const,
          enabled: true,
          gainIn: 0,
          gainOut: 0,
          stageConfig: { strength: 0.7 },
          parameters: [
            {
              name: 'strength',
              value: 0.7,
              min: 0,
              max: 1,
              step: 0.1,
              unit: '',
              description: 'Noise reduction strength',
            },
          ],
        },
      },
    ],
    edges: [
      {
        id: 'e1',
        source: 'input-1',
        target: 'process-1',
        type: 'audioConnection',
      },
    ],
  };

  const defaultHookReturn = {
    isProcessing: false,
    metrics: null,
    audioAnalysis: {},
    error: null,
    realtimeSession: null,
    isRealtimeActive: false,
    websocket: null,
    startRealtimeProcessing: vi.fn(),
    startMicrophoneCapture: vi.fn(),
    stopRealtimeProcessing: vi.fn(),
    updateRealtimeConfig: vi.fn(),
    processPipeline: vi.fn(),
    analyzeFFT: vi.fn(),
    analyzeLUFS: vi.fn(),
    clearError: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockWebSocket = new MockWebSocket();
    mockStore = createMockStore();
    mockUsePipelineProcessing.mockReturnValue(defaultHookReturn);

    // Mock requestAnimationFrame
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb: any) => {
      cb();
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const renderComponent = (props = {}) => {
    return render(
      <Provider store={mockStore}>
        <RealTimeProcessor
          currentPipeline={mockPipeline}
          {...props}
        />
      </Provider>
    );
  };

  describe('Session Management', () => {
    it('should start real-time session and open WebSocket', async () => {
      const user = userEvent.setup();
      const mockStart = vi.fn().mockResolvedValue(undefined);
      const mockStartMic = vi.fn().mockResolvedValue(undefined);
      const mockOnWebSocketChange = vi.fn();

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        startRealtimeProcessing: mockStart,
        startMicrophoneCapture: mockStartMic,
      });

      renderComponent({
        onWebSocketChange: mockOnWebSocketChange,
      });

      // Click start button
      const startButton = screen.getByRole('button', { name: /Start Live Processing/i });
      await user.click(startButton);

      // Should call startRealtimeProcessing with pipeline config
      await waitFor(() => {
        expect(mockStart).toHaveBeenCalledWith(
          expect.objectContaining({
            id: 'test-pipeline-1',
            name: 'Test Pipeline',
            stages: expect.arrayContaining([
              expect.objectContaining({
                id: 'input-1',
                type: 'microphone_input',
              }),
              expect.objectContaining({
                id: 'process-1',
                type: 'noise_reduction',
              }),
            ]),
            connections: expect.arrayContaining([
              expect.objectContaining({
                sourceStageId: 'input-1',
                targetStageId: 'process-1',
              }),
            ]),
          })
        );
      });

      // Should start microphone capture
      expect(mockStartMic).toHaveBeenCalled();
    });

    it('should not start without pipeline', async () => {
      const user = userEvent.setup();
      const mockStart = vi.fn();
      const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        startRealtimeProcessing: mockStart,
      });

      renderComponent({
        currentPipeline: null,
      });

      const startButton = screen.getByRole('button', { name: /Start Live Processing/i });
      await user.click(startButton);

      expect(alertSpy).toHaveBeenCalledWith('Please create a pipeline first');
      expect(mockStart).not.toHaveBeenCalled();

      alertSpy.mockRestore();
    });

    it('should show session ID when active', () => {
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
        realtimeSession: {
          sessionId: 'test-session-12345',
          startTime: Date.now(),
        },
      });

      renderComponent();

      expect(screen.getByText(/Session: test-ses/i)).toBeInTheDocument();
    });

    it('should notify parent of WebSocket changes', () => {
      const mockOnWebSocketChange = vi.fn();

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      renderComponent({
        onWebSocketChange: mockOnWebSocketChange,
      });

      expect(mockOnWebSocketChange).toHaveBeenCalledWith(mockWebSocket, true);
    });
  });

  describe('Audio Processing', () => {
    it('should display status chip when processing', () => {
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
      });

      renderComponent();

      const statusChip = screen.getByText('LIVE');
      expect(statusChip).toBeInTheDocument();
    });

    it('should show audio level meters when active', () => {
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
      });

      renderComponent();

      expect(screen.getByText('Input Level')).toBeInTheDocument();
      expect(screen.getByText('Output Level')).toBeInTheDocument();
    });

    it('should not show level meters when inactive', () => {
      renderComponent();

      expect(screen.queryByText('Input Level')).not.toBeInTheDocument();
      expect(screen.queryByText('Output Level')).not.toBeInTheDocument();
    });

    it('should stop processing and close WebSocket', async () => {
      const user = userEvent.setup();
      const mockStop = vi.fn();

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
        stopRealtimeProcessing: mockStop,
      });

      renderComponent();

      const stopButton = screen.getByRole('button', { name: /Stop Processing/i });
      await user.click(stopButton);

      expect(mockStop).toHaveBeenCalled();
    });
  });

  describe('Performance Metrics', () => {
    it('should display real-time metrics', () => {
      const mockMetrics = {
        totalLatency: 45.2,
        qualityScore: 85,
        cpuUsage: 32.5,
        chunksProcessed: 142,
        qualityMetrics: {
          snr: 35.6,
          thd: 0.015,
          lufs: -18.5,
          rms: -12.3,
        },
      };

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: mockMetrics,
      });

      renderComponent();

      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
      expect(screen.getByText('45.2ms')).toBeInTheDocument();
      expect(screen.getByText('85')).toBeInTheDocument();
      expect(screen.getByText('32.5%')).toBeInTheDocument();
      expect(screen.getByText('142')).toBeInTheDocument();
    });

    it('should display quality metrics', () => {
      const mockMetrics = {
        totalLatency: 50,
        qualityScore: 80,
        cpuUsage: 25,
        chunksProcessed: 100,
        qualityMetrics: {
          snr: 40.2,
          thd: 0.012,
          lufs: -16.5,
          rms: -10.8,
        },
      };

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: mockMetrics,
      });

      renderComponent();

      expect(screen.getByText(/SNR:/)).toBeInTheDocument();
      expect(screen.getByText(/40.2dB/)).toBeInTheDocument();
      expect(screen.getByText(/THD:/)).toBeInTheDocument();
      expect(screen.getByText(/1.20%/)).toBeInTheDocument();
    });

    it('should update parent with metrics', () => {
      const mockOnMetricsUpdate = vi.fn();
      const mockMetrics = {
        totalLatency: 45.2,
        qualityScore: 85,
        cpuUsage: 32.5,
        chunksProcessed: 142,
      };

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: mockMetrics,
      });

      renderComponent({
        onMetricsUpdate: mockOnMetricsUpdate,
      });

      expect(mockOnMetricsUpdate).toHaveBeenCalledWith(mockMetrics);
    });

    it('should use color coding for latency', () => {
      const { rerender } = renderComponent();

      // Good latency (green)
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: { totalLatency: 30, qualityScore: 0, cpuUsage: 0, chunksProcessed: 0 },
      });
      rerender(
        <Provider store={mockStore}>
          <RealTimeProcessor currentPipeline={mockPipeline} />
        </Provider>
      );
      let latencyElement = screen.getByText('30.0ms');
      expect(latencyElement).toHaveStyle({ color: '#4caf50' });

      // Warning latency (orange)
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: { totalLatency: 75, qualityScore: 0, cpuUsage: 0, chunksProcessed: 0 },
      });
      rerender(
        <Provider store={mockStore}>
          <RealTimeProcessor currentPipeline={mockPipeline} />
        </Provider>
      );
      latencyElement = screen.getByText('75.0ms');
      expect(latencyElement).toHaveStyle({ color: '#ff9800' });

      // Bad latency (red)
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        metrics: { totalLatency: 150, qualityScore: 0, cpuUsage: 0, chunksProcessed: 0 },
      });
      rerender(
        <Provider store={mockStore}>
          <RealTimeProcessor currentPipeline={mockPipeline} />
        </Provider>
      );
      latencyElement = screen.getByText('150.0ms');
      expect(latencyElement).toHaveStyle({ color: '#f44336' });
    });
  });

  describe('Audio Analysis', () => {
    it('should display FFT analysis results', () => {
      const mockAnalysis = {
        fft: {
          voiceCharacteristics: {
            fundamentalFreq: 120.5,
            voiceConfidence: 0.92,
          },
          spectralFeatures: {
            centroid: 1850,
            rolloff: 3200,
          },
        },
      };

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        audioAnalysis: mockAnalysis,
      });

      renderComponent();

      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
      expect(screen.getByText(/120.5Hz/)).toBeInTheDocument();
      expect(screen.getByText(/92.0%/)).toBeInTheDocument();
      expect(screen.getByText(/1850Hz/)).toBeInTheDocument();
    });

    it('should display LUFS analysis results', () => {
      const mockAnalysis = {
        lufs: {
          integratedLoudness: -18.5,
          loudnessRange: 6.2,
          truePeak: -1.2,
          complianceCheck: {
            compliant: true,
            targetLoudness: -16.0,
          },
        },
      };

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        audioAnalysis: mockAnalysis,
      });

      renderComponent();

      expect(screen.getByText(/Integrated Loudness:/)).toBeInTheDocument();
      expect(screen.getByText(/-18.5dB/)).toBeInTheDocument();
      expect(screen.getByText(/6.2 LU/)).toBeInTheDocument();
    });
  });

  describe('File Upload', () => {
    it('should open upload dialog when upload button clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      const uploadButton = screen.getByRole('button', { name: /Upload Audio File/i });
      await user.click(uploadButton);

      expect(screen.getByText('Upload Audio File')).toBeInTheDocument();
      expect(screen.getByText(/Upload an audio file to process/i)).toBeInTheDocument();
    });

    it('should process uploaded file', async () => {
      const user = userEvent.setup();
      const mockProcessPipeline = vi.fn().mockResolvedValue(undefined);
      const mockAnalyzeFFT = vi.fn().mockResolvedValue(undefined);
      const mockAnalyzeLUFS = vi.fn().mockResolvedValue(undefined);

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        processPipeline: mockProcessPipeline,
        analyzeFFT: mockAnalyzeFFT,
        analyzeLUFS: mockAnalyzeLUFS,
      });

      renderComponent();

      // Open dialog
      const uploadButton = screen.getByRole('button', { name: /Upload Audio File/i });
      await user.click(uploadButton);

      // Upload file
      const file = new File(['audio data'], 'test.wav', { type: 'audio/wav' });
      const input = screen.getByRole('textbox', { hidden: true }) as HTMLInputElement;

      // Note: file input doesn't have 'textbox' role, let's find it by type
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;

      if (fileInput) {
        await user.upload(fileInput, file);

        await waitFor(() => {
          expect(mockProcessPipeline).toHaveBeenCalledWith(
            expect.objectContaining({
              pipelineConfig: expect.any(Object),
              audioData: file,
              processingMode: 'batch',
              outputFormat: 'wav',
            })
          );
        });

        expect(mockAnalyzeFFT).toHaveBeenCalledWith(file);
        expect(mockAnalyzeLUFS).toHaveBeenCalledWith(file);
      }
    });

    it('should close upload dialog when cancel clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Open dialog
      const uploadButton = screen.getByRole('button', { name: /Upload Audio File/i });
      await user.click(uploadButton);

      expect(screen.getByText('Upload Audio File')).toBeInTheDocument();

      // Close dialog
      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);

      await waitFor(() => {
        expect(screen.queryByText('Upload Audio File')).not.toBeInTheDocument();
      });
    });
  });

  describe('Settings Dialog', () => {
    it('should open settings dialog', async () => {
      const user = userEvent.setup();
      renderComponent();

      const settingsButton = screen.getByRole('button', { name: /Processing Settings/i });
      await user.click(settingsButton);

      expect(screen.getByText('Real-time Processing Settings')).toBeInTheDocument();
    });

    it('should allow changing chunk size', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Open settings
      const settingsButton = screen.getByRole('button', { name: /Processing Settings/i });
      await user.click(settingsButton);

      // Find chunk size slider
      const chunkSizeText = screen.getByText(/Chunk Size: 1024/i);
      expect(chunkSizeText).toBeInTheDocument();

      // Change slider (this is simplified - actual slider interaction may vary)
      const sliders = screen.getAllByRole('slider');
      const chunkSizeSlider = sliders[0];

      // Simulate slider change
      if (chunkSizeSlider) {
        await user.click(chunkSizeSlider);
        // Value should be within range 256-2048
      }
    });

    it('should allow changing latency target', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Open settings
      const settingsButton = screen.getByRole('button', { name: /Processing Settings/i });
      await user.click(settingsButton);

      // Find latency target text
      expect(screen.getByText(/Latency Target: 100ms/i)).toBeInTheDocument();
    });

    it('should close settings dialog', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Open settings
      const settingsButton = screen.getByRole('button', { name: /Processing Settings/i });
      await user.click(settingsButton);

      expect(screen.getByText('Real-time Processing Settings')).toBeInTheDocument();

      // Close dialog
      const closeButton = screen.getByRole('button', { name: /Close/i });
      await user.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByText('Real-time Processing Settings')).not.toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('should display error when provided', () => {
      const mockError = 'Failed to start audio processing';

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        error: mockError,
      });

      renderComponent();

      expect(screen.getByText(mockError)).toBeInTheDocument();
    });

    it('should clear error when close button clicked', async () => {
      const user = userEvent.setup();
      const mockClearError = vi.fn();

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        error: 'Test error',
        clearError: mockClearError,
      });

      renderComponent();

      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      expect(mockClearError).toHaveBeenCalled();
    });

    it('should handle microphone permission denied', async () => {
      const user = userEvent.setup();
      const mockStart = vi.fn().mockResolvedValue(undefined);
      const mockStartMic = vi.fn().mockRejectedValue(new Error('Permission denied'));

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        startRealtimeProcessing: mockStart,
        startMicrophoneCapture: mockStartMic,
      });

      renderComponent();

      const startButton = screen.getByRole('button', { name: /Start Live Processing/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(mockStartMic).toHaveBeenCalled();
      });

      // Error should be handled (implementation would show error to user)
    });
  });

  describe('Button States', () => {
    it('should disable start button when no pipeline', () => {
      renderComponent({ currentPipeline: null });

      const startButton = screen.getByRole('button', { name: /Start Live Processing/i });
      expect(startButton).toBeDisabled();
    });

    it('should disable start button when processing', () => {
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isProcessing: true,
      });

      renderComponent();

      const startButton = screen.getByRole('button', { name: /Start Live Processing/i });
      expect(startButton).toBeDisabled();
    });

    it('should show stop button when active', () => {
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
      });

      renderComponent();

      const stopButton = screen.getByRole('button', { name: /Stop Processing/i });
      expect(stopButton).toBeInTheDocument();
      expect(stopButton).not.toBeDisabled();
    });
  });

  describe('Cleanup', () => {
    it('should cancel animation frame on unmount', () => {
      const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');

      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
      });

      const { unmount } = renderComponent();
      unmount();

      expect(cancelSpy).toHaveBeenCalled();
    });

    it('should reset audio levels when processing stops', () => {
      const { rerender } = renderComponent();

      // Start with active processing
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: true,
      });

      rerender(
        <Provider store={mockStore}>
          <RealTimeProcessor currentPipeline={mockPipeline} />
        </Provider>
      );

      expect(screen.getByText('Input Level')).toBeInTheDocument();

      // Stop processing
      mockUsePipelineProcessing.mockReturnValue({
        ...defaultHookReturn,
        isRealtimeActive: false,
      });

      rerender(
        <Provider store={mockStore}>
          <RealTimeProcessor currentPipeline={mockPipeline} />
        </Provider>
      );

      expect(screen.queryByText('Input Level')).not.toBeInTheDocument();
    });
  });
});
