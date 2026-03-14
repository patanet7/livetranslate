/**
 * Audio capture manager — handles mic and system audio via getUserMedia.
 *
 * Key design decisions:
 * - AudioWorklet (not ScriptProcessorNode) for glitch-free capture
 * - Native sample rate capture — downsampling is server-side
 * - No echoCancellation/noiseSuppression — these attenuate loopback audio
 */

export type AudioSourceType = 'mic' | 'system' | 'both';

export interface CaptureOptions {
  deviceId?: string;
  systemDeviceId?: string;  // Required when sourceType is 'both' — the loopback device ID
  sourceType: AudioSourceType;
  onChunk: (data: Float32Array) => void;
  onError: (error: Error) => void;
}

export class AudioCapture {
  private context: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private _systemStream: MediaStream | null = null;  // For 'both' mode — system audio stream
  private _isCapturing = false;

  get isCapturing(): boolean {
    return this._isCapturing;
  }

  get sampleRate(): number {
    return this.context?.sampleRate ?? 48000;
  }

  async start(options: CaptureOptions): Promise<void> {
    if (this._isCapturing) return;

    try {
      // Get audio stream(s) based on source type.
      // 'mic': getUserMedia for microphone input
      // 'system': getUserMedia targeting virtual loopback device (BlackHole/PulseAudio monitor)
      // 'both': capture mic and system audio as separate streams, merge into one source
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };

      if (options.sourceType === 'system') {
        // System audio requires a virtual loopback device — deviceId must be provided
        if (!options.deviceId) {
          throw new Error('System audio capture requires a loopback device ID (e.g., BlackHole)');
        }
        audioConstraints.deviceId = { exact: options.deviceId };
      } else if (options.sourceType === 'mic') {
        if (options.deviceId) {
          audioConstraints.deviceId = { exact: options.deviceId };
        }
      }

      if (options.sourceType === 'both') {
        // Capture both mic and system audio as separate streams, merge via ChannelMerger
        const micConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            ...(options.deviceId ? { deviceId: { exact: options.deviceId } } : {}),
          },
        };
        // System audio requires a separate loopback deviceId passed via options.systemDeviceId
        const systemConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            ...(options.systemDeviceId
              ? { deviceId: { exact: options.systemDeviceId } }
              : {}),
          },
        };

        const [micStream, systemStream] = await Promise.all([
          navigator.mediaDevices.getUserMedia(micConstraints),
          navigator.mediaDevices.getUserMedia(systemConstraints),
        ]);

        // Store both streams for cleanup
        this.stream = micStream;
        this._systemStream = systemStream;
        this.context = new AudioContext();

        // Merge both streams into a single mono source
        const micSource = this.context.createMediaStreamSource(micStream);
        const systemSource = this.context.createMediaStreamSource(systemStream);
        const merger = this.context.createChannelMerger(2);
        micSource.connect(merger, 0, 0);
        systemSource.connect(merger, 0, 1);

        await this.context.audioWorklet.addModule('/audio-worklet-processor.js');
        this.workletNode = new AudioWorkletNode(this.context, 'audio-chunk-processor');
        this.workletNode.port.onmessage = (event) => {
          if (event.data.type === 'audio_chunk') {
            options.onChunk(new Float32Array(event.data.data));
          }
        };
        merger.connect(this.workletNode);

        this._isCapturing = true;
        return;
      }

      // Single-source path (mic or system)
      const constraints: MediaStreamConstraints = { audio: audioConstraints };
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.context = new AudioContext();

      // Load AudioWorklet
      await this.context.audioWorklet.addModule('/audio-worklet-processor.js');

      // Create nodes
      const source = this.context.createMediaStreamSource(this.stream);
      this.workletNode = new AudioWorkletNode(this.context, 'audio-chunk-processor');

      // Handle chunks from worklet
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio_chunk') {
          const chunk = new Float32Array(event.data.data);
          options.onChunk(chunk);
        }
      };

      // Connect: source → worklet
      source.connect(this.workletNode);
      // Don't connect worklet to destination — we don't want to play back

      this._isCapturing = true;
    } catch (err) {
      options.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }

  async stop(): Promise<void> {
    if (!this._isCapturing) return;

    this.workletNode?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this._systemStream?.getTracks().forEach((t) => t.stop());
    await this.context?.close();

    this.workletNode = null;
    this.stream = null;
    this._systemStream = null;
    this.context = null;
    this._isCapturing = false;
  }

  static async getDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'audioinput');
  }
}
