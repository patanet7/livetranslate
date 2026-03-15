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

  get sampleRate(): number | null {
    return this.context?.sampleRate ?? null;
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
        // S4: Validate systemDeviceId — without it we'd capture the mic twice
        if (!options.systemDeviceId) {
          throw new Error('"both" mode requires systemDeviceId (loopback device for system audio)');
        }

        const micConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            ...(options.deviceId ? { deviceId: { exact: options.deviceId } } : {}),
          },
        };
        const systemConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            deviceId: { exact: options.systemDeviceId },
          },
        };

        const [micStream, systemStream] = await Promise.all([
          navigator.mediaDevices.getUserMedia(micConstraints),
          navigator.mediaDevices.getUserMedia(systemConstraints),
        ]);

        this.stream = micStream;
        this._systemStream = systemStream;
        this.context = new AudioContext();

        // C1 fix: Use per-source GainNodes at 0.7 feeding a DynamicsCompressor.
        // A single shared GainNode at 0.5 attenuates both sources unconditionally,
        // halving the amplitude when only one source is active — degrading Whisper
        // accuracy on quiet speech. Per-source gain + compressor handles clipping
        // adaptively while preserving signal integrity.
        const micSource = this.context.createMediaStreamSource(micStream);
        const systemSource = this.context.createMediaStreamSource(systemStream);
        const micGain = this.context.createGain();
        micGain.gain.value = 0.7;
        const systemGain = this.context.createGain();
        systemGain.gain.value = 0.7;
        const compressor = this.context.createDynamicsCompressor();
        compressor.threshold.value = -6;  // Compress at -6dB to prevent clipping
        compressor.ratio.value = 4;
        micSource.connect(micGain).connect(compressor);
        systemSource.connect(systemGain).connect(compressor);

        // I3: Log actual sample rates for drift debugging in long sessions
        const micTrack = micStream.getAudioTracks()[0];
        const systemTrack = systemStream.getAudioTracks()[0];
        console.info('Mic sample rate:', micTrack.getSettings().sampleRate);
        console.info('System sample rate:', systemTrack.getSettings().sampleRate);

        await this.context.audioWorklet.addModule('/audio-worklet-processor.js');
        this.workletNode = new AudioWorkletNode(this.context, 'audio-chunk-processor');
        this.workletNode.port.onmessage = (event) => {
          if (event.data.type === 'audio_chunk') {
            options.onChunk(new Float32Array(event.data.data));
          }
        };
        compressor.connect(this.workletNode);

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
      const error = err instanceof Error ? err : new Error(String(err));
      options.onError(error);
      throw error; // Re-throw so callers can await and know capture failed
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
