/**
 * AudioWorklet processor that forwards raw Float32Array chunks to the main thread.
 * Runs at native sample rate (typically 48kHz). Downsampling happens server-side.
 *
 * Only channel 0 is captured (mono). In 'both' mode, mic and system audio are
 * pre-mixed into a single mono channel by the AudioCapture class before reaching
 * this processor.
 */
class AudioChunkProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096; // ~85ms at 48kHz — good balance of latency vs overhead
    // Pre-allocated ring buffer to avoid GC pressure on the audio rendering thread.
    // A growable JS array would create garbage on every flush; a fixed Float32Array
    // ring buffer keeps allocations off the hot path entirely.
    this._ringBuffer = new Float32Array(this._bufferSize);
    this._writeIndex = 0;
    // M1: Track energy incrementally during accumulation to avoid a second pass.
    this._sumSquares = 0;
    // C2: Holdover counter — keep sending for N buffers after last non-silent buffer
    // to prevent mid-speech dropouts from quiet trailing syllables.
    this._silentBufferCount = 0;
    this._silentHoldover = 5; // ~425ms at 4096 samples/48kHz per buffer
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Take first channel (mono or left channel of stereo).
    // In 'both' mode, the upstream DynamicsCompressor already mixed mic + system
    // into mono, so channel 0 contains the combined signal.
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Accumulate samples into the pre-allocated ring buffer
    for (let i = 0; i < channelData.length; i++) {
      const sample = channelData[i];
      this._ringBuffer[this._writeIndex++] = sample;
      // M1: Track energy incrementally — no second pass needed
      this._sumSquares += sample * sample;

      // When buffer is full, check energy and send to main thread
      if (this._writeIndex >= this._bufferSize) {
        // C2: Use RMS energy instead of peak detection.
        // Peak detection (checking individual samples > threshold) drops quiet
        // speech where no single sample exceeds the threshold. RMS measures
        // actual energy content, which is what matters for speech presence.
        // Threshold of 0.001 ≈ -60dBFS — well above noise floor but catches
        // quiet speech that peak detection at 0.0001 (-80dBFS) would miss.
        const rms = Math.sqrt(this._sumSquares / this._bufferSize);
        const hasAudio = rms > 0.001;

        if (hasAudio) {
          this._silentBufferCount = 0;
          const chunk = new Float32Array(this._ringBuffer);
          this.port.postMessage({ type: 'audio_chunk', data: chunk.buffer }, [chunk.buffer]);
        } else {
          this._silentBufferCount++;
          // C2: Holdover — keep sending for N buffers after speech ends
          // to avoid clipping the tail of an utterance. Whisper needs
          // continuous audio; gaps cause garbled output.
          if (this._silentBufferCount <= this._silentHoldover) {
            const chunk = new Float32Array(this._ringBuffer);
            this.port.postMessage({ type: 'audio_chunk', data: chunk.buffer }, [chunk.buffer]);
          }
        }

        this._writeIndex = 0;
        this._sumSquares = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
