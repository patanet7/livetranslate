/**
 * AudioWorklet processor that forwards raw Float32Array chunks to the main thread.
 * Runs at native sample rate (typically 48kHz). Downsampling happens server-side.
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
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Take first channel (mono or left channel of stereo)
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Accumulate samples into the pre-allocated ring buffer
    for (let i = 0; i < channelData.length; i++) {
      this._ringBuffer[this._writeIndex++] = channelData[i];

      // When buffer is full, send to main thread and reset write index
      if (this._writeIndex >= this._bufferSize) {
        // Copy into a new buffer for transfer (original stays allocated)
        const chunk = new Float32Array(this._ringBuffer);
        this.port.postMessage({ type: 'audio_chunk', data: chunk.buffer }, [chunk.buffer]);
        this._writeIndex = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
