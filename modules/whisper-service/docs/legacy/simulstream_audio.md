SimulStreaming Audio Processing Pipeline - Complete Analysis

  Audio Format Transformations

  Server Input (PCM_16 bytes)
      ↓ soundfile + librosa
  NumPy array (float32, 1D, [-1, 1])
      ↓ torch.from_numpy()
  Torch tensor (float32, 1D, [-1, 1])
      ↓ torch.cat() multiple chunks
  Single torch tensor (float32, 1D, variable length)
      ↓ Append to segments list
  List of torch tensors
      ↓ torch.cat() all segments
  Concatenated tensor (float32, 1D, up to 30s)
      ↓ log_mel_spectrogram()
  Mel spectrogram (float32, 2D, [n_mels, n_frames])
      ↓ pad_or_trim()
  Fixed-size mel (float32, 3D, [1, n_mels, 3000])
      ↓ encoder()
  Encoder features (float32, 3D, [1, 1500, n_audio_state])

  Key Processing Stages

  Stage 1 - Entry Point (whisper_server.py):
  - Receives raw PCM_16 bytes
  - Converts to NumPy float32 array, range [-1.0, 1.0]

  Stage 2 - VAD Wrapper (vac_online_processor.py):
  - Buffers audio, runs Silero VAD
  - Forwards voice segments only to core processor

  Stage 3 - Chunk Accumulation (simulstreaming_whisper.py lines 151-217):
  def insert_audio_chunk(self, audio):
      self.audio_chunks.append(torch.from_numpy(audio))  # Store chunks

  def process_iter(self):
      audio = torch.cat(self.audio_chunks, dim=0)  # CONCATENATE ALL
      self.audio_chunks = []
      self.model.insert_audio(audio)  # Pass ONCE

  Stage 4 - Model Buffer (simul_whisper.py lines 269-285):
  def insert_audio(self, segment=None):
      if segment is not None:
          self.segments.append(segment)  # Add to segments list
      # Sliding window: drop oldest if > 30s

  Stage 5 - Inference (simul_whisper.py lines 346-357):
  # Concatenate ALL buffered segments
  input_segments = torch.cat(self.segments, dim=0)

  # Convert to mel spectrogram (padded to 30s)
  mel = log_mel_spectrogram(input_segments, device=self.model.device)

  Critical Insights:

  1. Buffer Strategy: 3 levels
    - audio_chunks[] → cleared each process_iter()
    - segments[] → sliding window (max 30s)
    - Mel conversion happens AFTER full concatenation
  2. Timing:
    - Min chunk: 1.2s
    - Max buffer: 30s
    - Processing triggered by process_iter() calls
  3. Key Difference vs Our Code:
    - SimulStreaming: Accumulates chunks in list → concatenates → calls insert_audio() ONCE per process_iter()
    - Our current code: Calls insert_audio() for EACH small chunk directly

