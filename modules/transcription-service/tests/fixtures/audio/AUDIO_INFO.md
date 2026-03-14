# Test Audio Fixtures

## Real Speech Audio Files

### JFK Speech (English)
- **File**: `jfk.wav`
- **Duration**: 11 seconds
- **Sample Rate**: 16kHz (Whisper native)
- **Channels**: Mono
- **Content**: "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country."
- **Source**: John F. Kennedy inaugural address (January 20, 1961)
- **Use**: English speech transcription testing

### Chinese Speech (Mandarin)
- **Files**: `OSR_cn_000_0072_8k.wav` through `OSR_cn_000_0075_8k.wav`
- **Duration**: ~20-25 seconds each
- **Sample Rate**: 8kHz (needs resampling to 16kHz for Whisper)
- **Channels**: Mono
- **Content**: Mandarin Chinese sentences
- **Transcriptions**: See `OSR_mandarin_test_sentences.doc` in `tests/audio/`
- **Use**: Multi-language transcription testing, code-switching tests

## Synthetic Audio Files

### hello_world.wav
- **Duration**: 3 seconds
- **Sample Rate**: 16kHz
- **Type**: Multi-formant synthesized speech-like audio
- **Use**: Basic transcription tests

### silence.wav
- **Duration**: 2 seconds
- **Sample Rate**: 16kHz
- **Type**: Pure silence (zeros)
- **Use**: VAD testing, silence detection

### noisy.wav
- **Duration**: 3 seconds
- **Sample Rate**: 16kHz
- **Type**: Synthesized speech + white noise (10dB SNR)
- **Use**: Noise robustness testing

### short_speech.wav
- **Duration**: 1 second
- **Sample Rate**: 16kHz
- **Type**: Short synthesized speech
- **Use**: Short audio handling

### long_speech.wav
- **Duration**: 5 seconds
- **Sample Rate**: 16kHz
- **Type**: Long synthesized speech
- **Use**: Long-form transcription

### white_noise.wav
- **Duration**: 2 seconds
- **Sample Rate**: 16kHz
- **Type**: Pure white noise
- **Use**: Noise-only testing

## Usage in Tests

```python
# Use JFK audio
def test_english_transcription(jfk_audio):
    audio, sr = jfk_audio
    result = service.transcribe(audio)
    assert "ask not what your country" in result['text'].lower()

# Use Chinese audio
def test_chinese_transcription(chinese_audio_1):
    audio, sr = chinese_audio_1
    result = service.transcribe(audio, language="zh")
    # Verify Chinese characters in result
```

## File Formats

All files are:
- **Format**: WAV (uncompressed)
- **Bit Depth**: 64-bit float (converted to float32 for Whisper)
- **Endianness**: Little-endian
- **Suitable for**: Direct use with Whisper models
