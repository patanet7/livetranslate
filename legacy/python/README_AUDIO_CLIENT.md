# LiveTranslate Audio Client

This is a WebSocket client that can stream audio data to the LiveTranslate transcription server for processing. It supports both microphone input and streaming from WAV files.

## Features

- Stream audio from your microphone in real-time
- Stream audio from WAV files with adjustable playback speed
- Receive and display transcription results
- Automatically handles resampling and mono conversion

## Requirements

- Python 3.7+
- websockets
- numpy
- sounddevice (for microphone streaming)
- soundfile (for WAV file handling)

## Installation

Install the required dependencies:

```bash
pip install websockets numpy sounddevice soundfile
```

## Usage

### Microphone Streaming

Stream audio from your microphone to the server:

```bash
python audio_client.py
```

### File Streaming

Stream audio from a WAV file to the server:

```bash
python audio_client.py --file test_audio.wav
```

You can adjust the playback speed:

```bash
python audio_client.py --file test_audio.wav --speed 2.0  # Play at 2x speed
```

## Command Line Arguments

- `--server`: WebSocket server URL (default: ws://localhost:8765)
- `--rate`: Sample rate in Hz (default: 16000)
- `--channels`: Number of channels (default: 1)
- `--blocksize`: Block size in samples (default: 8000)
- `--file`: Path to audio file to stream (WAV format)
- `--speed`: Playback speed factor (for file mode, default: 1.0)

## Generating Test Audio

The repository includes a script to generate test audio files:

```bash
python generate_test_wav.py --output test_audio.wav
```

This will generate a 5-second WAV file with a sequence of different tones.

### Test Audio Options

- `--output`: Output WAV file path (default: test_audio.wav)
- `--rate`: Sample rate in Hz (default: 16000)
- `--duration`: Duration in seconds (default: 5.0)

## Example Workflow

1. Start the transcription server:
   ```bash
   python server.py
   ```

2. Generate a test audio file:
   ```bash
   python generate_test_wav.py --output test_audio.wav
   ```

3. Stream the test file to the server:
   ```bash
   python audio_client.py --file test_audio.wav
   ```

4. Or use your microphone:
   ```bash
   python audio_client.py
   ```

## Notes

- The server must be running before connecting the client
- For microphone input, make sure your system's microphone is properly configured
- The client automatically handles connection errors and will disconnect cleanly 