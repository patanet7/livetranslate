# LiveTranslate

Real-time audio capture and processing with NVIDIA Audio Effects integration.

## Features

- Audio capture from input devices
- Audio playback to output devices
- NVIDIA Audio Effects integration
  - Noise Removal
  - Room Echo Removal
  - Combined Noise + Echo Removal
  - Audio Super-Resolution
  - Acoustic Echo Cancellation
- Voice Activity Detection (VAD)
- Modern UI with ImGui

## Requirements

- Windows 10 or newer
- NVIDIA GPU with Tensor Cores (Turing architecture or newer)
- NVIDIA Audio Effects SDK
- CMake 3.9 or later
- Visual Studio 2017 (MSVC15.0) or later

## Setup

### Setting Up NVIDIA Audio Effects SDK

1. Download the NVIDIA Audio Effects SDK from [NVIDIA's developer site](https://developer.nvidia.com/maxine-getting-started)
2. Extract the SDK to `cpp/external/NVIDIA Audio Effects SDK/` in this repository
3. The directory structure should look like:
   ```
   cpp/
     external/
       NVIDIA Audio Effects SDK/
         bin/
         include/
         models/
         ...
   ```

### Building the Application

1. Run `setup_dependencies.bat` to set up dependencies
2. Run `cpp/build.bat` to build the application
3. The built application will be in the `cpp/build` directory

## Usage

1. Launch the application
2. Select input and output audio devices
3. Choose an NVIDIA Audio Effect:
   - Noise Removal - Removes background noise from audio
   - Room Echo Removal - Removes reverberation/echo from audio
   - Noise + Echo Removal - Combines both effects
   - Audio Super-Resolution - Enhances audio quality by predicting higher frequencies
   - Acoustic Echo Cancellation - Removes acoustic feedback/echo
4. Adjust effect settings as needed
5. Click "Start Capture" to begin capturing and processing audio

## Effect Settings

### Noise Removal
- Intensity level (0.0 - 1.0): Controls the intensity of noise removal
- Higher values remove more noise but may affect voice quality

### Room Echo Removal
- Intensity level (0.0 - 1.0): Controls the level of echo/reverb suppression
- Higher values remove more echo but may affect voice quality

### Audio Super-Resolution
- Input/Output sample rates: Choose between 8kHz→16kHz, 16kHz→48kHz, or 8kHz→48kHz
- Works best with clean audio input

### Acoustic Echo Cancellation
- Requires both near-end and far-end audio inputs
- Removes acoustic feedback loops

### Voice Activity Detection (VAD)
- Available for all effects
- Zeros out non-speech audio frames
- Useful for further reducing noise during silence

## Troubleshooting

If the NVIDIA Audio Effects aren't working:
1. Make sure you have a compatible NVIDIA GPU
2. Check that the SDK is correctly installed
3. Verify the model files are in the correct locations
4. Check application logs for specific error messages

## License

This project is licensed under the terms of the included license file. 