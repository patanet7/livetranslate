try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    print(f"Warning: sounddevice not available: {e}")
    # Create a dummy sd object to prevent errors
    class DummySD:
        @staticmethod
        def query_devices(*args, **kwargs):
            return []
        @staticmethod
        def InputStream(*args, **kwargs):
            raise RuntimeError("sounddevice not available")
        @staticmethod
        def OutputStream(*args, **kwargs):
            raise RuntimeError("sounddevice not available")
    sd = DummySD()

import numpy as np
from typing import Callable, Optional
import queue
import threading

class AudioCapture:
    def __init__(self, 
                 callback: Optional[Callable[[np.ndarray], None]] = None,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 dtype=np.int16,
                 blocksize: int = 8000,  # 500ms chunks at 16kHz
                 device=None,            # None = automatic device selection
                 use_loopback=True,     # True = capture system audio
                 passthrough=False):    # True = play captured audio back to output
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.audio_queue = queue.Queue()
        self.stream = None
        self.output_stream = None
        self.is_running = False
        self.device = device
        self.use_loopback = use_loopback
        self.passthrough = passthrough
        
    @staticmethod
    def list_devices():
        """List all available audio devices"""
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
        return devices
    
    @staticmethod
    def find_loopback_device():
        """Find a suitable device for system audio capture"""
        devices = sd.query_devices()
        
        # First look for devices with "loopback" in the name
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and 'loopback' in dev['name'].lower():
                print(f"Found loopback device: [{i}] {dev['name']}")
                return i
                
        # Next look for "stereo mix" which is common on many systems
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and 'stereo mix' in dev['name'].lower():
                print(f"Found Stereo Mix device: [{i}] {dev['name']}")
                return i
        
        # Fall back to default output device
        try:
            default_output = sd.query_devices(kind='output')
            if isinstance(default_output, dict):
                idx = default_output.get('index', None)
                if idx is not None:
                    print(f"Using default output device: [{idx}] {default_output['name']}")
                    return idx
        except Exception as e:
            print(f"Error finding default output device: {e}")
            
        print("No suitable loopback device found. Using default input device.")
        return None

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to handle incoming audio data"""
        if status:
            print(f'Audio callback status: {status}')
        
        # Send audio to output if passthrough is enabled
        if self.passthrough and self.output_stream:
            self.output_stream.write(indata)
            
        if self.callback:
            # Convert to mono if needed
            if self.channels == 1 and indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            self.callback(audio_data.astype(self.dtype))
        else:
            self.audio_queue.put(indata.copy())

    def start_capture(self):
        """Start capturing system audio"""
        try:
            # Find appropriate device for system audio capture
            device = self.device
            if device is None and self.use_loopback:
                device = self.find_loopback_device()
            
            # Attempt to capture from default output device as a fallback
            if device is None:
                try:
                    default_output = sd.query_devices(kind='output')
                    if isinstance(default_output, dict):
                        device = default_output.get('index')
                        print(f"Capturing from default output device: [{device}] {default_output['name']}")
                except Exception as e:
                    print(f"Error finding default output device: {e}")
            
            device_info = None
            if device is not None:
                device_info = sd.query_devices(device)
                print(f"Using device: {device_info['name']}")
            
            # Get the right channels config from the device
            channels_in = 2  # Default stereo
            if device_info:
                channels_in = device_info.get('max_input_channels', 2)
                if channels_in == 0:  # If it's an output-only device
                    print("Warning: Selected device has no input channels. Falling back to default input.")
                    device = None
            
            # Create input stream
            self.stream = sd.InputStream(
                device=device,
                callback=self.audio_callback,
                channels=channels_in,
                samplerate=self.sample_rate,
                blocksize=self.blocksize
            )
            
            # Create output stream if passthrough is enabled
            if self.passthrough:
                output_device = sd.query_devices(kind='output').get('index')
                self.output_stream = sd.OutputStream(
                    device=output_device,
                    channels=channels_in,
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize
                )
                self.output_stream.start()
                
            self.stream.start()
            self.is_running = True
            print(f"Started audio capture: {self.sample_rate}Hz, {self.channels} channel(s)")
            if device_info:
                print(f"Device: {device_info['name']}")
            if self.passthrough:
                print("Audio passthrough enabled")
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            raise

    def stop_capture(self):
        """Stop capturing system audio"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("Stopped audio capture")
            
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            print("Stopped audio passthrough")

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the next chunk of audio data from the queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # List all available devices
    devices = AudioCapture.list_devices()
    
    # Let user choose a device
    print("\nEnter device number for system audio capture (press Enter for automatic selection):")
    choice = input("> ").strip()
    
    device = None
    if choice and choice.isdigit():
        device = int(choice)
        print(f"Selected device {device}")
    
    # Test audio capture
    def print_audio_stats(audio_data):
        # Only print if there's actual audio data
        if np.max(np.abs(audio_data)) > 0.01:
            print(f"Received audio chunk: shape={audio_data.shape}, "
                f"min={np.min(audio_data)}, max={np.max(audio_data)}")
    
    # Use loopback device for system audio if no manual selection
    if device is None:
        device = AudioCapture.find_loopback_device()
    
    # Enable passthrough for testing
    capture = AudioCapture(
        callback=print_audio_stats, 
        device=device,
        passthrough=True
    )
    
    capture.start_capture()
    
    try:
        print("Play some audio on your system to test. Press Enter to stop...")
        input()
    finally:
        capture.stop_capture() 