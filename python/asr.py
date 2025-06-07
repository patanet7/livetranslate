import numpy as np
import io
import soundfile as sf
import requests
import time
import os
from typing import Optional, Callable
import threading
import queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/transcription.log')
    ]
)
logger = logging.getLogger(__name__)

class ASRProcessor:
    def __init__(self,
                 server_url: Optional[str] = None,
                 model_name: str = "whisper-medium.en",
                 sample_rate: int = 16000,
                 chunk_duration: float = 2.0,  # Process audio every 2 seconds
                 on_transcript: Optional[Callable[[str, bool], None]] = None):
        """
        Initialize Whisper NPU ASR processor
        
        Args:
            server_url: URL of the whisper-npu-server instance
            model_name: Name of the Whisper model to use
            sample_rate: Audio sample rate (must match input audio)
            chunk_duration: Duration in seconds for each processing chunk
            on_transcript: Callback for transcript updates (text, is_final)
        """
        # Use environment variable or default to container network URL
        if server_url is None:
            server_url = os.getenv('WHISPER_SERVER_URL', 'http://whisper-npu-server:5000')
        
        self.server_url = server_url
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.on_transcript = on_transcript
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Processing thread
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = False
        
        # Start processing thread
        self.start_processing_thread()
        
        # Check if server is available
        self._check_server()
        
    def _check_server(self) -> None:
        """Check if the whisper-npu-server is running and accessible."""
        try:
            logger.info(f"Checking Whisper NPU server at {self.server_url}")
            response = requests.get(f"{self.server_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"Connected to Whisper NPU server. Available models: {models}")
            else:
                logger.warning(f"Whisper NPU server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Could not connect to whisper-npu-server at {self.server_url}. "
                "Transcription will not work until server is available."
            )
        except Exception as e:
            logger.warning(f"Error checking Whisper NPU server: {e}")

    def start_processing_thread(self):
        """Start the audio processing thread"""
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _processing_worker(self):
        """Worker thread that processes audio chunks"""
        while not self.stop_processing:
            try:
                # Wait for audio chunk or timeout
                audio_chunk = self.processing_queue.get(timeout=1.0)
                if audio_chunk is None:  # Stop signal
                    break
                    
                # Process the audio chunk
                self._process_audio_chunk(audio_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")

    def _process_audio_chunk(self, audio_data: np.ndarray):
        """Process a chunk of audio data with Whisper NPU"""
        try:
            # Convert audio to WAV format in memory
            audio_float = audio_data.astype(np.float32)
            if np.max(np.abs(audio_float)) > 0:
                audio_float = audio_float / np.max(np.abs(audio_float))
            
            # Create WAV data in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_float, self.sample_rate, format='WAV')
            buffer.seek(0)
            
            # Send to Whisper NPU server
            logger.debug(f"Sending audio chunk to Whisper NPU server: {self.server_url}")
            response = requests.post(
                f"{self.server_url}/transcribe/{self.model_name}",
                data=buffer.getvalue(),
                headers={'Content-Type': 'audio/wav'},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                if text and self.on_transcript:
                    logger.info(f"Transcribed text: '{text[:100]}...'")
                    # Whisper typically returns final results
                    self.on_transcript(text, True)
                else:
                    logger.debug("Empty transcription result")
            else:
                logger.error(f"Whisper NPU server error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def process_audio(self, audio_data: np.ndarray):
        """
        Process a chunk of audio data
        
        Args:
            audio_data: Numpy array of audio samples (mono, 16-bit or float)
        """
        with self.buffer_lock:
            # Convert to float if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Add to buffer
            self.audio_buffer.extend(audio_float)
            
            # Check if we have enough audio to process
            samples_needed = int(self.chunk_duration * self.sample_rate)
            if len(self.audio_buffer) >= samples_needed:
                # Extract chunk for processing
                chunk = np.array(self.audio_buffer[:samples_needed])
                # Keep some overlap (optional, for better continuity)
                overlap_samples = int(0.5 * self.sample_rate)  # 0.5 second overlap
                self.audio_buffer = self.audio_buffer[samples_needed - overlap_samples:]
                
                # Send to processing queue
                try:
                    self.processing_queue.put_nowait(chunk)
                except queue.Full:
                    logger.warning("Processing queue full, dropping audio chunk")

    def reset(self):
        """Reset the processor state"""
        with self.buffer_lock:
            self.audio_buffer.clear()
        
        # Clear processing queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        """Stop the ASR processor"""
        self.stop_processing = True
        self.processing_queue.put(None)  # Stop signal
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

if __name__ == "__main__":
    # Test ASR processing
    from capture import AudioCapture
    import time
    
    def on_transcript(text: str, is_final: bool):
        status = "FINAL" if is_final else "partial"
        print(f"[{status}] {text}")
    
    # First, let the user select an audio device
    devices = AudioCapture.list_devices()
    
    # Let user choose a device
    print("\nEnter device number for system audio capture (press Enter for automatic selection):")
    choice = input("> ").strip()
    
    device = None
    if choice and choice.isdigit():
        device = int(choice)
        print(f"Selected device {device}")
    else:
        device = AudioCapture.find_loopback_device()
        
    # Initialize ASR
    try:
        asr = ASRProcessor(on_transcript=on_transcript)
        
        # Initialize audio capture with ASR processing
        capture = AudioCapture(callback=asr.process_audio, device=device)
        
        # Start capture
        capture.start_capture()
        
        try:
            print("Speak or play audio (press Enter to stop)...")
            input()
        finally:
            capture.stop_capture()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download a Vosk model from https://alphacephei.com/vosk/models")
        print("Example: download vosk-model-small-en-us-0.15.zip and extract to models/vosk/")
        exit(1) 