import os
import json
import requests
import soundfile as sf
import numpy as np
from typing import Optional, Dict, Any

class WhisperNPUTranscriber:
    def __init__(self, server_url: str = "http://localhost:8009"):
        """Initialize the Whisper NPU transcriber.
        
        Args:
            server_url: URL of the whisper-npu-server instance
        """
        self.server_url = server_url
        self._check_server()

    def _check_server(self) -> None:
        """Check if the whisper-npu-server is running and accessible."""
        try:
            response = requests.get(f"{self.server_url}/models")
            if response.status_code != 200:
                raise ConnectionError(f"Server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to whisper-npu-server. "
                "Make sure it's running and accessible at {self.server_url}"
            )

    def transcribe(
        self, 
        audio_path: str, 
        model_name: str = "whisper-medium.en"
    ) -> Dict[str, Any]:
        """Transcribe audio using the NPU-accelerated Whisper model.
        
        Args:
            audio_path: Path to the audio file
            model_name: Name of the Whisper model to use
            
        Returns:
            Dict containing the transcription result
        """
        # Read audio file
        audio_data, sample_rate = sf.read(audio_path)
        
        # Ensure audio is mono and at 16kHz
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            # Resample to 16kHz if needed
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
        
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Save as temporary WAV file
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio_data, 16000)
        
        try:
            # Send to server
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{self.server_url}/transcribe/{model_name}",
                    data=f
                )
            
            if response.status_code != 200:
                raise Exception(f"Server returned error: {response.text}")
            
            result = response.json()
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_available_models(self) -> list:
        """Get list of available Whisper models."""
        response = requests.get(f"{self.server_url}/models")
        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")
        return response.json() 