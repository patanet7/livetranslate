import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from python.whisper_npu import WhisperNPUTranscriber

def main():
    # Initialize the transcriber
    transcriber = WhisperNPUTranscriber()
    
    # Get available models
    print("Available models:")
    models = transcriber.get_available_models()
    for model in models:
        print(f"- {model}")
    
    # Example audio file path - replace with your audio file
    audio_path = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"Please provide a valid audio file path. Current path '{audio_path}' does not exist.")
        return
    
    # Transcribe using default model (whisper-medium.en)
    print("\nTranscribing with default model (whisper-medium.en)...")
    result = transcriber.transcribe(audio_path)
    print(f"Transcription: {result['text']}")
    
    # Transcribe using a specific model
    model_name = "whisper-small.en"  # Example model
    print(f"\nTranscribing with {model_name}...")
    result = transcriber.transcribe(audio_path, model_name=model_name)
    print(f"Transcription: {result['text']}")

if __name__ == "__main__":
    main() 