import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

def generate_sine_wave(freq, duration, sample_rate=16000):
    """Generate a sine wave at the given frequency and duration"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def generate_test_audio(output_file, sample_rate=16000, duration=5.0):
    """Generate a test audio file with different tones"""
    # Create path if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate different frequency tones
    tones = []
    
    # Add a 1-second 440Hz tone (A4)
    tones.append(generate_sine_wave(440, 1.0, sample_rate))
    
    # Add a 1-second 523Hz tone (C5)
    tones.append(generate_sine_wave(523, 1.0, sample_rate))
    
    # Add a 1-second 659Hz tone (E5)
    tones.append(generate_sine_wave(659, 1.0, sample_rate))
    
    # Add a 1-second 784Hz tone (G5)
    tones.append(generate_sine_wave(784, 1.0, sample_rate))
    
    # Add a 1-second 880Hz tone (A5)
    tones.append(generate_sine_wave(880, 1.0, sample_rate))
    
    # Combine all tones
    audio = np.concatenate(tones)
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    # Write to file
    sf.write(output_file, audio, sample_rate)
    print(f"Generated test audio file: {output_file}")
    print(f"Duration: {len(audio)/sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

def main():
    parser = argparse.ArgumentParser(description="Generate test audio file")
    parser.add_argument("--output", default="test_audio.wav", help="Output WAV file path")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    
    args = parser.parse_args()
    
    generate_test_audio(args.output, args.rate, args.duration)

if __name__ == "__main__":
    main() 