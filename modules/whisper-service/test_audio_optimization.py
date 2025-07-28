#!/usr/bin/env python3
"""
Test script for optimized audio processing pipeline
"""

import sys
import os
import time
import numpy as np
import tempfile
import wave

# Add the src directory to path to import from api_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_audio_wav(duration=2.0, sample_rate=44100, frequency=440):
    """Create a test WAV audio file in bytes"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(frequency * 2 * np.pi * t) * 0.5
    
    # Create WAV in memory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        # Read back as bytes
        with open(tmp_file.name, 'rb') as f:
            wav_bytes = f.read()
        
        os.unlink(tmp_file.name)
        return wav_bytes

def test_audio_optimization():
    """Test the optimized audio processing functions"""
    print("Testing optimized audio processing pipeline...")
    
    try:
        # Import the optimized functions
        from api_server import (
            _detect_audio_format_optimized,
            _high_quality_resample,
            _calculate_audio_quality_metrics,
            _process_audio_data,
            AUDIO_CONFIG
        )
        
        # Test 1: Format detection
        print("\\n1. Testing format detection...")
        wav_data = create_test_audio_wav(duration=1.0, sample_rate=44100)
        format_detected = _detect_audio_format_optimized(wav_data)
        print(f"   Detected format: {format_detected}")
        assert format_detected == "wav", f"Expected 'wav', got '{format_detected}'"
        
        # Test 2: Quality metrics calculation
        print("\\n2. Testing quality metrics...")
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5
        metrics = _calculate_audio_quality_metrics(test_audio, 16000)
        print(f"   Duration: {metrics['duration']:.2f}s")
        print(f"   RMS: {metrics['rms']:.4f}")
        print(f"   Peak: {metrics['peak']:.4f}")
        print(f"   Zero crossing rate: {metrics['zero_crossing_rate']:.4f}")
        
        # Test 3: High-quality resampling
        print("\\n3. Testing high-quality resampling...")
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.5
        
        for quality in ['kaiser_fast', 'kaiser_best']:
            start_time = time.time()
            resampled = _high_quality_resample(original_audio, 44100, 16000, quality)
            resample_time = time.time() - start_time
            print(f"   {quality}: {len(original_audio)} -> {len(resampled)} samples in {resample_time:.3f}s")
        
        # Test 4: Full processing pipeline
        print("\\n4. Testing full processing pipeline...")
        test_cases = [
            ("WAV 44.1kHz", create_test_audio_wav(2.0, 44100)),
            ("WAV 48kHz", create_test_audio_wav(1.5, 48000)),
            ("WAV 16kHz", create_test_audio_wav(1.0, 16000)),
        ]
        
        for test_name, audio_data in test_cases:
            print(f"   Testing {test_name}...")
            start_time = time.time()
            
            # Test with different quality settings
            for quality in ['kaiser_fast', 'kaiser_best']:
                result = _process_audio_data(audio_data, enhance=False, quality=quality)
                process_time = time.time() - start_time
                
                print(f"     {quality}: {len(result)} samples, {process_time:.3f}s")
                assert len(result) > 0, "Processing returned empty array"
                assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
                
                start_time = time.time()  # Reset for next quality test
        
        # Test 5: Memory efficiency test
        print("\\n5. Testing memory efficiency...")
        large_audio_data = create_test_audio_wav(duration=10.0, sample_rate=44100)  # 10 seconds
        
        start_time = time.time()
        result = _process_audio_data(large_audio_data, enhance=True, quality='kaiser_fast')
        process_time = time.time() - start_time
        
        expected_samples = 10 * 16000  # 10 seconds at 16kHz
        actual_samples = len(result)
        print(f"   Large file: {len(large_audio_data)} bytes -> {actual_samples} samples")
        print(f"   Processing time: {process_time:.3f}s")
        print(f"   Expected ~{expected_samples} samples, got {actual_samples}")
        
        # Test 6: Configuration testing
        print("\\n6. Testing configuration...")
        print(f"   Default sample rate: {AUDIO_CONFIG['default_sample_rate']}")
        print(f"   Default resampling quality: {AUDIO_CONFIG['resampling_quality']}")
        print(f"   Format cache enabled: {AUDIO_CONFIG['enable_format_cache']}")
        
        print("\\n[PASS] All tests passed! Audio optimization is working correctly.")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        print("Make sure you're running this from the whisper-service directory")
        return False
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_audio_optimization()
    sys.exit(0 if success else 1)