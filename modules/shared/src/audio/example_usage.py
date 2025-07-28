"""
Example usage of the AudioValidator library

This script demonstrates various features of the audio validation library
including validation, corruption detection, quality assessment, and format conversion.
"""

import numpy as np
import time
from typing import List, Dict, Any

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import librosa
        import soundfile as sf
        import scipy
        import matplotlib
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install librosa soundfile scipy numpy matplotlib")
        return False

def generate_sample_audio_data():
    """Generate various types of sample audio for testing"""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    samples = {}
    
    # 1. Clean sine wave (high quality)
    samples['clean_sine'] = {
        'data': np.sin(2 * np.pi * 440 * t),  # 440 Hz sine wave
        'description': 'Clean 440Hz sine wave'
    }
    
    # 2. Noisy audio (medium quality)
    noise_level = 0.1
    samples['noisy_audio'] = {
        'data': np.sin(2 * np.pi * 440 * t) + np.random.normal(0, noise_level, len(t)),
        'description': 'Sine wave with 10% noise'
    }
    
    # 3. Clipped audio (corrupted)
    clipped = np.sin(2 * np.pi * 440 * t) * 2  # Amplify to cause clipping
    samples['clipped_audio'] = {
        'data': np.clip(clipped, -0.99, 0.99),
        'description': 'Clipped sine wave'
    }
    
    # 4. Audio with dropouts (corrupted)
    dropout_audio = np.sin(2 * np.pi * 440 * t)
    # Create random dropouts
    dropout_indices = np.random.choice(len(dropout_audio), size=len(dropout_audio)//50)
    dropout_audio[dropout_indices] = 0
    samples['dropout_audio'] = {
        'data': dropout_audio,
        'description': 'Sine wave with dropouts'
    }
    
    # 5. Mixed frequency content
    mixed = (np.sin(2 * np.pi * 440 * t) + 
             0.5 * np.sin(2 * np.pi * 880 * t) + 
             0.25 * np.sin(2 * np.pi * 1320 * t))
    samples['mixed_frequencies'] = {
        'data': mixed / np.max(np.abs(mixed)),  # Normalize
        'description': 'Mixed frequencies (440Hz, 880Hz, 1320Hz)'
    }
    
    # 6. Silent audio
    samples['silent_audio'] = {
        'data': np.zeros(int(sample_rate * 0.5)),  # 0.5 seconds of silence
        'description': 'Complete silence'
    }
    
    return samples, sample_rate

def demonstrate_basic_validation():
    """Demonstrate basic audio validation functionality"""
    print("=" * 60)
    print("BASIC AUDIO VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    from audio_validator import AudioValidator, AudioFormat
    
    # Initialize validator
    validator = AudioValidator(default_sample_rate=16000, quality_threshold=0.7)
    
    # Generate sample audio data
    samples, sample_rate = generate_sample_audio_data()
    
    print(f"\nTesting {len(samples)} different audio samples:")
    print("-" * 40)
    
    for sample_name, sample_info in samples.items():
        audio_data = sample_info['data']
        description = sample_info['description']
        
        print(f"\n📊 {sample_name.upper()}")
        print(f"Description: {description}")
        
        # Validate the audio
        start_time = time.time()
        result = validator.validate_audio_format(audio_data)
        validation_time = time.time() - start_time
        
        # Print results
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"Status: {status}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Quality Level: {result.quality_level.value}")
        print(f"Corruption Detected: {result.corruption_detected}")
        print(f"Processing Time: {validation_time:.3f}s")
        
        if result.errors:
            print(f"Errors: {', '.join(result.errors)}")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")
        
        if result.recommendations:
            print(f"Recommendations: {', '.join(result.recommendations)}")

def demonstrate_corruption_detection():
    """Demonstrate corruption detection capabilities"""
    print("\n" + "=" * 60)
    print("CORRUPTION DETECTION DEMONSTRATION")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    from audio_validator import AudioValidator
    
    validator = AudioValidator()
    samples, sample_rate = generate_sample_audio_data()
    
    # Focus on potentially corrupted samples
    test_samples = ['clean_sine', 'clipped_audio', 'dropout_audio', 'noisy_audio']
    
    print(f"\nAnalyzing corruption in {len(test_samples)} samples:")
    print("-" * 40)
    
    for sample_name in test_samples:
        if sample_name not in samples:
            continue
            
        audio_data = samples[sample_name]['data']
        description = samples[sample_name]['description']
        
        print(f"\n🔍 {sample_name.upper()}")
        print(f"Description: {description}")
        
        # Detect corruption
        corruption = validator.detect_audio_corruption(audio_data, sample_rate)
        
        print(f"Corrupted: {corruption.is_corrupted}")
        if corruption.is_corrupted:
            print(f"Corruption Type: {corruption.corruption_type}")
            print(f"Severity: {corruption.corruption_severity:.3f}")
            print(f"Confidence: {corruption.confidence:.3f}")
            
            if corruption.affected_regions:
                print(f"Affected Regions: {len(corruption.affected_regions)}")
                for i, (start, end) in enumerate(corruption.affected_regions[:3]):  # Show first 3
                    print(f"  Region {i+1}: {start:.2f}s - {end:.2f}s")
            
            # Show key corruption details
            details = corruption.details
            if 'clipping_ratio' in details:
                print(f"Clipping Ratio: {details['clipping_ratio']:.4f}")

def demonstrate_quality_assessment():
    """Demonstrate audio quality assessment"""
    print("\n" + "=" * 60)
    print("QUALITY ASSESSMENT DEMONSTRATION")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    from audio_validator import AudioValidator
    
    validator = AudioValidator()
    samples, sample_rate = generate_sample_audio_data()
    
    print(f"\nAssessing quality of {len(samples)} samples:")
    print("-" * 40)
    
    quality_results = []
    
    for sample_name, sample_info in samples.items():
        audio_data = sample_info['data']
        description = sample_info['description']
        
        print(f"\n📈 {sample_name.upper()}")
        print(f"Description: {description}")
        
        # Assess quality
        quality_score, quality_level, metrics = validator.validate_audio_quality(audio_data, sample_rate)
        
        print(f"Quality Score: {quality_score:.3f}")
        print(f"Quality Level: {quality_level.value}")
        
        # Show key metrics
        if 'snr_db' in metrics:
            print(f"SNR: {metrics['snr_db']:.1f} dB")
        if 'dynamic_range_db' in metrics:
            print(f"Dynamic Range: {metrics['dynamic_range_db']:.1f} dB")
        if 'silence_ratio' in metrics:
            print(f"Silence Ratio: {metrics['silence_ratio']:.3f}")
        
        quality_results.append((sample_name, quality_score, quality_level.value))
    
    # Summary
    print(f"\n📊 QUALITY SUMMARY")
    print("-" * 30)
    quality_results.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    for name, score, level in quality_results:
        print(f"{score:.3f} ({level:12}) - {name}")

def demonstrate_format_conversion():
    """Demonstrate audio format conversion"""
    print("\n" + "=" * 60)
    print("FORMAT CONVERSION DEMONSTRATION")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    from audio_validator import AudioValidator, AudioFormat
    
    validator = AudioValidator()
    samples, sample_rate = generate_sample_audio_data()
    
    # Use clean sine wave for conversion
    audio_data = samples['clean_sine']['data']
    
    print(f"Converting clean sine wave to different formats:")
    print("-" * 40)
    
    # Test different target formats
    target_formats = [AudioFormat.WAV, AudioFormat.FLAC]
    target_sample_rates = [16000, 44100]
    
    for target_format in target_formats:
        for target_sr in target_sample_rates:
            print(f"\n🔄 Converting to {target_format.value.upper()} @ {target_sr}Hz")
            
            try:
                start_time = time.time()
                converted_bytes, metadata = validator.standardize_audio_format(
                    audio_data, 
                    target_format, 
                    target_sample_rate=target_sr,
                    preserve_quality=True
                )
                conversion_time = time.time() - start_time
                
                print(f"✓ Conversion successful")
                print(f"Output size: {len(converted_bytes)} bytes")
                print(f"Processing time: {conversion_time:.3f}s")
                print(f"Metadata: {metadata.format} | {metadata.sample_rate}Hz | {metadata.channels}ch | {metadata.duration:.2f}s")
                
            except Exception as e:
                print(f"✗ Conversion failed: {e}")

def demonstrate_convenience_functions():
    """Demonstrate convenience functions"""
    print("\n" + "=" * 60)
    print("CONVENIENCE FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    from audio_validator import validate_audio, check_audio_corruption, convert_audio_format, AudioFormat
    
    samples, sample_rate = generate_sample_audio_data()
    
    # Test convenience functions
    audio_data = samples['mixed_frequencies']['data']
    
    print("Testing convenience functions with mixed frequency audio:")
    print("-" * 50)
    
    # 1. Quick validation
    print("\n1. Quick Validation (validate_audio)")
    result = validate_audio(audio_data, expected_sample_rate=sample_rate)
    print(f"   Valid: {result.is_valid}")
    print(f"   Quality: {result.quality_score:.3f} ({result.quality_level.value})")
    
    # 2. Quick corruption check
    print("\n2. Quick Corruption Check (check_audio_corruption)")
    corruption = check_audio_corruption(audio_data)
    print(f"   Corrupted: {corruption.is_corrupted}")
    if corruption.is_corrupted:
        print(f"   Type: {corruption.corruption_type}")
        print(f"   Severity: {corruption.corruption_severity:.3f}")
    
    # 3. Quick format conversion
    print("\n3. Quick Format Conversion (convert_audio_format)")
    try:
        converted_bytes, metadata = convert_audio_format(
            audio_data, 
            AudioFormat.WAV, 
            target_sample_rate=sample_rate
        )
        print(f"   Converted: {len(converted_bytes)} bytes")
        print(f"   Format: {metadata.format}")
    except Exception as e:
        print(f"   Conversion failed: {e}")

def main():
    """Run all demonstrations"""
    print("AUDIO VALIDATOR LIBRARY DEMONSTRATION")
    print("This script showcases the capabilities of the AudioValidator library")
    
    if not check_dependencies():
        print("\nPlease install the required dependencies to run this demonstration:")
        print("pip install librosa soundfile scipy numpy matplotlib")
        return
    
    try:
        # Run all demonstrations
        demonstrate_basic_validation()
        demonstrate_corruption_detection()
        demonstrate_quality_assessment()
        demonstrate_format_conversion()
        demonstrate_convenience_functions()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe AudioValidator library is ready for use in your LiveTranslate system.")
        print("See README.md for detailed API documentation and usage examples.")
        
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        print("Please check that all dependencies are properly installed.")

if __name__ == "__main__":
    main()