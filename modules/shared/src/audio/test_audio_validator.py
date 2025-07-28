"""
Comprehensive test suite for AudioValidator

This test file validates all functionality of the audio validation library
including format validation, corruption detection, quality assessment,
and format conversion.
"""

import numpy as np
import io
import tempfile
import os
from typing import List, Dict, Any
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from audio_validator import (
        AudioValidator, AudioFormat, QualityLevel,
        validate_audio, check_audio_corruption, convert_audio_format,
        AudioValidationError, AudioFormatError, AudioCorruptionError
    )
    import librosa
    import soundfile as sf
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Dependencies not available: {e}")
    print("Install required packages: pip install librosa soundfile scipy numpy matplotlib")


class AudioValidatorTester:
    """Comprehensive test suite for AudioValidator"""
    
    def __init__(self):
        self.validator = AudioValidator()
        self.test_results = []
        self.sample_rate = 16000
        
    def generate_test_audio(self, duration: float = 1.0, frequency: float = 440.0, 
                          sample_rate: int = 16000, noise_level: float = 0.0) -> np.ndarray:
        """Generate test audio with specified parameters"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(audio))
            audio += noise
        
        return audio
    
    def generate_corrupted_audio(self, base_audio: np.ndarray, corruption_type: str) -> np.ndarray:
        """Generate corrupted audio for testing corruption detection"""
        corrupted = base_audio.copy()
        
        if corruption_type == "clipping":
            # Introduce clipping
            corrupted = np.clip(corrupted * 2, -0.99, 0.99)
            
        elif corruption_type == "dropouts":
            # Introduce dropouts (zeros)
            dropout_indices = np.random.choice(len(corrupted), size=len(corrupted)//100)
            corrupted[dropout_indices] = 0
            
        elif corruption_type == "pops":
            # Introduce sudden amplitude spikes
            pop_indices = np.random.choice(len(corrupted), size=10)
            corrupted[pop_indices] = np.random.choice([-1, 1], size=len(pop_indices))
            
        elif corruption_type == "noise":
            # Add significant noise
            noise = np.random.normal(0, 0.5, len(corrupted))
            corrupted += noise
            
        elif corruption_type == "silence":
            # Make most of the audio silent
            silence_start = len(corrupted) // 4
            silence_end = 3 * len(corrupted) // 4
            corrupted[silence_start:silence_end] = 0
            
        return corrupted
    
    def create_test_file(self, audio_data: np.ndarray, file_format: AudioFormat, 
                        sample_rate: int = 16000) -> str:
        """Create a temporary test file"""
        with tempfile.NamedTemporaryFile(suffix=f'.{file_format.value}', delete=False) as f:
            if file_format == AudioFormat.WAV:
                sf.write(f.name, audio_data, sample_rate, format='WAV')
            elif file_format == AudioFormat.FLAC:
                sf.write(f.name, audio_data, sample_rate, format='FLAC')
            elif file_format == AudioFormat.OGG:
                sf.write(f.name, audio_data, sample_rate, format='OGG')
            else:
                # Default to WAV
                sf.write(f.name, audio_data, sample_rate, format='WAV')
            
            return f.name
    
    def test_basic_validation(self) -> Dict[str, Any]:
        """Test basic audio validation functionality"""
        print("Testing basic validation...")
        
        results = {}
        
        # Test with clean sine wave
        clean_audio = self.generate_test_audio(duration=2.0, frequency=440.0)
        result = self.validator.validate_audio_format(clean_audio)
        
        results['clean_audio'] = {
            'is_valid': result.is_valid,
            'quality_score': result.quality_score,
            'quality_level': result.quality_level.value,
            'corruption_detected': result.corruption_detected,
            'processing_time': result.processing_time
        }
        
        # Test with noisy audio
        noisy_audio = self.generate_test_audio(duration=1.0, noise_level=0.1)
        result = self.validator.validate_audio_format(noisy_audio)
        
        results['noisy_audio'] = {
            'is_valid': result.is_valid,
            'quality_score': result.quality_score,
            'quality_level': result.quality_level.value,
            'corruption_detected': result.corruption_detected
        }
        
        # Test with very short audio
        short_audio = self.generate_test_audio(duration=0.05)  # 50ms
        result = self.validator.validate_audio_format(short_audio)
        
        results['short_audio'] = {
            'is_valid': result.is_valid,
            'warnings': result.warnings,
            'recommendations': result.recommendations
        }
        
        return results
    
    def test_corruption_detection(self) -> Dict[str, Any]:
        """Test corruption detection capabilities"""
        print("Testing corruption detection...")
        
        results = {}
        base_audio = self.generate_test_audio(duration=1.0)
        
        corruption_types = ["clipping", "dropouts", "pops", "noise", "silence"]
        
        for corruption_type in corruption_types:
            corrupted_audio = self.generate_corrupted_audio(base_audio, corruption_type)
            analysis = self.validator.detect_audio_corruption(corrupted_audio, self.sample_rate)
            
            results[corruption_type] = {
                'is_corrupted': analysis.is_corrupted,
                'corruption_type': analysis.corruption_type,
                'corruption_severity': analysis.corruption_severity,
                'confidence': analysis.confidence,
                'affected_regions_count': len(analysis.affected_regions)
            }
        
        # Test clean audio (should not detect corruption)
        clean_analysis = self.validator.detect_audio_corruption(base_audio, self.sample_rate)
        results['clean'] = {
            'is_corrupted': clean_analysis.is_corrupted,
            'corruption_severity': clean_analysis.corruption_severity,
            'confidence': clean_analysis.confidence
        }
        
        return results
    
    def test_sample_rate_validation(self) -> Dict[str, Any]:
        """Test sample rate validation"""
        print("Testing sample rate validation...")
        
        results = {}
        audio = self.generate_test_audio()
        
        # Test various sample rates
        test_rates = [8000, 16000, 22050, 44100, 48000]
        
        for rate in test_rates:
            is_valid, actual_rate, analysis = self.validator.validate_sample_rate(audio, rate)
            results[f'rate_{rate}'] = {
                'is_valid': is_valid,
                'actual_rate': actual_rate,
                'needs_resampling': analysis.get('needs_resampling', False)
            }
        
        return results
    
    def test_quality_assessment(self) -> Dict[str, Any]:
        """Test audio quality assessment"""
        print("Testing quality assessment...")
        
        results = {}
        
        # Test different quality levels
        test_scenarios = [
            ('high_quality', self.generate_test_audio(duration=2.0, noise_level=0.001)),
            ('medium_quality', self.generate_test_audio(duration=1.0, noise_level=0.05)),
            ('low_quality', self.generate_test_audio(duration=0.5, noise_level=0.2)),
            ('very_low_quality', self.generate_test_audio(duration=0.2, noise_level=0.5))
        ]
        
        for scenario_name, audio in test_scenarios:
            quality_score, quality_level, metrics = self.validator.validate_audio_quality(audio, self.sample_rate)
            
            results[scenario_name] = {
                'quality_score': quality_score,
                'quality_level': quality_level.value,
                'snr_db': metrics.get('snr_db', 0),
                'dynamic_range_db': metrics.get('dynamic_range_db', 0),
                'silence_ratio': metrics.get('silence_ratio', 0)
            }
        
        return results
    
    def test_format_conversion(self) -> Dict[str, Any]:
        """Test audio format conversion"""
        print("Testing format conversion...")
        
        results = {}
        audio = self.generate_test_audio(duration=1.0)
        
        # Test conversion to different formats
        target_formats = [AudioFormat.WAV, AudioFormat.FLAC]
        
        for target_format in target_formats:
            try:
                converted_bytes, metadata = self.validator.standardize_audio_format(
                    audio, target_format, target_sample_rate=self.sample_rate
                )
                
                results[target_format.value] = {
                    'success': True,
                    'output_size': len(converted_bytes),
                    'metadata': {
                        'format': metadata.format,
                        'sample_rate': metadata.sample_rate,
                        'duration': metadata.duration,
                        'channels': metadata.channels
                    }
                }
            except Exception as e:
                results[target_format.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def test_file_operations(self) -> Dict[str, Any]:
        """Test file-based operations"""
        print("Testing file operations...")
        
        results = {}
        audio = self.generate_test_audio(duration=1.0)
        
        # Test with different file formats
        file_formats = [AudioFormat.WAV, AudioFormat.FLAC]
        
        for file_format in file_formats:
            try:
                # Create test file
                test_file = self.create_test_file(audio, file_format, self.sample_rate)
                
                # Test validation from file
                validation_result = self.validator.validate_audio_format(test_file, file_format)
                
                # Test metadata extraction
                metadata = self.validator.get_audio_metadata(test_file)
                
                results[file_format.value] = {
                    'validation_success': validation_result.is_valid,
                    'quality_score': validation_result.quality_score,
                    'metadata': {
                        'format': metadata.format,
                        'sample_rate': metadata.sample_rate,
                        'duration': metadata.duration,
                        'file_size': metadata.file_size
                    }
                }
                
                # Clean up
                os.unlink(test_file)
                
            except Exception as e:
                results[file_format.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def test_convenience_functions(self) -> Dict[str, Any]:
        """Test convenience functions"""
        print("Testing convenience functions...")
        
        results = {}
        audio = self.generate_test_audio(duration=1.0)
        
        # Test validate_audio convenience function
        try:
            validation_result = validate_audio(audio, expected_sample_rate=self.sample_rate)
            results['validate_audio'] = {
                'success': True,
                'is_valid': validation_result.is_valid,
                'quality_score': validation_result.quality_score
            }
        except Exception as e:
            results['validate_audio'] = {'success': False, 'error': str(e)}
        
        # Test check_audio_corruption convenience function
        try:
            corrupted_audio = self.generate_corrupted_audio(audio, "clipping")
            corruption_result = check_audio_corruption(corrupted_audio)
            results['check_corruption'] = {
                'success': True,
                'is_corrupted': corruption_result.is_corrupted,
                'corruption_severity': corruption_result.corruption_severity
            }
        except Exception as e:
            results['check_corruption'] = {'success': False, 'error': str(e)}
        
        # Test convert_audio_format convenience function
        try:
            converted_bytes, metadata = convert_audio_format(audio, AudioFormat.WAV, self.sample_rate)
            results['convert_format'] = {
                'success': True,
                'output_size': len(converted_bytes),
                'format': metadata.format
            }
        except Exception as e:
            results['convert_format'] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling"""
        print("Testing edge cases...")
        
        results = {}
        
        # Test empty audio
        try:
            empty_audio = np.array([])
            result = self.validator.validate_audio_format(empty_audio)
            results['empty_audio'] = {
                'handled': True,
                'is_valid': result.is_valid,
                'errors': result.errors
            }
        except Exception as e:
            results['empty_audio'] = {'handled': False, 'error': str(e)}
        
        # Test silent audio
        try:
            silent_audio = np.zeros(self.sample_rate)  # 1 second of silence
            result = self.validator.validate_audio_format(silent_audio)
            results['silent_audio'] = {
                'handled': True,
                'is_valid': result.is_valid,
                'quality_score': result.quality_score
            }
        except Exception as e:
            results['silent_audio'] = {'handled': False, 'error': str(e)}
        
        # Test extremely loud audio
        try:
            loud_audio = np.ones(self.sample_rate) * 10  # Clipped audio
            result = self.validator.validate_audio_format(loud_audio)
            results['loud_audio'] = {
                'handled': True,
                'corruption_detected': result.corruption_detected,
                'quality_score': result.quality_score
            }
        except Exception as e:
            results['loud_audio'] = {'handled': False, 'error': str(e)}
        
        # Test invalid input types
        try:
            result = self.validator.validate_audio_format("invalid_input")
            results['invalid_input'] = {
                'handled': True,
                'is_valid': result.is_valid,
                'errors': result.errors
            }
        except Exception as e:
            results['invalid_input'] = {'handled': True, 'error': str(e)}
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("=" * 60)
        print("RUNNING COMPREHENSIVE AUDIO VALIDATOR TEST SUITE")
        print("=" * 60)
        
        if not DEPENDENCIES_AVAILABLE:
            return {"error": "Required dependencies not available"}
        
        all_results = {}
        
        try:
            all_results['basic_validation'] = self.test_basic_validation()
            all_results['corruption_detection'] = self.test_corruption_detection()
            all_results['sample_rate_validation'] = self.test_sample_rate_validation()
            all_results['quality_assessment'] = self.test_quality_assessment()
            all_results['format_conversion'] = self.test_format_conversion()
            all_results['file_operations'] = self.test_file_operations()
            all_results['convenience_functions'] = self.test_convenience_functions()
            all_results['edge_cases'] = self.test_edge_cases()
            
            # Generate summary
            all_results['test_summary'] = self.generate_test_summary(all_results)
            
        except Exception as e:
            all_results['error'] = f"Test suite failed: {str(e)}"
            logger.error(f"Test suite failed: {str(e)}")
        
        return all_results
    
    def generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'total_test_categories': len([k for k in results.keys() if k != 'test_summary']),
            'successful_categories': 0,
            'failed_categories': 0,
            'key_findings': []
        }
        
        for category, category_results in results.items():
            if category == 'test_summary':
                continue
                
            if isinstance(category_results, dict) and 'error' not in category_results:
                summary['successful_categories'] += 1
            else:
                summary['failed_categories'] += 1
        
        # Key findings
        if 'corruption_detection' in results:
            corruption_results = results['corruption_detection']
            detected_corruptions = [k for k, v in corruption_results.items() 
                                  if isinstance(v, dict) and v.get('is_corrupted', False)]
            summary['key_findings'].append(f"Corruption detection: {len(detected_corruptions)} types detected")
        
        if 'quality_assessment' in results:
            quality_results = results['quality_assessment']
            quality_scores = [v.get('quality_score', 0) for v in quality_results.values() 
                            if isinstance(v, dict)]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                summary['key_findings'].append(f"Average quality score: {avg_quality:.3f}")
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        if 'error' in results:
            print(f"❌ Test suite failed: {results['error']}")
            return
        
        for category, category_results in results.items():
            if category == 'test_summary':
                continue
                
            print(f"\n📋 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    if isinstance(test_result, dict):
                        status = "✅" if test_result.get('success', test_result.get('is_valid', True)) else "❌"
                        print(f"  {status} {test_name}: {self._format_test_result(test_result)}")
                    else:
                        print(f"  📊 {test_name}: {test_result}")
        
        # Print summary
        if 'test_summary' in results:
            summary = results['test_summary']
            print(f"\n🎯 OVERALL SUMMARY")
            print("-" * 40)
            print(f"  Total categories: {summary['total_test_categories']}")
            print(f"  Successful: {summary['successful_categories']}")
            print(f"  Failed: {summary['failed_categories']}")
            
            for finding in summary['key_findings']:
                print(f"  📈 {finding}")
    
    def _format_test_result(self, result: Dict[str, Any]) -> str:
        """Format individual test result for display"""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        if 'quality_score' in result:
            return f"Quality: {result['quality_score']:.3f}"
        
        if 'is_corrupted' in result:
            return f"Corrupted: {result['is_corrupted']}"
        
        if 'output_size' in result:
            return f"Size: {result['output_size']} bytes"
        
        return "OK"


def main():
    """Run the comprehensive test suite"""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Cannot run tests - missing dependencies")
        print("Install with: pip install librosa soundfile scipy numpy matplotlib")
        return
    
    tester = AudioValidatorTester()
    results = tester.run_all_tests()
    tester.print_results(results)
    
    print(f"\n🏁 Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()