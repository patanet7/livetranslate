#!/usr/bin/env python3
"""
Advanced Audio Processing Pipeline

Comprehensive audio processing with format detection, conversion, enhancement,
and preprocessing optimized for Whisper transcription. Supports multiple
formats with intelligent fallback strategies.

Features:
- Multi-format audio detection and conversion (WAV, MP3, WebM, OGG, MP4)
- Fragment and streaming audio support
- Multiple conversion backends with fallback
- Audio enhancement and preprocessing
- Memory-efficient processing with temporary file management
- Comprehensive error handling and recovery
"""

import os
import io
import tempfile
import logging
import time
from typing import Tuple, Optional, Dict, Any, Union, BinaryIO
from pathlib import Path
from contextlib import contextmanager
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioFormatError(Exception):
    """Audio format related errors"""
    pass


class AudioConversionError(Exception):
    """Audio conversion related errors"""
    pass


class AudioProcessor:
    """
    Advanced audio processor with comprehensive format support and optimization.
    
    Handles audio format detection, conversion, and preprocessing with multiple
    fallback strategies for maximum compatibility.
    """
    
    def __init__(self, 
                 default_sample_rate: int = 16000,
                 enable_enhancement: bool = True,
                 temp_dir: Optional[str] = None,
                 ffmpeg_path: Optional[str] = None):
        """
        Initialize the audio processor.
        
        Args:
            default_sample_rate: Target sample rate for transcription
            enable_enhancement: Enable audio enhancement preprocessing
            temp_dir: Directory for temporary files
            ffmpeg_path: Path to FFmpeg binary
        """
        self.default_sample_rate = default_sample_rate
        self.enable_enhancement = enable_enhancement
        self.temp_dir = temp_dir
        
        # Configure FFmpeg
        self._configure_ffmpeg(ffmpeg_path)
        
        # Audio format signatures for detection
        self.format_signatures = {
            b'RIFF': 'wav',
            b'\x1a\x45\xdf\xa3': 'webm',
            b'OggS': 'ogg',
            b'ID3': 'mp3',
            b'\xff\xfb': 'mp3',
            b'\xff\xfa': 'mp3',
            b'\xff\xf3': 'mp3',
            b'\xff\xf2': 'mp3',
        }
        
        # Supported formats and their typical extensions
        self.supported_formats = {
            'wav': ['.wav', '.wave'],
            'mp3': ['.mp3'],
            'webm': ['.webm'],
            'ogg': ['.ogg', '.oga'],
            'mp4': ['.mp4', '.m4a', '.aac'],
            'flac': ['.flac'],
            'aiff': ['.aiff', '.aif']
        }
        
        logger.info(f"AudioProcessor initialized:")
        logger.info(f"  Default sample rate: {default_sample_rate}Hz")
        logger.info(f"  Enhancement enabled: {enable_enhancement}")
        logger.info(f"  Librosa available: {LIBROSA_AVAILABLE}")
        logger.info(f"  Soundfile available: {SOUNDFILE_AVAILABLE}")
        logger.info(f"  Pydub available: {PYDUB_AVAILABLE}")
    
    def _configure_ffmpeg(self, ffmpeg_path: Optional[str]):
        """Configure FFmpeg for pydub if available."""
        if not PYDUB_AVAILABLE:
            return
        
        try:
            if ffmpeg_path and os.path.exists(ffmpeg_path):
                # Set custom FFmpeg path
                AudioSegment.converter = ffmpeg_path
                AudioSegment.ffmpeg = ffmpeg_path
                ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
                if os.path.exists(ffprobe_path):
                    AudioSegment.ffprobe = ffprobe_path
                logger.info(f"✓ FFmpeg configured: {ffmpeg_path}")
            else:
                # Check if FFmpeg is available in PATH
                if which("ffmpeg"):
                    logger.info("✓ FFmpeg found in PATH")
                else:
                    logger.warning("⚠ FFmpeg not found - some formats may not be supported")
        except Exception as e:
            logger.warning(f"FFmpeg configuration failed: {e}")
    
    def detect_format(self, audio_data: bytes) -> str:
        """
        Detect audio format from binary data.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Detected format string
        """
        if len(audio_data) < 4:
            return 'unknown'
        
        # Check standard signatures
        for signature, format_name in self.format_signatures.items():
            if audio_data.startswith(signature):
                return format_name
        
        # Enhanced detection for containerized formats
        first_32 = audio_data[:32]
        first_100 = audio_data[:100] if len(audio_data) >= 100 else audio_data
        
        # MP4/M4A detection (including fragments)
        if b'ftyp' in first_32 or b'mdat' in first_32 or b'moov' in first_32:
            return 'mp4'
        elif b'moof' in first_100 or b'mfhd' in first_100:
            logger.debug("Detected MP4 fragment")
            return 'mp4_fragment'
        
        # Enhanced WebM detection
        if b'webm' in first_100.lower() or b'opus' in first_100.lower() or b'vorbis' in first_100.lower():
            return 'webm'
        
        # EBML header for WebM (without full signature)
        if audio_data.startswith(b'\x1a\x45') or audio_data[1:5] == b'\x1a\x45\xdf\xa3':
            return 'webm'
        
        # OGG variants
        if b'Ogg' in first_100:
            return 'ogg'
        
        # Check for common patterns in streaming chunks
        if first_32 == b'\x00\x00\x00\x00' * 8:
            logger.debug("Detected possible empty/padding chunk")
            return 'unknown'
        
        logger.debug(f"Unknown format - first 16 bytes: {first_32[:16].hex()}")
        return 'unknown'
    
    def process_audio(self, 
                     audio_data: bytes, 
                     target_sample_rate: Optional[int] = None,
                     enhance: Optional[bool] = None) -> Tuple[np.ndarray, int]:
        """
        Process audio data with comprehensive format support and fallback strategies.
        
        Args:
            audio_data: Raw audio data
            target_sample_rate: Target sample rate (uses default if None)
            enhance: Enable enhancement (uses instance setting if None)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            AudioFormatError: If audio format is unsupported
            AudioConversionError: If conversion fails
        """
        if not audio_data:
            raise AudioFormatError("Empty audio data")
        
        target_sr = target_sample_rate or self.default_sample_rate
        should_enhance = enhance if enhance is not None else self.enable_enhancement
        
        # Detect format
        detected_format = self.detect_format(audio_data)
        logger.debug(f"Detected format: {detected_format} ({len(audio_data)} bytes)")
        
        # Try multiple processing strategies
        strategies = [
            self._process_with_librosa,
            self._process_with_soundfile,
            self._process_with_pydub,
            self._process_raw_audio
        ]
        
        last_error = None
        
        for strategy in strategies:
            try:
                audio, sr = strategy(audio_data, detected_format, target_sr)
                if audio is not None and len(audio) > 0:
                    logger.debug(f"✓ Audio processed with {strategy.__name__}: {len(audio)} samples at {sr}Hz")
                    
                    # Apply enhancement if requested
                    if should_enhance:
                        audio = self._enhance_audio(audio, sr)
                    
                    return audio, sr
            except Exception as e:
                last_error = e
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        raise AudioConversionError(f"All processing strategies failed. Last error: {last_error}")
    
    def _process_with_librosa(self, audio_data: bytes, format_hint: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """Process audio using librosa."""
        if not LIBROSA_AVAILABLE:
            raise AudioConversionError("Librosa not available")
        
        with self._temp_audio_file(audio_data, format_hint) as temp_path:
            # Librosa can handle many formats directly
            audio, sr = librosa.load(temp_path, sr=target_sr, mono=True)
            
            if len(audio) == 0:
                raise AudioConversionError("Librosa returned empty audio")
            
            return audio.astype(np.float32), sr
    
    def _process_with_soundfile(self, audio_data: bytes, format_hint: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """Process audio using soundfile."""
        if not SOUNDFILE_AVAILABLE:
            raise AudioConversionError("Soundfile not available")
        
        try:
            # Try direct processing from BytesIO
            audio, sr = sf.read(io.BytesIO(audio_data))
            
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != target_sr and LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            return audio.astype(np.float32), sr
            
        except Exception as e:
            # Fallback to temporary file
            with self._temp_audio_file(audio_data, format_hint) as temp_path:
                audio, sr = sf.read(temp_path)
                
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                if sr != target_sr and LIBROSA_AVAILABLE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                
                return audio.astype(np.float32), sr
    
    def _process_with_pydub(self, audio_data: bytes, format_hint: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """Process audio using pydub."""
        if not PYDUB_AVAILABLE:
            raise AudioConversionError("Pydub not available")
        
        # Map format hints to pydub formats
        format_map = {
            'webm': 'webm',
            'mp4': 'mp4',
            'mp4_fragment': 'mp4',
            'ogg': 'ogg',
            'mp3': 'mp3',
            'wav': 'wav',
            'unknown': None
        }
        
        pydub_format = format_map.get(format_hint)
        
        try:
            # Load audio with pydub
            if pydub_format:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=pydub_format)
            else:
                # Let pydub auto-detect
                with self._temp_audio_file(audio_data, format_hint) as temp_path:
                    audio_segment = AudioSegment.from_file(temp_path)
            
            # Convert to target sample rate and mono
            audio_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            if audio_segment.sample_width == 2:  # 16-bit
                audio_array /= 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit
                audio_array /= 2147483648.0
            elif audio_segment.sample_width == 1:  # 8-bit
                audio_array = (audio_array - 128) / 128.0
            
            return audio_array, target_sr
            
        except Exception as e:
            raise AudioConversionError(f"Pydub processing failed: {e}")
    
    def _process_raw_audio(self, audio_data: bytes, format_hint: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """Process raw audio data as last resort."""
        logger.debug("Attempting raw audio interpretation")
        
        # Try to interpret as raw PCM data
        try:
            # Try 16-bit signed PCM
            audio_16 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_16) > 100:  # Reasonable length
                return audio_16, target_sr
        except:
            pass
        
        try:
            # Try 32-bit float PCM
            audio_32f = np.frombuffer(audio_data, dtype=np.float32)
            if len(audio_32f) > 100 and np.max(np.abs(audio_32f)) <= 1.0:
                return audio_32f, target_sr
        except:
            pass
        
        raise AudioConversionError("Could not interpret as raw audio data")
    
    def _enhance_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply audio enhancement and preprocessing."""
        if not self.enable_enhancement:
            return audio
        
        try:
            enhanced = audio.copy()
            
            # Normalize amplitude
            max_amp = np.max(np.abs(enhanced))
            if max_amp > 0:
                enhanced = enhanced / max_amp * 0.95  # Leave some headroom
            
            # Simple high-pass filter to remove DC offset and low-frequency noise
            if len(enhanced) > 100:
                # Calculate simple high-pass (remove very low frequencies)
                mean_value = np.mean(enhanced)
                enhanced = enhanced - mean_value
            
            # Simple noise gate (remove very quiet segments)
            noise_threshold = 0.01
            enhanced = np.where(np.abs(enhanced) < noise_threshold, 0, enhanced)
            
            logger.debug(f"Audio enhanced: {len(enhanced)} samples")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio
    
    @contextmanager
    def _temp_audio_file(self, audio_data: bytes, format_hint: str):
        """Create a temporary file with appropriate extension."""
        # Choose appropriate extension
        ext_map = {
            'wav': '.wav',
            'mp3': '.mp3',
            'webm': '.webm',
            'ogg': '.ogg',
            'mp4': '.mp4',
            'mp4_fragment': '.mp4',
            'flac': '.flac',
            'unknown': '.bin'
        }
        
        extension = ext_map.get(format_hint, '.bin')
        
        try:
            with tempfile.NamedTemporaryFile(
                suffix=extension,
                dir=self.temp_dir,
                delete=False
            ) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            yield temp_path
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
    
    def validate_audio(self, audio: np.ndarray, sample_rate: int, 
                      min_duration: float = 0.1, max_duration: float = 300.0) -> bool:
        """
        Validate processed audio data.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            True if audio is valid
        """
        try:
            if audio is None or len(audio) == 0:
                return False
            
            duration = len(audio) / sample_rate
            
            if duration < min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s < {min_duration}s")
                return False
            
            if duration > max_duration:
                logger.warning(f"Audio too long: {duration:.2f}s > {max_duration}s")
                return False
            
            # Check for reasonable amplitude range
            max_amp = np.max(np.abs(audio))
            if max_amp == 0:
                logger.warning("Audio is silent")
                return False
            
            if max_amp > 10.0:  # Unreasonably loud
                logger.warning(f"Audio amplitude too high: {max_amp}")
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(audio).all():
                logger.warning("Audio contains NaN or infinite values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Get information about audio data without full processing.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Dictionary with audio information
        """
        info = {
            'size_bytes': len(audio_data),
            'format': self.detect_format(audio_data),
            'valid': False,
            'error': None
        }
        
        try:
            audio, sr = self.process_audio(audio_data)
            duration = len(audio) / sr
            
            info.update({
                'valid': True,
                'duration_seconds': duration,
                'sample_rate': sr,
                'channels': 1,  # We always convert to mono
                'samples': len(audio),
                'max_amplitude': float(np.max(np.abs(audio))),
                'rms_amplitude': float(np.sqrt(np.mean(audio ** 2)))
            })
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def convert_format(self, 
                      audio_data: bytes, 
                      target_format: str = 'wav',
                      target_sample_rate: Optional[int] = None) -> bytes:
        """
        Convert audio to a specific format.
        
        Args:
            audio_data: Input audio data
            target_format: Target format ('wav', 'mp3', etc.)
            target_sample_rate: Target sample rate
            
        Returns:
            Converted audio data
        """
        if not PYDUB_AVAILABLE:
            raise AudioConversionError("Format conversion requires pydub")
        
        target_sr = target_sample_rate or self.default_sample_rate
        
        # Process to get normalized audio
        audio, sr = self.process_audio(audio_data, target_sr)
        
        # Convert numpy array back to AudioSegment
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # Export to target format
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format=target_format)
        
        return output_buffer.getvalue()


# Convenience functions
def create_audio_processor(**kwargs) -> AudioProcessor:
    """Create an audio processor with default settings."""
    return AudioProcessor(**kwargs)


def process_audio_data(audio_data: bytes, 
                      sample_rate: int = 16000,
                      enhance: bool = True) -> Tuple[np.ndarray, int]:
    """Quick audio processing function."""
    processor = AudioProcessor(
        default_sample_rate=sample_rate,
        enable_enhancement=enhance
    )
    return processor.process_audio(audio_data)


def detect_audio_format(audio_data: bytes) -> str:
    """Quick format detection function."""
    processor = AudioProcessor()
    return processor.detect_format(audio_data)


def validate_audio_data(audio_data: bytes) -> Dict[str, Any]:
    """Quick audio validation function."""
    processor = AudioProcessor()
    return processor.get_audio_info(audio_data)