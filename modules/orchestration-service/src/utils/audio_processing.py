"""
Audio Processing Utilities

Provides audio validation and processing utilities.
"""

from pathlib import Path
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()


class AudioProcessor:
    """
    Audio processing utilities
    """

    def __init__(self) -> None:
        self.supported_formats = {
            "wav": {"mime": "audio/wav", "extension": ".wav"},
            "mp3": {"mime": "audio/mpeg", "extension": ".mp3"},
            "webm": {"mime": "audio/webm", "extension": ".webm"},
            "ogg": {"mime": "audio/ogg", "extension": ".ogg"},
            "mp4": {"mime": "audio/mp4", "extension": ".mp4"},
            "m4a": {"mime": "audio/mp4", "extension": ".m4a"},
            "flac": {"mime": "audio/flac", "extension": ".flac"},
        }

    def validate_audio_file(self, file_data: bytes, filename: str) -> dict[str, Any]:
        """
        Validate audio file

        Args:
            file_data: Audio file data
            filename: Original filename

        Returns:
            Validation result dictionary
        """
        try:
            # Check file size (max 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if len(file_data) > max_size:
                return {
                    "valid": False,
                    "error": f"File too large: {len(file_data)} bytes (max {max_size})",
                }

            # Check minimum size (1KB)
            min_size = 1024  # 1KB
            if len(file_data) < min_size:
                return {
                    "valid": False,
                    "error": f"File too small: {len(file_data)} bytes (min {min_size})",
                }

            # Detect format
            format_info = self.detect_audio_format(file_data, filename)
            if not format_info["detected"]:
                return {
                    "valid": False,
                    "error": f"Unsupported audio format: {format_info.get('error', 'Unknown format')}",
                }

            # Basic header validation
            header_valid = self.validate_audio_header(file_data, format_info["format"])
            if not header_valid:
                return {
                    "valid": False,
                    "error": f"Invalid {format_info['format']} header",
                }

            return {
                "valid": True,
                "format": format_info["format"],
                "mime_type": format_info["mime_type"],
                "size": len(file_data),
                "estimated_duration": self.estimate_duration(file_data, format_info["format"]),
            }

        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return {"valid": False, "error": f"Validation failed: {e!s}"}

    def detect_audio_format(self, file_data: bytes, filename: str) -> dict[str, Any]:
        """
        Detect audio format from file data and filename

        Args:
            file_data: Audio file data
            filename: Original filename

        Returns:
            Format detection result
        """
        try:
            # Check file magic numbers
            if file_data.startswith(b"RIFF") and b"WAVE" in file_data[:12]:
                return {
                    "detected": True,
                    "format": "wav",
                    "mime_type": "audio/wav",
                    "method": "magic_number",
                }
            elif file_data.startswith(b"ID3") or file_data.startswith(b"\xff\xfb"):
                return {
                    "detected": True,
                    "format": "mp3",
                    "mime_type": "audio/mpeg",
                    "method": "magic_number",
                }
            elif file_data.startswith(b"\x1a\x45\xdf\xa3"):
                return {
                    "detected": True,
                    "format": "webm",
                    "mime_type": "audio/webm",
                    "method": "magic_number",
                }
            elif file_data.startswith(b"OggS"):
                return {
                    "detected": True,
                    "format": "ogg",
                    "mime_type": "audio/ogg",
                    "method": "magic_number",
                }
            elif file_data.startswith(b"\x00\x00\x00\x20\x66\x74\x79\x70"):
                return {
                    "detected": True,
                    "format": "mp4",
                    "mime_type": "audio/mp4",
                    "method": "magic_number",
                }
            elif file_data.startswith(b"fLaC"):
                return {
                    "detected": True,
                    "format": "flac",
                    "mime_type": "audio/flac",
                    "method": "magic_number",
                }

            # Fallback to filename extension
            extension = Path(filename).suffix.lower()
            if extension in [".wav", ".mp3", ".webm", ".ogg", ".mp4", ".m4a", ".flac"]:
                format_name = extension[1:]  # Remove dot
                if format_name == "m4a":
                    format_name = "mp4"

                return {
                    "detected": True,
                    "format": format_name,
                    "mime_type": self.supported_formats[format_name]["mime"],
                    "method": "filename_extension",
                }

            return {"detected": False, "error": "Unknown format"}

        except Exception as e:
            logger.error(f"Format detection error: {e}")
            return {"detected": False, "error": str(e)}

    def validate_audio_header(self, file_data: bytes, format_type: str) -> bool:
        """
        Validate audio file header

        Args:
            file_data: Audio file data
            format_type: Detected format type

        Returns:
            True if header is valid, False otherwise
        """
        try:
            if format_type == "wav":
                # Check WAV header structure
                if len(file_data) < 44:
                    return False

                # Check RIFF header
                if file_data[:4] != b"RIFF":
                    return False

                # Check WAVE format
                if file_data[8:12] != b"WAVE":
                    return False

                # Check fmt chunk
                return file_data[12:16] == b"fmt "

            elif format_type == "mp3":
                # Check MP3 header (ID3 or sync frame)
                if len(file_data) < 4:
                    return False

                # ID3 header
                if file_data[:3] == b"ID3":
                    return True

                # MP3 sync frame
                return bool(file_data[0] == 255 and file_data[1] & 224 == 224)

            elif format_type == "webm":
                # Check WebM/Matroska header
                if len(file_data) < 4:
                    return False

                return file_data.startswith(b"\x1a\x45\xdf\xa3")

            elif format_type == "ogg":
                # Check Ogg header
                if len(file_data) < 4:
                    return False

                return file_data.startswith(b"OggS")

            elif format_type == "mp4":
                # Check MP4 header
                if len(file_data) < 8:
                    return False

                # Check ftyp box
                return file_data[4:8] == b"ftyp"

            elif format_type == "flac":
                # Check FLAC header
                if len(file_data) < 4:
                    return False

                return file_data.startswith(b"fLaC")

            # Unknown format, assume valid
            return True

        except Exception as e:
            logger.error(f"Header validation error: {e}")
            return False

    def estimate_duration(self, file_data: bytes, format_type: str) -> float | None:
        """
        Estimate audio duration (rough calculation)

        Args:
            file_data: Audio file data
            format_type: Audio format

        Returns:
            Estimated duration in seconds, or None if cannot estimate
        """
        try:
            if format_type == "wav":
                # For WAV, we can read the header info
                if len(file_data) >= 44:
                    # Extract sample rate (bytes 24-27)
                    sample_rate = int.from_bytes(file_data[24:28], byteorder="little")

                    # Extract bits per sample (bytes 34-35)
                    bits_per_sample = int.from_bytes(file_data[34:36], byteorder="little")

                    # Extract channels (bytes 22-23)
                    channels = int.from_bytes(file_data[22:24], byteorder="little")

                    # Calculate duration
                    if sample_rate > 0 and bits_per_sample > 0 and channels > 0:
                        bytes_per_second = sample_rate * (bits_per_sample / 8) * channels
                        data_size = len(file_data) - 44  # Subtract header size
                        duration = data_size / bytes_per_second
                        return duration

            # For other formats, use rough estimation based on file size
            # This is very approximate and depends on bitrate
            file_size_mb = len(file_data) / (1024 * 1024)

            # Rough estimates based on typical bitrates
            estimates = {
                "mp3": file_size_mb * 8,  # ~128 kbps
                "webm": file_size_mb * 10,  # ~96 kbps
                "ogg": file_size_mb * 8,  # ~128 kbps
                "mp4": file_size_mb * 6,  # ~160 kbps
                "flac": file_size_mb * 1.5,  # ~1000 kbps
            }

            return estimates.get(format_type, file_size_mb * 8)

        except Exception as e:
            logger.error(f"Duration estimation error: {e}")
            return None

    def get_audio_info(self, file_data: bytes, filename: str) -> dict[str, Any]:
        """
        Get comprehensive audio file information

        Args:
            file_data: Audio file data
            filename: Original filename

        Returns:
            Audio file information dictionary
        """
        try:
            validation_result = self.validate_audio_file(file_data, filename)

            if not validation_result["valid"]:
                return validation_result

            info = {
                "filename": filename,
                "format": validation_result["format"],
                "mime_type": validation_result["mime_type"],
                "size_bytes": len(file_data),
                "size_mb": round(len(file_data) / (1024 * 1024), 2),
                "estimated_duration": validation_result.get("estimated_duration"),
                "supported_operations": ["transcribe", "analyze"],
            }

            # Add format-specific info
            if validation_result["format"] == "wav":
                info["supported_operations"].extend(["stream", "realtime"])

            return {"valid": True, "info": info}

        except Exception as e:
            logger.error(f"Audio info extraction error: {e}")
            return {"valid": False, "error": str(e)}

    def prepare_audio_for_processing(self, file_data: bytes, target_format: str = "wav") -> bytes:
        """
        Prepare audio data for processing (placeholder implementation)

        Args:
            file_data: Original audio data
            target_format: Target format for processing

        Returns:
            Processed audio data
        """
        # This would normally use a library like pydub or ffmpeg
        # For now, return the original data
        logger.info(f"Preparing audio for processing (target: {target_format})")
        return file_data
