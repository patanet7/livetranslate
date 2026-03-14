"""
Pytest configuration and fixtures for whisper-service tests.

This module provides:
- Audio fixture generation (real 16kHz mono float32 audio files)
- Pytest fixtures for loading audio
- Pytest markers (openvino, gpu, slow)
- Graceful handling of missing dependencies
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pytest
import soundfile as sf

# Test constants
SAMPLE_RATE = 16000
DURATION_SHORT = 1.0  # 1 second
DURATION_MEDIUM = 3.0  # 3 seconds
DURATION_LONG = 5.0  # 5 seconds
AUDIO_DTYPE = np.float32

# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pytest Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "openvino: requires OpenVINO (skip if not installed)")
    config.addinivalue_line("markers", "gpu: requires GPU (skip if not available)")
    config.addinivalue_line("markers", "slow: slow running tests (skip with -m 'not slow')")


# ============================================================================
# Audio Generation Helpers
# ============================================================================


def generate_sine_wave(
    frequency: float, duration: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.5
) -> np.ndarray:
    """
    Generate a sine wave audio signal.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Audio samples as float32 numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(AUDIO_DTYPE)


def generate_speech_like_audio(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate speech-like audio with multiple frequency components.

    This simulates formants and prosody patterns found in speech.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as float32 numpy array
    """
    # Generate formants (typical speech frequencies)
    formant1 = generate_sine_wave(500, duration, sample_rate, 0.3)  # F1
    formant2 = generate_sine_wave(1500, duration, sample_rate, 0.2)  # F2
    formant3 = generate_sine_wave(2500, duration, sample_rate, 0.1)  # F3

    # Add some amplitude modulation (prosody)
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation

    # Combine
    audio = (formant1 + formant2 + formant3) * modulation

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7

    return audio.astype(AUDIO_DTYPE)


def generate_silence(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate silence (zeros).

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as float32 numpy array
    """
    num_samples = int(sample_rate * duration)
    return np.zeros(num_samples, dtype=AUDIO_DTYPE)


def generate_white_noise(
    duration: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.1
) -> np.ndarray:
    """
    Generate white noise.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Audio samples as float32 numpy array
    """
    num_samples = int(sample_rate * duration)
    noise = np.random.randn(num_samples) * amplitude
    return noise.astype(AUDIO_DTYPE)


def add_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """
    Add white noise to audio at specified SNR.

    Args:
        audio: Input audio samples
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy audio samples
    """
    signal_power = np.mean(audio**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return (audio + noise).astype(AUDIO_DTYPE)


# ============================================================================
# Fixture Generation (Run once per test session)
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def generate_audio_fixtures():
    """
    Generate all audio fixtures once per test session.

    Creates:
    - hello_world.wav: Speech-like audio (3 seconds)
    - silence.wav: Pure silence (2 seconds)
    - noisy.wav: Noisy speech-like audio (3 seconds, SNR=10dB)
    - short_speech.wav: Short speech-like audio (1 second)
    - long_speech.wav: Long speech-like audio (5 seconds)
    - white_noise.wav: White noise (2 seconds)
    """
    fixtures = {
        "hello_world.wav": generate_speech_like_audio(DURATION_MEDIUM),
        "silence.wav": generate_silence(2.0),
        "noisy.wav": add_noise(generate_speech_like_audio(DURATION_MEDIUM), snr_db=10.0),
        "short_speech.wav": generate_speech_like_audio(DURATION_SHORT),
        "long_speech.wav": generate_speech_like_audio(DURATION_LONG),
        "white_noise.wav": generate_white_noise(2.0),
    }

    for filename, audio in fixtures.items():
        filepath = FIXTURES_DIR / filename
        sf.write(filepath, audio, SAMPLE_RATE)
        print(
            f"Generated fixture: {filepath} ({len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s)"
        )

    yield

    # Cleanup is optional - fixtures can be reused across test runs
    # Uncomment to delete after tests:
    # for filename in fixtures.keys():
    #     (FIXTURES_DIR / filename).unlink(missing_ok=True)


# ============================================================================
# Audio Loading Fixtures
# ============================================================================


@pytest.fixture
def hello_world_audio() -> tuple[np.ndarray, int]:
    """
    Load hello_world.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "hello_world.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def silence_audio() -> tuple[np.ndarray, int]:
    """
    Load silence.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "silence.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def noisy_audio() -> tuple[np.ndarray, int]:
    """
    Load noisy.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "noisy.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def short_speech_audio() -> tuple[np.ndarray, int]:
    """
    Load short_speech.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "short_speech.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def long_speech_audio() -> tuple[np.ndarray, int]:
    """
    Load long_speech.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "long_speech.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def white_noise_audio() -> tuple[np.ndarray, int]:
    """
    Load white_noise.wav fixture.

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    filepath = FIXTURES_DIR / "white_noise.wav"
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def all_audio_fixtures() -> dict[str, tuple[np.ndarray, int]]:
    """
    Load all audio fixtures as a dictionary.

    Returns:
        Dict mapping fixture name to (audio_samples, sample_rate)
    """
    fixtures = {}
    for filename in FIXTURES_DIR.glob("*.wav"):
        audio, sr = sf.read(filename, dtype=AUDIO_DTYPE)
        fixtures[filename.stem] = (audio, sr)
    return fixtures


# ============================================================================
# Hardware Detection Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def has_openvino() -> bool:
    """
    Check if OpenVINO is available.

    Returns:
        True if OpenVINO can be imported, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("openvino") is not None


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """
    Check if GPU (CUDA) is available.

    Returns:
        True if CUDA GPU is available, False otherwise
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def device_type(has_openvino: bool, has_gpu: bool) -> str:
    """
    Determine best available device type.

    Args:
        has_openvino: Whether OpenVINO is available
        has_gpu: Whether GPU is available

    Returns:
        Device type string: "openvino", "cuda", or "cpu"
    """
    if has_openvino:
        return "openvino"
    elif has_gpu:
        return "cuda"
    else:
        return "cpu"


# ============================================================================
# Skip Conditions (Decorators for use in tests)
# ============================================================================


def _check_openvino():
    """Check if OpenVINO is available."""
    import importlib.util

    return importlib.util.find_spec("openvino") is not None


def _check_gpu():
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


skip_if_no_openvino = pytest.mark.skipif(not _check_openvino(), reason="OpenVINO not available")

skip_if_no_gpu = pytest.mark.skipif(not _check_gpu(), reason="GPU not available")


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """
    Create temporary directory for audio file tests.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path to temporary audio directory
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    return audio_dir


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_whisper_model():
    """
    Create a mock Whisper model for testing.

    This uses unittest.mock which is part of Python standard library.
    For more advanced mocking, install pytest-mock: pip install pytest-mock

    Returns:
        Mock Whisper model with transcribe method
    """
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Hello world",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "tokens": [1, 2, 3],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "world",
                "tokens": [4, 5, 6],
            },
        ],
        "language": "en",
    }
    return mock_model


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def default_whisper_config() -> dict:
    """
    Provide default Whisper configuration for tests.

    Returns:
        Configuration dictionary
    """
    return {
        "model_name": "base",
        "device": "cpu",
        "compute_type": "float32",
        "language": None,
        "task": "transcribe",
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "vad_filter": True,
        "sample_rate": SAMPLE_RATE,
    }


# ============================================================================
# Session Info
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def print_test_environment():
    """Print test environment information at start of session."""
    import platform

    print("\n" + "=" * 80)
    print("WHISPER-SERVICE TEST ENVIRONMENT")
    print("=" * 80)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Fixtures directory: {FIXTURES_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Audio dtype: {AUDIO_DTYPE}")

    # Check dependencies
    deps = {
        "numpy": None,
        "soundfile": None,
        "torch": None,
        "openvino": None,
    }

    for dep in deps:
        try:
            mod = __import__(dep)
            deps[dep] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[dep] = "NOT INSTALLED"

    print("\nDependencies:")
    for dep, version in deps.items():
        print(f"  {dep}: {version}")

    print("=" * 80 + "\n")

    yield


# ============================================================================
# REAL Speech Audio Fixtures
# ============================================================================


@pytest.fixture
def jfk_audio() -> tuple[np.ndarray, int]:
    """
    Load JFK speech audio (REAL English speech!)

    Content: "And so, my fellow Americans, ask not what your country can do for you;
              ask what you can do for your country."

    Duration: 11 seconds
    Sample Rate: 16kHz (Whisper native)
    Language: English
    Source: JFK inaugural address, January 20, 1961
    """
    filepath = FIXTURES_DIR / "jfk.wav"
    if not filepath.exists():
        pytest.skip(f"JFK audio file not found: {filepath}")
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    return audio, sr


@pytest.fixture
def chinese_audio_1() -> tuple[np.ndarray, int]:
    """
    Load Chinese audio sample 1 (REAL Mandarin speech!)

    Duration: ~20 seconds
    Sample Rate: 8kHz (auto-resampled to 16kHz)
    Language: Mandarin Chinese
    """
    filepath = FIXTURES_DIR / "OSR_cn_000_0072_8k.wav"
    if not filepath.exists():
        pytest.skip(f"Chinese audio file not found: {filepath}")
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    # Resample to 16kHz for Whisper
    if sr != SAMPLE_RATE:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        except ImportError:
            pytest.skip("librosa required for resampling")
    return audio, sr


@pytest.fixture
def chinese_audio_2() -> tuple[np.ndarray, int]:
    """Load Chinese audio sample 2"""
    filepath = FIXTURES_DIR / "OSR_cn_000_0073_8k.wav"
    if not filepath.exists():
        pytest.skip(f"Chinese audio file not found: {filepath}")
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    return audio, sr


@pytest.fixture
def chinese_audio_3() -> tuple[np.ndarray, int]:
    """Load Chinese audio sample 3"""
    filepath = FIXTURES_DIR / "OSR_cn_000_0074_8k.wav"
    if not filepath.exists():
        pytest.skip(f"Chinese audio file not found: {filepath}")
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    return audio, sr


@pytest.fixture
def chinese_audio_4() -> tuple[np.ndarray, int]:
    """Load Chinese audio sample 4"""
    filepath = FIXTURES_DIR / "OSR_cn_000_0075_8k.wav"
    if not filepath.exists():
        pytest.skip(f"Chinese audio file not found: {filepath}")
    audio, sr = sf.read(filepath, dtype=AUDIO_DTYPE)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    return audio, sr
