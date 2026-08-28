import base64
import io
import tempfile
import os
from typing import Optional, Tuple
import numpy as np
from pydub import AudioSegment


def decode_base64_audio(base64_string: str) -> Tuple[np.ndarray, int]:
    """Decode base64 audio string to numpy array and sample rate."""
    audio_bytes = base64.b64decode(base64_string)
    audio_io = io.BytesIO(audio_bytes)
    
    # Use pydub to load (handles wav, mp3, m4a, etc.)
    audio = AudioSegment.from_file(audio_io)
    
    # Convert to mono if stereo
    if audio.channels == 2:
        audio = audio.set_channels(1)
    
    # Convert to 16kHz if needed
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    
    # Get samples as numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1, 1]
    if audio.sample_width == 2:  # 16-bit
        samples = samples / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples / 2147483648.0
    
    return samples, audio.frame_rate


def save_audio_temp(audio_data: np.ndarray, sample_rate: int, suffix: str = ".wav") -> str:
    """Save audio data to temporary file and return path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    # Convert back to int16 for saving
    int_data = (audio_data * 32767).astype(np.int16)
    
    # Create AudioSegment and export
    audio = AudioSegment(
        int_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )
    audio.export(temp_file.name, format="wav")
    temp_file.close()
    return temp_file.name


def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return numpy array and sample rate."""
    audio = AudioSegment.from_file(file_path)
    
    # Convert to mono if stereo
    if audio.channels == 2:
        audio = audio.set_channels(1)
    
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize
    if audio.sample_width == 2:
        samples = samples / 32768.0
    elif audio.sample_width == 4:
        samples = samples / 2147483648.0
    
    return samples, audio.frame_rate


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate using simple interpolation."""
    
    # Always ensure float32
    audio_data = np.asarray(audio_data, dtype=np.float32)

    if orig_sr == target_sr:
        return audio_data

    # Simple linear interpolation resampling
    ratio = target_sr / orig_sr
    new_length = int(len(audio_data) * ratio)

    indices = np.linspace(
        0,
        len(audio_data) - 1,
        new_length,
        dtype=np.float32
    )

    resampled = np.interp(
        indices,
        np.arange(len(audio_data), dtype=np.float32),
        audio_data
    )

    # IMPORTANT: np.interp returns float64 by default,
    # so convert it back to float32
    return resampled.astype(np.float32)


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    
    audio_data = np.asarray(audio_data, dtype=np.float32)
    max_val = np.max(np.abs(audio_data))

    if max_val > 0:
        return (audio_data / max_val).astype(np.float32)

    return audio_data.astype(np.float32)


def get_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
    """Get audio duration in seconds."""
    return len(audio_data) / sample_rate


def is_audio_valid(audio_data: np.ndarray, sample_rate: int, min_duration: float = 0.5, max_duration: float = 300) -> Tuple[bool, str]:
    """Validate audio meets requirements."""
    duration = get_audio_duration(audio_data, sample_rate)
    
    if duration < min_duration:
        return False, f"Audio too short: {duration:.1f}s (minimum {min_duration}s)"
    if duration > max_duration:
        return False, f"Audio too long: {duration:.1f}s (maximum {max_duration}s)"
    if np.max(np.abs(audio_data)) < 0.001:
        return False, "Audio appears to be silent"
    
    return True, "OK"