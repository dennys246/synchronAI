"""
Audio I/O and preprocessing utilities.

Handles:
- Audio extraction from video files via ffmpeg
- Audio normalization to 16kHz mono WAV
- Duration detection via ffprobe
- Chunked audio loading for per-second processing
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for extracted audio files.

    Uses SYNCHRONAI_CACHE_DIR env var if set, otherwise creates
    a .cache directory in the project root.
    """
    if cache_dir := os.environ.get("SYNCHRONAI_CACHE_DIR"):
        path = Path(cache_dir)
    else:
        # Find project root (where pyproject.toml is)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                path = parent / ".cache"
                break
        else:
            # Fallback to temp directory
            path = Path(tempfile.gettempdir()) / "synchronai_cache"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_audio_cache_dir() -> Path:
    """Get cache directory for extracted audio files."""
    path = get_cache_dir() / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_whisper_cache_dir() -> Path:
    """Get cache directory for Whisper model weights."""
    path = get_cache_dir() / "whisper"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class AudioInfo:
    """Basic audio metadata."""

    path: str
    sample_rate: int
    channels: int
    duration: float
    total_seconds: int  # Floor of duration


def _get_ffmpeg_path() -> Optional[str]:
    """Get ffmpeg executable path, checking system PATH and imageio-ffmpeg."""
    # Try system PATH first
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # Try imageio-ffmpeg (pip-installable bundled ffmpeg)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass

    return None


def _get_ffprobe_path() -> Optional[str]:
    """Get ffprobe executable path."""
    # Try system PATH first
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe

    # imageio-ffmpeg doesn't include ffprobe, so no fallback
    return None


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return _get_ffmpeg_path() is not None


def _check_ffprobe() -> bool:
    """Check if ffprobe is available."""
    return _get_ffprobe_path() is not None


def get_audio_duration(input_path: Union[str, Path]) -> float:
    """Get audio/video duration in seconds.

    Uses ffprobe if available, falls back to wave module for WAV files.

    Args:
        input_path: Path to audio or video file

    Returns:
        Duration in seconds
    """
    input_path = Path(input_path)

    ffprobe_path = _get_ffprobe_path()
    if ffprobe_path:
        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"ffprobe failed for {input_path}: {e}")

    # Fallback for WAV files
    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        except Exception as e:
            logger.warning(f"wave module failed for {input_path}: {e}")

    raise RuntimeError(f"Could not determine duration for {input_path}")


def get_audio_info(input_path: Union[str, Path]) -> AudioInfo:
    """Get audio metadata.

    Args:
        input_path: Path to audio file

    Returns:
        AudioInfo with sample_rate, channels, duration
    """
    input_path = Path(input_path)

    ffprobe_path = _get_ffprobe_path()
    if ffprobe_path:
        try:
            # Get all audio stream info
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels:format=duration",
                    "-of",
                    "csv=p=0",
                    str(input_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            # Parse stream info (sample_rate,channels)
            if len(lines) >= 1 and lines[0]:
                parts = lines[0].split(",")
                sample_rate = int(parts[0]) if len(parts) > 0 else 16000
                channels = int(parts[1]) if len(parts) > 1 else 1
            else:
                sample_rate, channels = 16000, 1

            duration = get_audio_duration(input_path)

            return AudioInfo(
                path=str(input_path),
                sample_rate=sample_rate,
                channels=channels,
                duration=duration,
                total_seconds=int(duration),
            )
        except Exception as e:
            logger.warning(f"ffprobe metadata extraction failed: {e}")

    # Fallback for WAV files
    if input_path.suffix.lower() == ".wav":
        with wave.open(str(input_path), "rb") as wf:
            return AudioInfo(
                path=str(input_path),
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                duration=wf.getnframes() / wf.getframerate(),
                total_seconds=int(wf.getnframes() / wf.getframerate()),
            )

    raise RuntimeError(f"Could not get audio info for {input_path}")


def extract_audio(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = True,
) -> Path:
    """Extract and normalize audio from video/audio file.

    Uses ffmpeg to convert to 16kHz mono WAV (Whisper's expected format).

    Args:
        input_path: Path to input video or audio file
        output_path: Path for output WAV file (temp file if None)
        sample_rate: Target sample rate (16000 for Whisper)
        mono: Convert to mono
        overwrite: Overwrite existing output file

    Returns:
        Path to extracted WAV file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    # Determine output path
    if output_path is None:
        # Create temp file in project cache directory (not system /tmp)
        cache_dir = get_audio_cache_dir()
        fd, output_path = tempfile.mkstemp(suffix=".wav", dir=cache_dir)
        os.close(fd)
        output_path = Path(output_path)
    else:
        output_path = Path(output_path)

    # Skip if already exists and not overwriting
    if output_path.exists() and not overwrite:
        logger.info(f"Audio already exists: {output_path}")
        return output_path

    # Build ffmpeg command
    ffmpeg_path = _get_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",  # Overwrite or fail if exists
        "-i",
        str(input_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        str(sample_rate),  # Sample rate
    ]

    if mono:
        cmd.extend(["-ac", "1"])  # Mono

    cmd.append(str(output_path))

    logger.info(f"Extracting audio: {input_path.name} -> {output_path.name}")

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

    logger.info(f"Audio extracted: {output_path} ({get_audio_duration(output_path):.1f}s)")
    return output_path


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
) -> np.ndarray:
    """Load entire audio file as numpy array.

    Args:
        audio_path: Path to audio file
        sample_rate: Expected sample rate

    Returns:
        Audio samples as float32 array normalized to [-1, 1]
    """
    audio_path = Path(audio_path)

    # Try soundfile first (handles many formats)
    try:
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != sample_rate:
            logger.warning(f"Sample rate mismatch: {sr} != {sample_rate}")
        return audio
    except ImportError:
        pass

    # Fallback to wave module for WAV files
    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            return audio.astype(np.float32) / 32768.0

    raise RuntimeError(f"Could not load audio: {audio_path}")


def load_audio_chunk(
    audio_path: Union[str, Path],
    start_sec: float,
    duration: float = 1.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Load a specific chunk of audio.

    Args:
        audio_path: Path to audio file
        start_sec: Start time in seconds
        duration: Duration in seconds
        sample_rate: Expected sample rate

    Returns:
        Audio chunk as float32 array, shape (n_samples,)
    """
    audio_path = Path(audio_path)
    start_sample = int(start_sec * sample_rate)
    n_samples = int(duration * sample_rate)

    # Try soundfile first
    try:
        import soundfile as sf

        audio, sr = sf.read(
            str(audio_path),
            start=start_sample,
            frames=n_samples,
            dtype="float32",
        )
        if sr != sample_rate:
            logger.warning(f"Sample rate mismatch: {sr} != {sample_rate}")
        # Ensure we have exactly n_samples
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        return audio[:n_samples]
    except ImportError:
        pass

    # Fallback: load entire file and slice
    audio = load_audio(audio_path, sample_rate)
    end_sample = start_sample + n_samples

    if start_sample >= len(audio):
        return np.zeros(n_samples, dtype=np.float32)

    chunk = audio[start_sample:end_sample]
    if len(chunk) < n_samples:
        chunk = np.pad(chunk, (0, n_samples - len(chunk)))

    return chunk


def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of audio.

    Args:
        audio: Audio samples

    Returns:
        RMS energy value
    """
    return float(np.sqrt(np.mean(audio**2)))


def compute_energy_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """Compute energy in decibels.

    Args:
        audio: Audio samples
        ref: Reference value for dB calculation

    Returns:
        Energy in dB (clamped to -80 dB minimum)
    """
    rms = compute_rms_energy(audio)
    if rms < 1e-10:
        return -80.0
    return float(20 * np.log10(rms / ref))


class AudioChunkIterator:
    """Iterator for loading audio in 1-second chunks.

    Useful for processing long audio files without loading entire file.
    """

    def __init__(
        self,
        audio_path: Union[str, Path],
        chunk_duration: float = 1.0,
        sample_rate: int = 16000,
    ):
        self.audio_path = Path(audio_path)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.info = get_audio_info(audio_path)
        self.total_chunks = self.info.total_seconds
        self._current = 0

    def __len__(self) -> int:
        return self.total_chunks

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self._current >= self.total_chunks:
            raise StopIteration

        chunk = load_audio_chunk(
            self.audio_path,
            start_sec=float(self._current),
            duration=self.chunk_duration,
            sample_rate=self.sample_rate,
        )
        second = self._current
        self._current += 1
        return second, chunk

    def __getitem__(self, second: int) -> np.ndarray:
        if second < 0 or second >= self.total_chunks:
            raise IndexError(f"Second {second} out of range [0, {self.total_chunks})")
        return load_audio_chunk(
            self.audio_path,
            start_sec=float(second),
            duration=self.chunk_duration,
            sample_rate=self.sample_rate,
        )
