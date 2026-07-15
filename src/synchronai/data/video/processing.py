"""
Video I/O and preprocessing utilities.

Handles:
- Video reading with multiple backend support (decord, pyav, opencv)
- Frame sampling by timestamp for variable FPS videos
- YOLO-style preprocessing (letterbox, BGR->RGB, NCHW, 0-1 scaling)
- DINOv2-style preprocessing (resize, BGR->RGB, NCHW, ImageNet normalization)
- Person crop extraction from bounding boxes
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Basic video metadata."""

    path: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


class VideoReader(Protocol):
    """Protocol for video reader backends."""

    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Get frame at specified timestamp in seconds."""
        ...

    def get_frame_at_index(self, index: int) -> np.ndarray:
        """Get frame at specified index."""
        ...

    def close(self) -> None:
        """Release video resources."""
        ...


class OpenCVReader:
    """OpenCV-based video reader (fallback)."""

    def __init__(self, video_path: str):
        import cv2

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._path = video_path

    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Get frame at specified timestamp."""
        import cv2

        frame_idx = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at timestamp {timestamp}")
        return frame

    def get_frame_at_index(self, index: int) -> np.ndarray:
        """Get frame at specified index."""
        import cv2

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {index}")
        return frame

    def close(self) -> None:
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()


class DecordReader:
    """Decord-based video reader (faster for sequential access)."""

    def __init__(self, video_path: str):
        from decord import VideoReader as DVideoReader
        from decord import cpu

        self.vr = DVideoReader(video_path, ctx=cpu(0))
        self.fps = self.vr.get_avg_fps()
        self.frame_count = len(self.vr)
        self._path = video_path

    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Get frame at specified timestamp."""
        frame_idx = min(int(timestamp * self.fps), self.frame_count - 1)
        frame = self.vr[frame_idx].asnumpy()
        # Decord returns RGB, convert to BGR for consistency with OpenCV
        return frame[:, :, ::-1]

    def get_frame_at_index(self, index: int) -> np.ndarray:
        """Get frame at specified index."""
        frame = self.vr[index].asnumpy()
        return frame[:, :, ::-1]

    def close(self) -> None:
        """Release video resources."""
        del self.vr


class PyAVReader:
    """PyAV-based video reader (accurate timestamp seeking for VFR)."""

    def __init__(self, video_path: str):
        import av

        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate or self.stream.guessed_rate or 30.0)
        self.frame_count = self.stream.frames or int(self.stream.duration * self.fps / self.stream.time_base.denominator)
        self._path = video_path
        self._frame_cache: dict[int, np.ndarray] = {}

    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Get frame at specified timestamp using PTS-based seeking."""
        import av

        # Seek to timestamp
        time_base = self.stream.time_base
        pts = int(timestamp / time_base)

        self.container.seek(pts, stream=self.stream)

        for frame in self.container.decode(video=0):
            return frame.to_ndarray(format="bgr24")

        raise RuntimeError(f"Failed to read frame at timestamp {timestamp}")

    def get_frame_at_index(self, index: int) -> np.ndarray:
        """Get frame at specified index."""
        timestamp = index / self.fps
        return self.get_frame_at_timestamp(timestamp)

    def close(self) -> None:
        """Release video resources."""
        self.container.close()


class VideoReaderPool:
    """Pool of video readers with LRU eviction for repeated access."""

    def __init__(self, max_readers: int = 8, backend: str = "auto"):
        self._readers: OrderedDict[str, VideoReader] = OrderedDict()
        self._max_readers = max_readers
        self._backend = backend

    def get_reader(self, video_path: str) -> VideoReader:
        """Get or create a video reader, evicting LRU if at capacity."""
        video_path = str(video_path)

        if video_path in self._readers:
            # Move to end (most recently used)
            self._readers.move_to_end(video_path)
            return self._readers[video_path]

        # Create new reader
        reader = create_video_reader(video_path, self._backend)

        # Evict LRU if at capacity
        while len(self._readers) >= self._max_readers:
            _, old_reader = self._readers.popitem(last=False)
            old_reader.close()

        self._readers[video_path] = reader
        return reader

    def close_all(self) -> None:
        """Close all readers."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def create_video_reader(video_path: str, backend: str = "auto") -> VideoReader:
    """Create a video reader with the specified backend.

    Args:
        video_path: Path to video file
        backend: Backend to use ("decord", "pyav", "opencv", or "auto")

    Returns:
        VideoReader instance
    """
    backends_to_try = []

    if backend == "auto":
        backends_to_try = ["decord", "opencv"]  # pyav can be slow, use only when needed
    else:
        backends_to_try = [backend]

    last_error = None

    for b in backends_to_try:
        try:
            if b == "decord":
                return DecordReader(video_path)
            elif b == "pyav":
                return PyAVReader(video_path)
            elif b == "opencv":
                return OpenCVReader(video_path)
        except ImportError as e:
            logger.debug(f"Backend {b} not available: {e}")
            last_error = e
        except Exception as e:
            logger.debug(f"Backend {b} failed to open {video_path}: {e}")
            last_error = e

    raise RuntimeError(f"All video backends failed for {video_path}: {last_error}")


def load_video_info(video_path: Union[str, Path]) -> VideoInfo:
    """Load basic video metadata.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with fps, frame_count, duration, dimensions
    """
    import cv2

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoInfo(
            path=video_path,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def sample_window_timestamps(
    second: float,
    window_seconds: float,
    sample_fps: float,
) -> list[float]:
    """Generate uniform timestamps for frame sampling.

    Args:
        second: Start time of the window in seconds (supports fractional offsets)
        window_seconds: Duration of window in seconds
        sample_fps: Target frames per second

    Returns:
        List of timestamps uniformly spaced in [second, second + window_seconds)
    """
    n_frames = int(sample_fps * window_seconds)
    return [second + i / sample_fps for i in range(n_frames)]


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Resize image with letterboxing to maintain aspect ratio.

    Args:
        image: Input image (H, W, C) in BGR format
        target_size: Target size (width, height)
        color: Padding color (BGR)

    Returns:
        Tuple of (letterboxed image, scale ratios, padding offsets)
    """
    import cv2

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Create padded image
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return padded, (scale, scale), (pad_w, pad_h)


def preprocess_yolo_frame(
    frame: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """Apply YOLO-style preprocessing to a frame.

    Preprocessing steps:
    1. Letterbox resize to target_size x target_size
    2. BGR to RGB conversion
    3. HWC to CHW transpose
    4. Normalize to [0, 1] range

    Args:
        frame: Input frame (H, W, C) in BGR format
        target_size: Target size for letterboxing

    Returns:
        Preprocessed frame (C, H, W) in RGB format, float32, [0, 1]
    """
    # Letterbox resize
    letterboxed, _, _ = letterbox(frame, (target_size, target_size))

    # BGR to RGB
    rgb = letterboxed[:, :, ::-1]

    # HWC to CHW
    chw = rgb.transpose(2, 0, 1)

    # Normalize to [0, 1]
    normalized = chw.astype(np.float32) / 255.0

    return normalized


def read_window_frames(
    video_path: str,
    second: float,
    sample_fps: float,
    window_seconds: float = 1.0,
    target_size: int = 640,
    reader: Optional[VideoReader] = None,
) -> np.ndarray:
    """Read and preprocess frames for a window.

    Args:
        video_path: Path to video file
        second: Start second of the window
        sample_fps: Target frames per second
        window_seconds: Duration of window in seconds
        target_size: Target size for YOLO preprocessing
        reader: Optional pre-created video reader

    Returns:
        Preprocessed frames array (n_frames, C, H, W) float32 [0, 1]
    """
    # Generate timestamps
    timestamps = sample_window_timestamps(second, window_seconds, sample_fps)

    # Create reader if not provided
    own_reader = reader is None
    if own_reader:
        reader = create_video_reader(video_path)

    try:
        frames = []
        for ts in timestamps:
            try:
                frame = reader.get_frame_at_timestamp(ts)
                processed = preprocess_yolo_frame(frame, target_size)
                frames.append(processed)
            except Exception as e:
                logger.warning(f"Failed to read frame at {ts}s from {video_path}: {e}")
                # Use black frame as fallback
                frames.append(np.zeros((3, target_size, target_size), dtype=np.float32))

        return np.stack(frames, axis=0)

    finally:
        if own_reader:
            reader.close()


def stack_window_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Stack preprocessed frames into a single array.

    Args:
        frames: List of preprocessed frames (C, H, W)

    Returns:
        Stacked frames (n_frames, C, H, W)
    """
    return np.stack(frames, axis=0)


# =============================================================================
# DINOv2 Preprocessing (ImageNet normalization, 224x224)
# =============================================================================

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_dinov2_frame(
    frame: np.ndarray,
    target_size: int = 224,
) -> np.ndarray:
    """Apply DINOv2-style preprocessing to a frame.

    Preprocessing steps:
    1. Resize to target_size x target_size (center crop or stretch)
    2. BGR to RGB conversion
    3. HWC to CHW transpose
    4. Normalize to [0, 1] then apply ImageNet mean/std

    Args:
        frame: Input frame (H, W, C) in BGR format
        target_size: Target size (default 224 for DINOv2)

    Returns:
        Preprocessed frame (C, H, W) in RGB format, float32, ImageNet-normalized
    """
    import cv2

    # Resize (simple resize, not letterbox — DINOv2 expects square input)
    resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # BGR to RGB
    rgb = resized[:, :, ::-1].copy()

    # HWC to CHW, normalize to [0, 1]
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    # ImageNet normalization: (x - mean) / std per channel
    for c in range(3):
        chw[c] = (chw[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return chw


def crop_and_preprocess_person(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    target_size: int = 224,
    padding: float = 0.1,
) -> np.ndarray:
    """Crop a person from a frame using bounding box and preprocess for DINOv2.

    Args:
        frame: Input frame (H, W, C) in BGR format
        bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates
        target_size: Target size for DINOv2 preprocessing
        padding: Fractional padding around bbox (0.1 = 10% on each side)

    Returns:
        Preprocessed crop (C, H, W) in RGB format, float32, ImageNet-normalized
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)

    # Crop
    crop = frame[int(y1):int(y2), int(x1):int(x2)]

    # Handle degenerate bboxes
    if crop.size == 0:
        crop = frame  # Fall back to full frame

    return preprocess_dinov2_frame(crop, target_size)


def read_window_frames_dinov2(
    video_path: str,
    second: float,
    sample_fps: float,
    window_seconds: float = 1.0,
    target_size: int = 224,
    reader: Optional[VideoReader] = None,
) -> np.ndarray:
    """Read and preprocess frames for a window using DINOv2 preprocessing.

    Args:
        video_path: Path to video file
        second: Start second of the window
        sample_fps: Target frames per second
        window_seconds: Duration of window in seconds
        target_size: Target size for DINOv2 preprocessing (default 224)
        reader: Optional pre-created video reader

    Returns:
        Preprocessed frames array (n_frames, C, H, W) float32, ImageNet-normalized
    """
    timestamps = sample_window_timestamps(second, window_seconds, sample_fps)

    own_reader = reader is None
    if own_reader:
        reader = create_video_reader(video_path)

    try:
        frames = []
        for ts in timestamps:
            try:
                frame = reader.get_frame_at_timestamp(ts)
                processed = preprocess_dinov2_frame(frame, target_size)
                frames.append(processed)
            except Exception as e:
                logger.warning(f"Failed to read frame at {ts}s from {video_path}: {e}")
                frames.append(np.zeros((3, target_size, target_size), dtype=np.float32))

        return np.stack(frames, axis=0)

    finally:
        if own_reader:
            reader.close()


def read_window_person_crops(
    video_path: str,
    second: float,
    bboxes: dict[str, list[tuple[float, float, float, float]]],
    sample_fps: float,
    window_seconds: float = 1.0,
    target_size: int = 224,
    reader: Optional[VideoReader] = None,
) -> dict[str, np.ndarray]:
    """Read frames and return per-person crops for a window.

    Args:
        video_path: Path to video file
        second: Start second of the window
        bboxes: Dict mapping role ("person_a", "person_b") to list of
                 per-frame bboxes (x1, y1, x2, y2). Each list should have
                 one bbox per frame (n_frames entries).
        sample_fps: Target frames per second
        window_seconds: Duration of window in seconds
        target_size: Target size for DINOv2 preprocessing
        reader: Optional pre-created video reader

    Returns:
        Dict with:
        - "person_a": (n_frames, C, H, W) crops for person A
        - "person_b": (n_frames, C, H, W) crops for person B
        - "full_frames": (n_frames, C, H, W) full-frame preprocessed
    """
    timestamps = sample_window_timestamps(second, window_seconds, sample_fps)
    n_frames = len(timestamps)

    own_reader = reader is None
    if own_reader:
        reader = create_video_reader(video_path)

    try:
        result = {role: [] for role in bboxes}
        result["full_frames"] = []

        for i, ts in enumerate(timestamps):
            try:
                frame = reader.get_frame_at_timestamp(ts)
            except Exception as e:
                logger.warning(f"Failed to read frame at {ts}s from {video_path}: {e}")
                # Black frames as fallback
                blank = np.zeros((3, target_size, target_size), dtype=np.float32)
                for role in bboxes:
                    result[role].append(blank)
                result["full_frames"].append(blank)
                continue

            # Full frame
            result["full_frames"].append(preprocess_dinov2_frame(frame, target_size))

            # Per-person crops
            for role, role_bboxes in bboxes.items():
                if i < len(role_bboxes) and role_bboxes[i] is not None:
                    crop = crop_and_preprocess_person(frame, role_bboxes[i], target_size)
                else:
                    crop = preprocess_dinov2_frame(frame, target_size)
                result[role].append(crop)

        # Stack into arrays
        return {key: np.stack(frames, axis=0) for key, frames in result.items()}

    finally:
        if own_reader:
            reader.close()
