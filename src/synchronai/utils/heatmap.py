"""
Heatmap visualization utilities for video synchrony classification.

Provides plotting functions for visualizing per-second synchrony predictions
as temporal heatmaps, confidence distributions, and video overlays.

Includes video overlay generation that renders heatmaps directly onto video frames.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from synchronai.inference.video.predict import (
        PredictionResult,
        VideoPredictionResult,
    )
    from synchronai.models.cv.YOLO_classifier import VideoClassifier

logger = logging.getLogger(__name__)


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""

    # Colormap settings
    colormap: str = "RdYlGn"
    vmin: float = 0.0
    vmax: float = 1.0

    # Figure settings
    figsize_timeline: tuple[int, int] = (16, 4)
    figsize_grid: tuple[int, int] = (16, 10)
    figsize_distribution: tuple[int, int] = (10, 6)
    dpi: int = 150

    # Labels
    cbar_label: str = "Synchrony Probability"
    xlabel: str = "Time (seconds)"

    # Styling
    grid_alpha: float = 0.3
    annotation_fontsize: int = 8
    title_fontsize: int = 14

    # Threshold for binary coloring
    threshold: float = 0.5


def _create_synchrony_colormap() -> LinearSegmentedColormap:
    """Create a custom colormap: red (async) -> yellow -> green (sync)."""
    colors = [
        (0.8, 0.2, 0.2),  # Red for low synchrony
        (0.9, 0.7, 0.3),  # Yellow/orange for middle
        (0.2, 0.7, 0.3),  # Green for high synchrony
    ]
    return LinearSegmentedColormap.from_list("synchrony", colors, N=256)


def plot_temporal_heatmap(
    predictions: List[PredictionResult],
    *,
    config: Optional[HeatmapConfig] = None,
    title: str = "Video Synchrony Timeline",
    save_path: Optional[str] = None,
    show_annotations: bool = True,
    show_threshold_line: bool = True,
) -> None:
    """
    Plot a single-row temporal heatmap showing synchrony probability per second.

    Args:
        predictions: List of PredictionResult from predict_video()
        config: Heatmap configuration
        title: Plot title
        save_path: Optional path to save the figure
        show_annotations: Whether to show probability values on cells
        show_threshold_line: Whether to show threshold indicator
    """
    if config is None:
        config = HeatmapConfig()

    if not predictions:
        raise ValueError("No predictions to plot")

    # Extract data
    seconds = [p.second for p in predictions]
    probs = np.array([p.probability for p in predictions])
    preds = np.array([p.prediction for p in predictions])

    # Create figure with two subplots: heatmap + line plot
    fig, (ax_heat, ax_line) = plt.subplots(
        2,
        1,
        figsize=config.figsize_timeline,
        height_ratios=[1, 2],
        sharex=True,
    )

    # === Heatmap subplot ===
    heatmap_data = probs.reshape(1, -1)
    cmap = plt.get_cmap(config.colormap)

    im = ax_heat.imshow(
        heatmap_data,
        aspect="auto",
        cmap=cmap,
        vmin=config.vmin,
        vmax=config.vmax,
        extent=[0, len(probs), 0, 1],
    )

    # Add cell annotations if requested and not too many cells
    if show_annotations and len(probs) <= 60:
        for i, prob in enumerate(probs):
            text_color = "white" if prob < 0.3 or prob > 0.7 else "black"
            ax_heat.text(
                i + 0.5,
                0.5,
                f"{prob:.2f}",
                ha="center",
                va="center",
                fontsize=config.annotation_fontsize,
                color=text_color,
            )

    ax_heat.set_yticks([])
    ax_heat.set_ylabel("Sync\nProb", fontsize=10, rotation=0, ha="right", va="center")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", pad=0.02, aspect=10)
    cbar.set_label(config.cbar_label, fontsize=10)

    # === Line plot subplot ===
    ax_line.fill_between(
        seconds,
        probs,
        alpha=0.3,
        color="steelblue",
        label="Probability",
    )
    ax_line.plot(seconds, probs, color="steelblue", linewidth=1.5)

    # Mark synchrony predictions
    sync_mask = preds == 1
    async_mask = preds == 0
    ax_line.scatter(
        np.array(seconds)[sync_mask],
        probs[sync_mask],
        color="green",
        s=30,
        zorder=5,
        label="Synchrony",
        alpha=0.7,
    )
    ax_line.scatter(
        np.array(seconds)[async_mask],
        probs[async_mask],
        color="red",
        s=30,
        zorder=5,
        label="Asynchrony",
        alpha=0.7,
    )

    # Threshold line
    if show_threshold_line:
        ax_line.axhline(
            y=config.threshold,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({config.threshold})",
        )

    ax_line.set_xlabel(config.xlabel, fontsize=11)
    ax_line.set_ylabel("Probability", fontsize=11)
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.set_xlim(0, len(probs) - 1)
    ax_line.grid(True, alpha=config.grid_alpha, linestyle="--")
    ax_line.legend(loc="upper right", fontsize=9)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)

    # Overall title
    sync_ratio = sum(preds) / len(preds)
    fig.suptitle(
        f"{title}\n(Synchrony: {sum(preds)}/{len(preds)} seconds = {sync_ratio:.1%})",
        fontsize=config.title_fontsize,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_heatmap_grid(
    predictions: List[PredictionResult],
    *,
    config: Optional[HeatmapConfig] = None,
    title: str = "Synchrony Heatmap Grid",
    save_path: Optional[str] = None,
    cols: int = 30,
) -> None:
    """
    Plot predictions as a 2D grid heatmap (useful for longer videos).

    Arranges seconds into a grid where each cell represents one second,
    making it easier to visualize patterns in longer videos.

    Args:
        predictions: List of PredictionResult from predict_video()
        config: Heatmap configuration
        title: Plot title
        save_path: Optional path to save the figure
        cols: Number of columns in the grid (seconds per row)
    """
    if config is None:
        config = HeatmapConfig()

    if not predictions:
        raise ValueError("No predictions to plot")

    probs = np.array([p.probability for p in predictions])
    n_seconds = len(probs)

    # Calculate grid dimensions
    rows = (n_seconds + cols - 1) // cols
    padded_size = rows * cols

    # Pad with NaN for incomplete rows
    padded_probs = np.full(padded_size, np.nan)
    padded_probs[:n_seconds] = probs
    grid_data = padded_probs.reshape(rows, cols)

    fig, ax = plt.subplots(figsize=config.figsize_grid)

    cmap = plt.get_cmap(config.colormap)
    cmap.set_bad(color="lightgray")  # Color for NaN values

    im = ax.imshow(
        grid_data,
        aspect="auto",
        cmap=cmap,
        vmin=config.vmin,
        vmax=config.vmax,
    )

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    # Labels
    ax.set_xticks(np.arange(0, cols, 5))
    ax.set_xticklabels([str(i) for i in range(0, cols, 5)])
    ax.set_xlabel(f"Second (within row, {cols}s per row)", fontsize=11)

    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([f"{i * cols}-{min((i + 1) * cols - 1, n_seconds - 1)}s" for i in range(rows)])
    ax.set_ylabel("Time Range", fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(config.cbar_label, fontsize=11)

    # Title with summary
    preds = [p.prediction for p in predictions]
    sync_ratio = sum(preds) / len(preds)
    ax.set_title(
        f"{title}\n({n_seconds} seconds total, {sync_ratio:.1%} synchrony)",
        fontsize=config.title_fontsize,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confidence_distribution(
    predictions: List[PredictionResult],
    *,
    config: Optional[HeatmapConfig] = None,
    title: str = "Synchrony Confidence Distribution",
    save_path: Optional[str] = None,
    bins: int = 20,
) -> None:
    """
    Plot histogram of synchrony probabilities.

    Helps visualize the distribution of model confidence and identify
    whether predictions are decisive or uncertain.

    Args:
        predictions: List of PredictionResult from predict_video()
        config: Heatmap configuration
        title: Plot title
        save_path: Optional path to save the figure
        bins: Number of histogram bins
    """
    if config is None:
        config = HeatmapConfig()

    if not predictions:
        raise ValueError("No predictions to plot")

    probs = np.array([p.probability for p in predictions])
    confidences = np.array([p.confidence for p in predictions])

    fig, (ax_prob, ax_conf) = plt.subplots(1, 2, figsize=config.figsize_distribution)

    # === Probability distribution ===
    n, bins_edges, patches = ax_prob.hist(
        probs,
        bins=bins,
        range=(0, 1),
        edgecolor="black",
        alpha=0.7,
    )

    # Color bars by value
    cmap = plt.get_cmap(config.colormap)
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    for patch, center in zip(patches, bin_centers):
        patch.set_facecolor(cmap(center))

    ax_prob.axvline(
        x=config.threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({config.threshold})",
    )
    ax_prob.axvline(
        x=np.mean(probs),
        color="blue",
        linestyle=":",
        linewidth=2,
        label=f"Mean ({np.mean(probs):.3f})",
    )

    ax_prob.set_xlabel("Synchrony Probability", fontsize=11)
    ax_prob.set_ylabel("Count (seconds)", fontsize=11)
    ax_prob.set_title("Probability Distribution", fontsize=12)
    ax_prob.legend(loc="upper right", fontsize=9)
    ax_prob.grid(True, alpha=config.grid_alpha, linestyle="--")
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)

    # === Confidence distribution ===
    ax_conf.hist(
        confidences,
        bins=bins,
        range=(0.5, 1),
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax_conf.axvline(
        x=np.mean(confidences),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean ({np.mean(confidences):.3f})",
    )

    ax_conf.set_xlabel("Prediction Confidence", fontsize=11)
    ax_conf.set_ylabel("Count (seconds)", fontsize=11)
    ax_conf.set_title("Confidence Distribution", fontsize=12)
    ax_conf.legend(loc="upper left", fontsize=9)
    ax_conf.grid(True, alpha=config.grid_alpha, linestyle="--")
    ax_conf.spines["top"].set_visible(False)
    ax_conf.spines["right"].set_visible(False)

    # Summary statistics
    preds = [p.prediction for p in predictions]
    sync_pct = sum(preds) / len(preds) * 100
    high_conf = sum(1 for c in confidences if c > 0.8) / len(confidences) * 100

    fig.suptitle(
        f"{title}\n(Synchrony: {sync_pct:.1f}%, High confidence (>0.8): {high_conf:.1f}%)",
        fontsize=config.title_fontsize,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_segment_summary(
    predictions: List[PredictionResult],
    *,
    config: Optional[HeatmapConfig] = None,
    title: str = "Synchrony Segments",
    save_path: Optional[str] = None,
    segment_duration: int = 10,
) -> None:
    """
    Plot synchrony summary by time segments.

    Aggregates predictions into segments (e.g., 10-second blocks) and shows
    the proportion of synchrony within each segment.

    Args:
        predictions: List of PredictionResult from predict_video()
        config: Heatmap configuration
        title: Plot title
        save_path: Optional path to save the figure
        segment_duration: Duration of each segment in seconds
    """
    if config is None:
        config = HeatmapConfig()

    if not predictions:
        raise ValueError("No predictions to plot")

    probs = np.array([p.probability for p in predictions])
    preds = np.array([p.prediction for p in predictions])
    n_seconds = len(probs)

    # Calculate segment statistics
    n_segments = (n_seconds + segment_duration - 1) // segment_duration
    segment_means = []
    segment_sync_ratios = []
    segment_labels = []

    for i in range(n_segments):
        start = i * segment_duration
        end = min((i + 1) * segment_duration, n_seconds)
        segment_probs = probs[start:end]
        segment_preds = preds[start:end]

        segment_means.append(np.mean(segment_probs))
        segment_sync_ratios.append(np.mean(segment_preds))
        segment_labels.append(f"{start}-{end - 1}s")

    fig, ax = plt.subplots(figsize=(max(12, n_segments * 0.8), 6))

    x = np.arange(n_segments)
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        segment_means,
        width,
        label="Mean Probability",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        segment_sync_ratios,
        width,
        label="Synchrony Ratio",
        color="green",
        alpha=0.8,
    )

    # Threshold line
    ax.axhline(
        y=config.threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({config.threshold})",
    )

    ax.set_xlabel(f"Time Segment ({segment_duration}s each)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=config.grid_alpha, linestyle="--", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    overall_sync = sum(preds) / len(preds)
    ax.set_title(
        f"{title}\n(Overall: {overall_sync:.1%} synchrony across {n_seconds}s)",
        fontsize=config.title_fontsize,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def export_heatmap_data(
    result: VideoPredictionResult,
    output_path: Union[str, Path],
) -> Path:
    """
    Export prediction data to JSON for external visualization tools.

    Args:
        result: VideoPredictionResult from predict_video()
        output_path: Output JSON path

    Returns:
        Path to saved JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "video_path": result.video_path,
        "predictions": [asdict(p) for p in result.predictions],
        "summary": {
            "total_seconds": result.total_seconds,
            "synchrony_seconds": result.synchrony_seconds,
            "synchrony_ratio": result.synchrony_ratio,
            "overall_probability": result.overall_probability,
            "overall_prediction": result.overall_prediction,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def generate_all_heatmaps(
    result: VideoPredictionResult,
    output_dir: Union[str, Path],
    *,
    config: Optional[HeatmapConfig] = None,
    prefix: str = "",
    include_video_overlay: bool = False,
) -> dict[str, Path]:
    """
    Generate all heatmap visualizations for a video prediction result.

    Args:
        result: VideoPredictionResult from predict_video()
        output_dir: Directory to save visualizations
        config: Heatmap configuration
        prefix: Optional prefix for output filenames
        include_video_overlay: Whether to generate video with heatmap overlay

    Returns:
        Dictionary mapping visualization type to file path
    """
    if config is None:
        config = HeatmapConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix:
        prefix = f"{prefix}_"

    video_name = Path(result.video_path).stem
    outputs = {}

    # Generate temporal heatmap (timeline)
    timeline_path = output_dir / f"{prefix}{video_name}_timeline.png"
    plot_temporal_heatmap(
        result.predictions,
        config=config,
        title=f"Synchrony Timeline: {video_name}",
        save_path=str(timeline_path),
    )
    outputs["timeline"] = timeline_path

    # Generate grid heatmap (for longer videos)
    if len(result.predictions) > 60:
        grid_path = output_dir / f"{prefix}{video_name}_grid.png"
        plot_heatmap_grid(
            result.predictions,
            config=config,
            title=f"Synchrony Grid: {video_name}",
            save_path=str(grid_path),
        )
        outputs["grid"] = grid_path

    # Generate confidence distribution
    dist_path = output_dir / f"{prefix}{video_name}_distribution.png"
    plot_confidence_distribution(
        result.predictions,
        config=config,
        title=f"Confidence Distribution: {video_name}",
        save_path=str(dist_path),
    )
    outputs["distribution"] = dist_path

    # Generate segment summary
    segment_path = output_dir / f"{prefix}{video_name}_segments.png"
    plot_segment_summary(
        result.predictions,
        config=config,
        title=f"Segment Summary: {video_name}",
        save_path=str(segment_path),
    )
    outputs["segments"] = segment_path

    # Export JSON data
    json_path = output_dir / f"{prefix}{video_name}_data.json"
    export_heatmap_data(result, json_path)
    outputs["data"] = json_path

    # Generate video with heatmap overlay (optional, can be slow)
    if include_video_overlay:
        video_path = output_dir / f"{prefix}{video_name}_overlay.mp4"
        render_video_with_heatmap_overlay(
            result.video_path,
            result.predictions,
            str(video_path),
            config=config,
        )
        outputs["video_overlay"] = video_path

    return outputs


# =============================================================================
# Video Overlay Functions
# =============================================================================


def probability_to_color(
    probability: float,
    colormap: str = "RdYlGn",
) -> Tuple[int, int, int]:
    """
    Convert a synchrony probability to a BGR color.

    Args:
        probability: Synchrony probability (0-1)
        colormap: Matplotlib colormap name

    Returns:
        BGR color tuple (0-255)
    """
    cmap = plt.get_cmap(colormap)
    rgba = cmap(probability)
    # Convert RGBA (0-1) to BGR (0-255)
    return (
        int(rgba[2] * 255),  # B
        int(rgba[1] * 255),  # G
        int(rgba[0] * 255),  # R
    )


def create_overlay_frame(
    frame: np.ndarray,
    probability: float,
    prediction: int,
    second: int,
    *,
    config: Optional[HeatmapConfig] = None,
    overlay_alpha: float = 0.25,
    show_border: bool = True,
    show_text: bool = True,
    show_bar: bool = True,
) -> np.ndarray:
    """
    Create a frame with heatmap overlay indicating synchrony probability.

    The overlay includes:
    - Semi-transparent color overlay (red=async, green=sync)
    - Colored border indicating current prediction
    - Text showing probability and classification
    - Progress bar showing probability level

    Args:
        frame: Input frame (H, W, C) in BGR format
        probability: Synchrony probability (0-1)
        prediction: Binary prediction (0 or 1)
        second: Current second in the video
        config: Heatmap configuration
        overlay_alpha: Transparency of color overlay (0-1)
        show_border: Whether to show colored border
        show_text: Whether to show probability text
        show_bar: Whether to show probability bar

    Returns:
        Frame with overlay (H, W, C) in BGR format
    """
    if config is None:
        config = HeatmapConfig()

    frame = frame.copy()
    h, w = frame.shape[:2]

    # Get color for this probability
    color_bgr = probability_to_color(probability, config.colormap)

    # Apply semi-transparent color overlay
    overlay = np.full_like(frame, color_bgr, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 1 - overlay_alpha, overlay, overlay_alpha, 0)

    # Draw colored border
    if show_border:
        border_width = max(4, int(min(h, w) * 0.01))
        border_color = (0, 200, 0) if prediction == 1 else (0, 0, 200)  # Green or red
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_width)

    # Draw text overlay
    if show_text:
        label = "SYNC" if prediction == 1 else "ASYNC"
        text = f"{label}: {probability:.1%}"

        # Background box for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(h, w) / 800)
        thickness = max(1, int(font_scale * 2))

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding = 10
        box_x1, box_y1 = 10, 10
        box_x2 = box_x1 + text_w + padding * 2
        box_y2 = box_y1 + text_h + padding * 2

        # Semi-transparent background
        sub_frame = frame[box_y1:box_y2, box_x1:box_x2]
        black_rect = np.zeros_like(sub_frame)
        frame[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(
            sub_frame, 0.5, black_rect, 0.5, 0
        )

        # Draw text
        text_x = box_x1 + padding
        text_y = box_y1 + padding + text_h
        text_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

        # Draw timestamp
        time_text = f"t={second}s"
        (time_w, time_h), _ = cv2.getTextSize(time_text, font, font_scale * 0.8, thickness)
        time_x = w - time_w - 20
        time_y = text_y

        # Background for timestamp
        ts_box_x1 = time_x - padding
        ts_box_y1 = box_y1
        ts_box_x2 = w - 10
        ts_box_y2 = box_y2

        sub_frame = frame[ts_box_y1:ts_box_y2, ts_box_x1:ts_box_x2]
        black_rect = np.zeros_like(sub_frame)
        frame[ts_box_y1:ts_box_y2, ts_box_x1:ts_box_x2] = cv2.addWeighted(
            sub_frame, 0.5, black_rect, 0.5, 0
        )
        cv2.putText(
            frame, time_text, (time_x, time_y), font, font_scale * 0.8, (255, 255, 255), thickness
        )

    # Draw probability bar at bottom
    if show_bar:
        bar_height = max(20, int(h * 0.03))
        bar_y = h - bar_height - 10
        bar_x = 10
        bar_width = w - 20

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Filled portion based on probability
        fill_width = int(bar_width * probability)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color_bgr, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Threshold marker
        threshold_x = bar_x + int(bar_width * config.threshold)
        cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)

    return frame


def render_video_with_heatmap_overlay(
    video_path: Union[str, Path],
    predictions: List[PredictionResult],
    output_path: Union[str, Path],
    *,
    config: Optional[HeatmapConfig] = None,
    overlay_alpha: float = 0.25,
    output_fps: Optional[float] = None,
    codec: str = "mp4v",
) -> Path:
    """
    Render a new video with heatmap overlay on each frame.

    Creates a new video file where each frame is overlaid with:
    - Color tint indicating synchrony probability (red=low, green=high)
    - Border color showing binary prediction
    - Text overlay with probability percentage
    - Progress bar showing probability level

    Args:
        video_path: Path to input video
        predictions: List of PredictionResult (one per second)
        output_path: Path to output video
        config: Heatmap configuration
        overlay_alpha: Transparency of color overlay
        output_fps: Output video FPS (defaults to input FPS)
        codec: Video codec fourcc code

    Returns:
        Path to saved video
    """
    if config is None:
        config = HeatmapConfig()

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup from second -> prediction
    pred_by_second = {p.second: p for p in predictions}

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_fps is None:
            output_fps = fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

        logger.info(
            f"Rendering heatmap overlay video: {width}x{height} @ {output_fps:.1f} FPS"
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get current second
            current_second = int(frame_idx / fps)

            # Get prediction for this second
            pred = pred_by_second.get(current_second)
            if pred is not None:
                frame = create_overlay_frame(
                    frame,
                    pred.probability,
                    pred.prediction,
                    current_second,
                    config=config,
                    overlay_alpha=overlay_alpha,
                )

            writer.write(frame)
            frame_idx += 1

            # Progress logging
            if frame_idx % int(fps * 10) == 0:
                progress = frame_idx / frame_count * 100
                logger.info(f"  Progress: {progress:.1f}% ({frame_idx}/{frame_count} frames)")

        writer.release()
        logger.info(f"Saved overlay video: {output_path}")

    finally:
        cap.release()

    return output_path


def create_thumbnail_grid(
    video_path: Union[str, Path],
    predictions: List[PredictionResult],
    output_path: Union[str, Path],
    *,
    config: Optional[HeatmapConfig] = None,
    cols: int = 6,
    thumb_size: Tuple[int, int] = (320, 180),
    max_thumbnails: int = 30,
) -> Path:
    """
    Create a grid of video thumbnails with heatmap overlays.

    Samples frames at regular intervals and creates a grid visualization
    showing how synchrony changes throughout the video.

    Args:
        video_path: Path to input video
        predictions: List of PredictionResult
        output_path: Path to output image
        config: Heatmap configuration
        cols: Number of columns in grid
        thumb_size: Size of each thumbnail (width, height)
        max_thumbnails: Maximum number of thumbnails to include

    Returns:
        Path to saved image
    """
    if config is None:
        config = HeatmapConfig()

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup from second -> prediction
    pred_by_second = {p.second: p for p in predictions}
    total_seconds = len(predictions)

    # Calculate which seconds to sample
    n_thumbs = min(max_thumbnails, total_seconds)
    if n_thumbs < total_seconds:
        sample_seconds = np.linspace(0, total_seconds - 1, n_thumbs, dtype=int)
    else:
        sample_seconds = np.arange(total_seconds)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Collect thumbnails
        thumbnails = []
        for second in sample_seconds:
            # Seek to middle of this second
            frame_idx = int((second + 0.5) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Resize to thumbnail size
            thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)

            # Apply overlay
            pred = pred_by_second.get(second)
            if pred is not None:
                thumb = create_overlay_frame(
                    thumb,
                    pred.probability,
                    pred.prediction,
                    second,
                    config=config,
                    overlay_alpha=0.3,
                    show_bar=False,  # Too small for bar
                )

            thumbnails.append(thumb)

    finally:
        cap.release()

    if not thumbnails:
        raise RuntimeError("Failed to extract any thumbnails")

    # Create grid
    n_thumbs = len(thumbnails)
    rows = (n_thumbs + cols - 1) // cols
    thumb_w, thumb_h = thumb_size

    # Create canvas
    grid_w = cols * thumb_w
    grid_h = rows * thumb_h
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Place thumbnails
    for i, thumb in enumerate(thumbnails):
        row = i // cols
        col = i % cols
        y = row * thumb_h
        x = col * thumb_w
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb

    cv2.imwrite(str(output_path), canvas)
    logger.info(f"Saved thumbnail grid: {output_path}")

    return output_path


# =============================================================================
# Grad-CAM Spatial Heatmap Functions
# =============================================================================


def render_video_with_gradcam_overlay(
    video_path: Union[str, Path],
    predictions: List[PredictionResult],
    output_path: Union[str, Path],
    model: "VideoClassifier",
    *,
    config: Optional[HeatmapConfig] = None,
    overlay_alpha: float = 0.4,
    output_fps: Optional[float] = None,
    codec: str = "mp4v",
    sample_fps: float = 12.0,
    frame_size: int = 640,
    cam_aggregate: str = "max",
    show_info_overlay: bool = True,
) -> Path:
    """
    Render a video with Grad-CAM spatial heatmaps overlaid on each frame.

    Unlike the basic heatmap overlay which tints the entire frame uniformly,
    this uses Grad-CAM to highlight the specific spatial regions that the
    model focuses on for its synchrony prediction.

    Args:
        video_path: Path to input video
        predictions: List of PredictionResult (one per second)
        output_path: Path to output video
        model: Trained VideoClassifier model (must be on correct device)
        config: Heatmap configuration
        overlay_alpha: Transparency of Grad-CAM overlay (0-1)
        output_fps: Output video FPS (defaults to input FPS)
        codec: Video codec fourcc code
        sample_fps: FPS used for model inference
        frame_size: Frame size used for model inference
        cam_aggregate: How to aggregate CAMs across frames in a window
                       ("max", "mean", "weighted")
        show_info_overlay: Whether to show probability text and bar

    Returns:
        Path to saved video
    """
    from synchronai.utils.gradcam import GradCAM, GradCAMConfig, apply_cam_to_frame

    if config is None:
        config = HeatmapConfig()

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup from second -> prediction
    pred_by_second = {p.second: p for p in predictions}

    # Initialize Grad-CAM extractor
    gradcam_config = GradCAMConfig(alpha=overlay_alpha)
    gradcam = GradCAM(model, gradcam_config)

    # Import video processing utilities
    from synchronai.data.video.processing import (
        VideoReaderPool,
        read_window_frames,
    )

    # Open input video for frame reading
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_fps is None:
            output_fps = fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

        logger.info(
            f"Rendering Grad-CAM overlay video: {width}x{height} @ {output_fps:.1f} FPS"
        )

        # Create reader pool for getting model input frames
        reader_pool = VideoReaderPool(max_readers=1)
        reader = reader_pool.get_reader(str(video_path))

        # Get device from model
        device = next(model.parameters()).device

        # Cache CAMs per second to avoid recomputation
        cam_cache: dict[int, np.ndarray] = {}

        frame_idx = 0
        import torch

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get current second
            current_second = int(frame_idx / fps)

            # Get prediction for this second
            pred = pred_by_second.get(current_second)

            if pred is not None:
                # Get or compute CAM for this second
                if current_second not in cam_cache:
                    try:
                        # Read frames for model (normalized, resized)
                        model_frames = read_window_frames(
                            video_path=str(video_path),
                            second=current_second,
                            sample_fps=sample_fps,
                            window_seconds=1.0,
                            target_size=frame_size,
                            reader=reader,
                        )

                        # Convert to tensor
                        frames_tensor = torch.from_numpy(model_frames).unsqueeze(0).to(device)

                        # Generate aggregated CAM for this window
                        with torch.enable_grad():
                            cam = gradcam.generate_cam_for_window(
                                frames_tensor,
                                aggregate=cam_aggregate,
                            )

                        cam_cache[current_second] = cam

                    except Exception as e:
                        logger.warning(f"Failed to generate CAM for second {current_second}: {e}")
                        cam_cache[current_second] = np.zeros((height, width), dtype=np.float32)

                cam = cam_cache[current_second]

                # Apply CAM overlay to frame
                frame = apply_cam_to_frame(frame, cam, gradcam_config)

                # Add info overlay if requested
                if show_info_overlay:
                    frame = _add_gradcam_info_overlay(
                        frame,
                        pred.probability,
                        pred.prediction,
                        current_second,
                        config,
                    )

            writer.write(frame)
            frame_idx += 1

            # Progress logging
            if frame_idx % int(fps * 10) == 0:
                progress = frame_idx / frame_count * 100
                logger.info(f"  Progress: {progress:.1f}% ({frame_idx}/{frame_count} frames)")

        writer.release()
        reader_pool.close_all()
        gradcam.remove_hooks()

        logger.info(f"Saved Grad-CAM overlay video: {output_path}")

    finally:
        cap.release()

    return output_path


def _add_gradcam_info_overlay(
    frame: np.ndarray,
    probability: float,
    prediction: int,
    second: int,
    config: HeatmapConfig,
) -> np.ndarray:
    """Add text and bar overlay to a Grad-CAM frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Text label
    label = "SYNC" if prediction == 1 else "ASYNC"
    text = f"{label}: {probability:.1%}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(h, w) / 800)
    thickness = max(1, int(font_scale * 2))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    box_x1, box_y1 = 10, 10
    box_x2 = box_x1 + text_w + padding * 2
    box_y2 = box_y1 + text_h + padding * 2

    # Semi-transparent background
    sub_frame = frame[box_y1:box_y2, box_x1:box_x2]
    black_rect = np.zeros_like(sub_frame)
    frame[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(
        sub_frame, 0.5, black_rect, 0.5, 0
    )

    # Draw text
    text_x = box_x1 + padding
    text_y = box_y1 + padding + text_h
    text_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    # Draw timestamp
    time_text = f"t={second}s"
    (time_w, time_h), _ = cv2.getTextSize(time_text, font, font_scale * 0.8, thickness)
    time_x = w - time_w - 20
    time_y = text_y

    ts_box_x1 = time_x - padding
    ts_box_y1 = box_y1
    ts_box_x2 = w - 10
    ts_box_y2 = box_y2

    sub_frame = frame[ts_box_y1:ts_box_y2, ts_box_x1:ts_box_x2]
    black_rect = np.zeros_like(sub_frame)
    frame[ts_box_y1:ts_box_y2, ts_box_x1:ts_box_x2] = cv2.addWeighted(
        sub_frame, 0.5, black_rect, 0.5, 0
    )
    cv2.putText(
        frame, time_text, (time_x, time_y), font, font_scale * 0.8, (255, 255, 255), thickness
    )

    # Draw probability bar at bottom
    bar_height = max(20, int(h * 0.03))
    bar_y = h - bar_height - 10
    bar_x = 10
    bar_width = w - 20

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

    # Filled portion - use colormap color
    fill_width = int(bar_width * probability)
    fill_color = (0, 200, 0) if prediction == 1 else (0, 0, 200)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    # Threshold marker
    threshold_x = bar_x + int(bar_width * config.threshold)
    cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)

    return frame


def create_gradcam_thumbnail_grid(
    video_path: Union[str, Path],
    predictions: List[PredictionResult],
    output_path: Union[str, Path],
    model: "VideoClassifier",
    *,
    config: Optional[HeatmapConfig] = None,
    cols: int = 6,
    thumb_size: Tuple[int, int] = (320, 180),
    max_thumbnails: int = 30,
    sample_fps: float = 12.0,
    frame_size: int = 640,
) -> Path:
    """
    Create a grid of video thumbnails with Grad-CAM overlays.

    Args:
        video_path: Path to input video
        predictions: List of PredictionResult
        output_path: Path to output image
        model: Trained VideoClassifier model
        config: Heatmap configuration
        cols: Number of columns in grid
        thumb_size: Size of each thumbnail (width, height)
        max_thumbnails: Maximum number of thumbnails to include
        sample_fps: FPS used for model inference
        frame_size: Frame size used for model inference

    Returns:
        Path to saved image
    """
    from synchronai.utils.gradcam import GradCAM, GradCAMConfig, apply_cam_to_frame
    from synchronai.data.video.processing import VideoReaderPool, read_window_frames
    import torch

    if config is None:
        config = HeatmapConfig()

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup
    pred_by_second = {p.second: p for p in predictions}
    total_seconds = len(predictions)

    # Initialize Grad-CAM
    gradcam_config = GradCAMConfig(alpha=0.5)
    gradcam = GradCAM(model, gradcam_config)
    device = next(model.parameters()).device

    # Calculate which seconds to sample
    n_thumbs = min(max_thumbnails, total_seconds)
    if n_thumbs < total_seconds:
        sample_seconds = np.linspace(0, total_seconds - 1, n_thumbs, dtype=int)
    else:
        sample_seconds = np.arange(total_seconds)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    reader_pool = VideoReaderPool(max_readers=1)
    reader = reader_pool.get_reader(str(video_path))

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        thumbnails = []
        for second in sample_seconds:
            # Get display frame
            frame_idx = int((second + 0.5) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Resize to thumbnail size
            thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)

            # Generate CAM
            pred = pred_by_second.get(second)
            if pred is not None:
                try:
                    model_frames = read_window_frames(
                        video_path=str(video_path),
                        second=second,
                        sample_fps=sample_fps,
                        window_seconds=1.0,
                        target_size=frame_size,
                        reader=reader,
                    )
                    frames_tensor = torch.from_numpy(model_frames).unsqueeze(0).to(device)

                    with torch.enable_grad():
                        cam = gradcam.generate_cam_for_window(frames_tensor, aggregate="max")

                    # Apply to thumbnail
                    thumb = apply_cam_to_frame(thumb, cam, gradcam_config)

                    # Add small label
                    label = "S" if pred.prediction == 1 else "A"
                    color = (0, 255, 0) if pred.prediction == 1 else (0, 0, 255)
                    cv2.putText(thumb, f"{label}:{pred.probability:.0%}", (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(thumb, f"t={second}s", (5, thumb_size[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                except Exception as e:
                    logger.warning(f"Failed to generate CAM for thumbnail at {second}s: {e}")

            thumbnails.append(thumb)

        gradcam.remove_hooks()
        reader_pool.close_all()

    finally:
        cap.release()

    if not thumbnails:
        raise RuntimeError("Failed to extract any thumbnails")

    # Create grid
    n_thumbs = len(thumbnails)
    rows = (n_thumbs + cols - 1) // cols
    thumb_w, thumb_h = thumb_size

    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbnails):
        row = i // cols
        col = i % cols
        y = row * thumb_h
        x = col * thumb_w
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb

    cv2.imwrite(str(output_path), canvas)
    logger.info(f"Saved Grad-CAM thumbnail grid: {output_path}")

    return output_path