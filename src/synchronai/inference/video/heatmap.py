"""
Heatmap generation for video synchrony classification.

Provides high-level functions for generating heatmap visualizations
from video files using the trained synchrony classifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from synchronai.inference.video.predict import (
    VideoPredictionResult,
    predict_video,
    predict_video_batch,
)
from synchronai.models.cv.video_classifier import (
    VideoClassifier,
    load_video_classifier,
)
from synchronai.utils.heatmap import (
    HeatmapConfig,
    create_thumbnail_grid,
    export_heatmap_data,
    generate_all_heatmaps,
    plot_confidence_distribution,
    plot_heatmap_grid,
    plot_segment_summary,
    plot_temporal_heatmap,
    render_video_with_heatmap_overlay,
    render_video_with_gradcam_overlay,
    create_gradcam_thumbnail_grid,
)

logger = logging.getLogger(__name__)


@dataclass
class HeatmapResult:
    """Result of heatmap generation for a single video."""

    video_path: str
    prediction_result: VideoPredictionResult
    output_files: dict[str, Path]


def generate_video_heatmap(
    video_path: Union[str, Path],
    weights_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    threshold: float = 0.5,
    device: Optional[str] = None,
    heatmap_config: Optional[HeatmapConfig] = None,
    visualizations: Optional[List[str]] = None,
    overlay_alpha: float = 0.25,
    use_gradcam: bool = False,
    gradcam_aggregate: str = "max",
) -> HeatmapResult:
    """
    Generate heatmap visualizations for a video file.

    This is the main entry point for heatmap generation. It runs inference
    on the video and generates the requested visualizations.

    Args:
        video_path: Path to video file
        weights_path: Path to model checkpoint
        output_dir: Directory to save visualizations
        threshold: Classification threshold
        device: Device to run on (auto-detected if None)
        heatmap_config: Configuration for heatmap styling
        visualizations: List of visualizations to generate. Options:
            - "timeline": Temporal heatmap with line plot
            - "grid": 2D grid heatmap (auto-generated for long videos)
            - "distribution": Confidence histogram
            - "segments": Segment summary bar chart
            - "data": JSON export
            - "video": Video with heatmap overlay on frames (uniform color tint)
            - "gradcam": Video with Grad-CAM spatial heatmap overlay
            - "thumbnails": Grid of thumbnail frames with overlays
            - "gradcam_thumbnails": Thumbnail grid with Grad-CAM overlays
            - "all": Generate all static visualizations (default)
            - "all_with_video": Generate all including video overlay
            - "all_with_gradcam": Generate all including Grad-CAM video overlay
        overlay_alpha: Transparency of video overlay (0-1)
        use_gradcam: If True, use Grad-CAM for video/thumbnail overlays
        gradcam_aggregate: How to aggregate Grad-CAM across frames ("max", "mean", "weighted")

    Returns:
        HeatmapResult with prediction results and output file paths
    """
    import torch

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if heatmap_config is None:
        heatmap_config = HeatmapConfig(threshold=threshold)

    if visualizations is None:
        visualizations = ["all"]

    # Determine if Grad-CAM is needed
    needs_gradcam = use_gradcam or any(
        v in visualizations for v in ["gradcam", "gradcam_thumbnails", "all_with_gradcam"]
    )

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Generating heatmaps for: {video_path.name}")
    if needs_gradcam:
        logger.info("Grad-CAM spatial heatmaps enabled")

    # Run inference
    logger.info("Running video classification inference...")
    result = predict_video(
        video_path=video_path,
        weights_path=weights_path,
        threshold=threshold,
        device=device,
    )

    logger.info(
        f"Inference complete: {result.synchrony_seconds}/{result.total_seconds}s "
        f"classified as synchrony ({result.synchrony_ratio:.1%})"
    )

    # Load model for Grad-CAM if needed
    model: Optional[VideoClassifier] = None
    if needs_gradcam:
        logger.info("Loading model for Grad-CAM generation...")
        model, model_config = load_video_classifier(str(weights_path), device=device)
        model.eval()

    # Generate visualizations
    video_name = video_path.stem
    output_files = {}

    # Determine what video overlays to include
    include_video = "video" in visualizations or "all_with_video" in visualizations
    include_gradcam = "gradcam" in visualizations or "all_with_gradcam" in visualizations
    include_gradcam_thumbs = "gradcam_thumbnails" in visualizations or "all_with_gradcam" in visualizations

    # Handle use_gradcam flag
    if use_gradcam:
        include_gradcam = include_video or include_gradcam
        include_gradcam_thumbs = True

    if "all" in visualizations or "all_with_video" in visualizations or "all_with_gradcam" in visualizations:
        logger.info("Generating all heatmap visualizations...")

        # Generate basic visualizations (not video overlay)
        output_files = generate_all_heatmaps(
            result,
            output_dir,
            config=heatmap_config,
            include_video_overlay=include_video and not include_gradcam,
        )

        # Generate standard thumbnail grid
        if not include_gradcam_thumbs:
            thumb_path = output_dir / f"{video_name}_thumbnails.png"
            create_thumbnail_grid(
                video_path,
                result.predictions,
                thumb_path,
                config=heatmap_config,
            )
            output_files["thumbnails"] = thumb_path

        # Generate Grad-CAM video if requested
        if include_gradcam and model is not None:
            logger.info("Generating Grad-CAM video overlay...")
            gradcam_video_path = output_dir / f"{video_name}_gradcam.mp4"
            render_video_with_gradcam_overlay(
                video_path,
                result.predictions,
                gradcam_video_path,
                model,
                config=heatmap_config,
                overlay_alpha=overlay_alpha,
                cam_aggregate=gradcam_aggregate,
            )
            output_files["gradcam_video"] = gradcam_video_path

        # Generate Grad-CAM thumbnails if requested
        if include_gradcam_thumbs and model is not None:
            logger.info("Generating Grad-CAM thumbnail grid...")
            gradcam_thumb_path = output_dir / f"{video_name}_gradcam_thumbnails.png"
            create_gradcam_thumbnail_grid(
                video_path,
                result.predictions,
                gradcam_thumb_path,
                model,
                config=heatmap_config,
            )
            output_files["gradcam_thumbnails"] = gradcam_thumb_path
    else:
        if "timeline" in visualizations:
            path = output_dir / f"{video_name}_timeline.png"
            plot_temporal_heatmap(
                result.predictions,
                config=heatmap_config,
                title=f"Synchrony Timeline: {video_name}",
                save_path=str(path),
            )
            output_files["timeline"] = path
            logger.info(f"  Saved timeline: {path}")

        if "grid" in visualizations:
            path = output_dir / f"{video_name}_grid.png"
            plot_heatmap_grid(
                result.predictions,
                config=heatmap_config,
                title=f"Synchrony Grid: {video_name}",
                save_path=str(path),
            )
            output_files["grid"] = path
            logger.info(f"  Saved grid: {path}")

        if "distribution" in visualizations:
            path = output_dir / f"{video_name}_distribution.png"
            plot_confidence_distribution(
                result.predictions,
                config=heatmap_config,
                title=f"Confidence Distribution: {video_name}",
                save_path=str(path),
            )
            output_files["distribution"] = path
            logger.info(f"  Saved distribution: {path}")

        if "segments" in visualizations:
            path = output_dir / f"{video_name}_segments.png"
            plot_segment_summary(
                result.predictions,
                config=heatmap_config,
                title=f"Segment Summary: {video_name}",
                save_path=str(path),
            )
            output_files["segments"] = path
            logger.info(f"  Saved segments: {path}")

        if "data" in visualizations:
            path = output_dir / f"{video_name}_data.json"
            export_heatmap_data(result, path)
            output_files["data"] = path
            logger.info(f"  Saved data: {path}")

        if "video" in visualizations:
            path = output_dir / f"{video_name}_overlay.mp4"
            render_video_with_heatmap_overlay(
                video_path,
                result.predictions,
                path,
                config=heatmap_config,
                overlay_alpha=overlay_alpha,
            )
            output_files["video"] = path
            logger.info(f"  Saved video overlay: {path}")

        if "thumbnails" in visualizations:
            path = output_dir / f"{video_name}_thumbnails.png"
            create_thumbnail_grid(
                video_path,
                result.predictions,
                path,
                config=heatmap_config,
            )
            output_files["thumbnails"] = path
            logger.info(f"  Saved thumbnails: {path}")

        if "gradcam" in visualizations and model is not None:
            path = output_dir / f"{video_name}_gradcam.mp4"
            render_video_with_gradcam_overlay(
                video_path,
                result.predictions,
                path,
                model,
                config=heatmap_config,
                overlay_alpha=overlay_alpha,
                cam_aggregate=gradcam_aggregate,
            )
            output_files["gradcam_video"] = path
            logger.info(f"  Saved Grad-CAM video: {path}")

        if "gradcam_thumbnails" in visualizations and model is not None:
            path = output_dir / f"{video_name}_gradcam_thumbnails.png"
            create_gradcam_thumbnail_grid(
                video_path,
                result.predictions,
                path,
                model,
                config=heatmap_config,
            )
            output_files["gradcam_thumbnails"] = path
            logger.info(f"  Saved Grad-CAM thumbnails: {path}")

    logger.info(f"Heatmap generation complete. Output directory: {output_dir}")

    return HeatmapResult(
        video_path=str(video_path),
        prediction_result=result,
        output_files=output_files,
    )


def generate_batch_heatmaps(
    video_paths: List[Union[str, Path]],
    weights_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    threshold: float = 0.5,
    device: Optional[str] = None,
    heatmap_config: Optional[HeatmapConfig] = None,
    visualizations: Optional[List[str]] = None,
) -> List[HeatmapResult]:
    """
    Generate heatmaps for multiple videos.

    Args:
        video_paths: List of video file paths
        weights_path: Path to model checkpoint
        output_dir: Directory to save visualizations
        threshold: Classification threshold
        device: Device to run on
        heatmap_config: Configuration for heatmap styling
        visualizations: List of visualizations to generate

    Returns:
        List of HeatmapResult for each video
    """
    results = []
    output_dir = Path(output_dir)

    logger.info(f"Processing {len(video_paths)} videos...")

    for i, video_path in enumerate(video_paths, 1):
        video_path = Path(video_path)
        logger.info(f"[{i}/{len(video_paths)}] Processing: {video_path.name}")

        try:
            # Create subdirectory for each video
            video_output_dir = output_dir / video_path.stem
            result = generate_video_heatmap(
                video_path=video_path,
                weights_path=weights_path,
                output_dir=video_output_dir,
                threshold=threshold,
                device=device,
                heatmap_config=heatmap_config,
                visualizations=visualizations,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")

    logger.info(f"Batch complete. Processed {len(results)}/{len(video_paths)} videos.")
    return results


def generate_comparison_heatmap(
    results: List[VideoPredictionResult],
    output_path: Union[str, Path],
    *,
    config: Optional[HeatmapConfig] = None,
    title: str = "Video Synchrony Comparison",
) -> Path:
    """
    Generate a comparison heatmap showing multiple videos side by side.

    Useful for comparing synchrony patterns across different videos
    or sessions.

    Args:
        results: List of VideoPredictionResult from multiple videos
        output_path: Path to save the comparison figure
        config: Heatmap configuration
        title: Plot title

    Returns:
        Path to saved figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if config is None:
        config = HeatmapConfig()

    if not results:
        raise ValueError("No results to compare")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_videos = len(results)
    max_seconds = max(r.total_seconds for r in results)

    fig, axes = plt.subplots(n_videos, 1, figsize=(16, 2 * n_videos + 1), sharex=True)
    if n_videos == 1:
        axes = [axes]

    cmap = plt.get_cmap(config.colormap)

    for i, result in enumerate(results):
        ax = axes[i]
        probs = np.array([p.probability for p in result.predictions])

        # Pad to max length
        padded = np.full(max_seconds, np.nan)
        padded[: len(probs)] = probs

        im = ax.imshow(
            padded.reshape(1, -1),
            aspect="auto",
            cmap=cmap,
            vmin=config.vmin,
            vmax=config.vmax,
            extent=[0, max_seconds, 0, 1],
        )

        video_name = Path(result.video_path).stem
        ax.set_ylabel(
            f"{video_name}\n({result.synchrony_ratio:.0%})",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
        )
        ax.set_yticks([])

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(config.cbar_label, fontsize=10)

    fig.suptitle(title, fontsize=config.title_fontsize, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison heatmap: {output_path}")
    return output_path