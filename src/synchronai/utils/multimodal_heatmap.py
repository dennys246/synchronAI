"""
Heatmap generation for multi-modal synchrony predictions.

Generates temporal heatmaps showing video-only, audio-only, and fused predictions.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def generate_multimodal_heatmap(
    model: nn.Module,
    video_path: Path,
    save_dir: Path,
    epoch: int,
    device: torch.device,
    sample_fps: float = 12.0,
    window_seconds: float = 2.0,
    frame_size: int = 640,
    sample_rate: int = 16000,
    chunk_duration: float = 1.0,
    threshold: float = 0.5,
    clip_duration: int = 10,
    labels_file: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate multi-modal heatmap visualization during training.

    Shows predictions from:
    - Video pathway only
    - Audio pathway only
    - Multi-modal fusion
    - Ground truth (if available)

    Args:
        model: Current multi-modal model state
        video_path: Path to sample video for visualization
        save_dir: Directory to save heatmaps
        epoch: Current epoch number
        device: Device to run inference on
        sample_fps: Video sampling FPS
        window_seconds: Video window duration
        frame_size: Video frame size
        sample_rate: Audio sample rate
        chunk_duration: Audio chunk duration
        threshold: Classification threshold
        clip_duration: Duration in seconds to analyze (from middle)

    Returns:
        Path to generated heatmap directory, or None if failed
    """
    from synchronai.data.video.processing import (
        VideoReaderPool,
        load_video_info,
        read_window_frames,
    )
    from synchronai.data.audio.processing import (
        extract_audio,
        load_audio_chunk,
    )

    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning(f"Heatmap video not found: {video_path}")
        return None

    heatmap_dir = save_dir / "heatmaps" / f"epoch_{epoch:04d}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get video info
        video_info = load_video_info(str(video_path))
        total_seconds = int(video_info.duration)

        # Select clip from middle
        clip_duration = min(clip_duration, total_seconds)
        start_second = max(0, (total_seconds - clip_duration) // 2)
        end_second = start_second + clip_duration

        logger.info(
            f"Generating multi-modal heatmap for epoch {epoch} "
            f"(seconds {start_second}-{end_second} of {total_seconds}s video)..."
        )

        # Create reader pool for video
        reader_pool = VideoReaderPool(max_readers=1)
        reader = reader_pool.get_reader(str(video_path))

        # Extract audio from video
        audio_path = heatmap_dir / f"{video_path.stem}_audio.wav"
        extract_audio(str(video_path), str(audio_path), sample_rate=sample_rate)

        # Storage for predictions
        video_only_preds = []
        audio_only_preds = []
        fused_preds = []
        seconds = []

        model.eval()

        try:
            with torch.no_grad():
                for second in range(start_second, end_second):
                    # Read video frames
                    frames = read_window_frames(
                        video_path=str(video_path),
                        second=second,
                        sample_fps=sample_fps,
                        window_seconds=window_seconds,
                        target_size=frame_size,
                        reader=reader,
                    )

                    # Read audio chunk
                    audio = load_audio_chunk(
                        audio_path=str(audio_path),
                        second=second,
                        chunk_duration=chunk_duration,
                        sample_rate=sample_rate,
                    )

                    # Convert to tensors
                    frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

                    # Get multi-modal predictions
                    outputs = model(frames_tensor, audio_tensor, return_features=True)

                    # Fused prediction
                    fused_prob = torch.sigmoid(outputs['sync_logits']).item()
                    fused_preds.append(fused_prob)

                    # Video-only prediction (bypass fusion)
                    video_features = outputs['video_features']
                    video_only_logit = model.sync_head(
                        model.fusion_module.video_proj(video_features)
                        if hasattr(model.fusion_module, 'video_proj')
                        else video_features
                    )
                    video_only_prob = torch.sigmoid(video_only_logit).item()
                    video_only_preds.append(video_only_prob)

                    # Audio-only prediction (bypass fusion)
                    audio_features = outputs['audio_features']
                    audio_only_logit = model.sync_head(
                        model.fusion_module.audio_proj(audio_features)
                        if hasattr(model.fusion_module, 'audio_proj')
                        else audio_features
                    )
                    audio_only_prob = torch.sigmoid(audio_only_logit).item()
                    audio_only_preds.append(audio_only_prob)

                    seconds.append(second)

        finally:
            reader_pool.close_all()

        # Load ground truth labels if labels_file is available
        ground_truth = None
        if labels_file:
            from synchronai.utils.heatmap import load_ground_truth_for_clip
            ground_truth = load_ground_truth_for_clip(
                labels_file, video_path, start_second, end_second
            )

        # Generate visualization
        plot_multimodal_comparison(
            seconds=seconds,
            video_preds=video_only_preds,
            audio_preds=audio_only_preds,
            fused_preds=fused_preds,
            threshold=threshold,
            video_name=video_path.stem,
            epoch=epoch,
            save_path=heatmap_dir / f"{video_path.stem}_comparison.png",
            ground_truth=ground_truth,
        )

        # Export data
        export_multimodal_data(
            seconds=seconds,
            video_preds=video_only_preds,
            audio_preds=audio_only_preds,
            fused_preds=fused_preds,
            save_path=heatmap_dir / f"{video_path.stem}_data.json"
        )

        logger.info(f"✓ Saved heatmap to {heatmap_dir}")
        return heatmap_dir

    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_multimodal_comparison(
    seconds: List[int],
    video_preds: List[float],
    audio_preds: List[float],
    fused_preds: List[float],
    threshold: float,
    video_name: str,
    epoch: int,
    save_path: Path,
    ground_truth: Optional[Dict[int, int]] = None,
):
    """
    Plot comparison of video-only, audio-only, and fused predictions.

    Args:
        seconds: List of second timestamps
        video_preds: Video-only predictions
        audio_preds: Audio-only predictions
        fused_preds: Multi-modal fused predictions
        threshold: Classification threshold
        video_name: Name of video
        epoch: Epoch number
        save_path: Path to save plot
        ground_truth: Optional dict mapping second -> label (0 or 1)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    has_gt = ground_truth is not None and len(ground_truth) > 0
    n_rows = 5 if has_gt else 4
    fig_height = 12 if has_gt else 10

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, fig_height), sharex=True)

    row_idx = 0

    # Ground truth row (if available)
    if has_gt:
        ax = axes[row_idx]
        for sec in seconds:
            label = ground_truth.get(sec, -1)
            if label >= 0:
                color = 'green' if label == 1 else 'red'
                ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=0.85, edgecolor='none')
            else:
                ax.barh(0, 1, left=sec, height=0.8, color='lightgray', alpha=0.5, edgecolor='none')
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel('Ground\nTruth', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title(f'Multi-Modal Predictions - {video_name} (Epoch {epoch})', fontweight='bold', fontsize=12)
        row_idx += 1

    # Video-only predictions
    ax = axes[row_idx]
    for i, (sec, prob) in enumerate(zip(seconds, video_preds)):
        color = 'green' if prob >= threshold else 'red'
        alpha = min(0.3 + 0.7 * abs(prob - 0.5) * 2, 1.0)
        ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=alpha, edgecolor='none')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel('Video\nOnly', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    if not has_gt:
        ax.set_title(f'Multi-Modal Predictions - {video_name} (Epoch {epoch})', fontweight='bold', fontsize=12)
    row_idx += 1

    # Audio-only predictions
    ax = axes[row_idx]
    for i, (sec, prob) in enumerate(zip(seconds, audio_preds)):
        color = 'green' if prob >= threshold else 'red'
        alpha = min(0.3 + 0.7 * abs(prob - 0.5) * 2, 1.0)
        ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=alpha, edgecolor='none')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel('Audio\nOnly', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    row_idx += 1

    # Fused predictions
    ax = axes[row_idx]
    for i, (sec, prob) in enumerate(zip(seconds, fused_preds)):
        color = 'green' if prob >= threshold else 'red'
        alpha = min(0.3 + 0.7 * abs(prob - 0.5) * 2, 1.0)
        ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=alpha, edgecolor='none')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel('Fused\n(Multi-Modal)', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    row_idx += 1

    # Probability timeline
    ax = axes[row_idx]
    ax.plot(seconds, video_preds, 'b-', label='Video Only', linewidth=2, alpha=0.7)
    ax.plot(seconds, audio_preds, 'orange', label='Audio Only', linewidth=2, alpha=0.7)
    ax.plot(seconds, fused_preds, 'g-', label='Fused', linewidth=2.5)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    # Show ground truth as step line
    if has_gt:
        gt_for_plot = [float(ground_truth.get(s, float('nan'))) for s in seconds]
        ax.step(seconds, gt_for_plot, where='mid', color='black', linewidth=2,
                linestyle='--', label='Ground Truth', alpha=0.7, zorder=4)
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Synchrony\nProbability', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_handles = [
        mpatches.Patch(color='green', label='Sync'),
        mpatches.Patch(color='red', label='No Sync'),
    ]
    if has_gt:
        legend_handles.append(mpatches.Patch(color='black', label='Ground Truth'))
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=len(legend_handles), bbox_to_anchor=(0.5, 0.98), framealpha=0.9)

    # Compute statistics
    video_sync_ratio = sum(1 for p in video_preds if p >= threshold) / len(video_preds)
    audio_sync_ratio = sum(1 for p in audio_preds if p >= threshold) / len(audio_preds)
    fused_sync_ratio = sum(1 for p in fused_preds if p >= threshold) / len(fused_preds)

    # Add summary text with accuracy if ground truth available
    summary_lines = [
        f"Predicted Sync Ratios:",
        f"  Video: {video_sync_ratio:.1%}",
        f"  Audio: {audio_sync_ratio:.1%}",
        f"  Fused: {fused_sync_ratio:.1%}",
    ]
    if has_gt:
        gt_labels = [ground_truth.get(s, -1) for s in seconds]
        gt_available = [i for i, g in enumerate(gt_labels) if g >= 0]
        if gt_available:
            gt_sync_count = sum(gt_labels[i] for i in gt_available)
            gt_total = len(gt_available)
            fused_binary = [1 if p >= threshold else 0 for p in fused_preds]
            correct = sum(fused_binary[i] == gt_labels[i] for i in gt_available)
            accuracy = correct / gt_total
            summary_lines.append(f"  Ground Truth: {gt_sync_count}/{gt_total} sync")
            summary_lines.append(f"  Fused Accuracy: {correct}/{gt_total} = {accuracy:.1%}")

    summary_text = "\n".join(summary_lines)
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_multimodal_data(
    seconds: List[int],
    video_preds: List[float],
    audio_preds: List[float],
    fused_preds: List[float],
    save_path: Path,
):
    """Export multi-modal predictions to JSON."""
    import json

    data = {
        'predictions': [
            {
                'second': sec,
                'video_probability': v_prob,
                'audio_probability': a_prob,
                'fused_probability': f_prob,
            }
            for sec, v_prob, a_prob, f_prob in zip(seconds, video_preds, audio_preds, fused_preds)
        ],
        'statistics': {
            'video_mean_prob': float(np.mean(video_preds)),
            'audio_mean_prob': float(np.mean(audio_preds)),
            'fused_mean_prob': float(np.mean(fused_preds)),
        }
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
