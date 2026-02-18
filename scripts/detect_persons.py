#!/usr/bin/env python3
"""
Offline person detection for dyadic videos.

Uses RTMDet (from MMDet, Apache 2.0 license) to detect persons in video
frames. Processes at 1fps (aligned with per-second synchrony labels) and
saves results as JSON sidecar files.

Usage:
    python scripts/detect_persons.py \
        --labels-file data/labels.csv \
        --output-dir data/person_bboxes/ \
        [--model rtmdet-m] \
        [--confidence-threshold 0.5] \
        [--device cuda]

Output format (per video JSON):
    {
        "0": [{"bbox": [x1, y1, x2, y2], "confidence": 0.95}, ...],
        "1": [{"bbox": [x1, y1, x2, y2], "confidence": 0.90}, ...],
        ...
    }

Requires: pip install mmengine mmdet (optional [person_detection] group)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_persons_in_video(
    video_path: str,
    model,
    confidence_threshold: float = 0.5,
    person_class_id: int = 0,
) -> dict[int, list[dict]]:
    """Detect persons in a video at 1fps.

    Args:
        video_path: Path to video file
        model: MMDet inference model
        confidence_threshold: Minimum detection confidence
        person_class_id: COCO class ID for person (0)

    Returns:
        Dict mapping second -> list of detections
    """
    import cv2
    from mmdet.apis import inference_detector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps) if fps > 0 else 0

    results = {}

    for second in range(duration):
        # Seek to middle of this second
        frame_idx = int((second + 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            results[str(second)] = []
            continue

        # Run detection
        result = inference_detector(model, frame)

        # Extract person detections
        detections = []
        pred_instances = result.pred_instances

        # Filter to person class with confidence threshold
        person_mask = (
            (pred_instances.labels == person_class_id)
            & (pred_instances.scores >= confidence_threshold)
        )

        person_bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
        person_scores = pred_instances.scores[person_mask].cpu().numpy()

        for bbox, score in zip(person_bboxes, person_scores):
            detections.append({
                "bbox": [float(x) for x in bbox],
                "confidence": float(score),
            })

        results[str(second)] = detections

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline person detection for dyadic videos")
    parser.add_argument("--labels-file", required=True, help="Path to labels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for bbox JSONs")
    parser.add_argument("--model", default="rtmdet-m", help="RTMDet model variant")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with existing bbox files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels to get video paths
    df = pd.read_csv(args.labels_file)
    video_paths = sorted(df["video_path"].unique())
    logger.info(f"Found {len(video_paths)} unique videos in {args.labels_file}")

    # Initialize MMDet model
    try:
        from mmdet.apis import init_detector

        # RTMDet configs from MMDetection
        config_map = {
            "rtmdet-tiny": "rtmdet_tiny_8xb32-300e_coco",
            "rtmdet-s": "rtmdet_s_8xb32-300e_coco",
            "rtmdet-m": "rtmdet_m_8xb32-300e_coco",
            "rtmdet-l": "rtmdet_l_8xb32-300e_coco",
        }

        model_name = config_map.get(args.model, args.model)
        logger.info(f"Loading RTMDet model: {model_name}")

        # Try to use mim download for config + checkpoint
        try:
            from mmengine.config import Config
            from mmdet.utils import register_all_modules
            register_all_modules()

            # Use mim to get config and checkpoint paths
            import mim
            config_path = mim.get_model_config(model_name, "mmdet")
            checkpoint_path = mim.download(model_name, "mmdet")[0]
            model = init_detector(config_path, checkpoint_path, device=args.device)
        except Exception:
            # Fallback: try direct path
            logger.warning("mim download failed, trying direct initialization...")
            model = init_detector(model_name, device=args.device)

    except ImportError:
        logger.error(
            "mmdet is required for person detection. "
            "Install with: pip install mmengine mmdet"
        )
        sys.exit(1)

    # Process each video
    stats = {"total": 0, "two_persons": 0, "one_person": 0, "zero_persons": 0}

    for i, video_path in enumerate(video_paths, 1):
        video_stem = Path(video_path).stem
        output_path = output_dir / f"{video_stem}_person_bboxes.json"

        if args.skip_existing and output_path.exists():
            logger.info(f"[{i}/{len(video_paths)}] Skipping (exists): {video_stem}")
            continue

        if not Path(video_path).exists():
            logger.warning(f"[{i}/{len(video_paths)}] Video not found: {video_path}")
            continue

        logger.info(f"[{i}/{len(video_paths)}] Processing: {video_stem}")

        detections = detect_persons_in_video(
            video_path,
            model,
            confidence_threshold=args.confidence_threshold,
        )

        # Save JSON sidecar
        with open(output_path, "w") as f:
            json.dump(detections, f, indent=2)

        # Collect stats
        for second, dets in detections.items():
            stats["total"] += 1
            n = len(dets)
            if n >= 2:
                stats["two_persons"] += 1
            elif n == 1:
                stats["one_person"] += 1
            else:
                stats["zero_persons"] += 1

        logger.info(f"  Saved {len(detections)} seconds to {output_path}")

    # Report statistics
    total = stats["total"]
    if total > 0:
        logger.info("\n=== Detection Statistics ===")
        logger.info(f"Total frames analyzed: {total}")
        logger.info(f"  2+ persons detected: {stats['two_persons']} ({stats['two_persons']/total:.1%})")
        logger.info(f"  1 person detected:   {stats['one_person']} ({stats['one_person']/total:.1%})")
        logger.info(f"  0 persons detected:  {stats['zero_persons']} ({stats['zero_persons']/total:.1%})")


if __name__ == "__main__":
    main()
