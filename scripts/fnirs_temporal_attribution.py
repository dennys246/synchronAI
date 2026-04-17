#!/usr/bin/env python3
"""
Compute Integrated Gradients temporal attributions for fNIRS child/adult
classification.

Combines a pretrained per-pair encoder + trained classifier and traces
gradients from the prediction back to the raw HbO/HbR input signal,
showing which time points drive the child-vs-adult decision.

Usage:
    python scripts/fnirs_temporal_attribution.py \
        --encoder-weights runs/fnirs_perpair_small/fnirs_unet_encoder.pt \
        --classifier-weights runs/fnirs_perpair_sweep/small_lstm64/best.pt \
        --recording /path/to/50001_V0_fNIRS \
        --output-dir results/attributions/ \
        --plot

    # Multiple recordings:
    python scripts/fnirs_temporal_attribution.py \
        --encoder-weights runs/fnirs_perpair_small/fnirs_unet_encoder.pt \
        --classifier-weights runs/fnirs_perpair_sweep/small_lstm64/best.pt \
        --recording /path/to/50001_V0_fNIRS /path/to/5000_V0_fNIRS \
        --output-dir results/attributions/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute temporal attributions for fNIRS child/adult classification",
    )
    parser.add_argument(
        "--encoder-weights", required=True,
        help="Path to pretrained encoder checkpoint (.pt from convert_fnirs_tf_to_pt.py)",
    )
    parser.add_argument(
        "--classifier-weights", required=True,
        help="Path to trained classifier checkpoint (best.pt from train_fnirs_from_features.py)",
    )
    parser.add_argument(
        "--recording", required=True, nargs="+",
        help="Path(s) to fNIRS recording(s) (NIRx directory, .snirf, or .fif)",
    )
    parser.add_argument(
        "--output-dir", default="results/attributions",
        help="Output directory for attribution results and plots",
    )
    parser.add_argument(
        "--n-steps", type=int, default=100,
        help="Number of Integrated Gradients interpolation steps (default: 100)",
    )
    parser.add_argument(
        "--stride-seconds", type=float, default=60.0,
        help="Stride between windows in seconds (default: 60.0, non-overlapping)",
    )
    parser.add_argument(
        "--target-class", type=int, default=None, choices=[0, 1],
        help="Force attribution toward a specific class (0=child, 1=adult). "
             "Default: use the model's predicted class per window.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate attribution plots (recording overview + per-pair detail)",
    )
    parser.add_argument(
        "--plot-pairs", action="store_true",
        help="Also generate individual per-pair detail plots (larger output)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from synchronai.inference.fnirs.temporal_attribution import FnirsTemporalAttribution

    attr = FnirsTemporalAttribution(
        encoder_weights=args.encoder_weights,
        classifier_weights=args.classifier_weights,
        device=args.device,
        n_steps=args.n_steps,
    )

    for rec_path in args.recording:
        rec_name = Path(rec_path).name
        logger.info("Processing recording: %s", rec_name)

        try:
            result = attr.attribute_recording(
                fnirs_path=rec_path,
                stride_seconds=args.stride_seconds,
                target_class=args.target_class,
            )
        except Exception as e:
            logger.error("Failed to attribute %s: %s", rec_name, e)
            continue

        # Save result summary as JSON
        summary = {
            "recording_path": result.recording_path,
            "n_pairs": len(result.pairs),
            "input_length": result.input_length,
            "sfreq_hz": result.sfreq_hz,
            "pairs": {},
        }
        for pair_name, pr in result.pairs.items():
            summary["pairs"][pair_name] = {
                "n_windows": pr.n_windows,
                "predicted_class": pr.predicted_class,
                "avg_logit": float(sum(pr.prediction_logits) / len(pr.prediction_logits)),
                "window_average_abs_mean": float(abs(pr.window_average).mean()),
            }

        json_path = out_dir / f"{rec_name}_attribution.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("  Summary → %s", json_path)

        # Save the raw attribution arrays
        import numpy as np
        npz_path = out_dir / f"{rec_name}_attribution.npz"
        arrays = {}
        if result.recording_average is not None:
            arrays["recording_average"] = result.recording_average
        for pair_name, pr in result.pairs.items():
            safe_name = pair_name.replace(" ", "_")
            arrays[f"{safe_name}_average"] = pr.window_average
            for i, w in enumerate(pr.window_attributions):
                arrays[f"{safe_name}_window_{i:03d}"] = w
        np.savez_compressed(npz_path, **arrays)
        logger.info("  Arrays → %s", npz_path)

        # Plots
        if args.plot or args.plot_pairs:
            attr.plot_recording(
                result,
                save_path=str(out_dir / f"{rec_name}_attribution.png"),
            )

        if args.plot_pairs:
            for pair_name, pr in result.pairs.items():
                safe_name = pair_name.replace(" ", "_")
                attr.plot_pair(
                    pr,
                    save_path=str(out_dir / f"{rec_name}_{safe_name}_attribution.png"),
                )

    logger.info("All attributions saved to %s/", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
