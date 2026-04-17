"""
Integrated Gradients temporal attribution for fNIRS child/adult classification.

Composes a pretrained per-pair encoder + trained classifier into one
differentiable pipeline and computes attributions w.r.t. the raw HbO/HbR
input signal, showing which time points in the hemodynamic trace drive
the child-vs-adult prediction.

Usage:
    from synchronai.inference.fnirs.temporal_attribution import FnirsTemporalAttribution

    attr = FnirsTemporalAttribution(
        encoder_weights="runs/fnirs_perpair_small/fnirs_unet_encoder.pt",
        classifier_weights="runs/fnirs_perpair_sweep/small_lstm64/best.pt",
    )
    result = attr.attribute_recording("/path/to/50001_V0_fNIRS")
    attr.plot_recording(result, save_path="attribution.png")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalAttributionResult:
    """Attribution results for a single source-detector pair."""
    pair_name: str
    window_attributions: list[np.ndarray]   # each (input_length, 2)
    window_average: np.ndarray               # (input_length, 2)
    raw_windows: list[np.ndarray]            # each (input_length, 2) — normalized signal
    prediction_logits: list[float]
    predicted_class: str                     # "child" or "adult"

    @property
    def n_windows(self) -> int:
        return len(self.window_attributions)


@dataclass
class RecordingAttributionResult:
    """Attribution results for a full recording (all pairs, all windows)."""
    recording_path: str
    pairs: dict[str, TemporalAttributionResult]
    recording_average: np.ndarray | None     # (input_length, 2) — avg across all pairs
    sfreq_hz: float
    input_length: int                        # samples per window (e.g. 472)
    encoder_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: differentiable encoder + classifier pipeline
# ---------------------------------------------------------------------------

class _EncoderClassifierPipeline(nn.Module):
    """Wraps encoder + classifier for end-to-end gradient computation."""

    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_length, 2) — raw normalized HbO/HbR per pair
        Returns:
            (B,) logits
        """
        # t=None defaults to t=0 (fully denoised) inside the encoder
        features = self.encoder(x, t=None)          # (B, T', D)
        logits = self.classifier(features)           # (B, 1)
        return logits.squeeze(-1)                    # (B,)


# ---------------------------------------------------------------------------
# Main attribution class
# ---------------------------------------------------------------------------

class FnirsTemporalAttribution:
    """Compute Integrated Gradients attributions for per-pair fNIRS classification.

    Loads a pretrained per-pair encoder and a trained classifier, composes
    them into one differentiable pipeline, and attributes the classifier's
    prediction back to the raw (input_length, 2) HbO/HbR input.
    """

    def __init__(
        self,
        encoder_weights: str,
        classifier_weights: str,
        device: str = "cpu",
        n_steps: int = 100,
        batch_size: int = 16,
    ):
        self.device = torch.device(device)
        self.n_steps = n_steps
        self.batch_size = batch_size

        # ---- Load encoder ----
        save_data = torch.load(encoder_weights, map_location=device, weights_only=False)
        self.encoder_config = save_data["encoder_config"]
        self.diffusion_config = save_data["config"]

        from synchronai.models.fnirs.unet_encoder_pt import FnirsUNetEncoderPT
        encoder = FnirsUNetEncoderPT(
            input_length=self.encoder_config["input_length"],
            feature_dim=self.encoder_config["feature_dim"],
            base_width=self.encoder_config["base_width"],
            depth=self.encoder_config["depth"],
            time_embed_dim=self.encoder_config["time_embed_dim"],
            dropout=self.encoder_config["dropout"],
        )
        encoder.load_state_dict(save_data["state_dict"])
        encoder.eval()

        # ---- Load classifier ----
        # FnirsFeatureClassifier is defined in the training script — import it
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scripts.train_fnirs_from_features import FnirsFeatureClassifier

        cls_data = torch.load(classifier_weights, map_location=device, weights_only=False)
        cls_config = cls_data["config"]
        classifier = FnirsFeatureClassifier(
            feature_dim=cls_config["feature_dim"],
            hidden_dim=cls_config["hidden_dim"],
            dropout=cls_config["dropout"],
            pool=cls_config["pool"],
        )
        classifier.load_state_dict(cls_data["model_state_dict"])
        classifier.eval()

        # ---- Compose pipeline ----
        self.pipeline = _EncoderClassifierPipeline(encoder, classifier)
        self.pipeline.to(self.device)
        self.pipeline.eval()
        # Freeze all parameters — only the input tensor needs gradients
        for p in self.pipeline.parameters():
            p.requires_grad_(False)

        # ---- Normalization stats ----
        self.feature_mean = np.array(self.diffusion_config["feature_mean"], dtype=np.float32)
        self.feature_std = np.array(self.diffusion_config["feature_std"], dtype=np.float32)
        self.input_length = self.encoder_config["input_length"]
        self.sfreq_hz = self.diffusion_config["sfreq_hz"]

        logger.info(
            "Attribution pipeline loaded: encoder=%s (bottleneck=%d), "
            "classifier=%s (pool=%s, hidden=%d), input_length=%d",
            Path(encoder_weights).parent.name,
            self.encoder_config["bottleneck_dim"],
            Path(classifier_weights).parent.name,
            cls_config["pool"], cls_config["hidden_dim"],
            self.input_length,
        )

    def attribute_window(
        self,
        window: np.ndarray,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Compute Integrated Gradients for a single (input_length, 2) window.

        Args:
            window: (input_length, 2) normalized HbO/HbR signal.
            target_class: 0=child, 1=adult, None=use predicted class.

        Returns:
            (attribution, logit) where attribution is (input_length, 2).
        """
        x = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        baseline = torch.zeros_like(x)

        # If target_class not specified, use the model's prediction
        with torch.no_grad():
            logit = self.pipeline(x).item()
        if target_class is None:
            target_class = 1 if logit > 0 else 0

        # Integrated Gradients: average gradients along interpolation path
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        accumulated_grads = torch.zeros_like(x)

        for batch_start in range(0, len(alphas), self.batch_size):
            batch_alphas = alphas[batch_start:batch_start + self.batch_size]
            # (batch, input_length, 2)
            interpolated = baseline + batch_alphas[:, None, None] * (x - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            logits = self.pipeline(interpolated)  # (batch,)

            # Gradient direction: positive toward target class
            if target_class == 0:
                score = -logits.sum()
            else:
                score = logits.sum()

            score.backward()
            accumulated_grads += interpolated.grad.sum(dim=0, keepdim=True)

        # IG formula: (x - baseline) * avg_gradient
        avg_grads = accumulated_grads / (self.n_steps + 1)
        attribution = ((x - baseline) * avg_grads).squeeze(0)

        return attribution.detach().cpu().numpy(), logit

    def attribute_recording(
        self,
        fnirs_path: str,
        stride_seconds: float = 60.0,
        target_class: int | None = None,
        signal_type: str = "hemodynamic",
    ) -> RecordingAttributionResult:
        """Compute attributions for all pairs and windows in a recording.

        Args:
            fnirs_path: Path to fNIRS recording (NIRx dir, .snirf, or .fif).
            stride_seconds: Stride between windows (60.0 = non-overlapping).
            target_class: 0=child, 1=adult, None=use predicted class per window.
            signal_type: 'hemodynamic' (default) or 'neural'.

        Returns:
            RecordingAttributionResult with per-pair and recording-level averages.
        """
        # Import here to avoid circular imports at module load
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scripts.extract_fnirs_features import (
            load_and_normalize_recording,
            window_recording,
        )

        pairs_data = load_and_normalize_recording(
            fnirs_path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_pairs=self.diffusion_config["pair_names"],
            hb_types=self.diffusion_config["hb_types"],
            target_sfreq=self.sfreq_hz,
            duration_seconds=self.diffusion_config["duration_seconds"],
            per_pair=True,
            signal_type=signal_type,
        )

        if pairs_data is None:
            raise ValueError(f"Failed to load recording: {fnirs_path}")

        pair_results: dict[str, TemporalAttributionResult] = {}
        all_averages = []

        for pair_name, pair_array in pairs_data:
            windows = window_recording(
                pair_array, self.input_length, stride_seconds, self.sfreq_hz,
            )
            if not windows:
                logger.warning("Pair %s: no windows, skipping", pair_name)
                continue

            # Check for zero-variance pairs
            pair_var = np.var(np.stack(windows))
            if pair_var < 1e-10:
                logger.warning("Pair %s: near-zero variance (%.2e), skipping", pair_name, pair_var)
                continue

            window_attrs = []
            raw_windows = []
            logits = []

            for window in windows:
                attr, logit = self.attribute_window(window, target_class=target_class)
                window_attrs.append(attr)
                raw_windows.append(window.copy())
                logits.append(logit)

            avg_attr = np.mean(window_attrs, axis=0)
            avg_logit = np.mean(logits)
            predicted = "adult" if avg_logit > 0 else "child"

            pair_result = TemporalAttributionResult(
                pair_name=pair_name,
                window_attributions=window_attrs,
                window_average=avg_attr,
                raw_windows=raw_windows,
                prediction_logits=logits,
                predicted_class=predicted,
            )
            pair_results[pair_name] = pair_result
            all_averages.append(avg_attr)

            logger.info(
                "  %s: %d windows, predicted=%s (avg logit=%.3f)",
                pair_name, len(windows), predicted, avg_logit,
            )

        # Recording-level average across pairs
        recording_avg = np.mean(all_averages, axis=0) if all_averages else None

        return RecordingAttributionResult(
            recording_path=fnirs_path,
            pairs=pair_results,
            recording_average=recording_avg,
            sfreq_hz=self.sfreq_hz,
            input_length=self.input_length,
            encoder_config=self.encoder_config,
        )

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    def plot_pair(
        self,
        pair_result: TemporalAttributionResult,
        sfreq_hz: float | None = None,
        save_path: str | None = None,
        show_windows: bool = False,
    ):
        """Detailed attribution plot for a single pair.

        Two panels:
          Top: raw HbO/HbR signal with attribution heatmap background
          Bottom: HbO and HbR attribution magnitude (filled curves)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sfreq = sfreq_hz or self.sfreq_hz
        n_samples = pair_result.window_average.shape[0]
        time_sec = np.arange(n_samples) / sfreq

        attr = pair_result.window_average   # (input_length, 2)
        raw = np.mean(pair_result.raw_windows, axis=0)  # (input_length, 2)

        fig, (ax_sig, ax_attr) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Top: raw signal with attribution-colored background
        attr_mag = np.abs(attr).sum(axis=1)  # (input_length,)
        attr_mag_norm = attr_mag / (attr_mag.max() + 1e-8)

        ax_sig.fill_between(time_sec, raw[:, 0].min(), raw[:, 0].max(),
                            alpha=attr_mag_norm * 0.4,
                            color="orange", label="attribution intensity")
        ax_sig.plot(time_sec, raw[:, 0], color="red", linewidth=1.2, label="HbO")
        ax_sig.plot(time_sec, raw[:, 1], color="blue", linewidth=1.2, label="HbR")
        ax_sig.set_ylabel("Normalized signal")
        ax_sig.legend(fontsize=8, loc="upper right")
        ax_sig.set_title(
            f"Pair {pair_result.pair_name} — predicted: {pair_result.predicted_class} "
            f"(avg logit: {np.mean(pair_result.prediction_logits):.3f})"
        )
        ax_sig.grid(alpha=0.2)

        # Bottom: attribution detail per channel
        ax_attr.fill_between(time_sec, 0, attr[:, 0],
                             where=attr[:, 0] > 0, color="red", alpha=0.4, label="HbO +")
        ax_attr.fill_between(time_sec, 0, attr[:, 0],
                             where=attr[:, 0] < 0, color="red", alpha=0.15, label="HbO −")
        ax_attr.fill_between(time_sec, 0, attr[:, 1],
                             where=attr[:, 1] > 0, color="blue", alpha=0.4, label="HbR +")
        ax_attr.fill_between(time_sec, 0, attr[:, 1],
                             where=attr[:, 1] < 0, color="blue", alpha=0.15, label="HbR −")
        ax_attr.axhline(0, color="gray", linewidth=0.5)
        ax_attr.set_xlabel("Time (seconds)")
        ax_attr.set_ylabel("Attribution")
        ax_attr.legend(fontsize=7, ncol=4, loc="upper right")
        ax_attr.grid(alpha=0.2)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved pair attribution plot → %s", save_path)
        plt.close(fig)
        return fig

    def plot_recording(
        self,
        result: RecordingAttributionResult,
        save_path: str | None = None,
        max_pairs: int = 12,
    ):
        """Multi-pair attribution overview as a grid of heatmap strips."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pairs = list(result.pairs.items())[:max_pairs]
        n_pairs = len(pairs)
        if n_pairs == 0:
            logger.warning("No pairs to plot")
            return None

        time_sec = np.arange(result.input_length) / result.sfreq_hz

        fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 1.8 * n_pairs), sharex=True)
        if n_pairs == 1:
            axes = [axes]

        vmax = max(
            np.abs(pr.window_average).max()
            for _, pr in pairs
        )

        for ax, (pair_name, pr) in zip(axes, pairs):
            attr = pr.window_average.T  # (2, input_length) — HbO, HbR
            ax.imshow(
                attr, aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                extent=[time_sec[0], time_sec[-1], -0.5, 1.5],
            )
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["HbR", "HbO"], fontsize=8)
            ax.set_ylabel(pair_name, fontsize=8, rotation=0, labelpad=50, va="center")
            logit = np.mean(pr.prediction_logits)
            pred = pr.predicted_class
            ax.text(
                0.99, 0.95, f"{pred} ({logit:+.2f})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

        axes[-1].set_xlabel("Time (seconds)")

        recording_name = Path(result.recording_path).name
        fig.suptitle(
            f"Temporal attribution — {recording_name}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved recording attribution plot → %s", save_path)
        plt.close(fig)
        return fig
