#!/usr/bin/env python3
"""
Convert TensorFlow fNIRS U-Net weights to PyTorch encoder state dict.

Reads Keras .weights.h5 file directly via h5py — NO TensorFlow dependency.
This avoids TF/PT environment conflicts entirely.

Usage:
    python scripts/convert_fnirs_tf_to_pt.py \
        --config-json runs/fnirs_diffusion_v3/fnirs_diffusion_config.json \
        --weights-path runs/fnirs_diffusion_v3/fnirs_unet.weights.h5 \
        --output runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt \
        --verify

Weight transposition rules:
    TF Conv1D kernel: (K, C_in, C_out) -> PyTorch: (C_out, C_in, K)
    TF Dense kernel:  (in, out)         -> PyTorch: (out, in)
    TF Dense bias:    (out,)            -> PyTorch: (out,)  [no change]
    TF LayerNorm gamma: (D,)            -> PyTorch weight: (D,)  [no change]
    TF LayerNorm beta:  (D,)            -> PyTorch bias: (D,)    [no change]

Requires: h5py, numpy, torch (all in ml-env, no tensorflow needed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_h5_weights(weights_path: str) -> list[tuple[str, np.ndarray]]:
    """Load all weight arrays from a Keras .weights.h5 file.

    Keras H5 format stores weights in groups organized by layer name.
    Each layer group contains datasets named 'kernel', 'bias', 'gamma',
    'beta', etc.

    Returns list of (full_path, array) tuples in file order.
    """
    weights = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            weights.append((name, obj[()]))

    with h5py.File(weights_path, "r") as f:
        f.visititems(visitor)

    logger.info(f"Loaded {len(weights)} weight arrays from {weights_path}")
    for path, arr in weights:
        logger.debug(f"  {path}: shape={arr.shape}, dtype={arr.dtype}")

    return weights


def _layer_sort_key(layer_name: str) -> tuple:
    """Sort Keras layer names by numeric suffix to match creation order.

    Keras names layers like: conv1d, conv1d_1, conv1d_2, ..., conv1d_10
    Alphabetical sort puts conv1d_10 before conv1d_2. We need numeric order.

    Returns a tuple (base_name, numeric_index) for proper sorting.
    """
    import re
    # Split "conv1d_12" into ("conv1d", 12)
    # "conv1d" (no suffix) -> ("conv1d", -1) so it sorts first
    match = re.match(r"^(.+?)_(\d+)$", layer_name)
    if match:
        return (match.group(1), int(match.group(2)))
    return (layer_name, -1)


def classify_weights(
    h5_weights: list[tuple[str, np.ndarray]],
) -> tuple[list, list, list]:
    """Classify H5 weights into Conv1D, Dense, and LayerNorm groups.

    Keras H5 paths look like:
        layers/conv1d/vars/0          (kernel)
        layers/conv1d/vars/1          (bias)
        layers/dense/vars/0           (kernel)
        layers/dense/vars/1           (bias)
        layers/layer_normalization/vars/0  (gamma)
        layers/layer_normalization/vars/1  (beta)

    IMPORTANT: H5 groups are stored alphabetically, NOT in creation order.
    conv1d_10 comes before conv1d_2 alphabetically. We must sort by the
    numeric suffix to recover the original TF layer creation order.
    """
    conv1d_weights = []   # [(kernel, bias), ...]
    dense_weights = []    # [(kernel, bias), ...]
    ln_weights = []       # [(gamma, beta), ...]

    # Group by layer path (everything except the last /vars/N)
    layer_groups = {}
    for path, arr in h5_weights:
        parts = path.rsplit("/", 2)
        if len(parts) >= 3 and parts[-2] == "vars":
            layer_path = "/".join(parts[:-2])
            var_idx = int(parts[-1])
        else:
            layer_path = "/".join(path.split("/")[:-1])
            var_idx = 0

        if layer_path not in layer_groups:
            layer_groups[layer_path] = {}
        layer_groups[layer_path][var_idx] = arr

    # Sort layer paths by numeric suffix to match TF creation order
    sorted_layer_paths = sorted(
        layer_groups.keys(),
        key=lambda p: _layer_sort_key(p.split("/")[-1])
    )

    for layer_path in sorted_layer_paths:
        group = layer_groups[layer_path]
        layer_name = layer_path.split("/")[-1].lower()

        kernel = group.get(0)
        bias_or_beta = group.get(1)

        if kernel is None:
            continue

        if "conv1d" in layer_name:
            conv1d_weights.append((kernel, bias_or_beta))
            logger.debug(f"  Conv1D: {layer_path} kernel={kernel.shape}")
        elif "dense" in layer_name:
            dense_weights.append((kernel, bias_or_beta))
            logger.debug(f"  Dense: {layer_path} kernel={kernel.shape}")
        elif "layer_normalization" in layer_name or "layer_norm" in layer_name:
            ln_weights.append((kernel, bias_or_beta))
            logger.debug(f"  LayerNorm: {layer_path} gamma={kernel.shape}")

    logger.info(
        f"Classified: {len(conv1d_weights)} Conv1D, "
        f"{len(dense_weights)} Dense, {len(ln_weights)} LayerNorm"
    )

    # Log shapes for debugging
    for i, (k, b) in enumerate(conv1d_weights):
        logger.info(f"  Conv1D[{i}]: kernel={k.shape}" + (f" bias={b.shape}" if b is not None else ""))
    for i, (k, b) in enumerate(dense_weights):
        logger.info(f"  Dense[{i}]: kernel={k.shape}" + (f" bias={b.shape}" if b is not None else ""))

    return conv1d_weights, dense_weights, ln_weights


def build_pytorch_state_dict(
    conv1d_weights: list,
    dense_weights: list,
    ln_weights: list,
    config_dict: dict,
) -> dict[str, torch.Tensor]:
    """Map classified TF weights to PyTorch state dict keys.

    The TF functional API creates layers in a deterministic order matching
    how build_unet_1d() calls them. We consume weights in that same order.
    """
    state_dict = {}
    conv_idx = 0
    dense_idx = 0
    ln_idx = 0

    def add_conv(prefix: str):
        nonlocal conv_idx
        kernel, bias = conv1d_weights[conv_idx]
        conv_idx += 1
        # TF Conv1D: (K, C_in, C_out) -> PT: (C_out, C_in, K)
        state_dict[f"{prefix}.weight"] = torch.tensor(kernel.transpose(2, 1, 0).copy())
        if bias is not None:
            state_dict[f"{prefix}.bias"] = torch.tensor(bias.copy())

    def add_dense(prefix: str):
        nonlocal dense_idx
        kernel, bias = dense_weights[dense_idx]
        dense_idx += 1
        # TF Dense: (in, out) -> PT: (out, in)
        state_dict[f"{prefix}.weight"] = torch.tensor(kernel.T.copy())
        if bias is not None:
            state_dict[f"{prefix}.bias"] = torch.tensor(bias.copy())

    def add_ln(prefix: str):
        nonlocal ln_idx
        gamma, beta = ln_weights[ln_idx]
        ln_idx += 1
        state_dict[f"{prefix}.weight"] = torch.tensor(gamma.copy())
        if beta is not None:
            state_dict[f"{prefix}.bias"] = torch.tensor(beta.copy())

    def add_resblock(prefix: str, has_residual_proj: bool):
        """Map one ResBlock's weights in TF layer order."""
        add_conv(f"{prefix}.conv1")       # Conv1D(3, same)
        add_ln(f"{prefix}.norm1")         # LayerNorm
        add_dense(f"{prefix}.temb_proj")  # Dense(temb -> filters)
        add_conv(f"{prefix}.conv2")       # Conv1D(3, same)
        add_ln(f"{prefix}.norm2")         # LayerNorm
        if has_residual_proj:
            add_conv(f"{prefix}.residual_proj")  # Conv1D(1) for channel mismatch

    # --- Timestep embedding MLP ---
    add_dense("temb_dense1")  # Dense(time_embed_dim * 4)
    add_dense("temb_dense2")  # Dense(time_embed_dim)

    # --- Encoder down-path ---
    depth = int(config_dict.get("unet_depth", 3))
    base_width = int(config_dict.get("unet_base_width", 64))
    feature_dim = int(config_dict["feature_dim"])
    widths = [base_width * (2 ** i) for i in range(depth)]

    prev_ch = feature_dim
    for level, filters in enumerate(widths):
        has_proj = prev_ch != filters
        add_resblock(f"encoder_blocks.{level}", has_residual_proj=has_proj)
        add_conv(f"downsample_convs.{level}")  # Conv1D(k=4, s=2)
        prev_ch = filters

    # --- Bottleneck ---
    bottleneck_ch = widths[-1] * 2
    add_resblock("bottleneck1", has_residual_proj=(prev_ch != bottleneck_ch))
    add_resblock("bottleneck2", has_residual_proj=False)  # Same channels

    logger.info(
        f"Consumed: {conv_idx}/{len(conv1d_weights)} Conv1D, "
        f"{dense_idx}/{len(dense_weights)} Dense, "
        f"{ln_idx}/{len(ln_weights)} LayerNorm"
    )
    logger.info(f"  (Remaining are decoder weights — not needed for encoder)")

    return state_dict


def verify_conversion(
    pt_state_dict: dict,
    config_dict: dict,
) -> bool:
    """Verify the PT encoder produces valid output (shape + sanity checks)."""
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from synchronai.models.fnirs.unet_encoder_pt import FnirsUNetEncoderPT

    model_len = int(config_dict["model_len"])
    feature_dim = int(config_dict["feature_dim"])
    depth = int(config_dict.get("unet_depth", 3))
    base_width = int(config_dict.get("unet_base_width", 64))

    pt_encoder = FnirsUNetEncoderPT(
        input_length=model_len,
        feature_dim=feature_dim,
        base_width=base_width,
        depth=depth,
        time_embed_dim=int(config_dict.get("unet_time_embed_dim", 128)),
        dropout=float(config_dict.get("unet_dropout", 0.15)),
    )
    pt_encoder.load_state_dict(pt_state_dict)
    pt_encoder.eval()

    # Random input
    np.random.seed(42)
    x_np = np.random.randn(2, model_len, feature_dim).astype(np.float32)

    x_pt = torch.tensor(x_np)
    with torch.no_grad():
        pt_output = pt_encoder(x_pt)
    pt_output_np = pt_output.numpy()

    # Check shape
    expected_t = model_len
    for _ in range(depth):
        expected_t = (expected_t + 1) // 2
    expected_shape = (2, expected_t, base_width * (2 ** depth))

    if pt_output_np.shape != expected_shape:
        logger.error(f"Shape mismatch! Expected {expected_shape}, got {pt_output_np.shape}")
        return False

    logger.info(f"Output shape correct: {pt_output_np.shape}")

    if np.any(np.isnan(pt_output_np)):
        logger.error("Output contains NaN!")
        return False

    if np.allclose(pt_output_np, 0):
        logger.error("Output is all zeros!")
        return False

    logger.info(
        f"Output stats: mean={pt_output_np.mean():.6f}, "
        f"std={pt_output_np.std():.6f}, "
        f"min={pt_output_np.min():.6f}, max={pt_output_np.max():.6f}"
    )

    # Also test multi-scale output
    with torch.no_grad():
        multi = pt_encoder(x_pt, return_all_levels=True)
    logger.info(f"Multi-scale outputs: {', '.join(f'{k}: {v.shape}' for k, v in multi.items())}")

    logger.info("Verification PASSED (shape + sanity checks)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert TF fNIRS U-Net weights to PyTorch encoder state dict. "
                    "Uses h5py only — no TensorFlow dependency."
    )
    parser.add_argument(
        "--config-json", required=True,
        help="Path to fnirs_diffusion_config.json",
    )
    parser.add_argument(
        "--weights-path", required=True,
        help="Path to fnirs_unet.weights.h5",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .pt file (default: same dir as config, fnirs_unet_encoder.pt)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run shape and sanity verification after conversion",
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.config_json).parent / "fnirs_unet_encoder.pt")

    # Load config
    with open(args.config_json) as f:
        config_dict = json.load(f)

    # Resolve weights path
    weights_p = Path(args.weights_path)
    if not weights_p.is_absolute():
        weights_p = Path(args.config_json).parent / weights_p
    weights_path = str(weights_p)

    # Load and classify weights from H5 (no TensorFlow needed!)
    logger.info(f"Loading weights from {weights_path} via h5py...")
    h5_weights = load_h5_weights(weights_path)

    logger.info("Classifying weights by layer type...")
    conv1d_weights, dense_weights, ln_weights = classify_weights(h5_weights)

    logger.info("Building PyTorch state dict...")
    state_dict = build_pytorch_state_dict(
        conv1d_weights, dense_weights, ln_weights, config_dict
    )

    # Save
    depth = int(config_dict.get("unet_depth", 3))
    base_width = int(config_dict.get("unet_base_width", 64))
    save_data = {
        "state_dict": state_dict,
        "config": config_dict,
        "encoder_config": {
            "input_length": int(config_dict["model_len"]),
            "feature_dim": int(config_dict["feature_dim"]),
            "base_width": base_width,
            "depth": depth,
            "time_embed_dim": int(config_dict.get("unet_time_embed_dim", 128)),
            "dropout": float(config_dict.get("unet_dropout", 0.15)),
            "bottleneck_dim": base_width * (2 ** depth),
            "multiscale_dim": (
                base_width * (2 ** depth)
                + sum(base_width * (2 ** i) for i in range(depth))
            ),
        },
    }
    torch.save(save_data, output_path)
    logger.info(f"Saved PyTorch encoder to: {output_path}")
    logger.info(f"State dict keys: {len(state_dict)}")

    # Verify
    if args.verify:
        logger.info("Running verification...")
        ok = verify_conversion(state_dict, config_dict)
        if ok:
            logger.info("Conversion verified successfully!")
        else:
            logger.error("Verification FAILED — check weight mapping")
            sys.exit(1)


if __name__ == "__main__":
    main()
