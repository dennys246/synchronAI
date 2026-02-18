"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for video synchrony classifier.

Generates spatial heatmaps showing which regions of each frame contributed
most to the model's synchrony prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization" (ICCV 2017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM generation."""

    # Target layer index in backbone (negative = from end)
    target_layer_idx: int = -1

    # Colormap for visualization
    colormap: int = cv2.COLORMAP_JET

    # Blend alpha for overlay (0=only original, 1=only heatmap)
    alpha: float = 0.4

    # Whether to apply ReLU to final CAM (standard Grad-CAM)
    relu: bool = True

    # Minimum activation threshold (below this = transparent)
    min_threshold: float = 0.0

    # Whether to normalize CAM to [0, 1]
    normalize: bool = True


class GradCAM:
    """
    Grad-CAM implementation for the VideoClassifier model.

    Hooks into the backbone (DINOv2 transformer layers or YOLO conv layers)
    to capture activations and gradients, then computes class activation maps
    showing important spatial regions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[GradCAMConfig] = None,
    ):
        """
        Initialize Grad-CAM extractor.

        Args:
            model: VideoClassifier model
            config: Grad-CAM configuration
        """
        self.model = model
        self.config = config or GradCAMConfig()

        # Storage for activations and gradients
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Detect backbone type
        self._backbone_type = self._detect_backbone_type()

        # Register hooks
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _detect_backbone_type(self) -> str:
        """Detect whether the model uses DINOv2 or YOLO backbone."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        if isinstance(self.model.feature_extractor, DINOv2FeatureExtractor):
            return "dinov2"
        return "yolo"

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        if self._backbone_type == "dinov2":
            self._register_hooks_dinov2()
        else:
            self._register_hooks_yolo()

    def _register_hooks_dinov2(self) -> None:
        """Register hooks on DINOv2 transformer encoder layer."""
        extractor = self.model.feature_extractor
        extractor._load_model()  # Ensure model is loaded

        # Hook into transformer encoder layer
        encoder_layers = extractor.dinov2.encoder.layer
        target_layer = encoder_layers[self.config.target_layer_idx]

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            # DINOv2 encoder layer output is a tuple (hidden_states, ...)
            if isinstance(output, (list, tuple)):
                self._activations = output[0].detach()
            else:
                self._activations = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (list, tuple)):
                self._gradients = grad_output[0].detach()
            else:
                self._gradients = grad_output.detach()

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        self._hook_handles = [handle_fwd, handle_bwd]
        logger.debug(f"Registered DINOv2 Grad-CAM hooks on layer: {type(target_layer).__name__}")

    def _register_hooks_yolo(self) -> None:
        """Register hooks on YOLO backbone conv layers."""
        # Get the YOLO backbone layers
        backbone = self.model.feature_extractor.backbone

        # Find target layer (last conv layer by default)
        target_layer = backbone[self.config.target_layer_idx]

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            # Handle different output types from YOLO layers
            if isinstance(output, (list, tuple)):
                self._activations = output[0].detach()
            else:
                self._activations = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (list, tuple)):
                self._gradients = grad_output[0].detach()
            else:
                self._gradients = grad_output.detach()

        # Register hooks
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        self._hook_handles = [handle_fwd, handle_bwd]
        logger.debug(f"Registered YOLO Grad-CAM hooks on layer: {type(target_layer).__name__}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()

    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute the class activation map from activations and gradients.

        Args:
            activations: Feature map activations (C, H, W)
            gradients: Gradients of target w.r.t. activations (C, H, W)

        Returns:
            CAM heatmap (H, W) normalized to [0, 1]
        """
        # Global average pooling of gradients -> channel weights
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)

        # Apply ReLU if configured (standard Grad-CAM)
        if self.config.relu:
            cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        if self.config.normalize and cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        # Apply minimum threshold
        if self.config.min_threshold > 0:
            cam[cam < self.config.min_threshold] = 0

        return cam

    def generate_cam(
        self,
        frames: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of frames.

        Args:
            frames: Input frames (batch, n_frames, C, H, W) or (n_frames, C, H, W)
            target_class: Class to compute CAM for (None = predicted class)

        Returns:
            List of CAM heatmaps, one per frame (each H, W array in [0, 1])
        """
        if self._backbone_type == "dinov2":
            return self._generate_cam_dinov2(frames, target_class)
        else:
            return self._generate_cam_yolo(frames, target_class)

    def _generate_cam_dinov2(
        self,
        frames: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Generate Grad-CAM heatmaps using DINOv2 backbone.

        DINOv2 outputs patch tokens (B, 257, D) where index 0 is CLS.
        We reshape the 256 patch tokens to a 16x16 spatial grid for the CAM.
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)

        batch_size, n_frames, C, H, W = frames.shape
        cams = []

        self.model.eval()

        for frame_idx in range(n_frames):
            frame_input = frames[:, frame_idx, :, :, :].requires_grad_(True)

            self._activations = None
            self._gradients = None

            # Forward through DINOv2 feature extractor (hooks capture activations)
            features = self.model.feature_extractor(frame_input)  # (1, D)

            # Simplified temporal + head for single-frame CAM
            features_temporal = features.unsqueeze(1)  # (1, 1, D)
            aggregated = features_temporal.mean(dim=1)  # (1, D)
            logits = self.model.head(aggregated)

            # Backward pass
            self.model.zero_grad()

            if target_class is None:
                target = logits
            else:
                target = logits[:, target_class] if logits.dim() > 1 else logits

            target.backward(retain_graph=True)

            if self._activations is None or self._gradients is None:
                logger.warning(f"No activations/gradients captured for frame {frame_idx}")
                cams.append(np.zeros((H, W), dtype=np.float32))
                continue

            # DINOv2 encoder layer output: (B, 257, D) — strip CLS token
            act_patches = self._activations[0, 1:, :]  # (256, D)
            grad_patches = self._gradients[0, 1:, :]  # (256, D)

            # Reshape to spatial grid: (256, D) → (16, 16, D) → (D, 16, 16)
            h = w = int(act_patches.shape[0] ** 0.5)
            act_spatial = act_patches.reshape(h, w, -1).permute(2, 0, 1)
            grad_spatial = grad_patches.reshape(h, w, -1).permute(2, 0, 1)

            # Standard Grad-CAM computation
            cam = self._compute_cam(act_spatial, grad_spatial)

            # Resize to original frame size
            cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
            cams.append(cam_resized)

        return cams

    def _generate_cam_yolo(
        self,
        frames: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Generate Grad-CAM heatmaps using YOLO backbone (legacy)."""
        # Ensure batch dimension
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)

        batch_size, n_frames, C, H, W = frames.shape
        cams = []

        self.model.eval()

        for frame_idx in range(n_frames):
            frame_input = frames[:, frame_idx, :, :, :].requires_grad_(True)

            self._activations = None
            self._gradients = None

            # Run forward pass through just the backbone
            x = frame_input
            for layer in self.model.feature_extractor.backbone:
                x = layer(x)

            # Store pre-pooled activations
            pre_pool_activations = self._activations

            if pre_pool_activations is None:
                logger.warning(f"No activations captured for frame {frame_idx}")
                cams.append(np.zeros((H, W), dtype=np.float32))
                continue

            # Global average pool then run through temporal + head
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)

            # Unsqueeze for temporal dimension (1, 1, features)
            features = pooled.unsqueeze(1)

            # Temporal aggregation (simplified - use mean for CAM)
            aggregated = features.mean(dim=1)

            # Classification head
            logits = self.model.head(aggregated)

            # Backward pass
            self.model.zero_grad()

            if target_class is None:
                target = logits
            else:
                target = logits[:, target_class] if logits.dim() > 1 else logits

            target.backward(retain_graph=True)

            if self._gradients is None:
                logger.warning(f"No gradients captured for frame {frame_idx}")
                cams.append(np.zeros((H, W), dtype=np.float32))
                continue

            # Compute CAM for this frame
            cam = self._compute_cam(
                pre_pool_activations[0],  # Remove batch dim
                self._gradients[0],
            )

            # Resize to original frame size
            cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
            cams.append(cam_resized)

        return cams

    def generate_cam_for_window(
        self,
        frames: torch.Tensor,
        aggregate: str = "max",
    ) -> np.ndarray:
        """
        Generate a single aggregated CAM for a window of frames.

        Args:
            frames: Input frames (batch, n_frames, C, H, W) or (n_frames, C, H, W)
            aggregate: How to aggregate per-frame CAMs ("max", "mean", "weighted")

        Returns:
            Single aggregated CAM heatmap (H, W) in [0, 1]
        """
        cams = self.generate_cam(frames)

        if not cams:
            return np.zeros((frames.shape[-2], frames.shape[-1]), dtype=np.float32)

        cams_array = np.stack(cams, axis=0)  # (n_frames, H, W)

        if aggregate == "max":
            aggregated = np.max(cams_array, axis=0)
        elif aggregate == "mean":
            aggregated = np.mean(cams_array, axis=0)
        elif aggregate == "weighted":
            # Weight by activation intensity
            weights = cams_array.sum(axis=(1, 2), keepdims=True)
            weights = weights / (weights.sum() + 1e-8)
            aggregated = (cams_array * weights).sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")

        # Re-normalize
        if aggregated.max() > 0:
            aggregated = aggregated / aggregated.max()

        return aggregated


def apply_cam_to_frame(
    frame: np.ndarray,
    cam: np.ndarray,
    config: Optional[GradCAMConfig] = None,
) -> np.ndarray:
    """
    Apply Grad-CAM heatmap overlay to a frame.

    Args:
        frame: Original frame (H, W, C) in BGR format (uint8)
        cam: CAM heatmap (H, W) normalized to [0, 1]
        config: Grad-CAM configuration

    Returns:
        Frame with heatmap overlay (H, W, C) in BGR format
    """
    if config is None:
        config = GradCAMConfig()

    # Ensure CAM is correct size
    if cam.shape[:2] != frame.shape[:2]:
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    # Convert CAM to uint8 for colormap
    cam_uint8 = (cam * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, config.colormap)

    # Blend with original frame
    overlay = cv2.addWeighted(frame, 1 - config.alpha, heatmap, config.alpha, 0)

    return overlay


def create_cam_comparison(
    frame: np.ndarray,
    cam: np.ndarray,
    probability: float,
    prediction: int,
    config: Optional[GradCAMConfig] = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison showing original frame and CAM overlay.

    Args:
        frame: Original frame (H, W, C) in BGR
        cam: CAM heatmap (H, W) in [0, 1]
        probability: Synchrony probability
        prediction: Binary prediction (0 or 1)
        config: Grad-CAM configuration

    Returns:
        Combined image with original and overlay side by side
    """
    if config is None:
        config = GradCAMConfig()

    h, w = frame.shape[:2]

    # Create overlay
    overlay = apply_cam_to_frame(frame, cam, config)

    # Create side-by-side
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = frame
    combined[:, w:] = overlay

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 800)
    thickness = max(1, int(font_scale * 2))

    # Label for original
    cv2.putText(combined, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)

    # Label for overlay with prediction
    label = "SYNC" if prediction == 1 else "ASYNC"
    color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
    text = f"Grad-CAM ({label}: {probability:.1%})"
    cv2.putText(combined, text, (w + 10, 30), font, font_scale, color, thickness)

    return combined