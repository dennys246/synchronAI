#!/bin/bash
# Generate heatmap visualizations from video synchrony classifier
# Usage: ./scripts/generate_heatmap.sh <video_path> <weights_path> [output_dir]
#
# This script generates:
# - Timeline heatmap showing synchrony probability over time
# - Confidence distribution histogram
# - Segment summary bar chart
# - Thumbnail grid with overlays
# - Video with heatmap overlay on frames (basic color tint or Grad-CAM spatial heatmap)
# - Grad-CAM spatial heatmaps highlighting important video regions
# - JSON data export for external tools
#
# Grad-CAM mode creates spatial heatmaps that show WHICH REGIONS of each frame
# the model focuses on, rather than just tinting the entire frame by probability.

set -e  # Exit on error

# =============================================================================
# Argument Parsing
# =============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <video_path> <weights_path> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  video_path    Path to input video file"
    echo "  weights_path  Path to trained model checkpoint (.pt file)"
    echo "  output_dir    Output directory (default: runs/heatmaps/<video_name>)"
    echo ""
    echo "Examples:"
    echo "  $0 data/videos/test.mp4 runs/classifier/best.pt"
    echo "  $0 /path/to/video.mp4 /path/to/model.pt /path/to/output/"
    exit 1
fi

VIDEO_PATH="$1"
WEIGHTS_PATH="$2"

# Extract video name for default output directory
VIDEO_NAME=$(basename "${VIDEO_PATH}" | sed 's/\.[^.]*$//')

if [ $# -ge 3 ]; then
    OUTPUT_DIR="$3"
else
    OUTPUT_DIR="runs/heatmaps/${VIDEO_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

# =============================================================================
# Configuration
# =============================================================================

# Visualization options (space-separated)
# Options: timeline, grid, distribution, segments, data, video, gradcam, thumbnails, gradcam_thumbnails
# Use "all" for all static visualizations
# Use "all_with_video" for all including basic video overlay
# Use "all_with_gradcam" for all including Grad-CAM spatial heatmap overlay
VISUALIZATIONS="all_with_gradcam"

# Classification threshold
THRESHOLD=0.5

# Video overlay transparency (0=invisible, 1=opaque)
OVERLAY_ALPHA=0.4

# Colormap for heatmaps (RdYlGn, RdYlBu, coolwarm, viridis, plasma)
COLORMAP="RdYlGn"

# Grad-CAM settings
USE_GRADCAM="true"  # Set to "true" to enable spatial Grad-CAM heatmaps
GRADCAM_AGGREGATE="max"  # How to aggregate CAMs across frames: max, mean, weighted

# =============================================================================
# Validation
# =============================================================================

echo "=== Video Synchrony Heatmap Generator ==="
echo ""

if [ ! -f "${VIDEO_PATH}" ]; then
    echo "Error: Video file not found: ${VIDEO_PATH}"
    exit 1
fi

if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "Error: Weights file not found: ${WEIGHTS_PATH}"
    exit 1
fi

echo "Input video:    ${VIDEO_PATH}"
echo "Model weights:  ${WEIGHTS_PATH}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Visualizations: ${VISUALIZATIONS}"
echo "Threshold:      ${THRESHOLD}"
echo "Overlay alpha:  ${OVERLAY_ALPHA}"
echo "Colormap:       ${COLORMAP}"
echo "Use Grad-CAM:   ${USE_GRADCAM}"
echo "Grad-CAM agg:   ${GRADCAM_AGGREGATE}"
echo ""

# =============================================================================
# Generate Heatmaps
# =============================================================================

echo "=== Generating Heatmaps ==="
echo ""

# Build command with optional Grad-CAM flag
GRADCAM_FLAG=""
if [ "${USE_GRADCAM}" = "true" ]; then
    GRADCAM_FLAG="--use-gradcam --gradcam-aggregate ${GRADCAM_AGGREGATE}"
fi

python -m synchronai.main --heatmap \
    --video-path "${VIDEO_PATH}" \
    --weights-path "${WEIGHTS_PATH}" \
    --save-dir "${OUTPUT_DIR}" \
    --visualizations ${VISUALIZATIONS} \
    --threshold "${THRESHOLD}" \
    --overlay-alpha "${OVERLAY_ALPHA}" \
    --colormap "${COLORMAP}" \
    ${GRADCAM_FLAG}

echo ""
echo "=== Heatmap Generation Complete ==="
echo ""
echo "Output files saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated visualizations:"
ls -la "${OUTPUT_DIR}/"