#!/bin/bash
# Audio synchrony classification script
# Processes video/audio files through the Whisper-based audio classifier

set -e

# Configuration - update these paths as needed
INPUT_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"
OUTPUT_DIRECTORY="data/audio_sync/"
WEIGHTS_PATH="runs/audio_classifier/best.pt"

# Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
WHISPER_MODEL="large-v3"

# Output format (csv, json, both)
OUTPUT_FORMAT="both"

echo "=========================================="
echo "Audio Synchrony Classification"
echo "=========================================="
echo "Input directory: $INPUT_DIRECTORY"
echo "Output directory: $OUTPUT_DIRECTORY"
echo "Weights: $WEIGHTS_PATH"
echo "Whisper model: $WHISPER_MODEL"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIRECTORY"

# Run audio classification
python -m synchronai.main \
    --audio \
    --mode predict \
    --audio-dir "$INPUT_DIRECTORY" \
    --save-dir "$OUTPUT_DIRECTORY" \
    --weights-path "$WEIGHTS_PATH" \
    --whisper-model "$WHISPER_MODEL" \
    --output-format "$OUTPUT_FORMAT" \
    --skip-existing

echo "=========================================="
echo "Audio classification complete!"
echo "Results saved to: $OUTPUT_DIRECTORY"
echo "=========================================="
