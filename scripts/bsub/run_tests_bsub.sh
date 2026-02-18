#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -J synchronai-tests
#BSUB -o scripts/bsub/logs/tests_%J.stdout
#BSUB -e scripts/bsub/logs/tests_%J.stderr
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# Save LSF working directory before .bashrc may change it
PROJECT_DIR="$(pwd)"

source /home/$USER/.bashrc
source "${SYNCHRONAI_DIR:-$PROJECT_DIR}"/ml-env/bin/activate

cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"

# =============================================================================
# Install Dependencies
# =============================================================================

echo "=== Installing test dependencies ==="

# Fix NumPy installation
pip install --force-reinstall --no-cache-dir "numpy>=2.0,<2.5"

# Clear any corrupted whisper wheel cache
rm -rf /home/$USER/.cache/pip/wheels/*/openai_whisper* 2>/dev/null || true

# Install audio dependencies
pip install --no-cache-dir openai-whisper soundfile imageio-ffmpeg
pip install --no-cache-dir transformers  # WavLM encoder support

# Install synchronAI package
pip install -e .

# Fix OpenCV for headless Docker container
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install --force-reinstall opencv-python-headless

# Install pytest
pip install --no-cache-dir pytest

# =============================================================================
# Run Tests
# =============================================================================

echo ""
echo "=== Running synchronAI Test Suite ==="
echo "Date: $(date)"
echo ""

# Run all tests by default, or specific tests if passed as arguments
# Usage:
#   bsub < scripts/bsub/run_tests_bsub.sh                    # run all tests
#   bsub -env "TEST_ARGS=tests/test_wavlm_encoder.py" < ...   # run specific test file
#   bsub -env "TEST_ARGS=-k wavlm" < ...                      # run tests matching pattern

# Default: run all tests (DINOv2 tests are mocked and lightweight)
# YOLO-specific tests in test_video_model.py use mocked ultralytics
TEST_ARGS="${TEST_ARGS:-tests/}"

echo "Test target: ${TEST_ARGS}"
echo ""

python -m pytest ${TEST_ARGS} -v --tb=short 2>&1

EXIT_CODE=$?

echo ""
echo "=== Test Suite Complete ==="
echo "Exit code: ${EXIT_CODE}"
echo "Date: $(date)"

exit ${EXIT_CODE}
