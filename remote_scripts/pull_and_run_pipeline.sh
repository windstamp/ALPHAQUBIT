#!/bin/bash
# =============================================================================
# pull_and_run_pipeline.sh
# =============================================================================
# Pull latest code from S3 and run the noise generation pipeline
# using Google experiment data (.dem/.stim files)
#
# Usage:
#   chmod +x pull_and_run_pipeline.sh
#   ./pull_and_run_pipeline.sh
#
# Or with options:
#   ./pull_and_run_pipeline.sh --exp-samples 50000 --soft-shots 100000
# =============================================================================

set -e

# Configuration
S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
S3_PATH="${S3_REMOTE}/ALPHAQUBIT"
WORK_DIR="${HOME}/work/ALPHAQUBIT"
GOOGLE_DATA_DIR="${HOME}/work/google_qec3v5_experiment_data"

# Default parameters (can be overridden via CLI)
EXP_SAMPLES=${EXP_SAMPLES:-50000}
SOFT_SHOTS=${SOFT_SHOTS:-100000}
DEM_SAMPLES=${DEM_SAMPLES:-200000}
SI1000_SAMPLES=${SI1000_SAMPLES:-285000}
DEVICE=${DEVICE:-auto}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-samples)
            EXP_SAMPLES="$2"
            shift 2
            ;;
        --soft-shots)
            SOFT_SHOTS="$2"
            shift 2
            ;;
        --dem-samples)
            DEM_SAMPLES="$2"
            shift 2
            ;;
        --si1000-samples)
            SI1000_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-pull)
            SKIP_PULL=1
            shift
            ;;
        --use-dem-stim)
            USE_DEM_STIM=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp-samples N     Samples per experiment (default: 50000)"
            echo "  --soft-shots N      Soft/IQ shots per experiment (default: 100000)"
            echo "  --dem-samples N     DEM samples (default: 200000)"
            echo "  --si1000-samples N  SI1000 samples per p per distance (default: 285000)"
            echo "  --device DEV        Device: auto|cpu|cuda|npu (default: auto)"
            echo "  --skip-pull         Skip pulling from S3"
            echo "  --use-dem-stim      Generate from .dem/.stim files directly"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "AlphaQubit Pipeline Runner"
echo "=============================================="
echo "Work directory: ${WORK_DIR}"
echo "Google data: ${GOOGLE_DATA_DIR}"
echo "Parameters:"
echo "  EXP_SAMPLES: ${EXP_SAMPLES}"
echo "  SOFT_SHOTS: ${SOFT_SHOTS}"
echo "  DEM_SAMPLES: ${DEM_SAMPLES}"
echo "  SI1000_SAMPLES: ${SI1000_SAMPLES}"
echo "  DEVICE: ${DEVICE}"
echo "=============================================="

# Step 1: Check rclone
if ! command -v rclone &> /dev/null; then
    echo "❌ rclone not found! Install with: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

# Step 2: Pull latest code from S3
if [[ -z "${SKIP_PULL}" ]]; then
    echo ""
    echo ">>> Step 1: Pulling latest code from S3..."
    mkdir -p "${WORK_DIR}"
    
    # Pull core Python files and configs
    rclone copy "${S3_PATH}/generate_data.py" "${WORK_DIR}/" -v
    rclone copy "${S3_PATH}/make_all_pretraining_noise.py" "${WORK_DIR}/" -v
    rclone copy "${S3_PATH}/run_create_all_samples.py" "${WORK_DIR}/" -v
    rclone copy "${S3_PATH}/download_google_experiment_data.py" "${WORK_DIR}/" -v
    
    # Pull directories
    rclone copy "${S3_PATH}/simulator" "${WORK_DIR}/simulator" -v
    rclone copy "${S3_PATH}/google_qec_simulator" "${WORK_DIR}/google_qec_simulator" -v
    rclone copy "${S3_PATH}/ai_models" "${WORK_DIR}/ai_models" -v
    rclone copy "${S3_PATH}/configs" "${WORK_DIR}/configs" -v
    
    echo "✓ Code pulled from S3"
else
    echo ">>> Skipping S3 pull (--skip-pull)"
fi

# Step 3: Check Google experiment data
echo ""
echo ">>> Step 2: Checking Google experiment data..."
if [[ -d "${GOOGLE_DATA_DIR}" ]]; then
    STIM_COUNT=$(find "${GOOGLE_DATA_DIR}" -name "*.stim" 2>/dev/null | wc -l)
    DEM_COUNT=$(find "${GOOGLE_DATA_DIR}" -name "*.dem" 2>/dev/null | wc -l)
    echo "✓ Found ${STIM_COUNT} .stim files, ${DEM_COUNT} .dem files"
else
    echo "⚠ Google experiment data not found at ${GOOGLE_DATA_DIR}"
    echo "  Expected structure: ~/work/google_qec3v5_experiment_data/"
    echo "  You may need to download it from Zenodo first."
fi

# Step 4: Setup Python environment
echo ""
echo ">>> Step 3: Checking Python environment..."
cd "${WORK_DIR}"

# Check for required packages
python3 -c "import stim" 2>/dev/null || {
    echo "Installing stim..."
    pip install stim
}
python3 -c "import torch" 2>/dev/null || {
    echo "Installing torch..."
    pip install torch
}
python3 -c "import yaml" 2>/dev/null || {
    echo "Installing pyyaml..."
    pip install pyyaml
}
python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy..."
    pip install numpy
}

echo "✓ Python environment ready"

# Step 5: Run the pipeline
echo ""
echo ">>> Step 4: Running noise generation pipeline..."
echo ""

# Build command
CMD="python3 make_all_pretraining_noise.py"
CMD="${CMD} --experiment-root ${GOOGLE_DATA_DIR}"
CMD="${CMD} --dem-samples ${DEM_SAMPLES}"
CMD="${CMD} --si1000-samples ${SI1000_SAMPLES}"
CMD="${CMD} --soft-shots ${SOFT_SHOTS}"
CMD="${CMD} --soft-device ${DEVICE}"
CMD="${CMD} --exp-samples ${EXP_SAMPLES}"

if [[ -n "${USE_DEM_STIM}" ]]; then
    CMD="${CMD} --use-dem-stim"
fi

echo "Running: ${CMD}"
echo ""

# Execute
eval ${CMD}

# Step 6: Summary
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Generated data location: ${WORK_DIR}/pretrain_data/"
ls -la "${WORK_DIR}/pretrain_data/" 2>/dev/null || echo "  (directory may not exist yet)"
echo ""
echo "To upload results back to S3:"
echo "  rclone copy pretrain_data ${S3_PATH}/pretrain_data -v --progress"
echo ""
echo "To run training:"
echo "  python3 ai_models/train.py --config configs/pauli_plus.yaml"
echo "=============================================="
