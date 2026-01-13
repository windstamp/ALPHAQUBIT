#!/bin/bash
# =============================================================================
# Full Decoder Comparison Script
# Runs both MWPM and AlphaQubit decoders to generate Figure 2 data
# =============================================================================

set -e

echo "=============================================="
echo "Full Decoder Comparison Pipeline"
echo "=============================================="

# Activate environment
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate alphaqubit 2>/dev/null || conda activate base
fi

# Enter working directory
cd ~/alphaqubit/ALPHAQUBIT

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./decoder_comparison_${TIMESTAMP}.log"
RESULTS_DIR="./benchmark_results"

echo "Log file: ${LOG_FILE}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "
import sys
deps = []
try:
    import stim
    print('✓ stim:', stim.__version__)
except ImportError:
    deps.append('stim')
    print('✗ stim: NOT INSTALLED')

try:
    import pymatching
    print('✓ pymatching:', pymatching.__version__)
except ImportError:
    deps.append('pymatching')
    print('✗ pymatching: NOT INSTALLED')

try:
    import torch
    print('✓ torch:', torch.__version__)
except ImportError:
    deps.append('torch')
    print('✗ torch: NOT INSTALLED')

if deps:
    print('\nMissing dependencies:', deps)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Installing missing dependencies..."
    pip install stim pymatching
fi
echo ""

# Parse arguments
MODE="test"
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            MODE="full"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Mode: ${MODE}"
if [ -n "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_PATH}"
fi
echo ""

# Run comparison
echo "=============================================="
echo "Starting Decoder Comparison..."
echo "=============================================="

mkdir -p ${RESULTS_DIR}
mkdir -p paper_figures/output

if [ "$MODE" == "full" ]; then
    # Full benchmark (run in background)
    echo "Running full benchmark (this may take several hours)..."
    
    CMD="python run_full_decoder_comparison.py --full --device auto"
    if [ -n "${MODEL_PATH}" ]; then
        CMD="${CMD} --model-path ${MODEL_PATH}"
    fi
    
    nohup ${CMD} > ${LOG_FILE} 2>&1 &
    PID=$!
    echo $PID > decoder_comparison.pid
    
    echo ""
    echo "Benchmark started in background"
    echo "PID: ${PID}"
    echo "Log: ${LOG_FILE}"
    echo ""
    echo "Commands:"
    echo "  tail -f ${LOG_FILE}        # View progress"
    echo "  kill ${PID}                # Stop benchmark"
else
    # Quick test (run in foreground)
    echo "Running quick test..."
    
    CMD="python run_full_decoder_comparison.py --test --device auto"
    if [ -n "${MODEL_PATH}" ]; then
        CMD="${CMD} --model-path ${MODEL_PATH}"
    fi
    
    ${CMD} 2>&1 | tee ${LOG_FILE}
fi

echo ""
echo "=============================================="
echo "Decoder comparison initiated!"
echo "=============================================="
echo ""
echo "Results will be saved to:"
echo "  - ${RESULTS_DIR}/decoder_comparison_results.json"
echo "  - paper_figures/output/decoder_comparison.png"
echo "=============================================="
