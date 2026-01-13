#!/bin/bash
# =============================================================================
# MWPM Threshold Benchmark Script for Server
# Run MWPM decoder benchmark to establish baseline threshold (~0.69%)
# =============================================================================

set -e

echo "=============================================="
echo "MWPM Threshold Benchmark"
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
LOG_FILE="./mwpm_benchmark_${TIMESTAMP}.log"

echo "Log file: ${LOG_FILE}"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import stim; import pymatching; print('✓ stim:', stim.__version__); print('✓ pymatching:', pymatching.__version__)"

if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install stim pymatching
fi
echo ""

# Parse arguments
MODE="test"
if [ "$1" == "--full" ]; then
    MODE="full"
elif [ "$1" == "--quick" ]; then
    MODE="test"
fi

echo "Mode: ${MODE}"
echo ""

# Run benchmark
echo "Starting MWPM benchmark..."
echo "=============================================="

if [ "$MODE" == "full" ]; then
    # Full benchmark (paper-aligned)
    nohup python run_mwpm_benchmark.py --full > ${LOG_FILE} 2>&1 &
    PID=$!
    echo $PID > mwpm_benchmark.pid
    echo "Full benchmark started in background (PID: ${PID})"
    echo "This will take several hours..."
else
    # Quick test
    python run_mwpm_benchmark.py --test 2>&1 | tee ${LOG_FILE}
fi

echo ""
echo "=============================================="
echo "MWPM benchmark initiated!"
echo "=============================================="
echo ""
echo "Results will be saved to:"
echo "  - benchmark_results/mwpm_threshold_data.json"
echo "  - paper_figures/output/mwpm_threshold_curve.png"
echo ""
echo "View logs: tail -f ${LOG_FILE}"
echo "=============================================="
