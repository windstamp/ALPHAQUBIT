#!/bin/bash
# test_experiment_noise_gen.sh
# Test noise generation using Google experiment .dem/.stim files
# Run on remote server with experiment data at /root/work/google_qec3v5_experiment_data

set -e

WORK_DIR="/root/work/ALPHAQUBIT"
EXPERIMENT_ROOT="/root/work/google_qec3v5_experiment_data"
OUTPUT_DIR="$WORK_DIR/test_output"

cd "$WORK_DIR"

echo "=============================================="
echo " ALPHAQUBIT - Experiment Noise Generation Test"
echo "=============================================="
echo "Experiment data: $EXPERIMENT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# 1. Pull latest code from S3
echo "[1/5] Pulling latest code from S3..."
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/ . --progress

# 2. Check experiment data exists
echo ""
echo "[2/5] Checking experiment data..."
if [ ! -d "$EXPERIMENT_ROOT" ]; then
    echo "ERROR: Experiment data not found at $EXPERIMENT_ROOT"
    exit 1
fi

echo "Found experiment directories:"
find "$EXPERIMENT_ROOT" -maxdepth 2 -type d | head -20

echo ""
echo "Found .dem files:"
find "$EXPERIMENT_ROOT" -name "*.dem" | head -10

echo ""
echo "Found .stim files:"
find "$EXPERIMENT_ROOT" -name "*.stim" | head -10

# 3. Quick test: generate_data.py with single experiment
echo ""
echo "[3/5] Testing generate_data.py with single experiment..."

# Find first experiment with .dem or .stim file
FIRST_EXP=$(find "$EXPERIMENT_ROOT" -name "*.dem" -o -name "*.stim" | head -1 | xargs dirname)
if [ -z "$FIRST_EXP" ]; then
    echo "ERROR: No .dem or .stim files found"
    exit 1
fi
echo "Using experiment: $FIRST_EXP"

mkdir -p "$OUTPUT_DIR"
python generate_data.py \
    --model experiment \
    --experiment "$FIRST_EXP" \
    --samples 1000 \
    --soft

echo "Single experiment test completed!"
ls -la output/

# 4. Test make_all_pretraining_noise.py with small sample size
echo ""
echo "[4/5] Testing make_all_pretraining_noise.py (small test run)..."

python make_all_pretraining_noise.py \
    --use-dem-stim \
    --experiment-root "$EXPERIMENT_ROOT" \
    --exp-samples 1000 \
    --skip-dem \
    --skip-si1000 \
    --skip-soft \
    --out-dir "$OUTPUT_DIR/test_pretrain"

echo "Pipeline test completed!"

# 5. Check outputs
echo ""
echo "[5/5] Checking generated outputs..."
echo ""
echo "=== Output directory contents ==="
ls -laR "$OUTPUT_DIR" | head -50

echo ""
echo "=== Generated .npz/.npy files ==="
find "$OUTPUT_DIR" -name "*.npz" -o -name "*.npy" | head -20

echo ""
echo "=== MANIFEST.json ==="
if [ -f "$OUTPUT_DIR/test_pretrain/MANIFEST.json" ]; then
    cat "$OUTPUT_DIR/test_pretrain/MANIFEST.json"
else
    echo "No manifest found (may not have been created in test mode)"
fi

echo ""
echo "=============================================="
echo " TEST COMPLETE"
echo "=============================================="
echo ""
echo "To run full generation, use:"
echo "  python make_all_pretraining_noise.py \\"
echo "      --use-dem-stim \\"
echo "      --experiment-root $EXPERIMENT_ROOT \\"
echo "      --exp-samples 100000 \\"
echo "      --soft-shots 100000 \\"
echo "      --soft-device auto \\"
echo "      --out-dir pretrain_data"
