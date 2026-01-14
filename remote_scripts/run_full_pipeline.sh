#!/bin/bash
# run_full_pipeline.sh
# Full noise generation pipeline using Google experiment data
# Run on remote server with experiment data at /root/work/google_qec3v5_experiment_data

set -e

WORK_DIR="/root/work/ALPHAQUBIT"
EXPERIMENT_ROOT="/root/work/google_qec3v5_experiment_data"
OUTPUT_DIR="pretrain_data"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

cd "$WORK_DIR"

echo "=============================================="
echo " ALPHAQUBIT - Full Noise Generation Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo "Experiment data: $EXPERIMENT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# 1. Pull latest code from S3
echo "[1/4] Pulling latest code from S3..."
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/ . --progress

# 2. Count experiments
echo ""
echo "[2/4] Counting experiments..."
EXP_COUNT=$(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d | wc -l)
DEM_COUNT=$(find "$EXPERIMENT_ROOT" -name "*.dem" | wc -l)
STIM_COUNT=$(find "$EXPERIMENT_ROOT" -name "*.stim" | wc -l)
echo "Found $EXP_COUNT experiment directories"
echo "Found $DEM_COUNT .dem files"
echo "Found $STIM_COUNT .stim files"

# 3. Run full pipeline
echo ""
echo "[3/4] Running full noise generation pipeline..."
echo "This will generate 100k samples per experiment (may take hours)..."
echo ""

python make_all_pretraining_noise.py \
    --use-dem-stim \
    --experiment-root "$EXPERIMENT_ROOT" \
    --exp-samples 100000 \
    --skip-dem \
    --skip-si1000 \
    --skip-soft \
    --out-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# 4. Summary
echo ""
echo "[4/4] Pipeline complete! Summary:"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.npz" | wc -l
echo " .npz files"
echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR"
echo ""
echo "Manifest:"
cat "$OUTPUT_DIR/MANIFEST.json" | head -100
echo ""
echo "=============================================="
