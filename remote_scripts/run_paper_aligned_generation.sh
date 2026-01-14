#!/bin/bash
# run_paper_aligned_generation.sh
# Generate noise data with EXACT paper-aligned sample counts
#
# Paper Specifications:
# - Pre-training: 8.5M total synthetic samples
# - Fine-tuning: 50K samples per experiment
# - Distances: d=3, d=5, d=7 (but Google data only has d3, d5 in v3/v5)
# - Rounds: 1 to 25 (odd numbers for d3, varying for d5)
# - Basis: X and Z
#
# Google Experiment Data Structure (google_qec3v5_experiment_data):
# - surface_code_bX_d3_r{01,03,...,25}_center_*: ~52 experiments (4 center locations × 13 rounds)
# - surface_code_bX_d5_r{01,03,...,25}_center_*: ~13 experiments (1 center × 13 rounds)
# - surface_code_bZ_d3_r{01,03,...,25}_center_*: ~52 experiments
# - surface_code_bZ_d5_r{01,03,...,25}_center_*: ~13 experiments
# Total: ~130 experiments (will vary based on actual data)
#
# Sample Distribution Strategy:
# For 8.5M total pretraining samples across ~130 experiments:
#   8,500,000 / 130 ≈ 65,385 samples per experiment
#
# For fine-tuning: 50,000 samples per experiment

set -e

WORK_DIR="/root/work/ALPHAQUBIT"
EXPERIMENT_ROOT="/root/work/google_qec3v5_experiment_data"
OUTPUT_DIR="pretrain_data_paper_aligned"
LOG_FILE="paper_aligned_generation_$(date +%Y%m%d_%H%M%S).log"

# Paper constants
TOTAL_PRETRAIN_SAMPLES=8500000
FINETUNE_SAMPLES_PER_EXP=50000

cd "$WORK_DIR"

echo "=============================================="
echo " ALPHAQUBIT - Paper-Aligned Data Generation"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# 1. Pull latest code
echo "[1/5] Pulling latest code from S3..."
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/ . --progress

# 2. Count experiments to calculate per-experiment samples
echo ""
echo "[2/5] Counting experiments..."

# Count experiment directories
NUM_EXPERIMENTS=$(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -name "surface_code_*" | wc -l)

if [ "$NUM_EXPERIMENTS" -eq 0 ]; then
    echo "ERROR: No experiments found in $EXPERIMENT_ROOT"
    exit 1
fi

# Calculate samples per experiment for pretraining
SAMPLES_PER_EXP=$((TOTAL_PRETRAIN_SAMPLES / NUM_EXPERIMENTS))

echo ""
echo "Paper-aligned configuration:"
echo "  Total experiments found: $NUM_EXPERIMENTS"
echo "  Total pretrain samples: $TOTAL_PRETRAIN_SAMPLES"
echo "  Samples per experiment: $SAMPLES_PER_EXP"
echo "  Fine-tune samples/exp: $FINETUNE_SAMPLES_PER_EXP"
echo ""

# Breakdown by type
echo "Experiment breakdown:"
echo "  bX_d3: $(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -name "*bX_d3*" | wc -l) experiments"
echo "  bX_d5: $(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -name "*bX_d5*" | wc -l) experiments"
echo "  bZ_d3: $(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -name "*bZ_d3*" | wc -l) experiments"
echo "  bZ_d5: $(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -name "*bZ_d5*" | wc -l) experiments"

# 3. Generate pretraining data
echo ""
echo "[3/5] Generating pre-training data ($TOTAL_PRETRAIN_SAMPLES samples)..."
echo "This will take a while..."

python make_all_pretraining_noise.py \
    --use-dem-stim \
    --experiment-root "$EXPERIMENT_ROOT" \
    --exp-samples $SAMPLES_PER_EXP \
    --exp-with-soft \
    --skip-dem \
    --skip-si1000 \
    --skip-soft \
    --out-dir "$OUTPUT_DIR/pretrain" \
    2>&1 | tee -a "$LOG_FILE"

# 4. Generate fine-tuning data (if needed separately with different sample count)
echo ""
echo "[4/5] Generating fine-tuning data ($FINETUNE_SAMPLES_PER_EXP samples per experiment)..."

python make_all_pretraining_noise.py \
    --use-dem-stim \
    --experiment-root "$EXPERIMENT_ROOT" \
    --exp-samples $FINETUNE_SAMPLES_PER_EXP \
    --exp-with-soft \
    --skip-dem \
    --skip-si1000 \
    --skip-soft \
    --out-dir "$OUTPUT_DIR/finetune" \
    2>&1 | tee -a "$LOG_FILE"

# 5. Summary
echo ""
echo "[5/5] Generation complete!"
echo ""
echo "=============================================="
echo " SUMMARY"
echo "=============================================="

# Count generated files
PRETRAIN_FILES=$(find "$OUTPUT_DIR/pretrain" -name "*.npz" | wc -l)
FINETUNE_FILES=$(find "$OUTPUT_DIR/finetune" -name "*.npz" | wc -l)

echo "Pre-training:"
echo "  Files generated: $PRETRAIN_FILES"
echo "  Target samples per file: $SAMPLES_PER_EXP"
echo "  Approx total samples: $((PRETRAIN_FILES * SAMPLES_PER_EXP))"
echo ""
echo "Fine-tuning:"
echo "  Files generated: $FINETUNE_FILES"
echo "  Samples per file: $FINETUNE_SAMPLES_PER_EXP"
echo "  Approx total samples: $((FINETUNE_FILES * FINETUNE_SAMPLES_PER_EXP))"
echo ""
echo "Total disk usage:"
du -sh "$OUTPUT_DIR"
du -sh "$OUTPUT_DIR"/*
echo ""
echo "End time: $(date)"
echo "Log file: $LOG_FILE"
echo "=============================================="
