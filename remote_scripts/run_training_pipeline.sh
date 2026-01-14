#!/bin/bash
# Full AlphaQubit Training Pipeline
# This script runs pre-training on all d3 data, then optionally fine-tunes

set -e

cd /root/work/ALPHAQUBIT

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

mkdir -p $LOG_DIR $CHECKPOINT_DIR

echo "============================================================"
echo "AlphaQubit Full Training Pipeline"
echo "Started: $(date)"
echo "============================================================"

# Check data
NPZ_COUNT=$(find output/ -name "*.npz" 2>/dev/null | wc -l)
echo "Found $NPZ_COUNT .npz files in output/"

if [ "$NPZ_COUNT" -eq 0 ]; then
    echo "ERROR: No data found. Please run data generation first."
    exit 1
fi

# ============================================================
# STAGE 1: Pre-training on all d3 data (all rounds)
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 1: Pre-training on d3 data (all rounds)"
echo "============================================================"

python3 train_full_pipeline.py \
    --mode pretrain \
    --data-dir output \
    --filter-distance 3 \
    --max-rounds 25 \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-layers 12 \
    --num-heads 8 \
    --npu \
    --save-dir $CHECKPOINT_DIR 2>&1 | tee ${LOG_DIR}/pretrain_d3_${TIMESTAMP}.log

echo ""
echo "Pre-training complete!"
echo ""

# ============================================================
# STAGE 2: Fine-tuning on d3 r25 data
# ============================================================
echo "============================================================"
echo "STAGE 2: Fine-tuning on d3, r=25 data"
echo "============================================================"

# Check if pretrain checkpoint exists
PRETRAIN_CKPT="${CHECKPOINT_DIR}/pretrain_d3_best.pth"
if [ -f "$PRETRAIN_CKPT" ]; then
    python3 train_full_pipeline.py \
        --mode finetune \
        --data-dir output \
        --filter-distance 3 \
        --filter-rounds 25 \
        --max-rounds 25 \
        --epochs 50 \
        --batch-size 256 \
        --lr 1e-5 \
        --checkpoint $PRETRAIN_CKPT \
        --npu \
        --save-dir $CHECKPOINT_DIR 2>&1 | tee ${LOG_DIR}/finetune_d3_r25_${TIMESTAMP}.log
else
    echo "WARNING: Pre-train checkpoint not found, skipping fine-tuning"
fi

# ============================================================
# STAGE 3: Fine-tuning on d5 data (if available)
# ============================================================
D5_COUNT=$(find output/ -name "*_d5_*.npz" 2>/dev/null | wc -l)
if [ "$D5_COUNT" -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 3: Fine-tuning on d5 data"
    echo "============================================================"
    
    python3 train_full_pipeline.py \
        --mode finetune \
        --data-dir output \
        --filter-distance 5 \
        --max-rounds 25 \
        --epochs 50 \
        --batch-size 256 \
        --lr 1e-5 \
        --checkpoint $PRETRAIN_CKPT \
        --npu \
        --save-dir $CHECKPOINT_DIR 2>&1 | tee ${LOG_DIR}/finetune_d5_${TIMESTAMP}.log
else
    echo ""
    echo "No d5 data found, skipping d5 fine-tuning"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Training Pipeline Complete!"
echo "============================================================"
echo "Finished: $(date)"
echo ""
echo "Checkpoints saved in: $CHECKPOINT_DIR/"
ls -la $CHECKPOINT_DIR/
echo ""
echo "Logs saved in: $LOG_DIR/"
ls -la $LOG_DIR/
echo "============================================================"
