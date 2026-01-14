#!/bin/bash
# Run training pipeline on generated noise data
# This script uses the pre-generated .npz files in output/ directory

set -e

cd /root/work/ALPHAQUBIT

# Check if data exists
echo "=============================================="
echo "Checking generated noise data..."
echo "=============================================="
NPZ_COUNT=$(find output/ -name "*.npz" 2>/dev/null | wc -l)
echo "Found $NPZ_COUNT .npz files in output/"

if [ "$NPZ_COUNT" -eq 0 ]; then
    echo "ERROR: No .npz files found in output/"
    echo "Please run noise generation first."
    exit 1
fi

# Show sample count
echo ""
echo "Calculating total samples..."
python3 -c "
import numpy as np
from pathlib import Path
total = 0
files = list(Path('output').glob('*.npz'))[:5]
for f in files:
    d = np.load(f)
    total += d['data'].shape[0]
print(f'Samples in first 5 files: {total:,}')
est = total // 5 * $NPZ_COUNT
print(f'Estimated total samples: {est:,}')
"

echo ""
echo "=============================================="
echo "Starting AlphaQubit Training"
echo "=============================================="
echo "Paper-aligned hyperparameters:"
echo "  - Batch size: 256"
echo "  - Learning rate: 1e-4"
echo "  - Epochs: 100"
echo "=============================================="
echo ""

# Run training with paper-aligned parameters
python3 train_soft_readout.py \
    --data-dir output \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-layers 12 \
    --num-heads 8 \
    --npu \
    --save-dir checkpoints

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved in: checkpoints/"
echo "  - alphaqubit_best.pth (best validation loss)"
echo "  - alphaqubit_final.pth (final epoch)"
echo "=============================================="
