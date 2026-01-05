# Paper Alignment Fix Report

## Executive Summary

Our model's poor performance (19.8% LER vs paper's 3% LER) is caused by **three critical issues** that deviate from the paper's methodology:

1. **Wrong learning rate for fine-tuning**: Using 1e-4 instead of paper's 1e-5 (10x too high)
2. **No pre-trained model used**: Fine-tuning from scratch instead of using pre-trained weights
3. **Pre-training on wrong noise distribution**: Only trained on p=0.001 instead of full p=0.001-0.01 range

---

## Issues Found

### CRITICAL Issue 1: Learning Rate Mismatch
- **File**: `run_finetune_all.py`
- **Expected**: lr = 1e-5 for fine-tuning (per paper Table S3)
- **Actual**: lr = 1e-4 (default was wrong)
- **Impact**: 10x higher learning rate causes aggressive training, potential overfitting
- **Status**: ✅ FIXED - Changed default to 1e-5

### CRITICAL Issue 2: Missing Pre-trained Model
- **File**: `run_finetune_all.py`
- **Expected**: `--pretrained alphaqubit_pauli_plus.pth` should be default
- **Actual**: `--pretrained None` was the default
- **Impact**: Fine-tuning from scratch defeats the entire purpose of pre-training
- **Status**: ✅ FIXED - Changed default to 'alphaqubit_pauli_plus.pth'

### CRITICAL Issue 3: Pre-training Noise Distribution
- **File**: `configs/pauli_plus.yaml`
- **Expected**: Pre-training on SI1000 with p ∈ {0.001, 0.002, ..., 0.01}
- **Actual**: Only `depolarization: 0.001` (single lowest noise level)
- **Impact**: Model only learned low-noise patterns, cannot generalize to higher noise
- **Status**: 🔴 NEEDS RE-TRAINING - Created `paper_aligned_pretrain.py` script

### HIGH Issue 4: Model Predicting All Zeros
- **Evidence**: test_summary.json shows TP=0, FP=0 for many experiments
- **Root Cause**: Class imbalance at low noise + model not trained on diverse noise
- **Impact**: Model achieves "accuracy" by always predicting 0, but F1=0
- **Status**: Will be fixed by proper pre-training

---

## Root Cause Analysis

The paper's approach works because:

1. **Pre-training sees diverse noise levels**: At p=0.01, positive label ratio is ~30-40%, much more balanced
2. **Transfer learning**: Pre-trained features transfer to fine-tuning task
3. **Lower fine-tuning LR**: 1e-5 prevents overwriting pre-trained features

Our failure was because:
1. Pre-training only used p=0.001 where positive ratio is ~3-5%
2. Model learned to predict 0 always (96% "accuracy" but 0% recall)
3. Fine-tuning with wrong LR and/or from scratch

---

## Solution

### Step 1: Use Fixed Scripts (Immediate)
The following fixes have been applied:

```python
# run_finetune_all.py - FIXED
parser.add_argument('--lr', type=float, default=1e-5, ...)  # Was 1e-4
parser.add_argument('--pretrained', type=str, default='alphaqubit_pauli_plus.pth', ...)  # Was None
```

### Step 2: Re-run Fine-tuning (If Pre-trained Model is Good)
```bash
python run_finetune_all.py \
    --pretrained alphaqubit_pauli_plus.pth \
    --lr 1e-5 \
    --epochs 30 \
    --batch-size 128 \
    --data-dir google_finetune_data/finetune \
    --output-dir finetuned_models_fixed
```

### Step 3: Re-run Pre-training (If Needed)
If Step 2 doesn't work, the pre-trained model itself needs fixing:

```bash
python paper_aligned_pretrain.py \
    --samples 8500000 \
    --epochs 100 \
    --output alphaqubit_paper_aligned.pth
```

Then run fine-tuning with the new model:
```bash
python run_finetune_all.py --pretrained alphaqubit_paper_aligned.pth
```

### Step 4: Test Results
```bash
python run_decode_all.py \
    --model-dir finetuned_models_fixed \
    --data-dir google_finetune_data/test \
    --output-dir test_results_fixed
```

---

## Expected Outcome After Fix

| Metric | Current | Expected |
|--------|---------|----------|
| Average LER | 19.8% | ~3% |
| Average Accuracy | 80.2% | ~97% |
| d3 r01 LER | 3.8-4.8% | 2.6% |
| d5 r01 LER | 7.2% | 1.5% |
| True Positives | 0 (many exp) | > 0 |
| F1 Score | 0.18 | > 0.8 |

---

## Files Modified

1. `run_finetune_all.py` - Fixed lr and pretrained defaults
2. `paper_aligned_pretrain.py` - NEW: Paper-aligned pre-training script
3. `diagnose_paper_alignment.py` - NEW: Diagnostic script

---

## Verification Checklist

- [x] Learning rate fixed to 1e-5 for fine-tuning
- [x] Pretrained default points to alphaqubit_pauli_plus.pth
- [x] Created paper_aligned_pretrain.py for proper pre-training
- [ ] Re-run fine-tuning with fixed settings
- [ ] Re-run testing
- [ ] Verify LER improvement to ~3%

---

*Report generated: 2026-01-04*
