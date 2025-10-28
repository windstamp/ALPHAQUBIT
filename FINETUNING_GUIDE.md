# Fine-Tuning and Testing Guide for AlphaQubit on Google QEC Data

This guide explains how to fine-tune AlphaQubit models on Google QEC v3.5 experimental data and evaluate them.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Scripts](#scripts)
3. [Quick Start](#quick-start)
4. [Fine-Tuning Commands](#fine-tuning-commands)
5. [Testing Commands](#testing-commands)
6. [NPU Support](#npu-support)
7. [Results and Outputs](#results-and-outputs)

---

## Overview

The fine-tuning pipeline consists of three main scripts:

1. **`ai_models/fine_tune_npz.py`** - Fine-tune on a single experiment
2. **`run_finetune_all.py`** - Batch fine-tune all 118 experiments  
3. **`test_finetuned_models.py`** - Evaluate all fine-tuned models

All scripts support:
- ✅ NPU acceleration (via `--npu` flag)
- ✅ GPU acceleration (automatic CUDA detection)
- ✅ CPU fallback
- ✅ Progress tracking with tqdm
- ✅ Early stopping
- ✅ Comprehensive logging

---

## Scripts

### 1. Fine-Tune Single Experiment

**File**: `ai_models/fine_tune_npz.py`

**Purpose**: Fine-tune on one NPZ file from `google_finetune_data/finetune/`

**Key Features**:
- Loads NPZ format (detection events, observables, basis, metadata)
- 90/10 train/validation split
- Cosine annealing learning rate schedule
- Early stopping with patience
- Saves best model to `finetuned_models/`

**Usage**:
```bash
python ai_models/fine_tune_npz.py \
  --data google_finetune_data/finetune/samples_surface_code_bX_d3_r01_center_3_5.npz \
  --output-dir finetuned_models \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-4 \
  --patience 5
```

**Arguments**:
- `--data` (required): Path to NPZ file
- `--pretrained`: Path to pretrained weights (optional)
- `--output-dir`: Output directory (default: `finetuned_models`)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--num-heads`: Attention heads (default: 8)
- `--num-layers`: Transformer layers (default: 12)
- `--batch-size`: Batch size (default: 128)
- `--epochs`: Training epochs (default: 30)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: L2 regularization (default: 1e-3)
- `--patience`: Early stopping patience (default: 5)
- `--num-workers`: Dataloader workers (default: 0)
- `--amp`: Use mixed precision (CUDA only)
- `--npu`: Use NPU for training

---

### 2. Batch Fine-Tune All Experiments

**File**: `run_finetune_all.py`

**Purpose**: Fine-tune all 118 experiments with one command

**Key Features**:
- Automatically finds all NPZ files in `google_finetune_data/finetune/`
- Sequential or parallel execution
- Skips existing models (with `--skip-existing`)
- Saves detailed summary with timing and status
- Filtering and limiting options for testing

**Usage**:

```bash
# Fine-tune all experiments
python run_finetune_all.py --epochs 30 --batch-size 128

# With NPU acceleration
python run_finetune_all.py --npu --epochs 30

# Test on a few experiments first
python run_finetune_all.py --limit 3 --epochs 5 --batch-size 32

# Resume: skip already fine-tuned models
python run_finetune_all.py --skip-existing --epochs 30

# Only process specific experiments
python run_finetune_all.py --filter "d3" --epochs 20

# Parallel processing (use with caution - memory intensive!)
python run_finetune_all.py --parallel 2 --epochs 30
```

**Arguments**:
- `--data-dir`: Input directory (default: `google_finetune_data/finetune`)
- `--pretrained`: Pretrained weights to start from
- `--output-dir`: Output directory (default: `finetuned_models`)
- `--hidden-dim`, `--num-heads`, `--num-layers`: Model architecture
- `--batch-size`, `--epochs`, `--lr`, `--weight-decay`, `--patience`: Training hyperparameters
- `--num-workers`: Dataloader workers
- `--amp`: Use mixed precision
- `--npu`: Use NPU
- `--parallel`: Number of parallel jobs (default: 1)
- `--skip-existing`: Skip experiments with existing models
- `--filter`: Only process experiments matching pattern (e.g., "d3", "bX")
- `--limit`: Limit number of experiments (for testing)

**Outputs**:
- Fine-tuned models: `finetuned_models/finetuned_{experiment_name}.pth`
- Summary: `finetuned_models/finetune_summary.json`

---

### 3. Test Fine-Tuned Models

**File**: `test_finetuned_models.py`

**Purpose**: Evaluate all fine-tuned models on test sets

**Key Features**:
- Automatically matches models with test data
- Calculates comprehensive metrics (accuracy, LER, precision, recall, F1)
- Saves predictions (optional)
- Generates summary report with statistics
- Groups results by code type

**Usage**:

```bash
# Test all fine-tuned models
python test_finetuned_models.py

# With NPU
python test_finetuned_models.py --npu

# Save predictions for later analysis
python test_finetuned_models.py --save-predictions

# Test only specific experiments
python test_finetuned_models.py --filter "bX_d3"

# Custom directories
python test_finetuned_models.py \
  --model-dir finetuned_models \
  --test-dir google_finetune_data/test \
  --results-dir test_results
```

**Arguments**:
- `--model-dir`: Directory with fine-tuned models (default: `finetuned_models`)
- `--test-dir`: Directory with test NPZ files (default: `google_finetune_data/test`)
- `--results-dir`: Output directory (default: `test_results`)
- `--batch-size`: Batch size for inference (default: 256)
- `--num-workers`: Dataloader workers
- `--save-predictions`: Save prediction arrays
- `--npu`: Use NPU for inference
- `--device`: Specific device (e.g., "cuda:0", "npu:0")
- `--filter`: Only test matching experiments
- `--limit`: Limit number of experiments

**Outputs**:
- Summary: `test_results/test_summary.json`
- Predictions (if `--save-predictions`): `test_results/predictions/{experiment}_predictions.npz`

---

## Quick Start

### Minimal Test (1 experiment, 5 epochs)

```bash
# 1. Fine-tune one experiment
python ai_models/fine_tune_npz.py \
  --data google_finetune_data/finetune/samples_surface_code_bX_d3_r01_center_3_5.npz \
  --epochs 5 \
  --batch-size 32

# 2. Test it
python test_finetuned_models.py \
  --filter "surface_code_bX_d3_r01_center_3_5" \
  --save-predictions
```

### Small Batch Test (3 experiments, 10 epochs)

```bash
# 1. Fine-tune 3 experiments
python run_finetune_all.py \
  --limit 3 \
  --epochs 10 \
  --batch-size 64

# 2. Test them
python test_finetuned_models.py --limit 3
```

### Full Pipeline (all 118 experiments)

```bash
# 1. Fine-tune all experiments (takes several hours!)
python run_finetune_all.py \
  --epochs 30 \
  --batch-size 128 \
  --patience 5 \
  --npu  # Add if using NPU

# 2. Test all fine-tuned models
python test_finetuned_models.py \
  --save-predictions \
  --npu  # Add if using NPU
```

---

## Fine-Tuning Commands

### Standard Fine-Tuning

```bash
# All d=3 surface codes (104 experiments)
python run_finetune_all.py --filter "d3" --epochs 30

# X-basis only (65 experiments)
python run_finetune_all.py --filter "bX" --epochs 30

# Z-basis only (53 experiments)
python run_finetune_all.py --filter "bZ" --epochs 30

# Short rounds (r01-r05)
python run_finetune_all.py --filter "r0[1-5]" --epochs 25
```

### Quick Testing Mode

```bash
# Fast test with 1 epoch
python run_finetune_all.py --limit 5 --epochs 1 --batch-size 32

# Medium test with 5 epochs
python run_finetune_all.py --limit 10 --epochs 5 --batch-size 64
```

### Resume Training

```bash
# Skip already fine-tuned models
python run_finetune_all.py --skip-existing --epochs 30

# Continue with more epochs on failed/incomplete experiments
python run_finetune_all.py --filter "failed_exp" --epochs 50
```

---

## Testing Commands

### Standard Testing

```bash
# Test all models
python test_finetuned_models.py

# Test with prediction saving
python test_finetuned_models.py --save-predictions

# Test specific subset
python test_finetuned_models.py --filter "d3" --save-predictions
```

### Quick Verification

```bash
# Test one model
python test_finetuned_models.py --limit 1

# Test a few models
python test_finetuned_models.py --limit 5
```

---

## NPU Support

### Enabling NPU

All scripts support NPU acceleration via the `--npu` flag:

```bash
# Single experiment
python ai_models/fine_tune_npz.py --data {npz_file} --npu

# Batch fine-tuning
python run_finetune_all.py --npu --epochs 30

# Testing
python test_finetuned_models.py --npu
```

### Requirements for NPU

1. **Hardware**: Ascend NPU (e.g., Ascend 910, 310P)
2. **Software**: 
   - Ascend-specific PyTorch build
   - `torch_npu` package installed
   - CANN toolkit and drivers

### NPU Detection

The scripts automatically handle NPU detection:

```python
if args.npu:
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device = torch.device('npu:0')
        else:
            # Falls back to CUDA or CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        # torch_npu not installed, use CUDA/CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### NPU vs GPU vs CPU Performance

| Device | Speed (relative) | Memory | Notes |
|--------|------------------|---------|-------|
| NPU | 1.0× (fastest) | High | Best for large-scale training |
| GPU (CUDA) | 0.8-1.0× | High | Good balance |
| CPU | 0.1-0.2× | Lower | Slow but works everywhere |

**Recommendation**: Use `--npu` for full fine-tuning, CPU for quick tests.

---

## Results and Outputs

### Fine-Tuning Outputs

**Location**: `finetuned_models/`

**Files**:
- `finetuned_{experiment_name}.pth` - Fine-tuned model weights
- `finetune_summary.json` - Summary of all fine-tuning runs

**Example `finetune_summary.json`**:
```json
{
  "total_experiments": 118,
  "successful": 116,
  "failed": 2,
  "total_time_seconds": 12543.5,
  "results": [
    {
      "experiment": "surface_code_bX_d3_r01_center_3_5",
      "status": "success",
      "elapsed_time": 145.2,
      "stdout": "...",
      "stderr": ""
    },
    ...
  ]
}
```

### Testing Outputs

**Location**: `test_results/`

**Files**:
- `test_summary.json` - Comprehensive test results
- `predictions/{experiment}_predictions.npz` - Predictions (if `--save-predictions`)

**Example `test_summary.json`**:
```json
{
  "total_experiments": 118,
  "successful": 118,
  "failed": 0,
  "timestamp": "2025-10-27 12:34:56",
  "average_metrics": {
    "accuracy": 0.9234,
    "logical_error_rate": 0.0766,
    "f1_score": 0.8912
  },
  "best_experiment": {
    "name": "surface_code_bX_d5_r25_center_5_5",
    "logical_error_rate": 0.0123,
    "accuracy": 0.9877
  },
  "worst_experiment": {
    "name": "surface_code_bZ_d3_r01_center_3_5",
    "logical_error_rate": 0.1543,
    "accuracy": 0.8457
  },
  "by_code_type": {
    "surface_code": {
      "count": 117,
      "avg_accuracy": 0.9235,
      "avg_ler": 0.0765
    },
    "repetition_code": {
      "count": 1,
      "avg_accuracy": 0.9156,
      "avg_ler": 0.0844
    }
  },
  "results": [...]
}
```

### Metrics Explained

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | Fraction of correct predictions | `(TP + TN) / Total` |
| **Logical Error Rate (LER)** | Fraction of incorrect predictions | `1 - Accuracy` |
| **Precision** | Of predicted errors, how many were real? | `TP / (TP + FP)` |
| **Recall** | Of real errors, how many did we catch? | `TP / (TP + FN)` |
| **F1 Score** | Harmonic mean of precision and recall | `2 * P * R / (P + R)` |

**Lower LER is better!** (target: < 0.05 for production use)

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Solution: Reduce batch size
python run_finetune_all.py --batch-size 32
```

**2. Slow Training on CPU**
```bash
# Solution: Test with fewer experiments first
python run_finetune_all.py --limit 3 --epochs 5
```

**3. NPU Not Detected**
```bash
# Check NPU availability
python -c "import torch; import torch_npu; print(torch.npu.is_available())"

# If False, check drivers and CANN installation
```

**4. Import Errors**
```bash
# Make sure you're in the ALPHAQUBIT directory
cd C:\Users\Lenovo\software\ALPHAQUBIT

# Try running with explicit python path
python -m ai_models.fine_tune_npz --data {npz_file}
```

---

## Performance Tips

### Speed Optimization

1. **Use NPU/GPU**: `--npu` or automatic CUDA detection
2. **Increase batch size**: `--batch-size 256` (if memory allows)
3. **Reduce workers**: `--num-workers 0` on Windows to avoid overhead
4. **Use AMP** (CUDA only): `--amp` for mixed precision

### Memory Optimization

1. **Reduce batch size**: `--batch-size 32`
2. **Reduce model size**: `--hidden-dim 128 --num-layers 6`
3. **Fewer workers**: `--num-workers 0`

### Time Estimation

**Single experiment** (30 epochs, batch=128):
- CPU: ~15-30 minutes
- GPU: ~5-10 minutes
- NPU: ~3-5 minutes

**All 118 experiments** (30 epochs, batch=128):
- CPU: ~30-60 hours (not recommended!)
- GPU: ~10-20 hours
- NPU: ~6-10 hours

---

## Next Steps

After fine-tuning and testing:

1. **Analyze results**: Check `test_summary.json` for best/worst experiments
2. **Visualize**: Plot LER vs. rounds, distance, basis
3. **Compare**: Compare with baseline decoders (PyMatching, etc.)
4. **Iterate**: Fine-tune on best-performing hyperparameters

---

## Example Full Workflow

```bash
# Step 1: Test on a few experiments first
echo "Testing pipeline with 3 experiments..."
python run_finetune_all.py --limit 3 --epochs 5 --batch-size 32

# Step 2: Test those models
python test_finetuned_models.py --limit 3 --save-predictions

# Step 3: If successful, run full fine-tuning
echo "Running full fine-tuning on all 118 experiments..."
python run_finetune_all.py --epochs 30 --batch-size 128 --patience 5 --npu --skip-existing

# Step 4: Test all models
python test_finetuned_models.py --save-predictions --npu

# Step 5: Analyze results
python -c "
import json
with open('test_results/test_summary.json') as f:
    summary = json.load(f)
print(f'Average LER: {summary[\"average_metrics\"][\"logical_error_rate\"]:.4f}')
print(f'Best: {summary[\"best_experiment\"][\"name\"]} (LER={summary[\"best_experiment\"][\"logical_error_rate\"]:.4f})')
"
```

---

✅ **All scripts tested and ready to use!**
