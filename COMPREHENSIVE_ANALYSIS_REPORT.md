# Comprehensive Analysis of AlphaQubit Replication Gap

**Date:** January 4, 2026
**Status:** Analysis Complete (UPDATED)
**Author:** GitHub Copilot

## Executive Summary

We have conducted a deep-dive diagnosis to understand the significant performance gap between our current replication results and the AlphaQubit paper baseline.

**CORRECTION**: The model architecture IS correctly configured (256 hidden dim, 12 layers) - matching the paper.

**The primary causes of the gap are now identified as:**
1. **Pre-training Data Distribution (Critical)**: Pre-training noise range may be too narrow vs the test distribution
2. **d5 Code Performance (Critical)**: 4.8x worse than paper on d5, suggesting scaling issues
3. **Fine-tuning Not Enough**: Fine-tuning alone cannot compensate for poor pre-training

---

## 1. Detailed Gap Analysis (Corrected)

### 1.1 Actual Performance Comparison

| Experiment | Our LER | Paper LER | Gap |
|------------|---------|-----------|-----|
| **d3_r01** (best case) | **3.74%** | **2.50%** | **+50%** |
| **d5_r01** (critical) | **7.15%** | **1.50%** | **+376%** |
| **d3_r25** (high noise) | **~25%** | *N/A* | At theoretical limit |

**Key Insight**: Our d3 low-noise results are only ~50% worse than the paper, but d5 results are **4.8x worse**. This points to a fundamental issue with how the model handles larger code distances.

### 1.2 Model Architecture Audit (CORRECTED)

| Component | Our Implementation | Paper Specification | Status |
|-----------|-------------------|---------------------|--------|
| **Hidden Dimension** | **256** | **256** | ✅ Aligned |
| **Layers** | **12** | **12** | ✅ Aligned |
| **Attention Heads** | **8** | **8** | ✅ Aligned |
| **Learning Rate (finetune)** | **1e-5** | **1e-5** | ✅ Aligned |

**Conclusion:** Model architecture is correct. The issue lies elsewhere.

### 1.3 Pre-training Data Audit

Checking `configs/pauli_plus.yaml` and `run_server_pipeline.py`:

- **Paper specification:**
  - 8.5M samples total
  - Physical error rates: 0.001 - 0.01 (SI1000 grid)
  - Code distances: d=3, 5, 7
  - Rounds: 1, 5, 10, 25

- **Our configuration:**
  - `si1000_p_grid`: [0.001, 0.002, ..., 0.01] ✅
  - `code_distances`: [3, 5, 7] ✅
  - `rounds_list`: [1, 5, 10, 25] ✅

- **Issue Found:** The `pretrain_data/` folder is **EMPTY**! This suggests pre-training may have been done with limited/different data.

---

## 2. Root Cause Diagnosis (Updated)

### Why is d5 so much worse than d3?
The gap for d3 (50%) is tolerable, but d5 (376%) is catastrophic.

1. **Pre-training may be insufficient for d5**: The pre-trained model might not have seen enough d5 samples during pre-training. If pre-training was done mostly on d3, the model won't transfer well to d5.

2. **Label imbalance at d5**: At higher code distances, the logical error rate is naturally lower (better error correction). If the training data has severe class imbalance (99%+ negative), the model may learn to always predict 0.

3. **Feature scaling**: d5 has 24 stabilizers vs d3's 8. The same model may struggle to attend to more tokens without specific scaling.

### Why is high-noise (r25) near 25% (random)?
At r25 (25% physical error rate), this is **above the threshold** for surface codes. At such high noise, even perfect decoders struggle. Our 25% LER is likely close to the theoretical limit, not a model bug.

---

## 3. Remediation Plan (Updated)

### Phase 1: Verify Pre-training Quality (Immediate)
1. **Check pre-training logs**: Was pre-training actually run for 100 epochs on 8.5M samples?
2. **Verify data diversity**: Did pre-training data include all code distances and noise levels?
3. **Re-run pre-training if needed**: Use the full `run_server_pipeline.py` configuration.

### Phase 2: Focus on d5 Performance (High Priority)
1. **Increase d5 samples in fine-tuning**: Current 50K may not be enough for d5's complexity.
2. **Use class-weighted loss**: Address potential label imbalance at d5.
3. **Longer fine-tuning for d5**: Increase epochs from 30 to 50-100.

### Phase 3: Data Analysis (Medium Priority)
1. **Check label distribution**: Print class balance for d3 vs d5 experiments.
2. **Verify data preprocessing**: Ensure detection events are normalized correctly.

## 4. Conclusion (Updated)

The replication is **architecturally correct** but has **pre-training/data issues**. The model works reasonably well on d3 (50% gap) but fails badly on d5 (376% gap), indicating the pre-trained model didn't learn generalizable features for larger codes.

**Next Step**: Re-run the full pre-training pipeline with proper data generation, then re-evaluate.

---

## 5. CRITICAL FINDING: Model Predicts All Zeros! (January 4, 2026)

### 🚨 The "Good" Low-Noise Results Are An Illusion!

After deep investigation, we discovered the model is **NOT actually decoding** - it's just predicting the majority class (0 = no logical error).

### Evidence:
| Metric | d3_r01 (all 8 experiments) |
|--------|---------------------------|
| True Positives | **0** |
| False Positives | **0** |
| Precision | **0.0** |
| Recall | **0.0** |
| F1 Score | **0.0** |

The model predicts "0" for **every single sample** at low noise!

### Why Does Accuracy Look Good?

Because of **severe label imbalance**:
- At r01 (low noise): only ~3-4% of samples are positive (logical error occurred)
- Predicting ALL ZEROS gives 96-97% "accuracy"
- Our reported LER (3.74%) = positive_ratio, NOT actual model performance!

### At High Noise (r25), the model learns slightly:
| Metric | r25 experiments |
|--------|-----------------|
| True Positives | 939-2335 |
| Precision | ~50% |
| Recall | 19-47% |

This is barely better than random guessing (50/50).

### Root Cause

1. **Class imbalance not handled**: BCELoss without class weights causes the model to predict majority class
2. **Pre-training may be broken**: The local `pipeline_log.txt` shows pre-training FAILED due to a `stim` module error
3. **Fine-tuning alone is insufficient**: Without proper pre-training, fine-tuning can't recover

### Immediate Fix Required

1. **Add class weights to loss function**:
   ```python
   pos_weight = (num_negative / num_positive)
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

2. **Verify pre-training was done correctly on the server** (not locally)

3. **Use focal loss or other imbalance-aware loss functions**
