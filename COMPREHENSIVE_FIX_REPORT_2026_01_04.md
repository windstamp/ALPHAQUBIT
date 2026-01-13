# Comprehensive AlphaQubit Fix Report - January 4, 2026

## Executive Summary

This document summarizes all fixes applied to address the performance gap between our implementation and the AlphaQubit Nature 2024 paper results.

**Root Cause Identified**: The model was predicting **all zeros** (no logical errors) for all samples at low noise levels. This happened because:
1. Severe class imbalance (96%+ negative samples at low noise)
2. Missing soft channel features in pre-training
3. Missing regularization (dropout)
4. Sub-optimal weight initialization

---

## Fixes Applied

### 1. ✅ Class Imbalance Handling (CRITICAL)

**File**: `ai_models/fine_tune_npz.py`

**Problem**: `BCEWithLogitsLoss()` without class weights causes model to predict majority class.

**Evidence from test results**:
```
All 111 experiments show:
- true_positives: 0
- false_positives: 0
- precision: 0.0
- recall: 0.0
- f1_score: 0.0
```

**Fix Applied**:
```python
# Before:
criterion = nn.BCEWithLogitsLoss()

# After:
labels_all = dataset.observables
num_pos = labels_all.sum().item()
num_neg = len(labels_all) - num_pos
if num_pos > 0:
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    print(f"Class balance: {num_neg:.0f} negative, {num_pos:.0f} positive")
    print(f"Using pos_weight = {pos_weight.item():.2f} for class imbalance")
else:
    pos_weight = torch.tensor([1.0], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Also applied to**: `paper_aligned_pretrain.py`

---

### 2. ✅ Learning Rate Scheduler (Medium Impact)

**File**: `ai_models/fine_tune_npz.py`

**Problem**: `CosineAnnealingWarmRestarts` causes LR to jump back up during training.

**Fix Applied**:
```python
# Before:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# After (paper-aligned):
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=0
)
```

---

### 3. ✅ Dropout Regularization (Medium Impact)

**File**: `ai_models/model.py`

**Problem**: No dropout in the main transformer model, causing potential overfitting.

**Fix Applied** (in `SyndromeTransformerLayer.__init__`):
```python
def __init__(
    self,
    hidden_dim: int,
    num_heads: int,
    num_stabilizers: int,
    grid_size: int,
    pair_embed_dim: int = 48,
    use_dilated_convs: bool = True,
    dropout: float = 0.1  # NEW: Paper-aligned regularization
):
    super().__init__()
    # ...
    
    # NEW: Dropout layers
    self.dropout = nn.Dropout(dropout)
    self.attn_dropout = nn.Dropout(dropout)
```

**Applied in forward pass**:
```python
# Attention dropout
attn = torch.softmax(scores, dim=-1)
attn = self.attn_dropout(attn)  # NEW

# Output projection dropout
state = residual + self.dropout(self.o_proj(out))  # NEW

# FFN dropout
state = residual_ffn + self.dropout(ff_out)  # NEW
```

---

### 4. ✅ Xavier/Glorot Weight Initialization (Medium Impact)

**File**: `ai_models/model.py`

**Problem**: PyTorch default initialization may not be optimal for this architecture.

**Fix Applied** (new method in `SyndromeTransformerLayer`):
```python
def _init_weights(self):
    """Paper-aligned Xavier/Glorot initialization."""
    # Initialize linear layers with Xavier uniform
    for module in [self.qkv_proj, self.o_proj, self.bias_proj, 
                   self.ff_proj, self.ff_gate, self.ff_out]:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    # Initialize pair MLP layers
    for layer in self.pair_mlp:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    # Initialize conv layers if present
    for conv in self.convs:
        nn.init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
```

---

### 5. ✅ Soft Channel Features in Pre-training (CRITICAL)

**File**: `paper_aligned_pretrain.py`

**Problem**: Pre-training only used 2 features (detection + basis) instead of paper's 3 features (detection, P(|1⟩), P(|L⟩)).

**Fix Applied**:
```python
# NEW: Import soft channels generator
try:
    from google_qec_simulator.data_helpers import soft_channels
    HAS_SOFT_CHANNELS = True
except ImportError:
    HAS_SOFT_CHANNELS = False

class PretrainDataset(Dataset):
    def __init__(self, syndromes, labels, basis_id, grid_size,
                 use_soft_channels=True, snr=10.0, tau=0.01):
        # ... reshape to (N, R, S, 1) ...
        
        # Paper-aligned: Add soft channels
        if use_soft_channels and HAS_SOFT_CHANNELS:
            total_elements = N * R * S
            p1_post, pL_post = soft_channels(total_elements, snr=snr, tau=tau)
            
            p1_feat = p1_post.reshape(N, R, S, 1)
            pL_feat = pL_post.reshape(N, R, S, 1)
            
            # [detection, P(|1⟩), P(|L⟩)] = 3 features
            x = np.concatenate([x, p1_feat, pL_feat], axis=-1)
```

---

### 6. ✅ Train/Val Split Consistency (Minor Impact)

**File**: `paper_aligned_pretrain.py`

**Problem**: Inconsistent split ratios (90/10 vs 95/5) across scripts.

**Fix Applied**:
```python
# Before:
train_size = int(0.9 * len(dataset))

# After (paper-aligned from configs/paper_aligned.yaml):
train_size = int(0.95 * len(dataset))  # 95/5 split
```

---

## Previously Applied Fixes (Jan 4, 2026 Earlier)

These were already documented in `PAPER_ALIGNMENT_FIXES_2026_01_04.md`:

1. **I/Q Readout Model** - Fixed symmetric means (±SNR/2)
2. **Leakage Prior** - Changed from 1% to 0.1%
3. **Pre-LN vs Post-LN** - Fixed LayerNorm placement
4. **SiLU Activation** - Changed GELU to SiLU in model_mla.py

---

## Files Modified

| File | Changes |
|------|---------|
| `ai_models/fine_tune_npz.py` | Class weighting, LR scheduler |
| `ai_models/model.py` | Dropout, Xavier init, _init_weights() |
| `paper_aligned_pretrain.py` | Soft channels, class weighting, 95/5 split |

---

## Data Verification Results

**CRITICAL FINDING**: Existing trained models are INVALID.

Test results show ALL models predict 0 for every sample:
- 111/111 experiments have `true_positives = 0`
- 111/111 experiments have `precision = 0.0, recall = 0.0, f1_score = 0.0`

The "accuracy" metrics are misleading - they reflect the baseline rate, not actual decoding performance.

---

## Recommended Actions

### Immediate (Must Do)

1. **Regenerate ALL pre-training data** with:
   - 3 features (detection, P(|1⟩), P(|L⟩))
   - Paper-aligned I/Q model parameters
   - Full noise range (p=0.001 to 0.01)

2. **Re-run pre-training** for 100 epochs on 8.5M samples

3. **Re-run fine-tuning** with the fixed code:
   - Class-weighted loss will be automatically computed
   - Dropout will prevent overfitting
   - Xavier init will improve convergence

### Verification

After retraining, verify:
- [ ] `true_positives > 0` at all noise levels
- [ ] `f1_score > 0.5` (better than random guessing)
- [ ] LER matches paper within ~20% for d3

---

## Technical Notes

### Why Class Imbalance Matters

At low noise (r01):
- Only ~3-4% of samples have logical errors
- Without weighting: model learns to predict 0 always → 96% "accuracy"
- With weighting: model must learn to distinguish positive cases

### Expected `pos_weight` Values

| Noise Level | Positive Ratio | pos_weight |
|-------------|---------------|------------|
| r01 (low)   | ~3%           | ~32        |
| r05         | ~10%          | ~9         |
| r10         | ~15%          | ~6         |
| r25 (high)  | ~25%          | ~3         |

---

## Summary

The primary cause of result discrepancy was **class imbalance** causing the model to predict all zeros. Combined with missing soft channels and proper regularization, these fixes should significantly improve performance.

**Next Step**: Re-run the full training pipeline on the server with the fixed code.
