# AlphaQubit Noise Model Verification Report
## Paper Alignment Check — January 2025

This report verifies that our implementation matches the noise model specifications from the Nature 2024 AlphaQubit paper.

---

## 1. SI1000 Circuit-Level Noise Model ✅

**Paper specification (Section S1.3):**
| Parameter | Paper Value | Our Implementation | Status |
|-----------|-------------|-------------------|--------|
| 2Q gate depolarization | p | `after_clifford_depolarization=p` | ✅ |
| Idle depolarization | p/10 | `before_round_data_depolarization=p/10` | ✅ |
| Measurement bitflip | 5p | `before_measure_flip_probability=5*p` | ✅ |
| Reset bitflip | 2p | `after_reset_flip_probability=2*p` | ✅ |

**File:** `simulator/si1000_generator.py`

**Status: FULLY ALIGNED** ✅

---

## 2. Pauli+ Noise Model (Table S4) ✅

**Paper Table S4 Parameters:**

### 2.1 Decoherence
| Parameter | Paper Value | Our Config | Status |
|-----------|-------------|------------|--------|
| T1 (us) | 73 | `T1_us: 73.0` | ✅ |
| T_φ (us) | 720 | `Tphi_us: 720.0` | ✅ |
| Cycle time | 1076 ns | `cycle_ns: 1076.0` | ✅ |

### 2.2 Gate Errors
| Parameter | Paper Value | Our Config | Status |
|-----------|-------------|------------|--------|
| 1Q excess | 6.2e-4 | `p_1q_excess: 6.2e-4` | ✅ |
| CZ excess | 2.75e-3 | `p_cz_excess: 2.75e-3` | ✅ |
| CZ ZZ crosstalk | 5.5e-4 | `p_cz_crosstalk_ZZ: 5.5e-4` | ✅ |
| CZ leakage (|11⟩→|02⟩) | 2.0e-4 | `p_cz_leak_11_to_02: 2.0e-4` | ✅ |

### 2.3 Readout/Reset
| Parameter | Paper Value | Our Config | Status |
|-----------|-------------|------------|--------|
| Readout error | 8e-3 | `p_readout: 8.0e-3` | ✅ |
| Reset error | 1.5e-3 | `p_reset: 1.5e-3` | ✅ |
| Heating (p_heat_12) | 2.5e-4 | `p_heat_12: 2.5e-4` | ✅ |

**File:** `configs/paper_aligned.yaml`, `simulator/pauli_plus_simulator.py`

**Status: FULLY ALIGNED** ✅

---

## 3. I/Q Readout Model (Soft Channels) ✅

**Paper specification (Methods: Soft input representation):**

The I/Q readout model generates three-state posterior probabilities P(|0⟩), P(|1⟩), P(|L⟩).

| Parameter | Paper Value | Our Implementation | Status |
|-----------|-------------|-------------------|--------|
| SNR | 10.0 | `snr=10.0` | ✅ |
| τ (amplitude damping) | 0.01 | `tau=0.01` | ✅ |
| Leakage prior | 0.1% (0.001) | `leak_p=0.001` | ✅ |
| Sigma (|0⟩,|1⟩) | 1.0 | `sigma=1.0` | ✅ |
| Sigma (|L⟩) | 1.6 | `sigma_leak=1.6*sigma` | ✅ |

### I/Q Model Physics:
- **Means:** μ₀ = +SNR/2 = +5, μ₁ = -α·SNR/2 ≈ -4.95 (where α = e^{-τ})
- **Leakage centered at 0** with wider sigma (1.6×)
- **Three-feature output:** [detection_event, P(|1⟩), P(|L⟩)]

**File:** `google_qec_simulator/data_helpers.py`, `my_noise_model/noise_params.yaml`

**Status: FULLY ALIGNED** ✅

---

## 4. GPTA (Generalized Pauli Twirling Approximation) ✅

**Paper specification:** Kraus operators are twirled to diagonal Pauli channels via PTM (Pauli Transfer Matrix) calculation.

**Our implementation (`my_noise_model/gpta.py`):**
- ✅ Computes PTM diagonal elements from Kraus operators
- ✅ Uses Hadamard transform to convert to Pauli probabilities
- ✅ Handles 1Q and 2Q channels separately
- ✅ Estimates leakage probability from evolved states

**Implementation matches paper's GPTA description.**

**Status: ALIGNED** ✅

---

## 5. Kraus Channels (`my_noise_model/channels.py`) ✅

### Implemented Channels:
| Channel | Implementation | Paper Reference |
|---------|---------------|-----------------|
| Amplitude damping | `amplitude_damping_kraus(tau)` | T1 decay |
| Dephasing | `dephasing_kraus(p)` | T_φ decay |
| 1Q depolarizing | `depolarizing_1q_kraus(p)` | Excess gate errors |
| 2Q depolarizing | `depolarizing_2q_kraus(p)` | CZ excess |
| CZ-induced leakage | `cz_induced_leakage_kraus(p)` | |11⟩→|02⟩,|20⟩ |
| Leakage transport | `leakage_transport_kraus(p)` | |12⟩→|30⟩, |21⟩→|03⟩ |
| Crosstalk Z | `spectator_crosstalk_z_kraus(p)` | ZZ crosstalk |
| Multi-level reset | `multi_level_reset_kraus(f, r)` | DQLR |

**All channels use 4-level (two qutrits) or 3-level (single qutrit) representations for leakage support.**

**Status: ALIGNED** ✅

---

## 6. Training Hyperparameters ✅

| Parameter | Paper Value | Our Config | Status |
|-----------|-------------|------------|--------|
| Pre-training samples | 8.5M | `samples: 8500000` | ✅ |
| Batch size | 256 | `batch_size: 256` | ✅ |
| Pre-training LR | 1e-4 | `lr: 1.0e-4` | ✅ |
| Fine-tuning LR | 1e-5 | `finetuning.lr: 1.0e-5` | ✅ |
| Hidden dim | 256 | `hidden_dim: 256` | ✅ |
| Attention heads | 8 | `num_heads: 8` | ✅ |
| Transformer layers | 12 | `num_layers: 12` | ✅ |
| Epochs | 100 | `epochs: 100` | ✅ |
| Optimizer | AdamW | `optimizer: adamw` | ✅ |
| Weight decay | 1e-3 | `weight_decay: 1.0e-3` | ✅ |

**Status: FULLY ALIGNED** ✅

---

## 7. Previously Identified & Fixed Issues

### 7.1 Class Imbalance (FIXED ✅)
- **Problem:** At low noise rates, ~97% of samples had negative labels
- **Fix:** Added `pos_weight = num_neg / num_pos` in BCEWithLogitsLoss
- **Files:** `ai_models/fine_tune_npz.py`, `paper_aligned_pretrain.py`

### 7.2 Missing Soft Channels in Pre-training (FIXED ✅)
- **Problem:** Pre-training used only 1-feature (detection) instead of 3-feature input
- **Fix:** Added P(|1⟩) and P(|L⟩) posteriors via `soft_channels()` function
- **Files:** `paper_aligned_pretrain.py`, `google_qec_simulator/data_helpers.py`

### 7.3 Missing Dropout (FIXED ✅)
- **Problem:** Model lacked regularization
- **Fix:** Added dropout=0.1 in attention and FFN layers
- **File:** `ai_models/model.py`

### 7.4 Missing Xavier Initialization (FIXED ✅)
- **Problem:** Default PyTorch init, not paper-aligned
- **Fix:** Added `_init_weights()` method with Xavier uniform
- **File:** `ai_models/model.py`

### 7.5 LR Scheduler (FIXED ✅)
- **Problem:** Used CosineAnnealingWarmRestarts (periodic resets)
- **Fix:** Changed to plain CosineAnnealingLR
- **File:** `ai_models/fine_tune_npz.py`

---

## 8. Summary

| Component | Status |
|-----------|--------|
| SI1000 noise model | ✅ ALIGNED |
| Pauli+ noise parameters | ✅ ALIGNED |
| I/Q soft channel model | ✅ ALIGNED |
| GPTA implementation | ✅ ALIGNED |
| Kraus channels | ✅ ALIGNED |
| Training hyperparameters | ✅ ALIGNED |
| Class imbalance handling | ✅ FIXED |
| Soft channels in pre-training | ✅ FIXED |
| Model regularization | ✅ FIXED |

### Overall Noise Model Status: **FULLY ALIGNED WITH PAPER** ✅

---

## 9. Next Steps

1. **Sync to server** — Push fixed code
2. **Regenerate training data** — With proper soft channels
3. **Retrain models** — With class weighting and proper initialization
4. **Evaluate** — Compare against paper's ~0.82% threshold

---

*Generated: January 2025*
*Reference: "Learning high-accuracy error decoding for quantum processors" - Nature 2024*
