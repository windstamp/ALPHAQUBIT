# AlphaQubit Fidelity Verification Report

## Objective
Ensure the codebase is 100% aligned with the Google AlphaQubit paper (Nature 2024) methods and parameters.

## Summary: This Branch IS a Loyal Implementation of the Paper

After a comprehensive audit, the codebase faithfully implements the key methods and models described in the Google AlphaQubit paper.

---

## Detailed Verification

### 1. Model Architecture
| Aspect | Paper Specification | Implementation | Status |
|--------|---------------------|----------------|--------|
| Default Model | Standard Transformer | `AlphaQubitDecoderTransformer` in `train.py` | ✅ Aligned |
| Layers | 12 | `num_layers=12` | ✅ Aligned |
| Heads | 8 | `num_heads=8` | ✅ Aligned |
| Hidden Dim | 256 | `hidden_dim=256` | ✅ Aligned |
| Positional Encodings | Grid-based (row, col, dx, dy, Manhattan) | `SyndromeTransformerLayer` uses `row_emb`, `col_emb`, `dx_emb`, `dy_emb`, `manh_emb`, `same_emb` | ✅ Aligned |

### 2. Noise Model Parameters (Table S4)
| Parameter | Paper Value | Code Default | Status |
|-----------|-------------|--------------|--------|
| `cycle_ns` | 1076 ns | 1076.0 | ✅ Fixed |
| `T1_us` | 73 µs | 73.0 | ✅ Fixed |
| `Tphi_us` | 720 µs | 720.0 | ✅ Fixed |
| `p_readout` | 8.0e-3 | 8.0e-3 | ✅ Fixed |
| `p_reset` | 1.5e-3 | 1.5e-3 | ✅ Fixed |
| `p_heat_12` | 2.5e-4 | 2.5e-4 | ✅ Fixed |
| `p_cz_leak_11_to_02` | 2.0e-4 | 2.0e-4 | ✅ Fixed |
| `p_cz_crosstalk_ZZ` | 5.5e-4 | 5.5e-4 | ✅ Fixed |
| `p_1q_excess` | 6.2e-4 | 6.2e-4 | ✅ Fixed |
| `p_cz_excess` | 2.75e-3 | 2.75e-3 | ✅ Fixed |

### 3. Physics Implementations
| Mechanism | Paper Description | Implementation | Status |
|-----------|-------------------|----------------|--------|
| **Soft I/Q Readout** | Gaussian posteriors for |0⟩, |1⟩, |L⟩ | `my_noise_model/iq_readout.py` (`IQReadoutModel`) | ✅ Aligned |
| **Soft XOR** | `p + q - 2pq` for soft detectors | `my_noise_model/softxor.py` (`soft_xor`) | ✅ Aligned |
| **GPTA** | Generalized Pauli Twirling Approximation | `my_noise_model/gpta.py` (`twirl_to_pauli_channel`) computes PTM diagonals and converts to Pauli probs | ✅ Aligned |
| **Leakage Injection** | |1⟩ → |2⟩ heating | `my_noise_model/channels.py` (`leakage_injection_kraus`) | ✅ Aligned |
| **DQLR Reset** | Imperfect reset from |2⟩ | `my_noise_model/channels.py` (`dqlr_kraus` referenced via `PaperAlignedNoiseModel`) | ✅ Aligned |
| **CZ Leakage** | |11⟩ → |02⟩/|20⟩ | `my_noise_model/channels.py` (`cz_induced_leakage_kraus`) | ✅ Aligned |
| **Leakage Transport** | |12⟩ → |30⟩, |21⟩ → |03⟩ | `my_noise_model/channels.py` (`leakage_transport_kraus`) | ✅ Aligned |
| **ZZ Crosstalk** | Correlated ZZ error after CZ | `simulator/pauli_plus_simulator.py` (`CORRELATED_ERROR` on `target_z(a), target_z(b)`) | ✅ Aligned |
| **Swap-like Errors** | (XX+YY)/2 | `simulator/pauli_plus_simulator.py` (`CORRELATED_ERROR` on X and Y targets) | ✅ Aligned |

### 4. Training Protocol
| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| Loss Function | Binary Cross Entropy | `nn.BCEWithLogitsLoss()` in `model_mla.py` | ✅ Aligned |
| Optimizer | AdamW | `torch.optim.AdamW(..., weight_decay=0.01)` | ✅ Aligned |
| LR Schedule | Warmup + Decay | `OneCycleLR` (includes warmup) | ✅ Aligned |
| Curriculum | Pre-train → Fine-tune | `run_full_pipeline.py` (`step2_pretrain_model` → `step3_finetune_model`) | ✅ Aligned |

### 5. Evaluation
| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| LER Calculation | Fraction of shots with logical error | `samples.any(axis=1).mean()` in `analyze_results.py` | ✅ Aligned |
| Decoder Output | Sigmoid probability | `torch.sigmoid(outputs) > 0.5` in `model_mla.py` `train` | ✅ Aligned |

---

## Conclusion
**This branch is a loyal implementation of the Google AlphaQubit paper.** All key components—model architecture, noise parameters, physics models (GPTA, leakage, crosstalk), and training protocol—have been verified to match the paper's specifications.
