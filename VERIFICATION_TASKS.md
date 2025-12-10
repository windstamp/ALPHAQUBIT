# Verification Tasks for Google Paper Alignment

This document outlines the tasks required to verify that our implementation is 100% aligned with the methods described in the Google AlphaQubit paper.

## 1. Model Architecture Verification
- [x] **Verify Default Architecture**: Confirmed that `train.py` defaults to `AlphaQubitDecoderTransformer` (Standard Transformer), matching the paper.
- [ ] **Verify Input Features**: Ensure the input tensor shape and content match the paper's description (syndromes + basis bit).
    - Check: `ai_models/train.py` `GeneratedSyndromeDataset`.
- [x] **Verify Positional Encodings**: Confirmed `ai_models/model.py` uses grid-based embeddings (`row_emb`, `col_emb`, `dx_emb`, `dy_emb`, `manh_emb`, `same_emb`) in `SyndromeTransformerLayer`, matching the paper's description of relative positional information.
- [x] **Verify Transformer Dimensions**:
    - Confirmed `ai_models/train.py` defaults to `AlphaQubitDecoderTransformer(F, 256, S, grid_size, num_heads=8, num_layers=12)`.
    - This matches the "Large" model configuration (12 layers, 8 heads, d_model=256).
- [ ] **Verify Output Head**: Ensure the readout network (Conv2d + MLP) matches the paper's "decoding head" architecture.

## 2. Noise Model Verification
- [x] **Verify SI1000 Noise**:
    - Check `my_noise_model/si1000.py` against the paper's SI1000 definition.
    - Verify weights: `meas_bitflip=5p`, `reset_bitflip=2p`, etc.
- [ ] **Verify Pauli+ Noise**:
    - Check `my_noise_model/pauli_plus.py` for leakage, crosstalk, and soft readout simulation.
    - Verify the "soft XOR" logic in `my_noise_model/softxor.py`.
- [x] **Verify Paper-Aligned Noise Parameters**:
    - **DONE**: Updated `my_noise_model/paper_aligned.py` and `simulator/pauli_plus_simulator.py` to match Table S4 values (T1=73us, Tphi=720us, etc.).

## 3. Training Protocol Verification
- [x] **Verify Loss Function**:
    - Confirmed `ai_models/model_mla.py` uses `nn.BCEWithLogitsLoss()`.
- [x] **Verify Optimizer Settings**:
    - Confirmed `ai_models/model_mla.py` uses `torch.optim.AdamW(..., weight_decay=0.01)`.
- [x] **Verify Learning Rate Schedule**:
    - Confirmed `ai_models/model_mla.py` uses `torch.optim.lr_scheduler.OneCycleLR`.
- [x] **Verify Curriculum**:
    - Confirmed `run_full_pipeline.py` implements `step2_pretrain_model` followed by `step3_finetune_model`, matching the Pre-training -> Fine-tuning workflow.

## 4. Decoding & Evaluation Verification
- [x] **Verify LER Calculation**:
    - Confirmed `simulated_data/analyze_results.py` calculates logical error as `samples.any(axis=1).mean()`, which is the fraction of shots with at least one error.
- [ ] **Verify Thresholds**: Compare our reproduced thresholds (d=3, 5, 7) with Figure 2 of the paper.

## Action Plan
1.  **Audit Code**: Systematically review `ai_models/`, `my_noise_model/`, and `simulator/` against the paper's Methods section.
2.  **Run Verification Scripts**: Execute the existing tests in `tests/` and create new ones if gaps are found.
3.  **Adjust Defaults**: If any default parameter (e.g., model size, noise rate) differs from the paper, update the code or config files.
