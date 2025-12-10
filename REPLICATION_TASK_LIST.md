# AlphaQubit Replication Task List

This document outlines the step-by-step process to replicate the results of the AlphaQubit paper using the provided codebase.

## Phase 1: Environment Setup & Verification
- [ ] **Check Python Version**: Ensure Python 3.8+ is installed.
- [ ] **Install Dependencies**: Install required packages from `requirements.txt`.
    - Key packages: `numpy`, `scipy`, `stim`, `pyyaml`, `torch`, `leakysim>=0.4.0`.
- [ ] **Verify Simulator Alignment**: Run verification tests to ensure the simulator matches the paper's specifications.
    - `pytest tests/test_si1000.py` (SI1000 Weights)
    - `pytest tests/test_iq_softxor.py` (I/Q Posteriors + SoftXOR)
    - `pytest tests/test_gpta.py` (GPTA Sanity)

## Phase 2: Data Generation (Pre-training)
- [ ] **Generate Noise Data**: Create synthetic data for pre-training across a range of noise levels.
    - Script: `make_all_pretraining_noise.py` (or via `run_full_pipeline.py`)
    - Config: `configs/pauli_plus.yaml`
    - *Note*: `run_full_pipeline.py` Step 1 handles this.

## Phase 3: Pre-training
- [ ] **Train Base Model**: Train the Transformer-based decoder on the generated synthetic data.
    - Script: `ai_models/train.py` (or via `run_full_pipeline.py`)
    - This creates the foundation for fine-tuning.

## Phase 4: Fine-tuning
- [ ] **Fine-tune on Experiments**: Fine-tune the pre-trained model on specific experimental configurations (118 experiments).
    - Script: `run_finetune_all.py`
    - Data Source: `google_finetune_data/finetune/` (Ensure this data is present or generated).

## Phase 5: Evaluation & Analysis
- [ ] **Evaluate Models**: Run inference on test sets to calculate Logical Error Rate (LER).
    - Script: `test_finetuned_models.py` or `run_decode_all.py`.
- [ ] **Compare with Paper**: Generate plots and statistics to compare against the paper's baseline.
    - Script: `plot_alphaqubit_results.py`
    - Script: `analyze_results.py`

## Phase 6: Diagnosis (Optional/As Needed)
- [ ] **Diagnose Results**: Use `diagnose_current_results.py` to identify underperforming models or configurations.

## Automation
The entire pipeline can be run using the master script:
```bash
python run_full_pipeline.py
```
Use `--quick-test` for a faster verification run with smaller data.
