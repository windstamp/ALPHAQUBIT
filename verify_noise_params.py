#!/usr/bin/env python3
"""
Comprehensive Paper Alignment Verification
==========================================

This script verifies ALL noise model parameters against the paper's specifications.
Run this before training to ensure 100% alignment.
"""

import sys

# =============================================================================
# Paper Specifications (from Table S4 and Methods)
# =============================================================================

PAPER_SI1000_PARAMS = {
    "description": "SI1000 Noise Model (Stim standard)",
    "meas_bitflip": "5p",          # before_measure_flip_probability
    "reset_bitflip": "2p",         # after_reset_flip_probability
    "twoq_depol": "p",             # after_clifford_depolarization
    "oneq_depol": "p/10",          # (not directly settable in stim.Circuit.generated)
    "idle": "p/10",                # before_round_data_depolarization
}

PAPER_PAULI_PLUS_PARAMS = {
    "description": "Pauli+ Noise Model (Table S4)",
    "cycle_ns": 1076.0,
    "T1_us": 73.0,
    "Tphi_us": 720.0,              # Note: NOT T2, but T_phi (pure dephasing)
    "p_readout": 8.0e-3,           # 0.8%
    "p_reset": 1.5e-3,             # 0.15%
    "p_heat_12": 2.5e-4,           # 0.025%
    "p_cz_leak_11_to_02": 2.0e-4,  # 0.02%
    "p_cz_crosstalk_ZZ": 5.5e-4,   # 0.055%
    "p_1q_excess": 6.2e-4,         # 0.062%
    "p_cz_excess": 2.75e-3,        # 0.275%
}

PAPER_PRETRAINING = {
    "total_samples": 8_500_000,
    "p_grid": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    "distances": [3, 5, 7],
    "rounds": [1, 5, 10, 25],
    "batch_size": 256,
    "learning_rate": 1e-4,
    "epochs": 100,
    "weight_decay": 1e-4,
}

PAPER_FINETUNING = {
    "batch_size": 128,
    "learning_rate": 1e-5,        # IMPORTANT: 10x smaller than pretraining!
    "epochs": 30,
    "early_stopping_patience": 5,
    "weight_decay": 1e-3,
}

PAPER_MODEL = {
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 12,
}

# =============================================================================
# Verification Functions
# =============================================================================

def check_si1000_implementation():
    """Check SI1000 noise model implementation."""
    print("\n" + "="*70)
    print("1. SI1000 NOISE MODEL VERIFICATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check simulator/si1000_generator.py
    try:
        with open("simulator/si1000_generator.py", "r") as f:
            content = f.read()
        
        checks = [
            ("before_measure_flip_probability=5 * p", "5p", "measurement flip"),
            ("before_measure_flip_probability=5*p", "5p", "measurement flip"),
            ("after_reset_flip_probability=2 * p", "2p", "reset flip"),
            ("after_reset_flip_probability=2*p", "2p", "reset flip"),
            ("before_round_data_depolarization=p / 10", "p/10", "idle depol"),
            ("before_round_data_depolarization=p/10", "p/10", "idle depol"),
            ("after_clifford_depolarization=p", "p", "2Q depol"),
        ]
        
        found = {name: False for _, _, name in checks}
        for pattern, expected, name in checks:
            if pattern in content:
                found[name] = True
                print(f"  ✅ {name}: {expected}")
        
        for name, ok in found.items():
            if not ok:
                errors.append(f"simulator/si1000_generator.py: Missing {name}")
                print(f"  ❌ {name}: NOT FOUND")
                
    except FileNotFoundError:
        errors.append("simulator/si1000_generator.py not found")
    
    # Check paper_aligned_pretrain.py
    try:
        with open("paper_aligned_pretrain.py", "r") as f:
            content = f.read()
        
        checks = [
            ("before_measure_flip_probability=5 * p", "5p", "measurement flip"),
            ("after_reset_flip_probability=2 * p", "2p", "reset flip"),
            ("before_round_data_depolarization=p / 10", "p/10", "idle depol"),
        ]
        
        print("\n  paper_aligned_pretrain.py:")
        for pattern, expected, name in checks:
            if pattern in content:
                print(f"    ✅ {name}: {expected}")
            else:
                errors.append(f"paper_aligned_pretrain.py: Wrong {name}")
                print(f"    ❌ {name}: WRONG (should be {expected})")
                
    except FileNotFoundError:
        errors.append("paper_aligned_pretrain.py not found")
    
    # Check generate_full_pretrain_data.py
    try:
        with open("generate_full_pretrain_data.py", "r") as f:
            content = f.read()
        
        print("\n  generate_full_pretrain_data.py:")
        for pattern, expected, name in checks:
            if pattern in content:
                print(f"    ✅ {name}: {expected}")
            else:
                errors.append(f"generate_full_pretrain_data.py: Wrong {name}")
                print(f"    ❌ {name}: WRONG (should be {expected})")
                
    except FileNotFoundError:
        warnings.append("generate_full_pretrain_data.py not found (optional)")
    
    return errors, warnings


def check_pauli_plus_implementation():
    """Check Pauli+ noise model implementation."""
    print("\n" + "="*70)
    print("2. PAULI+ NOISE MODEL VERIFICATION (Table S4)")
    print("="*70)
    
    errors = []
    warnings = []
    
    try:
        # Import and check the actual values
        sys.path.insert(0, ".")
        from my_noise_model.paper_aligned import PaperAlignedNoiseConfig
        
        cfg = PaperAlignedNoiseConfig()
        
        checks = [
            ("cycle_ns", cfg.cycle_ns, 1076.0),
            ("T1_us", cfg.T1_us, 73.0),
            ("Tphi_us", cfg.Tphi_us, 720.0),
            ("p_readout", cfg.p_readout, 8.0e-3),
            ("p_reset", cfg.p_reset, 1.5e-3),
            ("p_heat_12", cfg.p_heat_12, 2.5e-4),
            ("p_cz_leak_11_to_02", cfg.p_cz_leak_11_to_02, 2.0e-4),
            ("p_cz_crosstalk_ZZ", cfg.p_cz_crosstalk_ZZ, 5.5e-4),
            ("p_1q_excess", cfg.p_1q_excess, 6.2e-4),
            ("p_cz_excess", cfg.p_cz_excess, 2.75e-3),
        ]
        
        for name, actual, expected in checks:
            if abs(actual - expected) < 1e-10:
                print(f"  ✅ {name}: {actual} == {expected}")
            else:
                errors.append(f"{name}: {actual} != {expected}")
                print(f"  ❌ {name}: {actual} != {expected} (PAPER VALUE)")
        
        # Check DQLR matrix
        expected_dqlr = (
            (1.0, 0.0, 0.05),
            (0.0, 1.0, 0.90),
            (0.0, 0.0, 0.05),
        )
        if cfg.dqlr_matrix == expected_dqlr:
            print(f"  ✅ dqlr_matrix: correct")
        else:
            errors.append("dqlr_matrix incorrect")
            print(f"  ❌ dqlr_matrix: WRONG")
            print(f"     Expected: {expected_dqlr}")
            print(f"     Actual:   {cfg.dqlr_matrix}")
                
    except ImportError as e:
        warnings.append(f"Could not import PaperAlignedNoiseConfig: {e}")
        print(f"  ⚠️ Could not import: {e}")
        print("  Checking file content instead...")
        
        # Fallback: check file content
        try:
            with open("my_noise_model/paper_aligned.py", "r") as f:
                content = f.read()
            
            checks = [
                ("cycle_ns: float = 1076.0", "cycle_ns"),
                ("T1_us: float = 73.0", "T1_us"),
                ("Tphi_us: float = 720.0", "Tphi_us"),
                ("p_readout: float = 8.0e-3", "p_readout"),
                ("p_reset: float = 1.5e-3", "p_reset"),
                ("p_heat_12: float = 2.5e-4", "p_heat_12"),
                ("p_cz_leak_11_to_02: float = 2.0e-4", "p_cz_leak"),
                ("p_cz_crosstalk_ZZ: float = 5.5e-4", "p_cz_crosstalk"),
                ("p_1q_excess: float = 6.2e-4", "p_1q_excess"),
                ("p_cz_excess: float = 2.75e-3", "p_cz_excess"),
            ]
            
            for pattern, name in checks:
                if pattern in content:
                    print(f"  ✅ {name}: found in file")
                else:
                    errors.append(f"{name} not found with correct value")
                    print(f"  ❌ {name}: NOT FOUND or WRONG")
                    
        except FileNotFoundError:
            errors.append("my_noise_model/paper_aligned.py not found")
    
    return errors, warnings


def check_pretraining_params():
    """Check pretraining parameters."""
    print("\n" + "="*70)
    print("3. PRETRAINING PARAMETERS VERIFICATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check paper_aligned_pretrain.py
    try:
        with open("paper_aligned_pretrain.py", "r") as f:
            content = f.read()
        
        checks = [
            ("PAPER_TOTAL_SAMPLES = 8_500_000", "total_samples: 8.5M"),
            ("PAPER_TOTAL_SAMPLES = 8500000", "total_samples: 8.5M"),
            ("0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01", "p_grid: 10 levels"),
            ("PAPER_CODE_DISTANCES = [3, 5, 7]", "distances: [3,5,7]"),
            ("PAPER_ROUNDS = [1, 5, 10, 25]", "rounds: [1,5,10,25]"),
            ("PAPER_BATCH_SIZE = 256", "batch_size: 256"),
            ("PAPER_LR = 1e-4", "learning_rate: 1e-4"),
            ("PAPER_EPOCHS = 100", "epochs: 100"),
            ("PAPER_WEIGHT_DECAY = 1e-4", "weight_decay: 1e-4"),
        ]
        
        for pattern, name in checks:
            if pattern in content:
                print(f"  ✅ {name}")
            else:
                errors.append(f"Pretraining {name} not found")
                print(f"  ❌ {name}: NOT FOUND")
                
    except FileNotFoundError:
        errors.append("paper_aligned_pretrain.py not found")
    
    return errors, warnings


def check_finetuning_params():
    """Check finetuning parameters."""
    print("\n" + "="*70)
    print("4. FINETUNING PARAMETERS VERIFICATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check run_finetune_all.py
    try:
        with open("run_finetune_all.py", "r") as f:
            content = f.read()
        
        # Critical: lr must be 1e-5, not 1e-4!
        if "default=1e-5" in content or "default='1e-5'" in content:
            print(f"  ✅ learning_rate default: 1e-5")
        elif "default=1e-4" in content:
            errors.append("run_finetune_all.py: lr default is 1e-4, should be 1e-5!")
            print(f"  ❌ learning_rate default: 1e-4 (SHOULD BE 1e-5!)")
        else:
            warnings.append("Could not verify lr default in run_finetune_all.py")
            print(f"  ⚠️ learning_rate: could not verify")
        
        # Check other params
        checks = [
            ("--batch-size", "128", "batch_size"),
            ("--epochs", "30", "epochs"),
            ("--patience", "5", "early_stopping"),
        ]
        
        for arg, expected, name in checks:
            if f"{arg}" in content and expected in content:
                print(f"  ✅ {name}: {expected} (in argparse)")
            else:
                warnings.append(f"Could not verify {name}")
                
    except FileNotFoundError:
        errors.append("run_finetune_all.py not found")
    
    # Check ai_models/fine_tune_npz.py
    try:
        with open("ai_models/fine_tune_npz.py", "r") as f:
            content = f.read()
        
        if "BCEWithLogitsLoss" in content:
            print(f"  ✅ loss: BCEWithLogitsLoss")
        else:
            errors.append("BCEWithLogitsLoss not found in fine_tune_npz.py")
            print(f"  ❌ loss: NOT BCEWithLogitsLoss")
            
    except FileNotFoundError:
        warnings.append("ai_models/fine_tune_npz.py not found")
    
    return errors, warnings


def check_model_architecture():
    """Check model architecture parameters."""
    print("\n" + "="*70)
    print("5. MODEL ARCHITECTURE VERIFICATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    try:
        with open("ai_models/model.py", "r") as f:
            content = f.read()
        
        # These should be the defaults or commonly used
        checks = [
            ("hidden_dim", "256"),
            ("num_heads", "8"),
            ("num_layers", "12"),
        ]
        
        for param, expected in checks:
            # Just check if these values appear in the file
            if expected in content:
                print(f"  ✅ {param}: {expected} (found in file)")
            else:
                warnings.append(f"{param}={expected} not found")
                print(f"  ⚠️ {param}: {expected} not found (may be passed as argument)")
        
        # Check activation function
        if "SiLU" in content or "silu" in content:
            print(f"  ✅ activation: SiLU/Swish")
        else:
            warnings.append("SiLU activation not found")
            print(f"  ⚠️ activation: SiLU not found")
        
        # Check LayerNorm
        if "LayerNorm" in content:
            print(f"  ✅ normalization: LayerNorm")
        else:
            errors.append("LayerNorm not found")
            print(f"  ❌ normalization: LayerNorm NOT FOUND")
            
    except FileNotFoundError:
        errors.append("ai_models/model.py not found")
    
    return errors, warnings


def main():
    print("="*70)
    print("ALPHAQUBIT PAPER ALIGNMENT VERIFICATION")
    print("="*70)
    print("Checking all parameters against:")
    print("  - Nature 2024: 10.1038/s41586-024-08449-y")
    print("  - Table S4: Pauli+ noise parameters")
    print("  - Methods: Training hyperparameters")
    
    all_errors = []
    all_warnings = []
    
    # Run all checks
    e, w = check_si1000_implementation()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_pauli_plus_implementation()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_pretraining_params()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_finetuning_params()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_model_architecture()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_errors:
        print(f"\n❌ ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"   - {e}")
    
    if all_warnings:
        print(f"\n⚠️ WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"   - {w}")
    
    if not all_errors and not all_warnings:
        print("\n✅ ALL CHECKS PASSED! Parameters are 100% aligned with paper.")
    elif not all_errors:
        print(f"\n✅ No critical errors. {len(all_warnings)} warnings (non-critical).")
    else:
        print(f"\n❌ {len(all_errors)} CRITICAL ERRORS found! Fix before training.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
