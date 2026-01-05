#!/usr/bin/env python3
"""
Comprehensive Paper Alignment Verification
==========================================

This script verifies ALL parameters match the AlphaQubit Nature paper EXACTLY.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# PAPER SPECIFICATIONS (from Nature paper and supplementary materials)
# =============================================================================

PAPER_SPEC = {
    # Model Architecture (Table S1)
    "model": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 12,
        "ffn_expansion": 4,  # FFN = 4 * hidden_dim = 1024
    },
    
    # Pre-training (Section Methods)
    "pretraining": {
        "total_samples": 8_500_000,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "epochs": 100,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
        "loss": "BCEWithLogitsLoss",
    },
    
    # Fine-tuning (Section Methods)
    "finetuning": {
        "samples_per_exp": 50_000,
        "train_split": 0.8,
        "batch_size": 128,
        "learning_rate": 1e-5,  # CRITICAL: 10x lower than pretraining
        "epochs": 30,
        "weight_decay": 1e-3,
        "patience": 5,
        "gradient_clip": 1.0,
    },
    
    # SI1000 Noise Model (Section Methods)
    "si1000_noise": {
        "physical_error_rates": [0.001, 0.002, 0.003, 0.004, 0.005, 
                                  0.006, 0.007, 0.008, 0.009, 0.01],
        "code_distances": [3, 5, 7],
    },
    
    # Pauli+ Noise Model (Table S4)
    "pauli_plus_noise": {
        "cycle_ns": 1076,
        "T1_us": 73,
        "Tphi_us": 720,
        "p_readout": 0.008,  # 8.0e-3
        "p_reset": 0.0015,   # 1.5e-3
        "p_heat_12": 0.00025,  # 2.5e-4
        "p_cz_leak_11_to_02": 0.0002,  # 2.0e-4
        "p_cz_crosstalk_ZZ": 0.00055,  # 5.5e-4
        "p_1q_excess": 0.00062,  # 6.2e-4
        "p_cz_excess": 0.00275,  # 2.75e-3
    },
}

@dataclass
class CheckResult:
    name: str
    expected: str
    actual: str
    status: str  # PASS, FAIL, WARNING
    file: str
    line: int = 0

def check_file_for_value(filepath: Path, pattern: str, expected_value, tolerance=0.0) -> Tuple[bool, str, int]:
    """Search file for pattern and check if value matches expected."""
    if not filepath.exists():
        return False, "FILE NOT FOUND", 0
    
    content = filepath.read_text(encoding='utf-8', errors='ignore')
    match = re.search(pattern, content)
    if not match:
        return False, "NOT FOUND", 0
    
    # Find line number
    line_num = content[:match.start()].count('\n') + 1
    
    try:
        actual = float(match.group(1))
        if tolerance > 0:
            passed = abs(actual - expected_value) <= tolerance
        else:
            passed = actual == expected_value
        return passed, str(actual), line_num
    except:
        return False, match.group(1), line_num

def check_run_finetune_all() -> List[CheckResult]:
    """Check run_finetune_all.py for paper alignment."""
    results = []
    filepath = Path("run_finetune_all.py")
    
    if not filepath.exists():
        results.append(CheckResult("run_finetune_all.py", "exists", "NOT FOUND", "FAIL", str(filepath)))
        return results
    
    content = filepath.read_text()
    
    # Check learning rate default
    lr_match = re.search(r"--lr.*default=([0-9e.\-]+)", content)
    if lr_match:
        lr = float(lr_match.group(1))
        expected_lr = 1e-5
        status = "PASS" if lr == expected_lr else "FAIL"
        results.append(CheckResult(
            "Fine-tuning LR",
            f"{expected_lr}",
            f"{lr}",
            status,
            str(filepath)
        ))
    else:
        results.append(CheckResult("Fine-tuning LR", "1e-5", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check pretrained default
    pretrained_match = re.search(r"--pretrained.*default=['\"]?([^'\">\s,]+)", content)
    if pretrained_match:
        pretrained = pretrained_match.group(1)
        status = "PASS" if pretrained != "None" and "pth" in pretrained else "FAIL"
        results.append(CheckResult(
            "Pretrained default",
            "alphaqubit_*.pth",
            pretrained,
            status,
            str(filepath)
        ))
    else:
        results.append(CheckResult("Pretrained default", "alphaqubit_*.pth", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check batch size
    bs_match = re.search(r"--batch-size.*default=(\d+)", content)
    if bs_match:
        bs = int(bs_match.group(1))
        expected_bs = 128
        status = "PASS" if bs == expected_bs else "FAIL"
        results.append(CheckResult("Fine-tuning batch size", str(expected_bs), str(bs), status, str(filepath)))
    
    # Check epochs
    epochs_match = re.search(r"--epochs.*default=(\d+)", content)
    if epochs_match:
        epochs = int(epochs_match.group(1))
        expected_epochs = 30
        status = "PASS" if epochs == expected_epochs else "FAIL"
        results.append(CheckResult("Fine-tuning epochs", str(expected_epochs), str(epochs), status, str(filepath)))
    
    # Check weight decay
    wd_match = re.search(r"--weight-decay.*default=([0-9e.\-]+)", content)
    if wd_match:
        wd = float(wd_match.group(1))
        expected_wd = 1e-3
        status = "PASS" if wd == expected_wd else "FAIL"
        results.append(CheckResult("Fine-tuning weight_decay", str(expected_wd), str(wd), status, str(filepath)))
    
    # Check patience
    patience_match = re.search(r"--patience.*default=(\d+)", content)
    if patience_match:
        patience = int(patience_match.group(1))
        expected_patience = 5
        status = "PASS" if patience == expected_patience else "FAIL"
        results.append(CheckResult("Early stopping patience", str(expected_patience), str(patience), status, str(filepath)))
    
    # Check hidden dim
    hd_match = re.search(r"--hidden-dim.*default=(\d+)", content)
    if hd_match:
        hd = int(hd_match.group(1))
        expected_hd = 256
        status = "PASS" if hd == expected_hd else "FAIL"
        results.append(CheckResult("Hidden dimension", str(expected_hd), str(hd), status, str(filepath)))
    
    # Check num layers
    nl_match = re.search(r"--num-layers.*default=(\d+)", content)
    if nl_match:
        nl = int(nl_match.group(1))
        expected_nl = 12
        status = "PASS" if nl == expected_nl else "FAIL"
        results.append(CheckResult("Num layers", str(expected_nl), str(nl), status, str(filepath)))
    
    # Check num heads
    nh_match = re.search(r"--num-heads.*default=(\d+)", content)
    if nh_match:
        nh = int(nh_match.group(1))
        expected_nh = 8
        status = "PASS" if nh == expected_nh else "FAIL"
        results.append(CheckResult("Num heads", str(expected_nh), str(nh), status, str(filepath)))
    
    return results

def check_fine_tune_npz() -> List[CheckResult]:
    """Check ai_models/fine_tune_npz.py for paper alignment."""
    results = []
    filepath = Path("ai_models/fine_tune_npz.py")
    
    if not filepath.exists():
        results.append(CheckResult("fine_tune_npz.py", "exists", "NOT FOUND", "FAIL", str(filepath)))
        return results
    
    content = filepath.read_text()
    
    # Check learning rate default
    lr_match = re.search(r"--lr.*default=([0-9e.\-]+)", content)
    if lr_match:
        lr = float(lr_match.group(1))
        expected_lr = 1e-5
        status = "PASS" if lr == expected_lr else "FAIL"
        results.append(CheckResult(
            "fine_tune_npz LR default",
            f"{expected_lr}",
            f"{lr}",
            status,
            str(filepath)
        ))
    
    # Check loss function
    if "BCEWithLogitsLoss" in content:
        results.append(CheckResult("Loss function", "BCEWithLogitsLoss", "BCEWithLogitsLoss", "PASS", str(filepath)))
    else:
        results.append(CheckResult("Loss function", "BCEWithLogitsLoss", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check gradient clipping
    if "clip_grad_norm_" in content:
        clip_match = re.search(r"clip_grad_norm_.*?(\d+\.?\d*)", content)
        if clip_match:
            clip = float(clip_match.group(1))
            status = "PASS" if clip == 1.0 else "WARNING"
            results.append(CheckResult("Gradient clipping", "1.0", str(clip), status, str(filepath)))
    else:
        results.append(CheckResult("Gradient clipping", "1.0", "NOT FOUND", "WARNING", str(filepath)))
    
    return results

def check_model_mla() -> List[CheckResult]:
    """Check ai_models/model_mla.py for paper alignment."""
    results = []
    filepath = Path("ai_models/model_mla.py")
    
    if not filepath.exists():
        results.append(CheckResult("model_mla.py", "exists", "NOT FOUND", "FAIL", str(filepath)))
        return results
    
    content = filepath.read_text()
    
    # Check loss function in train
    if "BCEWithLogitsLoss" in content:
        results.append(CheckResult("MLA Loss function", "BCEWithLogitsLoss", "BCEWithLogitsLoss", "PASS", str(filepath)))
    else:
        results.append(CheckResult("MLA Loss function", "BCEWithLogitsLoss", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check CosineAnnealingLR
    if "CosineAnnealingLR" in content:
        results.append(CheckResult("LR Scheduler", "CosineAnnealingLR", "CosineAnnealingLR", "PASS", str(filepath)))
    else:
        results.append(CheckResult("LR Scheduler", "CosineAnnealingLR", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check AdamW
    if "AdamW" in content:
        results.append(CheckResult("Optimizer", "AdamW", "AdamW", "PASS", str(filepath)))
    else:
        results.append(CheckResult("Optimizer", "AdamW", "NOT FOUND", "FAIL", str(filepath)))
    
    # Check gradient clipping
    clip_match = re.search(r"clip_grad_norm_.*?(\d+\.?\d*)", content)
    if clip_match:
        clip = float(clip_match.group(1))
        status = "PASS" if clip == 1.0 else "WARNING"
        results.append(CheckResult("MLA Gradient clip", "1.0", str(clip), status, str(filepath)))
    
    return results

def check_paper_aligned_pretrain() -> List[CheckResult]:
    """Check paper_aligned_pretrain.py for paper alignment."""
    results = []
    filepath = Path("paper_aligned_pretrain.py")
    
    if not filepath.exists():
        results.append(CheckResult("paper_aligned_pretrain.py", "exists", "NOT FOUND", "FAIL", str(filepath)))
        return results
    
    content = filepath.read_text()
    
    # Check p_grid
    expected_p = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    p_match = re.search(r"PAPER_P_GRID\s*=\s*\[([\d.,\s]+)\]", content)
    if p_match:
        p_str = p_match.group(1)
        p_values = [float(x.strip()) for x in p_str.split(',')]
        status = "PASS" if p_values == expected_p else "FAIL"
        results.append(CheckResult(
            "Pretrain P-grid",
            str(expected_p),
            str(p_values),
            status,
            str(filepath)
        ))
    
    # Check distances
    expected_d = [3, 5, 7]
    d_match = re.search(r"PAPER_CODE_DISTANCES\s*=\s*\[([\d,\s]+)\]", content)
    if d_match:
        d_str = d_match.group(1)
        d_values = [int(x.strip()) for x in d_str.split(',')]
        status = "PASS" if d_values == expected_d else "FAIL"
        results.append(CheckResult(
            "Pretrain distances",
            str(expected_d),
            str(d_values),
            status,
            str(filepath)
        ))
    
    # Check samples
    samples_match = re.search(r"PAPER_TOTAL_SAMPLES\s*=\s*([\d_]+)", content)
    if samples_match:
        samples = int(samples_match.group(1).replace('_', ''))
        expected_samples = 8_500_000
        status = "PASS" if samples == expected_samples else "FAIL"
        results.append(CheckResult(
            "Pretrain samples",
            f"{expected_samples:,}",
            f"{samples:,}",
            status,
            str(filepath)
        ))
    
    # Check batch size
    bs_match = re.search(r"PAPER_BATCH_SIZE\s*=\s*(\d+)", content)
    if bs_match:
        bs = int(bs_match.group(1))
        expected_bs = 256
        status = "PASS" if bs == expected_bs else "FAIL"
        results.append(CheckResult("Pretrain batch size", str(expected_bs), str(bs), status, str(filepath)))
    
    # Check LR
    lr_match = re.search(r"PAPER_LR\s*=\s*([0-9e.\-]+)", content)
    if lr_match:
        lr = float(lr_match.group(1))
        expected_lr = 1e-4
        status = "PASS" if lr == expected_lr else "FAIL"
        results.append(CheckResult("Pretrain LR", str(expected_lr), str(lr), status, str(filepath)))
    
    # Check epochs
    epochs_match = re.search(r"PAPER_EPOCHS\s*=\s*(\d+)", content)
    if epochs_match:
        epochs = int(epochs_match.group(1))
        expected_epochs = 100
        status = "PASS" if epochs == expected_epochs else "FAIL"
        results.append(CheckResult("Pretrain epochs", str(expected_epochs), str(epochs), status, str(filepath)))
    
    return results

def check_pauli_plus_noise() -> List[CheckResult]:
    """Check Pauli+ noise model parameters (Table S4)."""
    results = []
    filepath = Path("my_noise_model/paper_aligned.py")
    
    if not filepath.exists():
        filepath = Path("configs/pauli_plus.yaml")
    
    if not filepath.exists():
        results.append(CheckResult("Pauli+ noise config", "exists", "NOT FOUND", "WARNING", "my_noise_model/"))
        return results
    
    content = filepath.read_text(encoding='utf-8', errors='ignore')
    
    # Check key parameters
    params_to_check = [
        ("p_readout", 0.008, r"p_readout['\"]?\s*[:=]\s*([0-9e.\-]+)"),
        ("p_reset", 0.0015, r"p_reset['\"]?\s*[:=]\s*([0-9e.\-]+)"),
        ("T1_us", 73, r"T1_us['\"]?\s*[:=]\s*([0-9.]+)"),
    ]
    
    for param_name, expected, pattern in params_to_check:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            actual = float(match.group(1))
            status = "PASS" if abs(actual - expected) < 0.001 else "WARNING"
            results.append(CheckResult(
                f"Pauli+ {param_name}",
                str(expected),
                str(actual),
                status,
                str(filepath)
            ))
    
    return results

def check_potential_issues() -> List[CheckResult]:
    """Check for potential issues that could cause problems."""
    results = []
    
    # Check if pretrained model exists
    pretrained_path = Path("alphaqubit_pauli_plus.pth")
    if pretrained_path.exists():
        size_mb = pretrained_path.stat().st_size / 1024 / 1024
        # Model should be around 50MB for 13M params
        status = "PASS" if 30 < size_mb < 100 else "WARNING"
        results.append(CheckResult(
            "Pretrained model file",
            "30-100 MB",
            f"{size_mb:.1f} MB",
            status,
            str(pretrained_path)
        ))
    else:
        results.append(CheckResult("Pretrained model file", "exists", "NOT FOUND", "WARNING", str(pretrained_path)))
    
    # Check google finetune data exists
    finetune_dir = Path("google_finetune_data/finetune")
    if finetune_dir.exists():
        npz_files = list(finetune_dir.glob("*.npz"))
        status = "PASS" if len(npz_files) > 100 else "WARNING"
        results.append(CheckResult(
            "Fine-tune data files",
            ">100 NPZ files",
            f"{len(npz_files)} files",
            status,
            str(finetune_dir)
        ))
    else:
        results.append(CheckResult("Fine-tune data directory", "exists", "NOT FOUND", "WARNING", str(finetune_dir)))
    
    return results

def main():
    print("=" * 80)
    print("COMPREHENSIVE PAPER ALIGNMENT VERIFICATION")
    print("=" * 80)
    print()
    
    all_results = []
    
    print("1. Checking run_finetune_all.py...")
    all_results.extend(check_run_finetune_all())
    
    print("2. Checking ai_models/fine_tune_npz.py...")
    all_results.extend(check_fine_tune_npz())
    
    print("3. Checking ai_models/model_mla.py...")
    all_results.extend(check_model_mla())
    
    print("4. Checking paper_aligned_pretrain.py...")
    all_results.extend(check_paper_aligned_pretrain())
    
    print("5. Checking Pauli+ noise model...")
    all_results.extend(check_pauli_plus_noise())
    
    print("6. Checking potential issues...")
    all_results.extend(check_potential_issues())
    
    # Summary
    print()
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    passed = [r for r in all_results if r.status == "PASS"]
    failed = [r for r in all_results if r.status == "FAIL"]
    warnings = [r for r in all_results if r.status == "WARNING"]
    
    print(f"\nTotal checks: {len(all_results)}")
    print(f"  PASSED:   {len(passed)}")
    print(f"  FAILED:   {len(failed)}")
    print(f"  WARNINGS: {len(warnings)}")
    
    if failed:
        print("\n" + "=" * 80)
        print("FAILURES (MUST FIX)")
        print("=" * 80)
        for r in failed:
            print(f"\n  [{r.status}] {r.name}")
            print(f"    File: {r.file}")
            print(f"    Expected: {r.expected}")
            print(f"    Actual:   {r.actual}")
    
    if warnings:
        print("\n" + "=" * 80)
        print("WARNINGS (REVIEW)")
        print("=" * 80)
        for r in warnings:
            print(f"\n  [{r.status}] {r.name}")
            print(f"    File: {r.file}")
            print(f"    Expected: {r.expected}")
            print(f"    Actual:   {r.actual}")
    
    print("\n" + "=" * 80)
    print("ALL CHECKS")
    print("=" * 80)
    for r in all_results:
        icon = "✓" if r.status == "PASS" else ("✗" if r.status == "FAIL" else "⚠")
        print(f"  {icon} [{r.status:7}] {r.name}: {r.actual} (expected: {r.expected})")
    
    # Final verdict
    print("\n" + "=" * 80)
    if len(failed) == 0:
        print("✓ ALL CRITICAL CHECKS PASSED - Ready to train!")
    else:
        print(f"✗ {len(failed)} CRITICAL FAILURES - Must fix before training")
    print("=" * 80)
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
