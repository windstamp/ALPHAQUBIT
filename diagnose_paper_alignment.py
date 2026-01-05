#!/usr/bin/env python3
"""
Diagnose the exact issues causing our model to underperform vs the paper.
Produces a clear action plan to replicate the paper's results EXACTLY.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Paper specifications (from PAPER_ALIGNMENT_SPEC.md and paper_data.py)
PAPER_SPEC = {
    "pretraining": {
        "total_samples": 8_500_000,
        "noise_model": "SI1000 + Pauli+",
        "physical_error_rates": [0.001, 0.002, 0.003, 0.004, 0.005, 
                                  0.006, 0.007, 0.008, 0.009, 0.01],
        "code_distances": [3, 5, 7],
        "rounds": [1, 5, 10, 25],
        "batch_size": 256,
        "learning_rate": 1e-4,
        "epochs": 100,
        "weight_decay": 1e-4,
        "scheduler": "cosine_annealing",
        "loss": "BCEWithLogitsLoss (no class weights)",
    },
    "finetuning": {
        "samples_per_experiment": 50_000,
        "train_val_split": "80/20",
        "batch_size": 128,
        "learning_rate": 1e-5,  # CRITICAL: 10x lower than pretraining!
        "epochs": 30,
        "weight_decay": 1e-3,
        "patience": 5,
        "gradient_clipping": 1.0,
    },
    "model": {
        "architecture": "Transformer (Large)",
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 12,
        "total_params": "~8M",
    },
    "expected_results": {
        "average_ler": 0.03,  # 3%
        "average_accuracy": 0.97,  # 97%
        "d3_r01_ler": 0.026,  # 2.6%
        "d5_r01_ler": 0.015,  # 1.5%
    }
}

@dataclass
class Issue:
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    component: str
    expected: str
    actual: str
    impact: str
    fix: str

def check_run_finetune_all():
    """Check run_finetune_all.py for paper alignment."""
    issues = []
    
    # Check learning rate
    script_path = Path("run_finetune_all.py")
    if script_path.exists():
        content = script_path.read_text()
        
        # Check default lr
        if "'--lr', type=float, default=1e-4" in content or "default=1e-4" in content:
            issues.append(Issue(
                severity="CRITICAL",
                component="run_finetune_all.py",
                expected="Fine-tuning lr = 1e-5",
                actual="lr = 1e-4 (10x too high!)",
                impact="Model trains too aggressively, may overfit or diverge",
                fix="Change --lr default from 1e-4 to 1e-5"
            ))
        
        # Check pretrained default
        if "default=None" in content and "--pretrained" in content:
            issues.append(Issue(
                severity="CRITICAL", 
                component="run_finetune_all.py",
                expected="Pre-trained model should be provided",
                actual="--pretrained defaults to None",
                impact="Fine-tuning from scratch without pre-training defeats the purpose!",
                fix="Set default to 'alphaqubit_pauli_plus.pth' or require --pretrained"
            ))
    
    return issues

def check_s3_results():
    """Check the S3 results for issues."""
    issues = []
    
    # Load test results
    results_file = Path("s3_results/20251230_220521/test_results_v2/test_summary.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        # Count zero-prediction experiments
        zero_pred_count = 0
        for exp in results.get("results", []):
            metrics = exp.get("metrics", {})
            if metrics.get("true_positives", 0) == 0 and metrics.get("false_positives", 0) == 0:
                zero_pred_count += 1
        
        if zero_pred_count > 0:
            issues.append(Issue(
                severity="CRITICAL",
                component="Model predictions",
                expected="Model should predict some positives",
                actual=f"{zero_pred_count}/{len(results.get('results', []))} experiments have ALL ZEROS predictions",
                impact="Model never learned to predict errors, just predicts majority class",
                fix="1. Use pre-training across full noise range\n"
                    "   2. Use correct fine-tuning lr (1e-5)\n"
                    "   3. Consider class-weighted loss if needed"
            ))
        
        # Check average LER
        avg_ler = results.get("average_metrics", {}).get("logical_error_rate", 0)
        if avg_ler > 0.05:  # More than 5%
            issues.append(Issue(
                severity="HIGH",
                component="Overall performance",
                expected=f"Average LER ~ {PAPER_SPEC['expected_results']['average_ler']:.1%}",
                actual=f"Average LER = {avg_ler:.1%}",
                impact=f"Performance is {avg_ler/PAPER_SPEC['expected_results']['average_ler']:.1f}x worse than paper",
                fix="Follow paper's training pipeline exactly"
            ))
    
    return issues

def check_pretrained_model():
    """Check if pretrained model exists and is valid."""
    issues = []
    
    model_path = Path("alphaqubit_pauli_plus.pth")
    if not model_path.exists():
        issues.append(Issue(
            severity="CRITICAL",
            component="Pre-trained model",
            expected="alphaqubit_pauli_plus.pth should exist with 8.5M samples pre-training",
            actual="File not found",
            impact="Cannot fine-tune without pre-training",
            fix="Run pre-training with make_all_pretraining_noise.py + ai_models/train.py"
        ))
    else:
        import torch
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            n_params = sum(v.numel() for v in state_dict.values())
            
            # Paper Large model should have ~8M params
            if n_params < 7_000_000 or n_params > 10_000_000:
                issues.append(Issue(
                    severity="HIGH",
                    component="Pre-trained model size",
                    expected="~8M parameters",
                    actual=f"{n_params:,} parameters",
                    impact="Model may be wrong architecture",
                    fix="Verify model was trained with hidden_dim=256, num_layers=12, num_heads=8"
                ))
        except Exception as e:
            issues.append(Issue(
                severity="HIGH",
                component="Pre-trained model",
                expected="Valid PyTorch state dict",
                actual=f"Load error: {e}",
                impact="Cannot use pre-trained model",
                fix="Check model file integrity"
            ))
    
    return issues

def check_pretraining_data():
    """Check if pre-training was done with correct noise distribution."""
    issues = []
    
    # Check if pre-training data covers full range
    # This is inferred from the config files
    
    configs_to_check = [
        ("configs/pauli_plus.yaml", "depolarization", 0.001),
    ]
    
    for config_path, key, suspicious_value in configs_to_check:
        path = Path(config_path)
        if path.exists():
            content = path.read_text()
            if f"{key}: {suspicious_value}" in content or f'"{key}": {suspicious_value}' in content:
                issues.append(Issue(
                    severity="HIGH",
                    component=f"Pre-training config ({config_path})",
                    expected=f"Noise should cover p=0.001 to p=0.01 range",
                    actual=f"Only {key}={suspicious_value} (single low noise level)",
                    impact="Model only sees low-noise data, cannot generalize",
                    fix="Pre-train with full SI1000 p-grid: 0.001, 0.002, ..., 0.01"
                ))
    
    return issues

def generate_action_plan(issues: List[Issue]) -> str:
    """Generate a prioritized action plan."""
    
    critical = [i for i in issues if i.severity == "CRITICAL"]
    high = [i for i in issues if i.severity == "HIGH"]
    medium = [i for i in issues if i.severity == "MEDIUM"]
    
    plan = ["=" * 80]
    plan.append("ACTION PLAN TO REPLICATE PAPER RESULTS")
    plan.append("=" * 80)
    plan.append("")
    
    if critical:
        plan.append("🔴 CRITICAL ISSUES (Must Fix)")
        plan.append("-" * 40)
        for i, issue in enumerate(critical, 1):
            plan.append(f"\n{i}. {issue.component}")
            plan.append(f"   Expected: {issue.expected}")
            plan.append(f"   Actual: {issue.actual}")
            plan.append(f"   Impact: {issue.impact}")
            plan.append(f"   Fix: {issue.fix}")
        plan.append("")
    
    if high:
        plan.append("🟠 HIGH PRIORITY ISSUES")
        plan.append("-" * 40)
        for i, issue in enumerate(high, 1):
            plan.append(f"\n{i}. {issue.component}")
            plan.append(f"   Expected: {issue.expected}")
            plan.append(f"   Actual: {issue.actual}")
            plan.append(f"   Fix: {issue.fix}")
        plan.append("")
    
    plan.append("=" * 80)
    plan.append("STEP-BY-STEP FIX PROCEDURE")
    plan.append("=" * 80)
    plan.append("""
Step 1: Fix run_finetune_all.py learning rate
-------------------------------------------------
Change line: parser.add_argument('--lr', type=float, default=1e-4, ...)
To:          parser.add_argument('--lr', type=float, default=1e-5, ...)

Step 2: Verify pre-trained model
-------------------------------------------------
The pre-trained model should have been trained on:
- 8.5M samples
- SI1000 noise with p ∈ {0.001, 0.002, ..., 0.01}
- Code distances d ∈ {3, 5, 7}
- 100 epochs

If pre-training was incorrect, re-run:
```
python make_all_pretraining_noise.py --si1000-samples 8500000 \\
    --si1000-p-grid 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01

python ai_models/train.py --config configs/si1000.yaml --epochs 100 \\
    --batch-size 256 --lr 1e-4 --model-path alphaqubit_pretrained.pth
```

Step 3: Re-run fine-tuning with correct settings
-------------------------------------------------
```
python run_finetune_all.py \\
    --pretrained alphaqubit_pauli_plus.pth \\
    --lr 1e-5 \\
    --epochs 30 \\
    --batch-size 128 \\
    --patience 5 \\
    --data-dir google_finetune_data/finetune \\
    --output-dir finetuned_models_fixed
```

Step 4: Re-run testing
-------------------------------------------------
```
python run_decode_all.py --model-dir finetuned_models_fixed \\
    --data-dir google_finetune_data/test \\
    --output-dir test_results_fixed
```

Expected outcome:
- Average LER should be ~3% (not ~20%)
- Model should predict some positive errors (TP > 0)
- F1 score should be meaningful (> 0.5)
""")
    
    return "\n".join(plan)

def main():
    print("=" * 80)
    print("ALPHAQUBIT PAPER ALIGNMENT DIAGNOSTIC")
    print("=" * 80)
    print()
    
    all_issues = []
    
    print("Checking run_finetune_all.py...")
    all_issues.extend(check_run_finetune_all())
    
    print("Checking S3 results...")
    all_issues.extend(check_s3_results())
    
    print("Checking pre-trained model...")
    all_issues.extend(check_pretrained_model())
    
    print("Checking pre-training data config...")
    all_issues.extend(check_pretraining_data())
    
    print()
    print(f"Found {len(all_issues)} issues:")
    print(f"  - CRITICAL: {len([i for i in all_issues if i.severity == 'CRITICAL'])}")
    print(f"  - HIGH: {len([i for i in all_issues if i.severity == 'HIGH'])}")
    print(f"  - MEDIUM: {len([i for i in all_issues if i.severity == 'MEDIUM'])}")
    print()
    
    # Generate and print action plan
    action_plan = generate_action_plan(all_issues)
    print(action_plan)
    
    # Save report
    report_path = Path("PAPER_ALIGNMENT_FIX_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Paper Alignment Fix Report\n\n")
        f.write(f"Generated: {__import__('datetime').datetime.now()}\n\n")
        f.write("## Issues Found\n\n")
        for issue in all_issues:
            f.write(f"### [{issue.severity}] {issue.component}\n")
            f.write(f"- **Expected**: {issue.expected}\n")
            f.write(f"- **Actual**: {issue.actual}\n")
            f.write(f"- **Impact**: {issue.impact}\n")
            f.write(f"- **Fix**: {issue.fix}\n\n")
        f.write("## Action Plan\n\n")
        f.write("```\n")
        f.write(action_plan)
        f.write("\n```\n")
    
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
