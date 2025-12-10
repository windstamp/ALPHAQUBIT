#!/usr/bin/env python
"""
Run fine-tuned model evaluation on NPU and generate comparison plots with paper baseline.
Usage: python run_evaluation_and_plot.py
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
from pathlib import Path

def run_evaluation():
    """Run test_finetuned_models.py with NPU support."""
    print("="*80)
    print("STEP 1: Running fine-tuned model evaluation on NPU")
    print("="*80)
    
    cmd = [
        sys.executable,
        "test_finetuned_models.py",
        "--model-dir", "finetuned_models",
        "--test-dir", "google_finetune_data/test",
        "--results-dir", "test_results",
        "--npu",  # Use NPU for inference
        "--batch-size", "512",
        "--save-predictions"
    ]
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[FAIL] Evaluation failed with return code {result.returncode}")
        return False
    
    print("\n[OK] Evaluation completed successfully")
    return True

def generate_comparison_plot():
    """Generate comparison plot with paper baseline."""
    print("\n" + "="*80)
    print("STEP 2: Generating comparison plot with paper baseline")
    print("="*80)
    
    summary_path = Path('test_results/test_summary.json')
    
    if not summary_path.exists():
        print(f"✗ Error: {summary_path} not found")
        return False
    
    # Load results
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    successful = [r for r in results['results'] if r['status'] == 'success']
    
    if not successful:
        print("✗ No successful experiments found")
        return False
    
    exps = [r['experiment'] for r in successful]
    lers = [r['metrics']['logical_error_rate'] for r in successful]
    accuracies = [r['metrics']['accuracy'] for r in successful]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Logical Error Rate
    x = np.arange(len(exps))
    bars1 = ax1.bar(x, lers, color='steelblue', alpha=0.8, edgecolor='navy')
    ax1.axhline(y=0.03, color='red', linestyle='--', linewidth=2.5, label='Paper Baseline (3%)', zorder=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exps, rotation=75, ha='right', fontsize=9)
    ax1.set_ylabel('Logical Error Rate (LER)', fontsize=13, fontweight='bold')
    ax1.set_title('Fine-tuned Model Performance vs AlphaQubit Paper Baseline', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.4, linestyle='--')
    ax1.set_ylim(0, max(max(lers) * 1.15, 0.04))
    
    # Highlight bars better/worse than baseline
    for i, (bar, ler) in enumerate(zip(bars1, lers)):
        if ler < 0.03:
            bar.set_color('green')
            bar.set_alpha(0.7)
    
    # Plot 2: Accuracy
    bars2 = ax2.bar(x, accuracies, color='coral', alpha=0.8, edgecolor='darkred')
    ax2.axhline(y=0.97, color='red', linestyle='--', linewidth=2.5, label='Paper Baseline (97% accuracy)', zorder=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exps, rotation=75, ha='right', fontsize=9)
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    ax2.set_ylim(min(accuracies) * 0.95, 1.0)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path('test_results/ler_comparison_with_paper.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Generate detailed statistics
    stats = {
        'summary': {
            'total_experiments': len(exps),
            'average_ler': float(np.mean(lers)),
            'median_ler': float(np.median(lers)),
            'std_ler': float(np.std(lers)),
            'average_accuracy': float(np.mean(accuracies)),
            'paper_baseline_ler': 0.03,
            'paper_baseline_accuracy': 0.97
        },
        'best_model': {
            'experiment': exps[np.argmin(lers)],
            'ler': float(min(lers)),
            'accuracy': float(accuracies[np.argmin(lers)])
        },
        'worst_model': {
            'experiment': exps[np.argmax(lers)],
            'ler': float(max(lers)),
            'accuracy': float(accuracies[np.argmax(lers)])
        },
        'models_beating_paper': {
            'count': int(sum(1 for l in lers if l < 0.03)),
            'percentage': float(sum(1 for l in lers if l < 0.03) / len(lers) * 100),
            'experiments': [exp for exp, ler in zip(exps, lers) if ler < 0.03]
        },
        'comparison_by_experiment': [
            {
                'experiment': exp,
                'ler': float(ler),
                'accuracy': float(acc),
                'beats_paper': ler < 0.03,
                'improvement_over_paper': float((0.03 - ler) / 0.03 * 100)
            }
            for exp, ler, acc in zip(exps, lers, accuracies)
        ]
    }
    
    # Save detailed comparison
    comparison_path = Path('test_results/paper_comparison_analysis.json')
    with open(comparison_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Detailed analysis saved to: {comparison_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Total Experiments: {len(exps)}")
    print(f"\nAverage LER (Your Models):  {np.mean(lers):.6f}")
    print(f"Paper Baseline:              0.030000 (3%)")
    print(f"Improvement:                 {(0.03 - np.mean(lers)) / 0.03 * 100:+.2f}%")
    print(f"\nBest Model:  {exps[np.argmin(lers)]}")
    print(f"  LER:       {min(lers):.6f}")
    print(f"  Accuracy:  {accuracies[np.argmin(lers)]:.6f}")
    print(f"\nWorst Model: {exps[np.argmax(lers)]}")
    print(f"  LER:       {max(lers):.6f}")
    print(f"  Accuracy:  {accuracies[np.argmax(lers)]:.6f}")
    print(f"\nModels beating paper baseline: {sum(1 for l in lers if l < 0.03)}/{len(lers)} ({sum(1 for l in lers if l < 0.03)/len(lers)*100:.1f}%)")
    
    if sum(1 for l in lers if l < 0.03) > 0:
        print("\nExperiments beating paper baseline:")
        for exp, ler in sorted(zip(exps, lers), key=lambda x: x[1]):
            if ler < 0.03:
                improvement = (0.03 - ler) / 0.03 * 100
                print(f"  ✓ {exp:60s} LER={ler:.6f} ({improvement:+.1f}%)")
    
    print("="*80)
    
    return True

def main():
    print("AlphaQubit Fine-tuned Model Evaluation and Paper Comparison")
    print("="*80)
    
    # Check if test_summary.json already exists
    summary_path = Path('test_results/test_summary.json')
    
    if summary_path.exists():
        print(f"\n✓ Found existing results at {summary_path}")
        response = input("Re-run evaluation? (y/N): ").strip().lower()
        if response == 'y':
            success = run_evaluation()
            if not success:
                print("\n✗ Evaluation failed. Exiting.")
                return 1
    else:
        # Run evaluation
        success = run_evaluation()
        if not success:
            print("\n✗ Evaluation failed. Exiting.")
            return 1
    
    # Generate comparison plot
    success = generate_comparison_plot()
    if not success:
        print("\n✗ Plot generation failed. Exiting.")
        return 1
    
    print("\n✓ All steps completed successfully!")
    print(f"\nResults saved in test_results/:")
    print(f"  - test_summary.json (raw results)")
    print(f"  - paper_comparison_analysis.json (detailed comparison)")
    print(f"  - ler_comparison_with_paper.png (visualization)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
