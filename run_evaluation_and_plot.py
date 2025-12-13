#!/usr/bin/env python
"""
Run fine-tuned model evaluation and generate comparison plots with paper baseline.
Fully automated - no user prompts.
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


def find_directory(candidates, must_have_files=None):
    """Find the first existing directory from candidates."""
    for d in candidates:
        path = Path(d)
        if path.exists():
            if must_have_files:
                # Check if directory has the required files
                has_files = any(path.glob(must_have_files))
                if has_files:
                    return path
            else:
                return path
    return None


def run_evaluation():
    """Run test_finetuned_models.py - fully automated."""
    print("="*80)
    print("STEP 1: Running fine-tuned model evaluation")
    print("="*80)
    
    # Find model directory
    model_dir = find_directory([
        'finetuned_models_v2',
        'finetuned_models',
    ], must_have_files='*.pth')
    
    if not model_dir:
        print("✗ No model directory found with .pth files")
        return False
    
    # Find test data directory
    test_dir = find_directory([
        'google_finetune_data/test',
        'pretrain_data',
        'simulated_data',
    ], must_have_files='*.npz')
    
    if not test_dir:
        print("✗ No test data directory found with .npz files")
        return False
    
    # Output directory
    results_dir = 'test_results_v2'
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Model directory: {model_dir}")
    print(f"Test data directory: {test_dir}")
    print(f"Results directory: {results_dir}")
    
    cmd = [
        sys.executable,
        "test_finetuned_models.py",
        "--model-dir", str(model_dir),
        "--test-dir", str(test_dir),
        "--results-dir", results_dir,
        "--batch-size", "512",
        "--save-predictions"
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
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
    
    # Find summary file in possible locations
    summary_paths = [
        Path('test_results_v2/test_summary.json'),
        Path('test_results/test_summary.json'),
    ]
    
    summary_path = None
    for p in summary_paths:
        if p.exists():
            summary_path = p
            break
    
    if not summary_path:
        print(f"✗ Error: No test_summary.json found")
        return False
    
    print(f"Using summary: {summary_path}")
    results_dir = summary_path.parent
    
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
    plot_path = results_dir / 'ler_comparison_with_paper.png'
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
    comparison_path = results_dir / 'paper_comparison_analysis.json'
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
    
    # Always run evaluation - no prompts, fully automated
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
    print(f"\nResults saved:")
    print(f"  - test_summary.json (raw results)")
    print(f"  - paper_comparison_analysis.json (detailed comparison)")
    print(f"  - ler_comparison_with_paper.png (visualization)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
