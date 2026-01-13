#!/usr/bin/env python3
"""
Quick Analysis Script for AlphaQubit Results
Generates paper-style plots from downloaded S3 results
"""
print("Script starting...", flush=True)

import json
import os
from pathlib import Path
import numpy as np

print("Imports done...", flush=True)

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    exit(1)

# Configuration
RESULTS_DIR = Path("./s3_results/20251230_220521")
OUTPUT_DIR = Path("./analysis_output")

def load_results():
    """Load all JSON results"""
    results = {}
    
    # Test summary - check both possible locations
    test_summary_path = RESULTS_DIR / "test_results_v2" / "test_summary.json"
    if not test_summary_path.exists():
        test_summary_path = RESULTS_DIR / "test_summary.json"
    if test_summary_path.exists():
        with open(test_summary_path) as f:
            results['test_summary'] = json.load(f)
    
    # Paper comparison - check both possible locations
    paper_comp_path = RESULTS_DIR / "test_results_v2" / "paper_comparison_analysis.json"
    if not paper_comp_path.exists():
        paper_comp_path = RESULTS_DIR / "paper_comparison_analysis.json"
    if paper_comp_path.exists():
        with open(paper_comp_path) as f:
            results['paper_comparison'] = json.load(f)
    
    # Benchmark results - check both possible locations
    benchmark_path = RESULTS_DIR / "benchmark_results" / "benchmark_results.json"
    if not benchmark_path.exists():
        benchmark_path = RESULTS_DIR / "benchmark_results.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            results['benchmarks'] = json.load(f)
    
    return results


def plot_ler_by_rounds(results, output_dir):
    """Plot LER vs number of rounds (like paper Figure 2)"""
    
    test_results = results.get('test_summary', {}).get('results', [])
    if not test_results:
        print("No test results found")
        return
    
    # Extract data by rounds
    data_bZ = {}  # basis Z
    data_bX = {}  # basis X
    
    for r in test_results:
        exp = r.get('experiment', '')
        metrics = r.get('metrics', {})
        ler = metrics.get('logical_error_rate', 0)
        
        # Parse experiment name: surface_code_bZ_d3_r05_center_5_7
        parts = exp.split('_')
        if len(parts) >= 5:
            basis = parts[2]  # bZ or bX
            try:
                rounds = int(parts[4][1:])  # r05 -> 5
            except:
                continue
            
            target = data_bZ if basis == 'bZ' else data_bX
            if rounds not in target:
                target[rounds] = []
            target[rounds].append(ler)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Z basis
    if data_bZ:
        rounds_z = sorted(data_bZ.keys())
        avg_ler_z = [np.mean(data_bZ[r]) for r in rounds_z]
        std_ler_z = [np.std(data_bZ[r]) for r in rounds_z]
        
        ax1.errorbar(rounds_z, avg_ler_z, yerr=std_ler_z, 
                    fmt='o-', capsize=3, label='AlphaQubit (Z basis)', 
                    color='#2ecc71', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Rounds', fontsize=12)
        ax1.set_ylabel('Logical Error Rate', fontsize=12)
        ax1.set_title('(a) Z-Basis Logical Error vs Rounds', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 0.3])
    
    # Plot X basis
    if data_bX:
        rounds_x = sorted(data_bX.keys())
        avg_ler_x = [np.mean(data_bX[r]) for r in rounds_x]
        std_ler_x = [np.std(data_bX[r]) for r in rounds_x]
        
        ax2.errorbar(rounds_x, avg_ler_x, yerr=std_ler_x,
                    fmt='s-', capsize=3, label='AlphaQubit (X basis)',
                    color='#3498db', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Rounds', fontsize=12)
        ax2.set_ylabel('Logical Error Rate', fontsize=12)
        ax2.set_title('(b) X-Basis Logical Error vs Rounds', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 0.3])
    
    plt.tight_layout()
    
    output_path = output_dir / "ler_vs_rounds.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path


def plot_accuracy_distribution(results, output_dir):
    """Plot accuracy distribution histogram"""
    
    test_results = results.get('test_summary', {}).get('results', [])
    if not test_results:
        return
    
    accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in test_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(accuracies, bins=20, color='steelblue', 
                               edgecolor='white', alpha=0.8)
    
    # Add statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_acc:.3f}')
    ax.axvline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=1.5,
              label=f'Std: {std_acc:.3f}')
    ax.axvline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('AlphaQubit Accuracy Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "accuracy_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path


def plot_decoder_comparison(results, output_dir):
    """Plot decoder comparison (like paper Figure 3)"""
    
    benchmarks = results.get('benchmarks', {}).get('benchmarks', [])
    if not benchmarks:
        print("No benchmark data found")
        return
    
    # Get first benchmark
    bench = benchmarks[0]
    decoders = bench.get('decoders', {})
    
    if not decoders:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(decoders.keys())
    lers = [decoders[d].get('logical_error_rate', 0) for d in names]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(names)]
    
    bars = ax.bar(names, lers, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, ler in zip(bars, lers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{ler:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title(f'Decoder Comparison (d={bench.get("distance", "?")}, '
                f'rounds={bench.get("rounds", "?")}, '
                f'p={bench.get("physical_error_rate", "?")})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "decoder_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path


def plot_paper_comparison(results, output_dir):
    """Plot comparison with paper baseline"""
    
    paper_comp = results.get('paper_comparison', {})
    if not paper_comp:
        print("No paper comparison data found")
        return
    
    summary = paper_comp.get('summary', {})
    best = paper_comp.get('best_model', {})
    worst = paper_comp.get('worst_model', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Summary comparison
    categories = ['Our Best', 'Our Average', 'Our Worst', 'Paper Baseline']
    lers = [
        best.get('ler', 0),
        summary.get('average_ler', 0),
        worst.get('ler', 0),
        summary.get('paper_baseline_ler', 0.03)
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(categories, lers, color=colors, edgecolor='white')
    
    for bar, ler in zip(bars, lers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{ler:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Logical Error Rate', fontsize=12)
    ax1.set_title('(a) LER Comparison with Paper', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: LER distribution vs paper threshold
    comparisons = paper_comp.get('comparison_by_experiment', [])
    if comparisons:
        exp_lers = [c.get('ler', 0) for c in comparisons]
        paper_threshold = summary.get('paper_baseline_ler', 0.03)
        
        ax2.hist(exp_lers, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.axvline(paper_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Paper Baseline: {paper_threshold:.3f}')
        ax2.axvline(np.mean(exp_lers), color='green', linestyle='--', linewidth=2,
                   label=f'Our Mean: {np.mean(exp_lers):.3f}')
        
        ax2.set_xlabel('Logical Error Rate', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('(b) LER Distribution vs Paper Threshold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "paper_comparison_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path


def generate_summary_report(results, output_dir):
    """Generate text summary report"""
    
    report_path = output_dir / "ANALYSIS_REPORT.md"
    
    summary = results.get('paper_comparison', {}).get('summary', {})
    best = results.get('paper_comparison', {}).get('best_model', {})
    worst = results.get('paper_comparison', {}).get('worst_model', {})
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AlphaQubit Results Analysis Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Experiments**: {summary.get('total_experiments', 'N/A')}\n")
        f.write(f"- **Average LER**: {summary.get('average_ler', 'N/A'):.4f}\n")
        f.write(f"- **Average Accuracy**: {summary.get('average_accuracy', 'N/A'):.4f}\n")
        f.write(f"- **Median LER**: {summary.get('median_ler', 'N/A'):.4f}\n")
        f.write(f"- **Std LER**: {summary.get('std_ler', 'N/A'):.4f}\n")
        f.write("\n")
        
        f.write("## Best & Worst Models\n\n")
        f.write(f"- **Best Model**: `{best.get('experiment', 'N/A')}`\n")
        f.write(f"  - LER: {best.get('ler', 'N/A'):.4f}\n")
        f.write(f"  - Accuracy: {best.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"- **Worst Model**: `{worst.get('experiment', 'N/A')}`\n")
        f.write(f"  - LER: {worst.get('ler', 'N/A'):.4f}\n")
        f.write(f"  - Accuracy: {worst.get('accuracy', 'N/A'):.4f}\n")
        f.write("\n")
        
        f.write("## Paper Comparison\n\n")
        f.write(f"- **Paper Baseline LER**: {summary.get('paper_baseline_ler', 0.03):.4f}\n")
        f.write(f"- **Paper Baseline Accuracy**: {summary.get('paper_baseline_accuracy', 0.97):.4f}\n")
        f.write(f"- **Models Beating Paper**: {results.get('paper_comparison', {}).get('models_beating_paper', {}).get('count', 0)}\n")
        f.write("\n")
        
        f.write("## Generated Plots\n\n")
        f.write("- `ler_vs_rounds.png/pdf` - LER vs number of rounds\n")
        f.write("- `accuracy_distribution.png/pdf` - Accuracy histogram\n")
        f.write("- `decoder_comparison.png/pdf` - Decoder comparison\n")
        f.write("- `paper_comparison_summary.png/pdf` - Paper baseline comparison\n")
    
    print(f"✓ Saved: {report_path}")
    return report_path


def main():
    import sys
    sys.stdout.flush()
    print("=" * 60, flush=True)
    print("  AlphaQubit Results Analysis", flush=True)
    print("=" * 60, flush=True)
    print(f"Results Dir: {RESULTS_DIR}", flush=True)
    print(f"Output Dir:  {OUTPUT_DIR}", flush=True)
    print(flush=True)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...", flush=True)
    try:
        results = load_results()
        print(f"  - Test summary: {'✓' if 'test_summary' in results else '✗'}", flush=True)
        print(f"  - Paper comparison: {'✓' if 'paper_comparison' in results else '✗'}", flush=True)
        print(f"  - Benchmarks: {'✓' if 'benchmarks' in results else '✗'}", flush=True)
    except Exception as e:
        print(f"Error loading results: {e}", flush=True)
        return
    print(flush=True)
    
    # Generate plots
    print("Generating plots...", flush=True)
    
    try:
        plot_ler_by_rounds(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in plot_ler_by_rounds: {e}", flush=True)
    
    try:
        plot_accuracy_distribution(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in plot_accuracy_distribution: {e}", flush=True)
    
    try:
        plot_decoder_comparison(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in plot_decoder_comparison: {e}", flush=True)
    
    try:
        plot_paper_comparison(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in plot_paper_comparison: {e}", flush=True)
    
    try:
        generate_summary_report(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in generate_summary_report: {e}", flush=True)
    
    print(flush=True)
    print("=" * 60, flush=True)
    print("  Analysis Complete!", flush=True)
    print("=" * 60, flush=True)
    print(f"Output saved to: {OUTPUT_DIR.absolute()}", flush=True)


if __name__ == "__main__":
    main()
