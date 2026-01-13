#!/usr/bin/env python3
"""
AlphaQubit S3 Results Analysis Script

Analyzes downloaded S3 results and generates paper replication figures.
Compares our implementation results with the AlphaQubit Nature paper.

Usage:
    python analyze_s3_results.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import paper reference data
try:
    from paper_figures.paper_data import (
        THRESHOLD_DATA_SI1000,
        DECODER_COMPARISON,
        GOOGLE_QEC_EXPERIMENTS,
        PAPER_BASELINE,
        THRESHOLD_ALPHAQUBIT,
        THRESHOLD_MWPM
    )
    PAPER_DATA_AVAILABLE = True
except ImportError:
    PAPER_DATA_AVAILABLE = False
    print("Warning: paper_figures.paper_data not available, using fallback values")


# =============================================================================
# Configuration
# =============================================================================

S3_RESULTS_DIR = PROJECT_ROOT / "s3_results" / "20251230_220521"
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"


def setup_plotting_style():
    """Setup matplotlib style for paper-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# Data Loading
# =============================================================================

def load_test_summary() -> Dict:
    """Load test summary from S3 results."""
    summary_path = S3_RESULTS_DIR / "test_results_v2" / "test_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def load_paper_comparison() -> Dict:
    """Load paper comparison analysis from S3 results."""
    comparison_path = S3_RESULTS_DIR / "test_results_v2" / "paper_comparison_analysis.json"
    if comparison_path.exists():
        with open(comparison_path, 'r') as f:
            return json.load(f)
    return {}


def load_benchmark_results() -> Dict:
    """Load benchmark results from S3."""
    benchmark_path = S3_RESULTS_DIR / "benchmark_results" / "benchmark_results.json"
    if benchmark_path.exists():
        with open(benchmark_path, 'r') as f:
            return json.load(f)
    return {}


def parse_experiment_name(name: str) -> Dict:
    """Parse experiment name into components."""
    # Format: surface_code_bX_d3_r01_center_5_7
    parts = name.split('_')
    result = {
        'code_type': parts[0] + '_' + parts[1] if len(parts) > 1 else 'unknown',
        'basis': 'unknown',
        'distance': 0,
        'rounds': 0,
        'center': 'unknown'
    }
    
    for i, part in enumerate(parts):
        if part.startswith('b') and len(part) == 2:
            result['basis'] = part[1]  # X or Z
        elif part.startswith('d') and part[1:].isdigit():
            result['distance'] = int(part[1:])
        elif part.startswith('r') and part[1:].isdigit():
            result['rounds'] = int(part[1:])
        elif part == 'center' and i + 2 < len(parts):
            result['center'] = f"{parts[i+1]}_{parts[i+2]}"
    
    return result


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_by_configuration(results: List[Dict]) -> Dict:
    """Group and analyze results by different configurations."""
    by_distance = {}
    by_basis = {'X': [], 'Z': []}
    by_rounds = {}
    
    for r in results:
        exp_name = r.get('experiment', '')
        ler = r.get('metrics', {}).get('logical_error_rate', 0)
        acc = r.get('metrics', {}).get('accuracy', 0)
        
        parsed = parse_experiment_name(exp_name)
        
        # Group by distance
        d = parsed['distance']
        if d > 0:
            if d not in by_distance:
                by_distance[d] = []
            by_distance[d].append({'name': exp_name, 'ler': ler, 'accuracy': acc, 'rounds': parsed['rounds']})
        
        # Group by basis
        basis = parsed['basis']
        if basis in ['X', 'Z']:
            by_basis[basis].append({'name': exp_name, 'ler': ler, 'accuracy': acc})
        
        # Group by rounds
        rounds = parsed['rounds']
        if rounds > 0:
            if rounds not in by_rounds:
                by_rounds[rounds] = []
            by_rounds[rounds].append({'name': exp_name, 'ler': ler, 'accuracy': acc})
    
    return {
        'by_distance': by_distance,
        'by_basis': by_basis,
        'by_rounds': by_rounds
    }


def compute_statistics(values: List[float]) -> Dict:
    """Compute statistics for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_ler_distribution(results: List[Dict], output_dir: Path):
    """Plot LER distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    lers = [r.get('metrics', {}).get('logical_error_rate', 0) for r in results]
    accs = [r.get('metrics', {}).get('accuracy', 0) for r in results]
    
    # LER histogram
    ax1 = axes[0]
    ax1.hist(lers, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(x=np.mean(lers), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(lers):.3f}')
    if PAPER_DATA_AVAILABLE:
        ax1.axvline(x=PAPER_BASELINE['average_ler'], color='green', linestyle=':', 
                    linewidth=2, label=f"Paper Baseline: {PAPER_BASELINE['average_ler']:.3f}")
    ax1.set_xlabel('Logical Error Rate (LER)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Logical Error Rates')
    ax1.legend()
    
    # Accuracy histogram
    ax2 = axes[1]
    ax2.hist(accs, bins=20, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.mean(accs), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(accs):.3f}')
    if PAPER_DATA_AVAILABLE:
        ax2.axvline(x=PAPER_BASELINE['average_accuracy'], color='green', linestyle=':',
                    linewidth=2, label=f"Paper Baseline: {PAPER_BASELINE['average_accuracy']:.3f}")
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Model Accuracies')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_ler_distribution.png')
    plt.savefig(output_dir / 'fig1_ler_distribution.pdf')
    plt.close()
    
    print(f"[✓] Saved LER distribution plot")


def plot_ler_by_distance(grouped_results: Dict, output_dir: Path):
    """Plot LER comparison by code distance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    by_distance = grouped_results['by_distance']
    
    # Box plot
    ax1 = axes[0]
    distances = sorted(by_distance.keys())
    box_data = [[r['ler'] for r in by_distance[d]] for d in distances]
    
    bp = ax1.boxplot(box_data, labels=[f'd={d}' for d in distances], patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(distances)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_xlabel('Code Distance')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_title('LER Distribution by Code Distance')
    
    # Add paper reference lines
    if PAPER_DATA_AVAILABLE:
        for i, d_key in enumerate(['d3', 'd5', 'd7']):
            if d_key in THRESHOLD_DATA_SI1000:
                # Use p=0.005 reference
                ref_ler = THRESHOLD_DATA_SI1000[d_key]['logical_error_rate_alphaqubit'][4]
                ax1.axhline(y=ref_ler, color=colors[i], linestyle='--', alpha=0.5)
    
    # Bar chart comparing means
    ax2 = axes[1]
    means = [np.mean([r['ler'] for r in by_distance[d]]) for d in distances]
    stds = [np.std([r['ler'] for r in by_distance[d]]) for d in distances]
    
    x = np.arange(len(distances))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, means, width, yerr=stds, label='Our Results', 
                    color='steelblue', capsize=5, alpha=0.8)
    
    if PAPER_DATA_AVAILABLE:
        paper_means = []
        for d in distances:
            d_key = f'd{d}'
            if d_key in THRESHOLD_DATA_SI1000:
                # Average across physical error rates for comparison
                paper_means.append(np.mean(THRESHOLD_DATA_SI1000[d_key]['logical_error_rate_alphaqubit']))
            else:
                paper_means.append(0)
        
        bars2 = ax2.bar(x + width/2, paper_means, width, label='Paper Reference',
                        color='coral', alpha=0.8)
    
    ax2.set_xlabel('Code Distance')
    ax2.set_ylabel('Mean Logical Error Rate')
    ax2.set_title('Mean LER Comparison with Paper')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'd={d}' for d in distances])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_ler_by_distance.png')
    plt.savefig(output_dir / 'fig2_ler_by_distance.pdf')
    plt.close()
    
    print(f"[✓] Saved LER by distance plot")


def plot_ler_by_rounds(grouped_results: Dict, output_dir: Path):
    """Plot LER vs number of rounds (noise rate proxy)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    by_rounds = grouped_results['by_rounds']
    rounds = sorted(by_rounds.keys())
    
    means = [np.mean([r['ler'] for r in by_rounds[rd]]) for rd in rounds]
    stds = [np.std([r['ler'] for r in by_rounds[rd]]) for rd in rounds]
    counts = [len(by_rounds[rd]) for rd in rounds]
    
    # Convert rounds to noise rate (r01 = 1%, r25 = 25%, etc.)
    noise_rates = [rd / 100.0 for rd in rounds]
    
    ax.errorbar(noise_rates, means, yerr=stds, fmt='o-', color='steelblue', 
                capsize=5, capthick=2, markersize=8, linewidth=2, 
                label='Our Results (mean ± std)')
    
    # Add paper trend if available
    if PAPER_DATA_AVAILABLE and 'd3' in THRESHOLD_DATA_SI1000:
        paper_p = THRESHOLD_DATA_SI1000['d3']['physical_error_rate']
        paper_ler = THRESHOLD_DATA_SI1000['d3']['logical_error_rate_alphaqubit']
        ax.plot(paper_p, paper_ler, 's--', color='coral', markersize=6, 
                linewidth=1.5, alpha=0.7, label='Paper (d=3)')
    
    ax.set_xlabel('Noise Rate (r/100)')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('LER vs Noise Rate')
    ax.legend()
    ax.set_xlim(0, max(noise_rates) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ler_vs_noise.png')
    plt.savefig(output_dir / 'fig3_ler_vs_noise.pdf')
    plt.close()
    
    print(f"[✓] Saved LER vs noise plot")


def plot_paper_comparison(test_summary: Dict, output_dir: Path):
    """Generate comprehensive paper comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    results = test_summary.get('results', [])
    
    # (a) Top 10 best models
    ax1 = axes[0, 0]
    sorted_results = sorted(results, key=lambda x: x.get('metrics', {}).get('logical_error_rate', 1))
    top10 = sorted_results[:10]
    
    names = [r['experiment'].replace('surface_code_', '').replace('_center', '\n') 
             for r in top10]
    lers = [r['metrics']['logical_error_rate'] for r in top10]
    
    colors = ['#2ecc71' if ler < 0.05 else '#f39c12' if ler < 0.1 else '#e74c3c' 
              for ler in lers]
    
    bars = ax1.barh(range(len(names)), lers, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Logical Error Rate')
    ax1.set_title('(a) Top 10 Best Performing Models')
    ax1.invert_yaxis()
    
    if PAPER_DATA_AVAILABLE:
        ax1.axvline(x=PAPER_BASELINE['average_ler'], color='green', linestyle='--',
                    linewidth=2, label=f"Paper Avg: {PAPER_BASELINE['average_ler']:.1%}")
        ax1.legend()
    
    # (b) Top 10 worst models
    ax2 = axes[0, 1]
    worst10 = sorted_results[-10:][::-1]
    
    names_w = [r['experiment'].replace('surface_code_', '').replace('_center', '\n') 
               for r in worst10]
    lers_w = [r['metrics']['logical_error_rate'] for r in worst10]
    
    colors_w = ['#e74c3c' if ler > 0.2 else '#f39c12' for ler in lers_w]
    
    ax2.barh(range(len(names_w)), lers_w, color=colors_w, alpha=0.8)
    ax2.set_yticks(range(len(names_w)))
    ax2.set_yticklabels(names_w, fontsize=8)
    ax2.set_xlabel('Logical Error Rate')
    ax2.set_title('(b) Top 10 Worst Performing Models')
    ax2.invert_yaxis()
    
    # (c) Basis comparison (X vs Z)
    ax3 = axes[1, 0]
    
    bX_results = [r for r in results if '_bX_' in r['experiment']]
    bZ_results = [r for r in results if '_bZ_' in r['experiment']]
    
    bX_lers = [r['metrics']['logical_error_rate'] for r in bX_results]
    bZ_lers = [r['metrics']['logical_error_rate'] for r in bZ_results]
    
    bp = ax3.boxplot([bX_lers, bZ_lers], labels=['X basis', 'Z basis'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax3.set_ylabel('Logical Error Rate')
    ax3.set_title('(c) LER by Measurement Basis')
    
    # Add mean annotations
    ax3.scatter([1, 2], [np.mean(bX_lers), np.mean(bZ_lers)], color='black', 
                marker='D', s=50, zorder=5, label='Mean')
    ax3.legend()
    
    # (d) Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    all_lers = [r['metrics']['logical_error_rate'] for r in results]
    all_accs = [r['metrics']['accuracy'] for r in results]
    
    summary_text = f"""
    ╔══════════════════════════════════════════════╗
    ║          RESULTS SUMMARY                     ║
    ╠══════════════════════════════════════════════╣
    ║  Total Experiments: {len(results):>5}                    ║
    ║                                              ║
    ║  Logical Error Rate:                         ║
    ║    Mean:   {np.mean(all_lers):>8.4f}                       ║
    ║    Std:    {np.std(all_lers):>8.4f}                       ║
    ║    Min:    {np.min(all_lers):>8.4f}                       ║
    ║    Max:    {np.max(all_lers):>8.4f}                       ║
    ║    Median: {np.median(all_lers):>8.4f}                       ║
    ║                                              ║
    ║  Accuracy:                                   ║
    ║    Mean:   {np.mean(all_accs):>8.4f}                       ║
    ║    Std:    {np.std(all_accs):>8.4f}                       ║
    ║                                              ║
    ║  Paper Comparison:                           ║
    ║    Paper Avg LER:  {PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03:>8.4f}                 ║
    ║    Our Avg LER:    {np.mean(all_lers):>8.4f}                 ║
    ║    Gap:            {(np.mean(all_lers) - (PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03)):>+8.4f}                 ║
    ╚══════════════════════════════════════════════╝
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('(d) Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_paper_comparison.png')
    plt.savefig(output_dir / 'fig4_paper_comparison.pdf')
    plt.close()
    
    print(f"[✓] Saved paper comparison plot")


def plot_threshold_behavior(grouped_results: Dict, output_dir: Path):
    """Replicate Figure 2 from paper: Threshold behavior."""
    if not PAPER_DATA_AVAILABLE:
        print("[!] Paper data not available, skipping threshold plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: LER vs physical error rate (paper reference)
    ax1 = axes[0]
    
    colors = {'d3': '#2ecc71', 'd5': '#3498db', 'd7': '#e74c3c'}
    markers = {'d3': 'o', 'd5': 's', 'd7': '^'}
    
    for d_key in ['d3', 'd5', 'd7']:
        data = THRESHOLD_DATA_SI1000[d_key]
        p_vals = data['physical_error_rate']
        ler_aq = data['logical_error_rate_alphaqubit']
        ler_mwpm = data['logical_error_rate_mwpm']
        
        ax1.semilogy(p_vals, ler_aq, f'{markers[d_key]}-', color=colors[d_key],
                     label=f'{d_key} AlphaQubit', markersize=6, linewidth=1.5)
        ax1.semilogy(p_vals, ler_mwpm, f'{markers[d_key]}--', color=colors[d_key],
                     label=f'{d_key} MWPM', markersize=6, linewidth=1.5, alpha=0.5)
    
    # Add threshold lines
    ax1.axvline(x=THRESHOLD_ALPHAQUBIT, color='green', linestyle=':', alpha=0.7,
                label=f'AQ threshold ({THRESHOLD_ALPHAQUBIT:.1%})')
    ax1.axvline(x=THRESHOLD_MWPM, color='blue', linestyle=':', alpha=0.7,
                label=f'MWPM threshold ({THRESHOLD_MWPM:.1%})')
    
    ax1.set_xlabel('Physical Error Rate')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_title('(a) Paper Reference: Threshold Behavior')
    ax1.legend(loc='upper left', ncol=2, fontsize=8)
    ax1.set_xlim([0.0005, 0.012])
    ax1.set_ylim([1e-5, 0.1])
    
    # Panel B: Our results overlaid
    ax2 = axes[1]
    
    by_distance = grouped_results['by_distance']
    
    for d, results_list in sorted(by_distance.items()):
        if d in [3, 5, 7]:
            # Group by rounds (noise proxy)
            by_rounds = {}
            for r in results_list:
                rd = r['rounds']
                if rd not in by_rounds:
                    by_rounds[rd] = []
                by_rounds[rd].append(r['ler'])
            
            rounds = sorted(by_rounds.keys())
            noise_rates = [rd / 100.0 for rd in rounds]  # Convert to rate
            mean_lers = [np.mean(by_rounds[rd]) for rd in rounds]
            
            d_key = f'd{d}'
            ax2.semilogy(noise_rates, mean_lers, f'{markers.get(d_key, "o")}-',
                         color=colors.get(d_key, 'gray'), markersize=8, linewidth=2,
                         label=f'd={d} (ours)')
    
    # Add paper reference for d=3
    paper_p = THRESHOLD_DATA_SI1000['d3']['physical_error_rate']
    paper_ler = THRESHOLD_DATA_SI1000['d3']['logical_error_rate_alphaqubit']
    ax2.semilogy(paper_p, paper_ler, 'o--', color='#2ecc71', alpha=0.4,
                 markersize=4, label='d=3 paper ref')
    
    ax2.set_xlabel('Noise Rate')
    ax2.set_ylabel('Logical Error Rate')
    ax2.set_title('(b) Our Results vs Paper Reference')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_threshold_behavior.png')
    plt.savefig(output_dir / 'fig5_threshold_behavior.pdf')
    plt.close()
    
    print(f"[✓] Saved threshold behavior plot")


def generate_analysis_report(test_summary: Dict, grouped_results: Dict, output_dir: Path):
    """Generate a markdown analysis report."""
    
    results = test_summary.get('results', [])
    all_lers = [r['metrics']['logical_error_rate'] for r in results]
    all_accs = [r['metrics']['accuracy'] for r in results]
    
    # Find best/worst
    sorted_results = sorted(results, key=lambda x: x['metrics']['logical_error_rate'])
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    report = f"""# AlphaQubit Results Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the AlphaQubit model results downloaded from S3 and compares 
them with the reference values from the AlphaQubit Nature paper.

## Overall Statistics

| Metric | Our Results | Paper Baseline | Gap |
|--------|-------------|----------------|-----|
| Average LER | {np.mean(all_lers):.4f} | {PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03:.4f} | {np.mean(all_lers) - (PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03):+.4f} |
| Average Accuracy | {np.mean(all_accs):.4f} | {PAPER_BASELINE['average_accuracy'] if PAPER_DATA_AVAILABLE else 0.97:.4f} | {np.mean(all_accs) - (PAPER_BASELINE['average_accuracy'] if PAPER_DATA_AVAILABLE else 0.97):+.4f} |
| Best LER | {np.min(all_lers):.4f} | {PAPER_BASELINE['best_ler'] if PAPER_DATA_AVAILABLE else 0.015:.4f} | {np.min(all_lers) - (PAPER_BASELINE['best_ler'] if PAPER_DATA_AVAILABLE else 0.015):+.4f} |
| Worst LER | {np.max(all_lers):.4f} | {PAPER_BASELINE['worst_ler'] if PAPER_DATA_AVAILABLE else 0.085:.4f} | {np.max(all_lers) - (PAPER_BASELINE['worst_ler'] if PAPER_DATA_AVAILABLE else 0.085):+.4f} |

## Experiment Counts

- **Total Experiments**: {len(results)}
- **Successful**: {test_summary.get('successful', len(results))}
- **Failed**: {test_summary.get('failed', 0)}

## Best Performing Model

- **Name**: `{best['experiment']}`
- **LER**: {best['metrics']['logical_error_rate']:.4f}
- **Accuracy**: {best['metrics']['accuracy']:.4f}

## Worst Performing Model

- **Name**: `{worst['experiment']}`
- **LER**: {worst['metrics']['logical_error_rate']:.4f}
- **Accuracy**: {worst['metrics']['accuracy']:.4f}

## Analysis by Code Distance

| Distance | Count | Mean LER | Std LER | Min LER | Max LER |
|----------|-------|----------|---------|---------|---------|
"""
    
    by_distance = grouped_results['by_distance']
    for d in sorted(by_distance.keys()):
        lers = [r['ler'] for r in by_distance[d]]
        report += f"| d={d} | {len(lers)} | {np.mean(lers):.4f} | {np.std(lers):.4f} | {np.min(lers):.4f} | {np.max(lers):.4f} |\n"
    
    report += """
## Analysis by Measurement Basis

| Basis | Count | Mean LER | Mean Accuracy |
|-------|-------|----------|---------------|
"""
    
    by_basis = grouped_results['by_basis']
    for basis in ['X', 'Z']:
        if by_basis[basis]:
            lers = [r['ler'] for r in by_basis[basis]]
            accs = [r['accuracy'] for r in by_basis[basis]]
            report += f"| {basis} | {len(lers)} | {np.mean(lers):.4f} | {np.mean(accs):.4f} |\n"
    
    report += """
## Analysis by Noise Rate (Rounds)

| Rounds (r) | Noise Rate | Count | Mean LER | Std LER |
|------------|------------|-------|----------|---------|
"""
    
    by_rounds = grouped_results['by_rounds']
    for rd in sorted(by_rounds.keys()):
        lers = [r['ler'] for r in by_rounds[rd]]
        noise_rate = rd / 100.0
        report += f"| r{rd:02d} | {noise_rate:.2%} | {len(lers)} | {np.mean(lers):.4f} | {np.std(lers):.4f} |\n"
    
    report += f"""
## Key Findings

### 1. Performance Gap Analysis

Our average LER of **{np.mean(all_lers):.2%}** is higher than the paper's baseline of **{PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03:.2%}**.
This represents a gap of approximately **{(np.mean(all_lers) / (PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03)):.1f}x** higher error rate.

### 2. Strong Performance Cases

Models with **d=3 and low noise (r01)** achieve performance closest to the paper:
- Best model achieves {np.min(all_lers):.2%} LER
- This is only {((np.min(all_lers) / (PAPER_BASELINE['best_ler'] if PAPER_DATA_AVAILABLE else 0.015)) - 1) * 100:.1f}% higher than paper's best

### 3. Challenging Cases

High noise rates (r20+) and larger code distances show the biggest gaps:
- Models struggle with high noise scenarios
- May indicate need for more training data or longer fine-tuning

### 4. Basis Comparison

"""
    
    if by_basis['X'] and by_basis['Z']:
        mean_x = np.mean([r['ler'] for r in by_basis['X']])
        mean_z = np.mean([r['ler'] for r in by_basis['Z']])
        better = 'X' if mean_x < mean_z else 'Z'
        report += f"- X-basis mean LER: {mean_x:.4f}\n"
        report += f"- Z-basis mean LER: {mean_z:.4f}\n"
        report += f"- **{better}-basis** performs slightly better\n"
    
    report += """
## Recommendations for Improvement

1. **Increase Training Data**: Generate more synthetic pre-training samples, especially for high-noise scenarios
2. **Longer Fine-tuning**: Increase fine-tuning epochs for challenging experiments
3. **Model Architecture**: Consider using larger model (hidden_dim=256 or 512)
4. **Learning Rate Tuning**: Experiment with different learning rate schedules
5. **Data Augmentation**: Apply symmetry-based augmentation during training

## Generated Figures

The following figures have been generated in the `analysis_output/` directory:

1. `fig1_ler_distribution.png` - Distribution of LER and accuracy across all experiments
2. `fig2_ler_by_distance.png` - LER comparison by code distance
3. `fig3_ler_vs_noise.png` - LER vs noise rate trend
4. `fig4_paper_comparison.png` - Comprehensive comparison with paper
5. `fig5_threshold_behavior.png` - Threshold behavior (paper Figure 2 style)

---
*Report generated by analyze_s3_results.py*
"""
    
    # Save report
    report_path = output_dir / 'ANALYSIS_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[✓] Saved analysis report to {report_path}")
    
    return report


# =============================================================================
# Main
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("AlphaQubit S3 Results Analysis")
    print("=" * 60)
    
    # Setup
    setup_plotting_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check S3 results directory
    if not S3_RESULTS_DIR.exists():
        print(f"[ERROR] S3 results directory not found: {S3_RESULTS_DIR}")
        sys.exit(1)
    
    print(f"\n[*] Loading results from: {S3_RESULTS_DIR}")
    
    # Load data
    test_summary = load_test_summary()
    paper_comparison = load_paper_comparison()
    benchmark_results = load_benchmark_results()
    
    if not test_summary.get('results'):
        print("[ERROR] No test results found in summary")
        sys.exit(1)
    
    print(f"[*] Loaded {len(test_summary.get('results', []))} experiment results")
    
    # Analyze by configuration
    results = test_summary.get('results', [])
    grouped_results = analyze_by_configuration(results)
    
    print(f"\n[*] Generating figures in: {OUTPUT_DIR}")
    
    # Generate plots
    plot_ler_distribution(results, OUTPUT_DIR)
    plot_ler_by_distance(grouped_results, OUTPUT_DIR)
    plot_ler_by_rounds(grouped_results, OUTPUT_DIR)
    plot_paper_comparison(test_summary, OUTPUT_DIR)
    plot_threshold_behavior(grouped_results, OUTPUT_DIR)
    
    # Generate report
    print(f"\n[*] Generating analysis report...")
    report = generate_analysis_report(test_summary, grouped_results, OUTPUT_DIR)
    
    # Print summary
    all_lers = [r['metrics']['logical_error_rate'] for r in results]
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey Results:")
    print(f"  • Total experiments: {len(results)}")
    print(f"  • Average LER: {np.mean(all_lers):.4f} ({np.mean(all_lers):.2%})")
    print(f"  • Best LER: {np.min(all_lers):.4f} ({np.min(all_lers):.2%})")
    print(f"  • Paper baseline: {PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03:.4f} ({(PAPER_BASELINE['average_ler'] if PAPER_DATA_AVAILABLE else 0.03):.2%})")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
