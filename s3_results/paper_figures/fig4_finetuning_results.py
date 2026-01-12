"""
Figure 4: Fine-tuning Results on Google QEC Device
==================================================

Reproduces Figure 4 from the AlphaQubit paper showing:
- Fine-tuning performance on Google Sycamore QEC v3.5 data
- LER across different experimental configurations
- Comparison with paper baseline

Usage:
    python -m paper_figures.fig4_finetuning_results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

from paper_figures.paper_data import (
    GOOGLE_QEC_EXPERIMENTS,
    PAPER_BASELINE,
)


def generate_figure4(output_dir: Path = None, show: bool = False):
    """
    Generate Figure 4: Fine-tuning results on Google QEC device data.
    
    Args:
        output_dir: Directory to save the figure
        show: Whether to display the figure interactively
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    experiments = list(GOOGLE_QEC_EXPERIMENTS.keys())
    lers = [GOOGLE_QEC_EXPERIMENTS[exp]['ler'] * 100 for exp in experiments]
    accuracies = [GOOGLE_QEC_EXPERIMENTS[exp]['accuracy'] * 100 for exp in experiments]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(experiments))
    
    # =========================================================================
    # Panel A: Logical Error Rate
    # =========================================================================
    colors = ['green' if ler < PAPER_BASELINE['average_ler'] * 100 else 'steelblue' 
              for ler in lers]
    
    bars1 = ax1.bar(x, lers, color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    ax1.axhline(y=PAPER_BASELINE['average_ler'] * 100, color='red', linestyle='--', 
                linewidth=2.5, label=f"Paper Baseline ({PAPER_BASELINE['average_ler']*100:.1f}%)")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([exp.replace('surface_code_', '') for exp in experiments], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Logical Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Fine-tuned Model Performance: Logical Error Rate', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_ylim(0, max(lers) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, ler in zip(bars1, lers):
        ax1.annotate(f'{ler:.2f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # =========================================================================
    # Panel B: Accuracy
    # =========================================================================
    colors2 = ['green' if acc > PAPER_BASELINE['average_accuracy'] * 100 else 'coral' 
               for acc in accuracies]
    
    bars2 = ax2.bar(x, accuracies, color=colors2, edgecolor='black', linewidth=1, alpha=0.8)
    ax2.axhline(y=PAPER_BASELINE['average_accuracy'] * 100, color='red', linestyle='--', 
                linewidth=2.5, label=f"Paper Baseline ({PAPER_BASELINE['average_accuracy']*100:.1f}%)")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([exp.replace('surface_code_', '') for exp in experiments], 
                        rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Fine-tuned Model Performance: Accuracy', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.set_ylim(min(accuracies) * 0.95, 100.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "fig4_finetuning_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    pdf_path = output_dir / "fig4_finetuning_results.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF saved to: {pdf_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def generate_grouped_by_config(output_dir: Path = None, show: bool = False):
    """
    Generate plots grouped by code distance and noise rate.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parse experiment names
    by_distance = {'d3': [], 'd5': []}
    by_noise = {'r01': [], 'r05': [], 'r10': [], 'r25': []}
    
    for exp, data in GOOGLE_QEC_EXPERIMENTS.items():
        # Extract distance
        d_match = re.search(r'_d(\d)_', exp)
        if d_match:
            dist = f'd{d_match.group(1)}'
            if dist in by_distance:
                by_distance[dist].append(data['ler'] * 100)
        
        # Extract noise rate
        r_match = re.search(r'_r(\d+)_', exp)
        if r_match:
            rate = f'r{r_match.group(1)}'
            if rate in by_noise:
                by_noise[rate].append(data['ler'] * 100)
    
    # =========================================================================
    # Panel A: By Code Distance
    # =========================================================================
    ax1 = axes[0]
    distances = list(by_distance.keys())
    avg_lers = [np.mean(by_distance[d]) if by_distance[d] else 0 for d in distances]
    std_lers = [np.std(by_distance[d]) if len(by_distance[d]) > 1 else 0 for d in distances]
    
    bars = ax1.bar(distances, avg_lers, yerr=std_lers, capsize=5,
                   color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.8)
    ax1.axhline(y=PAPER_BASELINE['average_ler'] * 100, color='red', linestyle='--', 
                linewidth=2, label='Paper Baseline')
    
    ax1.set_xlabel('Code Distance', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax1.set_title('LER by Code Distance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel B: By Noise Rate
    # =========================================================================
    ax2 = axes[1]
    noise_labels = {'r01': '1%', 'r05': '5%', 'r10': '10%', 'r25': '25%'}
    rates = [k for k in by_noise.keys() if by_noise[k]]
    avg_lers2 = [np.mean(by_noise[r]) for r in rates]
    std_lers2 = [np.std(by_noise[r]) if len(by_noise[r]) > 1 else 0 for r in rates]
    
    bars2 = ax2.bar([noise_labels[r] for r in rates], avg_lers2, yerr=std_lers2, capsize=5,
                    color='#2ecc71', edgecolor='black', alpha=0.8)
    ax2.axhline(y=PAPER_BASELINE['average_ler'] * 100, color='red', linestyle='--', 
                linewidth=2, label='Paper Baseline')
    
    ax2.set_xlabel('Physical Noise Rate', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax2.set_title('LER by Physical Noise Rate', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "fig4_grouped_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def generate_heatmap(output_dir: Path = None, show: bool = False):
    """
    Generate a heatmap of LER across code distance and noise rate.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create matrix
    distances = ['d3', 'd5']
    noise_rates = ['r01', 'r05', 'r10', 'r25']
    
    matrix = np.zeros((len(distances), len(noise_rates)))
    matrix[:] = np.nan
    
    for exp, data in GOOGLE_QEC_EXPERIMENTS.items():
        d_match = re.search(r'_d(\d)_', exp)
        r_match = re.search(r'_r(\d+)_', exp)
        
        if d_match and r_match:
            dist = f'd{d_match.group(1)}'
            rate = f'r{r_match.group(1)}'
            
            if dist in distances and rate in noise_rates:
                d_idx = distances.index(dist)
                r_idx = noise_rates.index(rate)
                if np.isnan(matrix[d_idx, r_idx]):
                    matrix[d_idx, r_idx] = data['ler'] * 100
                else:
                    matrix[d_idx, r_idx] = (matrix[d_idx, r_idx] + data['ler'] * 100) / 2
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Logical Error Rate (%)', fontsize=12)
    
    # Labels
    ax.set_xticks(np.arange(len(noise_rates)))
    ax.set_yticks(np.arange(len(distances)))
    ax.set_xticklabels(['1%', '5%', '10%', '25%'])
    ax.set_yticklabels(['Distance 3', 'Distance 5'])
    ax.set_xlabel('Physical Noise Rate', fontsize=12)
    ax.set_ylabel('Code Distance', fontsize=12)
    ax.set_title('Logical Error Rate Heatmap\n(AlphaQubit Fine-tuned on Google QEC)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(distances)):
        for j in range(len(noise_rates)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}%',
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "fig4_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Figure 4 from AlphaQubit paper")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("=" * 60)
    print("Generating Figure 4: Fine-tuning Results")
    print("=" * 60)
    
    generate_figure4(output_dir, show=args.show)
    generate_grouped_by_config(output_dir, show=args.show)
    generate_heatmap(output_dir, show=args.show)
    
    print("\n✓ All Figure 4 variants generated successfully!")
