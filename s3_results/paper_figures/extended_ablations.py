"""
Extended Data Figures: Ablation Studies
=======================================

Reproduces Extended Data Figures from the AlphaQubit paper showing:
- Impact of soft vs hard syndrome readout
- Effect of pre-training strategies
- Model size scaling

Usage:
    python -m paper_figures.extended_ablations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from paper_figures.paper_data import (
    ABLATION_SOFT_READOUT,
    ABLATION_PRETRAINING,
    ABLATION_MODEL_SIZE,
)


def generate_soft_readout_ablation(output_dir: Path = None, show: bool = False):
    """
    Generate ablation study: Soft vs Hard syndrome readout.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ABLATION_SOFT_READOUT['conditions']
    x = np.arange(len(conditions))
    width = 0.35
    
    # Data for two noise levels
    vals_low = [ABLATION_SOFT_READOUT['d5_p0.005'][c] * 100 for c in conditions]
    vals_high = [ABLATION_SOFT_READOUT['d5_p0.01'][c] * 100 for c in conditions]
    
    bars1 = ax.bar(x - width/2, vals_low, width, label='p=0.5%', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, vals_high, width, label='p=1.0%', color='#e74c3c', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax.set_title('Impact of Soft vs Hard Syndrome Readout\n(d=5 Surface Code)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add improvement annotation
    improvement = ABLATION_SOFT_READOUT['improvement']
    ax.annotate(f'Soft readout improves by {improvement}', 
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = output_dir / "ext_ablation_soft_readout.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def generate_pretraining_ablation(output_dir: Path = None, show: bool = False):
    """
    Generate ablation study: Pre-training strategies.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ABLATION_PRETRAINING['conditions']
    values = [ABLATION_PRETRAINING['d5_finetuned'][c] * 100 for c in conditions]
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(conditions, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f}%', 
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax.set_title('Impact of Pre-training Strategy\n(d=5 Surface Code, Fine-tuned)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add improvement arrows
    ax.annotate('', xy=(2, values[2]), xytext=(0, values[0]),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    improvement = (values[0] - values[2]) / values[0] * 100
    ax.annotate(f'{improvement:.0f}% better', xy=(1, (values[0] + values[2])/2),
               ha='center', fontsize=11, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "ext_ablation_pretraining.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def generate_model_size_ablation(output_dir: Path = None, show: bool = False):
    """
    Generate ablation study: Model size scaling.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = ABLATION_MODEL_SIZE['configurations']
    params = [c['params'] for c in configs]
    lers = [c['ler'] * 100 for c in configs]
    hidden_dims = [c['hidden_dim'] for c in configs]
    num_layers = [c['num_layers'] for c in configs]
    
    # =========================================================================
    # Panel A: LER vs Model Size
    # =========================================================================
    ax1.plot(range(len(params)), lers, 'o-', color='#3498db', linewidth=2.5, markersize=12)
    ax1.set_xticks(range(len(params)))
    ax1.set_xticklabels(params)
    ax1.set_xlabel('Model Size (Parameters)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax1.set_title('LER vs Model Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for each point
    for i, (p, ler) in enumerate(zip(params, lers)):
        ax1.annotate(f'{ler:.2f}%', xy=(i, ler), xytext=(5, 10),
                    textcoords="offset points", fontsize=10, fontweight='bold')
    
    # =========================================================================
    # Panel B: LER vs Hidden Dimension
    # =========================================================================
    ax2.plot(hidden_dims, lers, 's-', color='#e74c3c', linewidth=2.5, markersize=12)
    ax2.set_xlabel('Hidden Dimension', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (%)', fontsize=12)
    ax2.set_title('LER vs Hidden Dimension', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # Add layer count annotations
    for hd, ler, nl in zip(hidden_dims, lers, num_layers):
        ax2.annotate(f'{nl} layers', xy=(hd, ler), xytext=(5, -15),
                    textcoords="offset points", fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    output_path = output_dir / "ext_ablation_model_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Extended Data ablation figures")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("=" * 60)
    print("Generating Extended Data: Ablation Studies")
    print("=" * 60)
    
    generate_soft_readout_ablation(output_dir, show=args.show)
    generate_pretraining_ablation(output_dir, show=args.show)
    generate_model_size_ablation(output_dir, show=args.show)
    
    print("\n✓ All ablation figures generated successfully!")
