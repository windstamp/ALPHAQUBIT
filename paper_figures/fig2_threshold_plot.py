"""
Figure 2: Threshold and Scaling Behavior
========================================

Reproduces Figure 2 from the AlphaQubit paper showing:
- Logical error rate vs physical error rate
- Threshold crossing for different code distances
- Comparison between AlphaQubit and MWPM decoder

Usage:
    python -m paper_figures.fig2_threshold_plot
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import paper reference data
from paper_figures.paper_data import (
    THRESHOLD_DATA_SI1000,
    THRESHOLD_ALPHAQUBIT,
    THRESHOLD_MWPM,
)


def generate_figure2(output_dir: Path = None, show: bool = False):
    """
    Generate Figure 2: Threshold plot showing logical vs physical error rate.
    
    Args:
        output_dir: Directory to save the figure (default: paper_figures/output/)
        show: Whether to display the figure interactively
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure with publication-quality settings
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different code distances
    colors = {'d3': '#1f77b4', 'd5': '#ff7f0e', 'd7': '#2ca02c'}
    markers = {'d3': 'o', 'd5': 's', 'd7': '^'}
    
    # =========================================================================
    # Panel A: AlphaQubit decoder
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title('(a) AlphaQubit Decoder', fontsize=14, fontweight='bold')
    
    for dist in ['d3', 'd5', 'd7']:
        data = THRESHOLD_DATA_SI1000[dist]
        p = np.array(data['physical_error_rate'])
        ler = np.array(data['logical_error_rate_alphaqubit'])
        
        ax1.semilogy(p * 100, ler, 
                     marker=markers[dist], 
                     color=colors[dist],
                     linewidth=2, 
                     markersize=8,
                     label=f'Distance {dist[1]}')
    
    # Add threshold line
    ax1.axvline(x=THRESHOLD_ALPHAQUBIT * 100, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Threshold ≈ {THRESHOLD_ALPHAQUBIT*100:.2f}%')
    
    ax1.set_xlabel('Physical Error Rate (%)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(1e-5, 1e-1)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel B: MWPM decoder (for comparison)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title('(b) MWPM Decoder', fontsize=14, fontweight='bold')
    
    for dist in ['d3', 'd5', 'd7']:
        data = THRESHOLD_DATA_SI1000[dist]
        p = np.array(data['physical_error_rate'])
        ler = np.array(data['logical_error_rate_mwpm'])
        
        ax2.semilogy(p * 100, ler, 
                     marker=markers[dist], 
                     color=colors[dist],
                     linewidth=2, 
                     markersize=8,
                     label=f'Distance {dist[1]}')
    
    # Add threshold line
    ax2.axvline(x=THRESHOLD_MWPM * 100, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Threshold ≈ {THRESHOLD_MWPM*100:.2f}%')
    
    ax2.set_xlabel('Physical Error Rate (%)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(1e-5, 1e-1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "fig2_threshold_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_dir / "fig2_threshold_plot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF saved to: {pdf_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


def generate_threshold_comparison(output_dir: Path = None, show: bool = False):
    """
    Generate a combined threshold comparison plot.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'d3': '#1f77b4', 'd5': '#ff7f0e', 'd7': '#2ca02c'}
    
    # Plot AlphaQubit (solid lines)
    for dist in ['d3', 'd5', 'd7']:
        data = THRESHOLD_DATA_SI1000[dist]
        p = np.array(data['physical_error_rate'])
        ler = np.array(data['logical_error_rate_alphaqubit'])
        ax.semilogy(p * 100, ler, '-', color=colors[dist], linewidth=2.5, 
                    marker='o', markersize=8, label=f'AlphaQubit d={dist[1]}')
    
    # Plot MWPM (dashed lines)
    for dist in ['d3', 'd5', 'd7']:
        data = THRESHOLD_DATA_SI1000[dist]
        p = np.array(data['physical_error_rate'])
        ler = np.array(data['logical_error_rate_mwpm'])
        ax.semilogy(p * 100, ler, '--', color=colors[dist], linewidth=2, 
                    marker='s', markersize=6, alpha=0.7, label=f'MWPM d={dist[1]}')
    
    # Threshold lines
    ax.axvline(x=THRESHOLD_ALPHAQUBIT * 100, color='darkgreen', linestyle=':', 
               linewidth=2, label=f'AlphaQubit threshold ({THRESHOLD_ALPHAQUBIT*100:.2f}%)')
    ax.axvline(x=THRESHOLD_MWPM * 100, color='darkred', linestyle=':', 
               linewidth=2, label=f'MWPM threshold ({THRESHOLD_MWPM*100:.2f}%)')
    
    ax.set_xlabel('Physical Error Rate (%)', fontsize=14)
    ax.set_ylabel('Logical Error Rate', fontsize=14)
    ax.set_title('AlphaQubit vs MWPM: Threshold Comparison\n(SI1000 Noise Model)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(1e-5, 1e-1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "fig2_threshold_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Figure 2 from AlphaQubit paper")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("=" * 60)
    print("Generating Figure 2: Threshold and Scaling Behavior")
    print("=" * 60)
    
    generate_figure2(output_dir, show=args.show)
    generate_threshold_comparison(output_dir, show=args.show)
    
    print("\n✓ All Figure 2 variants generated successfully!")
