#!/usr/bin/env python3
"""
Generate Paper Figures from Benchmark Results
==============================================

This script generates publication-quality figures comparing all decoders,
similar to Figure 3 of the AlphaQubit Nature 2024 paper.

It can use either:
1. Reference data from paper_data.py (default)
2. Actual benchmark results from run_decoder_comparison_all.py

Usage:
    # Generate figures from paper reference data
    python generate_paper_figures.py
    
    # Generate figures from benchmark results
    python generate_paper_figures.py --results decoder_comparison_results/comparison_results.json
    
    # Show figures interactively
    python generate_paper_figures.py --show
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Reference data from paper
from paper_figures.paper_data import (
    DECODER_COMPARISON,
    THRESHOLD_DATA_SI1000,
    THRESHOLD_ALPHAQUBIT,
    THRESHOLD_MWPM
)


def load_benchmark_results(results_path: str) -> Optional[Dict]:
    """Load benchmark results from JSON file."""
    path = Path(results_path)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def generate_figure3_bar_chart(
    data: Dict[str, Dict[str, float]],
    title: str,
    output_path: Path,
    show: bool = False
):
    """
    Generate bar chart comparing decoders (like paper Figure 3a).
    
    Args:
        data: Dict mapping condition name to {decoder: LER}
        title: Figure title
        output_path: Path to save figure
        show: Whether to display interactively
    """
    if not HAS_MATPLOTLIB:
        return
    
    conditions = list(data.keys())
    decoders = ['AlphaQubit', 'MWPM', 'Tensor Network', 'Belief Propagation', 'Union Find']
    
    # Filter to available decoders
    available_decoders = []
    for d in decoders:
        if any(d in data[c] for c in conditions):
            available_decoders.append(d)
    
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 6))
    if len(conditions) == 1:
        axes = [axes]
    
    colors = {
        'AlphaQubit': '#2ecc71',
        'MWPM': '#3498db',
        'Tensor Network': '#9b59b6',
        'Belief Propagation': '#e74c3c',
        'Union Find': '#f39c12'
    }
    
    for ax, condition in zip(axes, conditions):
        cond_data = data[condition]
        
        x = np.arange(len(available_decoders))
        values = []
        bar_colors = []
        
        for decoder in available_decoders:
            val = cond_data.get(decoder, 0) * 100  # Convert to %
            values.append(val)
            bar_colors.append(colors.get(decoder, 'gray'))
        
        bars = ax.bar(x, values, color=bar_colors, edgecolor='black', linewidth=1.2)
        
        # Highlight best performer
        min_idx = np.argmin(values)
        bars[min_idx].set_edgecolor('gold')
        bars[min_idx].set_linewidth(3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_decoders, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Logical Error Rate (%)', fontsize=11)
        ax.set_title(condition, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_figure3_improvement_chart(
    data: Dict[str, Dict[str, float]],
    baseline: str = 'MWPM',
    output_path: Path = None,
    show: bool = False
):
    """
    Generate improvement chart showing how each decoder compares to baseline.
    """
    if not HAS_MATPLOTLIB:
        return
    
    decoders = ['AlphaQubit', 'Tensor Network', 'Belief Propagation', 'Union Find']
    decoders = [d for d in decoders if d != baseline]
    
    # Calculate average improvement over baseline
    improvements = {}
    for decoder in decoders:
        imp_list = []
        for condition, cond_data in data.items():
            if decoder in cond_data and baseline in cond_data:
                baseline_ler = cond_data[baseline]
                decoder_ler = cond_data[decoder]
                if baseline_ler > 0:
                    imp = (baseline_ler - decoder_ler) / baseline_ler * 100
                    imp_list.append(imp)
        if imp_list:
            improvements[decoder] = np.mean(imp_list)
    
    if not improvements:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'AlphaQubit': '#2ecc71',
        'Tensor Network': '#9b59b6',
        'Belief Propagation': '#e74c3c',
        'Union Find': '#f39c12'
    }
    
    sorted_decoders = sorted(improvements.keys(), key=lambda x: improvements[x], reverse=True)
    values = [improvements[d] for d in sorted_decoders]
    bar_colors = [colors.get(d, 'gray') for d in sorted_decoders]
    
    bars = ax.barh(range(len(sorted_decoders)), values, color=bar_colors, 
                   edgecolor='black', height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, values):
        color = 'green' if val > 0 else 'red'
        ax.annotate(f'{val:+.1f}%',
                   xy=(val, bar.get_y() + bar.get_height() / 2),
                   xytext=(5 if val >= 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if val >= 0 else 'right',
                   va='center', fontsize=11, fontweight='bold', color=color)
    
    ax.set_yticks(range(len(sorted_decoders)))
    ax.set_yticklabels(sorted_decoders, fontsize=11)
    ax.set_xlabel(f'Improvement over {baseline} (%)', fontsize=12)
    ax.set_title(f'Decoder Performance Relative to {baseline}\n(Average across conditions)', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_threshold_plot(
    data: Dict = None,
    output_path: Path = None,
    show: bool = False
):
    """
    Generate threshold plot (LER vs physical error rate) like paper Figure 2.
    """
    if not HAS_MATPLOTLIB:
        return
    
    if data is None:
        data = THRESHOLD_DATA_SI1000
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'d3': 'C0', 'd5': 'C1', 'd7': 'C2'}
    
    for d_key, d_data in data.items():
        d = int(d_key[1:])
        color = colors.get(d_key, 'gray')
        
        p_vals = np.array(d_data['physical_error_rate']) * 100
        
        # AlphaQubit
        if 'logical_error_rate_alphaqubit' in d_data:
            ler = d_data['logical_error_rate_alphaqubit']
            ax.semilogy(p_vals, ler, 's-', color=color, 
                       label=f'AlphaQubit d={d}', linewidth=2, markersize=8)
        
        # MWPM
        if 'logical_error_rate_mwpm' in d_data:
            ler = d_data['logical_error_rate_mwpm']
            ax.semilogy(p_vals, ler, 'o--', color=color, alpha=0.7,
                       label=f'MWPM d={d}', linewidth=1.5, markersize=6)
    
    # Threshold lines
    ax.axvline(x=THRESHOLD_ALPHAQUBIT * 100, color='green', linestyle=':', 
               alpha=0.7, linewidth=2, label=f'AlphaQubit threshold ({THRESHOLD_ALPHAQUBIT*100:.2f}%)')
    ax.axvline(x=THRESHOLD_MWPM * 100, color='blue', linestyle=':', 
               alpha=0.7, linewidth=2, label=f'MWPM threshold ({THRESHOLD_MWPM*100:.2f}%)')
    
    ax.set_xlabel('Physical Error Rate (%)', fontsize=13)
    ax.set_ylabel('Logical Error Rate', fontsize=13)
    ax.set_title('Threshold Comparison: AlphaQubit vs MWPM\n(SI1000 Noise Model)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_summary_table(
    data: Dict[str, Dict[str, float]],
    output_path: Path = None
):
    """
    Generate a summary table as a figure.
    """
    if not HAS_MATPLOTLIB:
        return
    
    conditions = list(data.keys())
    decoders = ['AlphaQubit', 'MWPM', 'Tensor Network', 'Belief Propagation', 'Union Find']
    
    # Filter to available decoders
    available_decoders = []
    for d in decoders:
        if any(d in data[c] for c in conditions):
            available_decoders.append(d)
    
    # Build table data
    table_data = []
    for condition in conditions:
        row = [condition]
        for decoder in available_decoders:
            val = data[condition].get(decoder, None)
            if val is not None:
                row.append(f'{val*100:.3f}%')
            else:
                row.append('N/A')
        table_data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4 + 0.5 * len(conditions)))
    ax.axis('off')
    
    # Create table
    col_labels = ['Condition'] + available_decoders
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best values in each row
    for row_idx in range(len(table_data)):
        values = []
        for col_idx, decoder in enumerate(available_decoders):
            val_str = table_data[row_idx][col_idx + 1]
            if val_str != 'N/A':
                values.append((col_idx + 1, float(val_str.rstrip('%'))))
            else:
                values.append((col_idx + 1, float('inf')))
        
        if values:
            best_col = min(values, key=lambda x: x[1])[0]
            table[(row_idx + 1, best_col)].set_facecolor('#90EE90')
    
    plt.title('Decoder Comparison Summary\n(Logical Error Rate)', 
             fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def generate_all_figures(
    results_path: Optional[str] = None,
    output_dir: Path = None,
    show: bool = False
):
    """
    Generate all paper figures.
    
    Args:
        results_path: Path to benchmark results JSON (optional)
        output_dir: Output directory for figures
        show: Whether to display figures interactively
    """
    if output_dir is None:
        output_dir = Path('paper_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)
    
    # Load data
    if results_path:
        results = load_benchmark_results(results_path)
        if results:
            print(f"Using benchmark results from: {results_path}")
            # Convert results to comparison format
            comparison_data = {}
            for d_key, d_results in results.get('results', {}).items():
                for p_key, p_results in d_results.items():
                    p_val = float(p_key[1:]) * 100
                    condition = f'SI1000, {d_key}, p={p_val:.1f}%'
                    comparison_data[condition] = {
                        name: r['logical_error_rate'] 
                        for name, r in p_results.items()
                    }
        else:
            print(f"Could not load results from {results_path}, using paper data")
            comparison_data = DECODER_COMPARISON
    else:
        print("Using reference data from paper")
        comparison_data = {
            'SI1000, p=0.5%, d=5': DECODER_COMPARISON['si1000_p0.005_d5'],
            'SI1000, p=1.0%, d=5': DECODER_COMPARISON['si1000_p0.01_d5'],
            'Pauli+, p=0.5%, d=5': DECODER_COMPARISON['pauli_plus_p0.005_d5'],
        }
    
    # Generate Figure 3a: Bar chart comparison
    print("\n[1/4] Generating bar chart comparison...")
    generate_figure3_bar_chart(
        data=comparison_data,
        title='Decoder Comparison (Figure 3 Style)',
        output_path=output_dir / 'fig3_decoder_comparison.png',
        show=show
    )
    
    # Generate Figure 3b: Improvement chart
    print("[2/4] Generating improvement chart...")
    generate_figure3_improvement_chart(
        data=comparison_data,
        baseline='MWPM',
        output_path=output_dir / 'fig3_improvement_over_mwpm.png',
        show=show
    )
    
    # Generate Figure 2: Threshold plot
    print("[3/4] Generating threshold plot...")
    generate_threshold_plot(
        output_path=output_dir / 'fig2_threshold_plot.png',
        show=show
    )
    
    # Generate summary table
    print("[4/4] Generating summary table...")
    generate_summary_table(
        data=comparison_data,
        output_path=output_dir / 'decoder_summary_table.png'
    )
    
    print("\n" + "=" * 60)
    print(f"✅ All figures saved to: {output_dir}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures from benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_paper_figures.py
  python generate_paper_figures.py --results decoder_comparison_results/comparison_results.json
  python generate_paper_figures.py --show
        """
    )
    
    parser.add_argument('--results', type=str, default=None,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output-dir', type=str, default='paper_figures/output',
                       help='Output directory for figures')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib is required. Install with: pip install matplotlib")
        return
    
    generate_all_figures(
        results_path=args.results,
        output_dir=Path(args.output_dir),
        show=args.show
    )


if __name__ == '__main__':
    main()
