"""
Generate All Paper Figures and Tables
=====================================

Master script to generate all figures and tables from the AlphaQubit paper.

Usage:
    python -m paper_figures.generate_all
    
Or from command line:
    python paper_figures/generate_all.py
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures and tables from the AlphaQubit paper"
    )
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Output directory (default: paper_figures/output/)")
    parser.add_argument("--show", action="store_true", 
                       help="Display figures interactively")
    parser.add_argument("--figures-only", action="store_true",
                       help="Generate only figures (skip tables)")
    parser.add_argument("--tables-only", action="store_true",
                       help="Generate only tables (skip figures)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AlphaQubit Paper Figure & Table Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    generated_files = []
    
    # =========================================================================
    # Generate Figures
    # =========================================================================
    if not args.tables_only:
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)
        
        # Figure 2: Threshold plots
        try:
            from paper_figures.fig2_threshold_plot import (
                generate_figure2, 
                generate_threshold_comparison
            )
            print("\n[Figure 2] Threshold and Scaling Behavior...")
            generated_files.append(generate_figure2(output_dir, show=args.show))
            generated_files.append(generate_threshold_comparison(output_dir, show=args.show))
        except Exception as e:
            print(f"  ✗ Error generating Figure 2: {e}")
        
        # Figure 3: Decoder comparison
        try:
            from paper_figures.fig3_decoder_comparison import (
                generate_figure3,
                generate_improvement_chart,
                generate_radar_chart
            )
            print("\n[Figure 3] Decoder Comparison...")
            generated_files.append(generate_figure3(output_dir, show=args.show))
            generated_files.append(generate_improvement_chart(output_dir, show=args.show))
            generated_files.append(generate_radar_chart(output_dir, show=args.show))
        except Exception as e:
            print(f"  ✗ Error generating Figure 3: {e}")
        
        # Figure 4: Fine-tuning results
        try:
            from paper_figures.fig4_finetuning_results import (
                generate_figure4,
                generate_grouped_by_config,
                generate_heatmap
            )
            print("\n[Figure 4] Fine-tuning Results...")
            generated_files.append(generate_figure4(output_dir, show=args.show))
            generated_files.append(generate_grouped_by_config(output_dir, show=args.show))
            generated_files.append(generate_heatmap(output_dir, show=args.show))
        except Exception as e:
            print(f"  ✗ Error generating Figure 4: {e}")
        
        # Extended Data: Ablation studies
        try:
            from paper_figures.extended_ablations import (
                generate_soft_readout_ablation,
                generate_pretraining_ablation,
                generate_model_size_ablation
            )
            print("\n[Extended Data] Ablation Studies...")
            generated_files.append(generate_soft_readout_ablation(output_dir, show=args.show))
            generated_files.append(generate_pretraining_ablation(output_dir, show=args.show))
            generated_files.append(generate_model_size_ablation(output_dir, show=args.show))
        except Exception as e:
            print(f"  ✗ Error generating ablation figures: {e}")
    
    # =========================================================================
    # Generate Tables
    # =========================================================================
    if not args.figures_only:
        print("\n" + "=" * 70)
        print("GENERATING TABLES")
        print("=" * 70)
        
        try:
            from paper_figures.generate_tables import generate_all_tables
            generate_all_tables(output_dir)
        except ImportError as e:
            # tabulate might not be installed - use fallback
            print(f"  Note: {e}")
            print("  Installing tabulate or using fallback...")
            try:
                from paper_figures.generate_tables_simple import generate_all_tables
                generate_all_tables(output_dir)
            except:
                print("  ✗ Could not generate tables. Install tabulate: pip install tabulate")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated: {len([f for f in generated_files if f])}")
    
    if output_dir.exists():
        print("\nGenerated files:")
        for f in sorted(output_dir.iterdir()):
            print(f"  - {f.name}")
    
    print("\n✓ All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
