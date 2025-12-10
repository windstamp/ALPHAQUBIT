"""
Paper Figures Package
=====================

This package contains scripts to generate all figures and tables from the AlphaQubit paper:
"Accurate neural network decoding of surface codes for quantum error correction"
(Nature 2024)

Key Figures:
- Figure 1: Architecture overview and surface code schematic
- Figure 2: Logical error rate vs physical error rate (threshold plots)
- Figure 3: Decoder comparison (AlphaQubit vs MWPM, tensor network, etc.)
- Figure 4: Fine-tuning results on Google QEC device data
- Extended Data Figures: Ablation studies, noise model comparisons

Usage:
    python -m paper_figures.generate_all
    
Or individual figures:
    python -m paper_figures.fig2_threshold_plot
    python -m paper_figures.fig3_decoder_comparison
    python -m paper_figures.fig4_finetuning_results
"""

__version__ = "1.0.0"
