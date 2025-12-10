"""
Generate Tables from the AlphaQubit Paper
=========================================

Generates all tables from the paper as formatted output:
- Table 1: Model architecture configurations
- Table 2: Training hyperparameters
- Table 3: Decoder performance comparison
- Table 4: Fine-tuning results summary

Usage:
    python -m paper_figures.generate_tables
"""

import numpy as np
from pathlib import Path
from tabulate import tabulate
import json

from paper_figures.paper_data import (
    MODEL_ARCHITECTURE,
    TRAINING_CONFIG,
    DECODER_COMPARISON,
    GOOGLE_QEC_EXPERIMENTS,
    PAPER_BASELINE,
    THRESHOLD_ALPHAQUBIT,
    THRESHOLD_MWPM,
)


def generate_table1_architecture(output_dir: Path = None):
    """
    Generate Table 1: Model Architecture Configurations
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 1: AlphaQubit Model Architecture Configurations")
    print("=" * 70)
    
    headers = ["Config", "Hidden Dim", "Num Heads", "Num Layers", "Total Params"]
    rows = []
    
    for name, config in MODEL_ARCHITECTURE.items():
        rows.append([
            name.upper(),
            config['hidden_dim'],
            config['num_heads'],
            config['num_layers'],
            config['total_params']
        ])
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(table)
    
    # Save to file
    output_path = output_dir / "table1_architecture.txt"
    with open(output_path, 'w') as f:
        f.write("Table 1: AlphaQubit Model Architecture Configurations\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
    print(f"\n✓ Table saved to: {output_path}")
    
    return table


def generate_table2_training(output_dir: Path = None):
    """
    Generate Table 2: Training Hyperparameters
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 2: Training Hyperparameters")
    print("=" * 70)
    
    headers = ["Parameter", "Pre-training", "Fine-tuning"]
    rows = [
        ["Samples", f"{TRAINING_CONFIG['pretraining']['samples']:,}", 
         f"{TRAINING_CONFIG['finetuning']['samples']:,}"],
        ["Batch Size", TRAINING_CONFIG['pretraining']['batch_size'], 
         TRAINING_CONFIG['finetuning']['batch_size']],
        ["Learning Rate", TRAINING_CONFIG['pretraining']['learning_rate'], 
         TRAINING_CONFIG['finetuning']['learning_rate']],
        ["Epochs", TRAINING_CONFIG['pretraining']['epochs'], 
         TRAINING_CONFIG['finetuning']['epochs']],
        ["Optimizer", TRAINING_CONFIG['pretraining']['optimizer'], "AdamW"],
        ["Weight Decay", TRAINING_CONFIG['pretraining']['weight_decay'], 
         TRAINING_CONFIG['finetuning']['weight_decay']],
        ["LR Scheduler", TRAINING_CONFIG['pretraining']['scheduler'], "cosine_annealing"],
        ["Early Stopping", "N/A", f"patience={TRAINING_CONFIG['finetuning']['patience']}"],
    ]
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(table)
    
    output_path = output_dir / "table2_training.txt"
    with open(output_path, 'w') as f:
        f.write("Table 2: Training Hyperparameters\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
    print(f"\n✓ Table saved to: {output_path}")
    
    return table


def generate_table3_decoder_comparison(output_dir: Path = None):
    """
    Generate Table 3: Decoder Performance Comparison
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 3: Decoder Performance Comparison (Logical Error Rate %)")
    print("=" * 70)
    
    decoders = DECODER_COMPARISON['decoders']
    headers = ["Decoder"] + ["SI1000 p=0.5%", "SI1000 p=1.0%", "Pauli+ p=0.5%", "Avg"]
    rows = []
    
    for decoder in decoders:
        vals = [
            DECODER_COMPARISON['si1000_p0.005_d5'][decoder] * 100,
            DECODER_COMPARISON['si1000_p0.01_d5'][decoder] * 100,
            DECODER_COMPARISON['pauli_plus_p0.005_d5'][decoder] * 100,
        ]
        avg = np.mean(vals)
        rows.append([decoder] + [f"{v:.2f}%" for v in vals] + [f"{avg:.2f}%"])
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(table)
    
    # Add summary
    print(f"\nThreshold Comparison:")
    print(f"  AlphaQubit: {THRESHOLD_ALPHAQUBIT*100:.2f}%")
    print(f"  MWPM:       {THRESHOLD_MWPM*100:.2f}%")
    print(f"  Improvement: {(THRESHOLD_ALPHAQUBIT - THRESHOLD_MWPM) / THRESHOLD_MWPM * 100:.1f}%")
    
    output_path = output_dir / "table3_decoder_comparison.txt"
    with open(output_path, 'w') as f:
        f.write("Table 3: Decoder Performance Comparison (Logical Error Rate %)\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
        f.write(f"\n\nThreshold Comparison:\n")
        f.write(f"  AlphaQubit: {THRESHOLD_ALPHAQUBIT*100:.2f}%\n")
        f.write(f"  MWPM:       {THRESHOLD_MWPM*100:.2f}%\n")
    print(f"\n✓ Table saved to: {output_path}")
    
    return table


def generate_table4_finetuning(output_dir: Path = None):
    """
    Generate Table 4: Fine-tuning Results Summary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 4: Fine-tuning Results on Google QEC v3.5")
    print("=" * 70)
    
    headers = ["Experiment", "LER (%)", "Accuracy (%)", "vs Baseline"]
    rows = []
    
    for exp, data in GOOGLE_QEC_EXPERIMENTS.items():
        ler = data['ler'] * 100
        acc = data['accuracy'] * 100
        vs_baseline = "✓ BETTER" if ler < PAPER_BASELINE['average_ler'] * 100 else "✗ WORSE"
        
        # Shorten experiment name
        short_name = exp.replace('surface_code_', '')
        rows.append([short_name, f"{ler:.2f}", f"{acc:.1f}", vs_baseline])
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(table)
    
    # Summary statistics
    lers = [d['ler'] * 100 for d in GOOGLE_QEC_EXPERIMENTS.values()]
    accs = [d['accuracy'] * 100 for d in GOOGLE_QEC_EXPERIMENTS.values()]
    
    print(f"\nSummary Statistics:")
    print(f"  Average LER:      {np.mean(lers):.2f}% (Baseline: {PAPER_BASELINE['average_ler']*100:.1f}%)")
    print(f"  Best LER:         {np.min(lers):.2f}%")
    print(f"  Worst LER:        {np.max(lers):.2f}%")
    print(f"  Average Accuracy: {np.mean(accs):.1f}%")
    
    output_path = output_dir / "table4_finetuning.txt"
    with open(output_path, 'w') as f:
        f.write("Table 4: Fine-tuning Results on Google QEC v3.5\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
        f.write(f"\n\nSummary Statistics:\n")
        f.write(f"  Average LER:      {np.mean(lers):.2f}%\n")
        f.write(f"  Best LER:         {np.min(lers):.2f}%\n")
        f.write(f"  Worst LER:        {np.max(lers):.2f}%\n")
        f.write(f"  Average Accuracy: {np.mean(accs):.1f}%\n")
    print(f"\n✓ Table saved to: {output_path}")
    
    return table


def generate_all_tables(output_dir: Path = None):
    """Generate all tables."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_table1_architecture(output_dir)
    generate_table2_training(output_dir)
    generate_table3_decoder_comparison(output_dir)
    generate_table4_finetuning(output_dir)
    
    print("\n" + "=" * 70)
    print("✓ All tables generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tables from AlphaQubit paper")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    generate_all_tables(output_dir)
