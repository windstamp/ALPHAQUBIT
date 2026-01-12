"""
Generate Tables (Simple Version - No External Dependencies)
===========================================================

Generates tables without requiring tabulate package.
"""

import numpy as np
from pathlib import Path

from paper_figures.paper_data import (
    MODEL_ARCHITECTURE,
    TRAINING_CONFIG,
    DECODER_COMPARISON,
    GOOGLE_QEC_EXPERIMENTS,
    PAPER_BASELINE,
    THRESHOLD_ALPHAQUBIT,
    THRESHOLD_MWPM,
)


def format_table(headers, rows, col_widths=None):
    """Simple table formatter."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                      for i, h in enumerate(headers)]
    
    # Header
    header_line = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    separator = "+".join("-" * w for w in col_widths)
    
    lines = [separator, header_line, separator]
    
    # Rows
    for row in rows:
        row_line = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        lines.append(row_line)
    
    lines.append(separator)
    return "\n".join(lines)


def generate_table1_architecture(output_dir: Path = None):
    """Generate Table 1: Model Architecture Configurations"""
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
    
    table = format_table(headers, rows)
    print(table)
    
    output_path = output_dir / "table1_architecture.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Table 1: AlphaQubit Model Architecture Configurations\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
    print(f"\n✓ Table saved to: {output_path}")


def generate_table2_training(output_dir: Path = None):
    """Generate Table 2: Training Hyperparameters"""
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
        ["Optimizer", "AdamW", "AdamW"],
        ["Weight Decay", TRAINING_CONFIG['pretraining']['weight_decay'], 
         TRAINING_CONFIG['finetuning']['weight_decay']],
        ["Early Stopping", "N/A", f"patience={TRAINING_CONFIG['finetuning']['patience']}"],
    ]
    
    table = format_table(headers, rows)
    print(table)
    
    output_path = output_dir / "table2_training.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Table 2: Training Hyperparameters\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
    print(f"\n✓ Table saved to: {output_path}")


def generate_table3_decoder_comparison(output_dir: Path = None):
    """Generate Table 3: Decoder Performance Comparison"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 3: Decoder Performance Comparison (Logical Error Rate %)")
    print("=" * 70)
    
    decoders = DECODER_COMPARISON['decoders']
    headers = ["Decoder", "SI1000 p=0.5%", "SI1000 p=1.0%", "Pauli+ p=0.5%", "Average"]
    rows = []
    
    for decoder in decoders:
        vals = [
            DECODER_COMPARISON['si1000_p0.005_d5'][decoder] * 100,
            DECODER_COMPARISON['si1000_p0.01_d5'][decoder] * 100,
            DECODER_COMPARISON['pauli_plus_p0.005_d5'][decoder] * 100,
        ]
        avg = np.mean(vals)
        rows.append([decoder] + [f"{v:.2f}%" for v in vals] + [f"{avg:.2f}%"])
    
    table = format_table(headers, rows)
    print(table)
    
    print(f"\nThreshold Comparison:")
    print(f"  AlphaQubit: {THRESHOLD_ALPHAQUBIT*100:.2f}%")
    print(f"  MWPM:       {THRESHOLD_MWPM*100:.2f}%")
    
    output_path = output_dir / "table3_decoder_comparison.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Table 3: Decoder Performance Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
        f.write(f"\n\nThreshold: AlphaQubit={THRESHOLD_ALPHAQUBIT*100:.2f}%, MWPM={THRESHOLD_MWPM*100:.2f}%\n")
    print(f"\n✓ Table saved to: {output_path}")


def generate_table4_finetuning(output_dir: Path = None):
    """Generate Table 4: Fine-tuning Results Summary"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Table 4: Fine-tuning Results on Google QEC v3.5")
    print("=" * 70)
    
    headers = ["Experiment", "LER (%)", "Accuracy (%)", "Status"]
    rows = []
    
    for exp, data in GOOGLE_QEC_EXPERIMENTS.items():
        ler = data['ler'] * 100
        acc = data['accuracy'] * 100
        status = "BETTER" if ler < PAPER_BASELINE['average_ler'] * 100 else "WORSE"
        short_name = exp.replace('surface_code_', '')
        rows.append([short_name, f"{ler:.2f}", f"{acc:.1f}", status])
    
    table = format_table(headers, rows)
    print(table)
    
    lers = [d['ler'] * 100 for d in GOOGLE_QEC_EXPERIMENTS.values()]
    accs = [d['accuracy'] * 100 for d in GOOGLE_QEC_EXPERIMENTS.values()]
    
    print(f"\nSummary: Avg LER={np.mean(lers):.2f}%, Best={np.min(lers):.2f}%, Worst={np.max(lers):.2f}%")
    
    output_path = output_dir / "table4_finetuning.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Table 4: Fine-tuning Results on Google QEC v3.5\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
        f.write(f"\n\nSummary: Avg LER={np.mean(lers):.2f}%, Best={np.min(lers):.2f}%, Worst={np.max(lers):.2f}%\n")
    print(f"\n✓ Table saved to: {output_path}")


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
