#!/usr/bin/env python3
"""
Generate FULL pre-training data for AlphaQubit (Paper-Aligned)
==============================================================

This script generates the complete 8.5M samples required by the paper.

Paper specifications:
- Total: 8,500,000 samples
- p ∈ {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01}
- d ∈ {3, 5, 7}
- r ∈ {1, 5, 10, 25}
- Basis: X and Z

Usage:
    python generate_full_pretrain_data.py
    
Or with custom samples:
    python generate_full_pretrain_data.py --samples 1000000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import stim
except ImportError:
    print("ERROR: stim not installed. Run: pip install stim")
    sys.exit(1)


# =============================================================================
# Paper Constants
# =============================================================================

PAPER_P_GRID = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
PAPER_CODE_DISTANCES = [3, 5, 7]
PAPER_ROUNDS = [1, 5, 10, 25]
PAPER_TOTAL_SAMPLES = 8_500_000


def create_si1000_circuit(d: int, rounds: int, p: float, basis: str = "Z") -> stim.Circuit:
    """Create an SI1000 surface code circuit.
    
    SI1000 noise model parameters (per Google/Stim standard):
    - meas_bitflip: 5p (before measurement)
    - reset_bitflip: 2p (after reset)  
    - twoq_depol: p (after 2Q Clifford gates)
    - oneq_depol: p/10 (after 1Q Clifford gates)
    - idle: p/10 (before round data depolarization)
    
    Args:
        d: Code distance (3, 5, or 7)
        rounds: Number of QEC rounds
        p: Physical error rate
        basis: Measurement basis ("X" or "Z")
    
    Returns:
        stim.Circuit with SI1000 noise
    """
    # IMPORTANT: These parameters must match the paper exactly!
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_{}".format(basis.lower()),
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,          # p for 2Q gates (DEPOLARIZE2)
        after_reset_flip_probability=2 * p,       # 2p for reset (SI1000 spec)
        before_measure_flip_probability=5 * p,    # 5p for measurement (SI1000 spec)
        before_round_data_depolarization=p / 10,  # p/10 for idle (SI1000 spec)
    )
    return circuit


def sample_circuit(circuit: stim.Circuit, n_shots: int) -> tuple:
    """Sample detection events and observables from a circuit."""
    sampler = circuit.compile_detector_sampler()
    detection_events, observables = sampler.sample(n_shots, separate_observables=True)
    return detection_events.astype(np.float32), observables.astype(np.float32).flatten()


def generate_and_save_data(
    total_samples: int,
    p_grid: list,
    distances: list,
    rounds_list: list,
    output_dir: Path,
):
    """Generate and save pre-training data as NPZ files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per configuration
    n_configs = len(p_grid) * len(distances) * len(rounds_list) * 2  # *2 for X and Z basis
    samples_per_config = total_samples // n_configs
    
    print("=" * 70)
    print("GENERATING PAPER-ALIGNED PRE-TRAINING DATA")
    print("=" * 70)
    print(f"Total samples target: {total_samples:,}")
    print(f"Configurations: {n_configs}")
    print(f"Samples per config: {samples_per_config:,}")
    print(f"Physical error rates (p): {p_grid}")
    print(f"Code distances (d): {distances}")
    print(f"Rounds (r): {rounds_list}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    manifest = []
    total_generated = 0
    
    # Progress bar for all configurations
    configs = [(p, d, r, b) for p in p_grid for d in distances for r in rounds_list for b in ["X", "Z"]]
    
    for p, d, r, basis in tqdm(configs, desc="Generating data"):
        try:
            circuit = create_si1000_circuit(d, r, p, basis)
            syndromes, labels = sample_circuit(circuit, samples_per_config)
            
            # Save as NPZ file
            filename = f"pretrain_p{p:.4f}_d{d}_r{r}_{basis}.npz"
            filepath = output_dir / filename
            
            np.savez_compressed(
                filepath,
                syndromes=syndromes,
                labels=labels,
                p=p,
                d=d,
                r=r,
                basis=basis,
            )
            
            pos_ratio = labels.mean()
            manifest.append({
                'filename': filename,
                'p': p,
                'd': d,
                'r': r,
                'basis': basis,
                'samples': len(labels),
                'positive_ratio': float(pos_ratio),
            })
            
            total_generated += len(labels)
            
        except Exception as e:
            print(f"\nWARNING: Failed p={p}, d={d}, r={r}, basis={basis}: {e}")
            continue
    
    # Save manifest
    manifest_data = {
        'total_samples': total_generated,
        'target_samples': total_samples,
        'n_configs': len(manifest),
        'p_grid': p_grid,
        'distances': distances,
        'rounds': rounds_list,
        'timestamp': datetime.now().isoformat(),
        'configs': manifest,
    }
    
    manifest_path = output_dir / "pretrain_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total samples generated: {total_generated:,}")
    print(f"Total configurations: {len(manifest)}")
    print(f"Files saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    
    # Show positive ratio distribution
    pos_ratios = [m['positive_ratio'] for m in manifest]
    print(f"\nPositive ratio statistics:")
    print(f"  Min:  {min(pos_ratios):.2%}")
    print(f"  Max:  {max(pos_ratios):.2%}")
    print(f"  Mean: {np.mean(pos_ratios):.2%}")
    
    # Show breakdown by p
    print(f"\nSamples per noise level:")
    for p in p_grid:
        samples_at_p = sum(m['samples'] for m in manifest if m['p'] == p)
        avg_pos = np.mean([m['positive_ratio'] for m in manifest if m['p'] == p])
        print(f"  p={p:.3f}: {samples_at_p:,} samples, avg_pos_ratio={avg_pos:.2%}")
    
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Generate full pre-training data for AlphaQubit")
    
    parser.add_argument("--samples", type=int, default=PAPER_TOTAL_SAMPLES,
                        help=f"Total samples (paper: {PAPER_TOTAL_SAMPLES:,})")
    parser.add_argument("--output-dir", type=str, default="pretrain_data",
                        help="Output directory for data")
    parser.add_argument("--p-grid", type=str, default=None,
                        help="Custom p-grid (comma-separated). Default: paper values")
    parser.add_argument("--distances", type=str, default=None,
                        help="Custom distances (comma-separated). Default: 3,5,7")
    parser.add_argument("--rounds", type=str, default=None,
                        help="Custom rounds (comma-separated). Default: 1,5,10,25")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 100K samples")
    
    args = parser.parse_args()
    
    # Parse parameters
    if args.p_grid:
        p_grid = [float(x) for x in args.p_grid.split(",")]
    else:
        p_grid = PAPER_P_GRID
    
    if args.distances:
        distances = [int(x) for x in args.distances.split(",")]
    else:
        distances = PAPER_CODE_DISTANCES
    
    if args.rounds:
        rounds_list = [int(x) for x in args.rounds.split(",")]
    else:
        rounds_list = PAPER_ROUNDS
    
    # Quick mode
    if args.quick:
        args.samples = 100_000
        print("=== QUICK MODE: 100K samples ===\n")
    
    output_dir = Path(args.output_dir)
    
    generate_and_save_data(
        total_samples=args.samples,
        p_grid=p_grid,
        distances=distances,
        rounds_list=rounds_list,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
