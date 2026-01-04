#!/usr/bin/env python3
"""
MWPM Threshold Benchmark Script
===============================

This script runs the MWPM (Minimum Weight Perfect Matching) decoder using
PyMatching and stim to generate proper threshold data comparable to the
AlphaQubit paper.

Paper reference thresholds (SI1000 noise model):
- MWPM: ~0.69%
- AlphaQubit: ~0.82%

Usage:
    # Quick test (small scale)
    python run_mwpm_benchmark.py --test
    
    # Full benchmark (paper-aligned)
    python run_mwpm_benchmark.py --full
    
    # Custom configuration
    python run_mwpm_benchmark.py --distances 3 5 7 --p-values 0.001 0.005 0.01 --samples 10000

Output:
    - benchmark_results/mwpm_threshold_data.json
    - paper_figures/output/mwpm_threshold_curve.png
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Check dependencies
try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False
    print("ERROR: stim is required. Install with: pip install stim")

try:
    import pymatching
    from pymatching import Matching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    print("ERROR: pymatching is required. Install with: pip install pymatching")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available for plotting")


# Paper reference data
PAPER_THRESHOLDS = {
    'MWPM': 0.0069,      # ~0.69%
    'AlphaQubit': 0.0082  # ~0.82%
}

# SI1000 noise model (from paper)
# after_clifford_depolarization = p
# before_round_data_depolarization = p/10
# before_measure_flip_probability = 5p
# after_reset_flip_probability = 2p


def generate_si1000_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    basis: str = 'z'
) -> stim.Circuit:
    """
    Generate surface code circuit with SI1000 noise model.
    
    SI1000 is the standard circuit-level noise model from the paper.
    """
    if not HAS_STIM:
        raise ImportError("stim is required for circuit generation")
    
    p = physical_error_rate
    
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,           # p for 2Q gates
        before_round_data_depolarization=p / 10,   # p/10 for idle
        before_measure_flip_probability=5 * p,     # 5p for measurement
        after_reset_flip_probability=2 * p,        # 2p for reset
    )
    
    return circuit


def run_mwpm_simulation(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    num_samples: int,
    basis: str = 'z'
) -> Dict[str, Any]:
    """
    Run MWPM decoder on stim-generated samples.
    
    Args:
        distance: Code distance d
        rounds: Number of syndrome measurement rounds
        physical_error_rate: Physical error probability p
        num_samples: Number of shots to simulate
        basis: Measurement basis ('x' or 'z')
        
    Returns:
        Dictionary with logical error rate and statistics
    """
    if not HAS_STIM or not HAS_PYMATCHING:
        raise ImportError("stim and pymatching are required")
    
    # Generate circuit
    circuit = generate_si1000_circuit(distance, rounds, physical_error_rate, basis)
    
    # Create detector error model and matching graph
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = Matching.from_detector_error_model(dem)
    
    # Sample from circuit
    sampler = circuit.compile_detector_sampler()
    
    # Run in batches for memory efficiency
    batch_size = min(num_samples, 10000)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_errors = 0
    total_samples = 0
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - total_samples)
        
        # Sample detection events and observables
        detection_events, observable_flips = sampler.sample(
            current_batch_size,
            separate_observables=True
        )
        
        # Decode each sample
        predicted_observables = matching.decode_batch(detection_events)
        
        # Count logical errors
        errors = (predicted_observables != observable_flips).sum()
        total_errors += errors
        total_samples += current_batch_size
    
    logical_error_rate = total_errors / total_samples
    
    return {
        'distance': distance,
        'rounds': rounds,
        'physical_error_rate': physical_error_rate,
        'num_samples': total_samples,
        'num_errors': int(total_errors),
        'logical_error_rate': float(logical_error_rate),
        'basis': basis
    }


def run_threshold_sweep(
    distances: List[int],
    p_values: List[float],
    num_samples: int,
    rounds_factor: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run MWPM threshold sweep across distances and error rates.
    
    Args:
        distances: List of code distances to test
        p_values: List of physical error rates
        num_samples: Samples per configuration
        rounds_factor: rounds = distance * rounds_factor
        verbose: Print progress
        
    Returns:
        Complete results dictionary
    """
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'distances': distances,
            'p_values': p_values,
            'num_samples': num_samples,
            'rounds_factor': rounds_factor,
            'noise_model': 'SI1000',
            'decoder': 'MWPM (PyMatching)'
        },
        'data': {}
    }
    
    total_configs = len(distances) * len(p_values)
    current_config = 0
    
    for d in distances:
        d_key = f'd{d}'
        results['data'][d_key] = {
            'distance': d,
            'rounds': d * rounds_factor,
            'physical_error_rates': [],
            'logical_error_rates': [],
            'num_errors': [],
            'num_samples': []
        }
        
        for p in p_values:
            current_config += 1
            rounds = d * rounds_factor
            
            if verbose:
                print(f"[{current_config}/{total_configs}] d={d}, p={p:.4f}, "
                      f"rounds={rounds}, samples={num_samples}...", end=' ', flush=True)
            
            start_time = time.time()
            
            try:
                result = run_mwpm_simulation(
                    distance=d,
                    rounds=rounds,
                    physical_error_rate=p,
                    num_samples=num_samples
                )
                
                results['data'][d_key]['physical_error_rates'].append(p)
                results['data'][d_key]['logical_error_rates'].append(result['logical_error_rate'])
                results['data'][d_key]['num_errors'].append(result['num_errors'])
                results['data'][d_key]['num_samples'].append(result['num_samples'])
                
                elapsed = time.time() - start_time
                
                if verbose:
                    print(f"LER={result['logical_error_rate']:.4f} ({elapsed:.1f}s)")
                    
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results['data'][d_key]['physical_error_rates'].append(p)
                results['data'][d_key]['logical_error_rates'].append(None)
                results['data'][d_key]['num_errors'].append(0)
                results['data'][d_key]['num_samples'].append(0)
    
    # Estimate threshold
    results['threshold_estimate'] = estimate_threshold(results['data'])
    
    return results


def estimate_threshold(data: Dict) -> Dict[str, Any]:
    """
    Estimate threshold from LER vs p data.
    
    At threshold, LER is independent of distance.
    """
    threshold_info = {
        'method': 'crossing_point',
        'estimated_threshold': None,
        'confidence': 'low'
    }
    
    # Find crossing points between different distances
    distances = sorted([int(k[1:]) for k in data.keys()])
    if len(distances) < 2:
        return threshold_info
    
    # Simple estimation: find where curves cross
    d1_key = f'd{distances[0]}'
    d2_key = f'd{distances[-1]}'
    
    p1 = np.array(data[d1_key]['physical_error_rates'])
    ler1 = np.array([x if x is not None else np.nan for x in data[d1_key]['logical_error_rates']])
    p2 = np.array(data[d2_key]['physical_error_rates'])
    ler2 = np.array([x if x is not None else np.nan for x in data[d2_key]['logical_error_rates']])
    
    # Find where smaller distance has lower LER (below threshold)
    # and larger distance has lower LER (above threshold)
    valid = ~np.isnan(ler1) & ~np.isnan(ler2)
    
    if valid.sum() >= 2:
        diff = ler1[valid] - ler2[valid]
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation
            p_low = p1[valid][idx]
            p_high = p1[valid][idx + 1]
            threshold_info['estimated_threshold'] = (p_low + p_high) / 2
            threshold_info['confidence'] = 'medium'
    
    return threshold_info


def plot_threshold_curve(results: Dict, output_path: Path):
    """
    Generate threshold curve plot (paper Figure 2 style).
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {3: 'blue', 5: 'green', 7: 'red', 9: 'purple', 11: 'orange'}
    markers = {3: 'o', 5: 's', 7: '^', 9: 'v', 11: 'D'}
    
    data = results['data']
    
    for d_key in sorted(data.keys()):
        d = data[d_key]['distance']
        p_vals = data[d_key]['physical_error_rates']
        ler_vals = data[d_key]['logical_error_rates']
        
        # Filter out None values
        valid = [(p, l) for p, l in zip(p_vals, ler_vals) if l is not None]
        if not valid:
            continue
            
        p_valid, ler_valid = zip(*valid)
        
        color = colors.get(d, 'gray')
        marker = markers.get(d, 'o')
        
        ax.semilogy(
            np.array(p_valid) * 100,  # Convert to percentage
            ler_valid,
            f'{marker}-',
            color=color,
            label=f'd={d}',
            markersize=8,
            linewidth=2
        )
    
    # Add threshold lines
    ax.axvline(
        x=PAPER_THRESHOLDS['MWPM'] * 100,
        color='blue',
        linestyle='--',
        alpha=0.7,
        label=f"MWPM threshold ({PAPER_THRESHOLDS['MWPM']*100:.2f}%)"
    )
    ax.axvline(
        x=PAPER_THRESHOLDS['AlphaQubit'] * 100,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f"AlphaQubit threshold ({PAPER_THRESHOLDS['AlphaQubit']*100:.2f}%)"
    )
    
    # Estimated threshold
    est = results.get('threshold_estimate', {}).get('estimated_threshold')
    if est is not None:
        ax.axvline(
            x=est * 100,
            color='green',
            linestyle=':',
            alpha=0.7,
            label=f"Measured threshold ({est*100:.2f}%)"
        )
    
    ax.set_xlabel('Physical Error Rate (%)', fontsize=12)
    ax.set_ylabel('Logical Error Rate (per round)', fontsize=12)
    ax.set_title('MWPM Decoder Threshold (SI1000 Noise Model)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(1e-4, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='MWPM Threshold Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python run_mwpm_benchmark.py --test
    
    # Full paper-aligned benchmark
    python run_mwpm_benchmark.py --full
    
    # Custom
    python run_mwpm_benchmark.py --distances 3 5 7 --p-values 0.002 0.004 0.006 0.008 --samples 50000
        """
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (small scale)')
    parser.add_argument('--full', action='store_true',
                        help='Full paper-aligned benchmark')
    
    parser.add_argument('--distances', '-d', type=int, nargs='+', default=None,
                        help='Code distances to test (default: [3, 5, 7])')
    parser.add_argument('--p-values', '-p', type=float, nargs='+', default=None,
                        help='Physical error rates to test')
    parser.add_argument('--samples', '-n', type=int, default=None,
                        help='Number of samples per configuration')
    parser.add_argument('--rounds-factor', type=int, default=1,
                        help='rounds = distance * rounds_factor')
    
    parser.add_argument('--output-dir', '-o', type=str, default='benchmark_results',
                        help='Output directory')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_STIM or not HAS_PYMATCHING:
        print("\n❌ Missing required dependencies!")
        print("Install with: pip install stim pymatching")
        sys.exit(1)
    
    # Set configuration based on mode
    if args.test:
        distances = [3, 5]
        p_values = [0.002, 0.004, 0.006, 0.008, 0.01]
        num_samples = 1000
        print("\n🧪 TEST MODE (small scale)")
    elif args.full:
        distances = [3, 5, 7, 9]
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        num_samples = 100000
        print("\n🚀 FULL BENCHMARK MODE (paper-aligned)")
    else:
        distances = args.distances or [3, 5, 7]
        p_values = args.p_values or [0.002, 0.004, 0.006, 0.008, 0.01]
        num_samples = args.samples or 10000
        print("\n⚙️ CUSTOM MODE")
    
    # Override with command line args if provided
    if args.distances:
        distances = args.distances
    if args.p_values:
        p_values = args.p_values
    if args.samples:
        num_samples = args.samples
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = Path('paper_figures/output')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Distances: {distances}")
    print(f"  P values: {p_values}")
    print(f"  Samples: {num_samples:,}")
    print(f"  Rounds factor: {args.rounds_factor}")
    print(f"  Output: {output_dir}")
    print()
    
    # Run benchmark
    print("=" * 60)
    print("MWPM THRESHOLD BENCHMARK")
    print("=" * 60)
    
    start_time = time.time()
    
    results = run_threshold_sweep(
        distances=distances,
        p_values=p_values,
        num_samples=num_samples,
        rounds_factor=args.rounds_factor,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    results_path = output_dir / 'mwpm_threshold_data.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate plot
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_path = figures_dir / 'mwpm_threshold_curve.png'
        plot_threshold_curve(results, plot_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds")
    
    est = results.get('threshold_estimate', {}).get('estimated_threshold')
    if est:
        print(f"Estimated threshold: {est*100:.3f}%")
        print(f"Paper MWPM threshold: {PAPER_THRESHOLDS['MWPM']*100:.2f}%")
        print(f"Paper AlphaQubit threshold: {PAPER_THRESHOLDS['AlphaQubit']*100:.2f}%")
    
    print("\nLogical Error Rates:")
    for d_key in sorted(results['data'].keys()):
        d_data = results['data'][d_key]
        print(f"\n  {d_key} (rounds={d_data['rounds']}):")
        for p, ler in zip(d_data['physical_error_rates'], d_data['logical_error_rates']):
            if ler is not None:
                print(f"    p={p:.4f}: LER={ler:.5f}")
    
    print("\n✅ MWPM benchmark complete!")
    return results


if __name__ == '__main__':
    main()
