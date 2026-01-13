#!/usr/bin/env python3
"""
Full Decoder Comparison Pipeline
================================

This script runs a complete comparison between MWPM and AlphaQubit decoders
using proper stim simulation with SI1000 noise model.

This generates data for paper Figure 2 (threshold comparison).

Usage:
    # Quick test
    python run_full_decoder_comparison.py --test
    
    # Full paper-aligned benchmark  
    python run_full_decoder_comparison.py --full
    
    # With pre-trained AlphaQubit model
    python run_full_decoder_comparison.py --full --model-path models/alphaqubit_d5.pth
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Check dependencies
try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False
    print("Warning: stim not available")

try:
    import pymatching
    from pymatching import Matching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    print("Warning: pymatching not available")

try:
    import torch
    HAS_TORCH = True
    try:
        import torch_npu
        HAS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
    except ImportError:
        HAS_NPU = False
except ImportError:
    HAS_TORCH = False
    HAS_NPU = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Paper reference thresholds
PAPER_THRESHOLDS = {
    'MWPM': 0.0069,
    'AlphaQubit': 0.0082,
}


def get_device(device_str: str = 'auto') -> str:
    """Get available compute device."""
    if device_str != 'auto':
        return device_str
    
    if HAS_NPU:
        return 'npu'
    elif HAS_TORCH and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def generate_si1000_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    basis: str = 'z'
) -> 'stim.Circuit':
    """Generate surface code circuit with SI1000 noise."""
    p = physical_error_rate
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p / 10,
        before_measure_flip_probability=5 * p,
        after_reset_flip_probability=2 * p,
    )


def run_mwpm_benchmark(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    num_samples: int
) -> Dict:
    """Run MWPM decoder on stim samples."""
    if not HAS_STIM or not HAS_PYMATCHING:
        return {'error': 'Missing stim or pymatching'}
    
    circuit = generate_si1000_circuit(distance, rounds, physical_error_rate)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    
    # Sample and decode
    detection_events, observable_flips = sampler.sample(
        num_samples, separate_observables=True
    )
    predictions = matching.decode_batch(detection_events)
    
    errors = (predictions != observable_flips).sum()
    ler = errors / num_samples
    
    return {
        'decoder': 'MWPM',
        'distance': distance,
        'rounds': rounds,
        'physical_error_rate': physical_error_rate,
        'num_samples': num_samples,
        'num_errors': int(errors),
        'logical_error_rate': float(ler)
    }


def run_alphaqubit_benchmark(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    num_samples: int,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> Dict:
    """Run AlphaQubit decoder on stim samples."""
    if not HAS_STIM or not HAS_TORCH:
        return {'error': 'Missing stim or torch'}
    
    # Import model
    from ai_models.model import AlphaQubitDecoder
    
    circuit = generate_si1000_circuit(distance, rounds, physical_error_rate)
    sampler = circuit.compile_detector_sampler()
    
    # Sample
    detection_events, observable_flips = sampler.sample(
        num_samples, separate_observables=True
    )
    
    # Prepare model
    num_stabilizers = distance * distance - 1
    
    dev = torch.device(device)
    model = AlphaQubitDecoder(
        num_features=1,  # Detection events only
        hidden_dim=256,
        num_stabilizers=num_stabilizers,
        grid_size=distance,
        num_heads=8,
        num_layers=12
    ).to(dev)
    
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=dev))
        model_status = 'loaded'
    else:
        model_status = 'random_init'
    
    model.eval()
    
    # Reshape detections to (N, R, S, 1)
    syndromes = detection_events.reshape(num_samples, rounds, num_stabilizers)
    inputs = torch.from_numpy(syndromes[..., np.newaxis].astype(np.float32)).to(dev)
    basis = torch.zeros(num_samples, dtype=torch.long, device=dev)
    final_mask = torch.zeros(num_samples, num_stabilizers, device=dev)
    
    # Decode in batches
    batch_size = 256
    predictions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_inputs = inputs[i:batch_end]
            batch_basis = basis[i:batch_end]
            batch_mask = final_mask[i:batch_end]
            
            logits = model(batch_inputs, batch_basis, batch_mask)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
            predictions.extend(preds.flatten())
    
    predictions = np.array(predictions)
    errors = (predictions != observable_flips.flatten()).sum()
    ler = errors / num_samples
    
    return {
        'decoder': 'AlphaQubit',
        'distance': distance,
        'rounds': rounds,
        'physical_error_rate': physical_error_rate,
        'num_samples': num_samples,
        'num_errors': int(errors),
        'logical_error_rate': float(ler),
        'model_status': model_status,
        'device': str(device)
    }


def run_comparison(
    distances: List[int],
    p_values: List[float],
    num_samples: int,
    model_path: Optional[str] = None,
    device: str = 'auto',
    verbose: bool = True
) -> Dict:
    """Run full comparison between MWPM and AlphaQubit."""
    
    device = get_device(device)
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'distances': distances,
            'p_values': p_values,
            'num_samples': num_samples,
            'device': device,
            'model_path': model_path
        },
        'mwpm': {},
        'alphaqubit': {}
    }
    
    total = len(distances) * len(p_values)
    current = 0
    
    for d in distances:
        d_key = f'd{d}'
        rounds = d
        
        results['mwpm'][d_key] = {'p': [], 'ler': []}
        results['alphaqubit'][d_key] = {'p': [], 'ler': []}
        
        for p in p_values:
            current += 1
            
            if verbose:
                print(f"\n[{current}/{total}] d={d}, p={p:.4f}")
            
            # Run MWPM
            if HAS_STIM and HAS_PYMATCHING:
                if verbose:
                    print(f"  MWPM...", end=' ', flush=True)
                start = time.time()
                mwpm_result = run_mwpm_benchmark(d, rounds, p, num_samples)
                elapsed = time.time() - start
                
                if 'error' not in mwpm_result:
                    results['mwpm'][d_key]['p'].append(p)
                    results['mwpm'][d_key]['ler'].append(mwpm_result['logical_error_rate'])
                    if verbose:
                        print(f"LER={mwpm_result['logical_error_rate']:.5f} ({elapsed:.1f}s)")
            
            # Run AlphaQubit
            if HAS_STIM and HAS_TORCH:
                if verbose:
                    print(f"  AlphaQubit...", end=' ', flush=True)
                start = time.time()
                aq_result = run_alphaqubit_benchmark(d, rounds, p, num_samples, model_path, device)
                elapsed = time.time() - start
                
                if 'error' not in aq_result:
                    results['alphaqubit'][d_key]['p'].append(p)
                    results['alphaqubit'][d_key]['ler'].append(aq_result['logical_error_rate'])
                    if verbose:
                        print(f"LER={aq_result['logical_error_rate']:.5f} ({elapsed:.1f}s)")
    
    return results


def plot_comparison(results: Dict, output_path: Path):
    """Generate comparison plot."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3'}
    
    # Plot MWPM
    for d_key, data in results['mwpm'].items():
        d = int(d_key[1:])
        if data['p'] and data['ler']:
            ax.semilogy(
                np.array(data['p']) * 100,
                data['ler'],
                'o--',
                color=colors.get(d, 'gray'),
                label=f'MWPM d={d}',
                alpha=0.7
            )
    
    # Plot AlphaQubit
    for d_key, data in results['alphaqubit'].items():
        d = int(d_key[1:])
        if data['p'] and data['ler']:
            ax.semilogy(
                np.array(data['p']) * 100,
                data['ler'],
                's-',
                color=colors.get(d, 'gray'),
                label=f'AlphaQubit d={d}',
                linewidth=2
            )
    
    # Threshold lines
    ax.axvline(x=PAPER_THRESHOLDS['MWPM'] * 100, color='blue', linestyle=':', 
               alpha=0.5, label=f"MWPM threshold ({PAPER_THRESHOLDS['MWPM']*100:.2f}%)")
    ax.axvline(x=PAPER_THRESHOLDS['AlphaQubit'] * 100, color='red', linestyle=':', 
               alpha=0.5, label=f"AlphaQubit threshold ({PAPER_THRESHOLDS['AlphaQubit']*100:.2f}%)")
    
    ax.set_xlabel('Physical Error Rate (%)', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Decoder Comparison: AlphaQubit vs MWPM (SI1000)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Full decoder comparison')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--full', action='store_true', help='Full benchmark')
    parser.add_argument('--distances', type=int, nargs='+', default=None)
    parser.add_argument('--p-values', type=float, nargs='+', default=None)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='benchmark_results')
    
    args = parser.parse_args()
    
    # Configuration
    if args.test:
        distances = [3, 5]
        p_values = [0.003, 0.005, 0.007, 0.009]
        num_samples = 1000
        print("\n🧪 TEST MODE")
    elif args.full:
        distances = [3, 5, 7]
        p_values = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        num_samples = 50000
        print("\n🚀 FULL BENCHMARK")
    else:
        distances = args.distances or [3, 5]
        p_values = args.p_values or [0.003, 0.005, 0.007]
        num_samples = args.samples or 5000
    
    if args.distances:
        distances = args.distances
    if args.p_values:
        p_values = args.p_values
    if args.samples:
        num_samples = args.samples
    
    print(f"\nConfiguration:")
    print(f"  Distances: {distances}")
    print(f"  P values: {p_values}")
    print(f"  Samples: {num_samples:,}")
    print(f"  Device: {args.device}")
    if args.model_path:
        print(f"  Model: {args.model_path}")
    
    # Check dependencies
    print(f"\nDependencies:")
    print(f"  stim: {'✓' if HAS_STIM else '✗'}")
    print(f"  pymatching: {'✓' if HAS_PYMATCHING else '✗'}")
    print(f"  torch: {'✓' if HAS_TORCH else '✗'}")
    print(f"  NPU: {'✓' if HAS_NPU else '✗'}")
    
    if not HAS_STIM:
        print("\n❌ stim is required. Install with: pip install stim")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path('paper_figures/output')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    print("\n" + "=" * 60)
    print("DECODER COMPARISON")
    print("=" * 60)
    
    start = time.time()
    results = run_comparison(
        distances=distances,
        p_values=p_values,
        num_samples=num_samples,
        model_path=args.model_path,
        device=args.device,
        verbose=True
    )
    elapsed = time.time() - start
    
    # Save results
    results_path = output_dir / 'decoder_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate plot
    if HAS_MATPLOTLIB:
        plot_path = figures_dir / 'decoder_comparison.png'
        plot_comparison(results, plot_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds")
    
    print("\nMWPM Results:")
    for d_key, data in results['mwpm'].items():
        if data['p']:
            avg_ler = np.mean(data['ler'])
            print(f"  {d_key}: avg LER = {avg_ler:.5f}")
    
    print("\nAlphaQubit Results:")
    for d_key, data in results['alphaqubit'].items():
        if data['p']:
            avg_ler = np.mean(data['ler'])
            print(f"  {d_key}: avg LER = {avg_ler:.5f}")
    
    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
