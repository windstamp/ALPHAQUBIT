"""
Decoder Comparison Benchmark Script
===================================

This script benchmarks all implemented decoders against each other
and compares with the AlphaQubit neural network decoder.

Usage:
    python -m decoders.benchmark [--distance 5] [--samples 1000]
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import json

# Import decoders
from decoders.base import BaseDecoder
from decoders.belief_propagation import BeliefPropagationDecoder
from decoders.union_find import UnionFindDecoder
from decoders.tensor_network import TensorNetworkDecoder

# Try to import MWPM (requires pymatching)
try:
    from decoders.mwpm_decoder import MWPMDecoder
    HAS_MWPM = True
except ImportError:
    HAS_MWPM = False
    print("Warning: MWPM decoder not available (install pymatching)")

# Try to import AlphaQubit
try:
    import torch
    from ai_models.model import AlphaQubitDecoder
    HAS_ALPHAQUBIT = True
except ImportError:
    HAS_ALPHAQUBIT = False
    print("Warning: AlphaQubit decoder not available")


def generate_synthetic_data(
    distance: int,
    rounds: int,
    num_samples: int,
    physical_error_rate: float = 0.01
) -> tuple:
    """
    Generate synthetic syndrome data for benchmarking.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome rounds
        num_samples: Number of samples to generate
        physical_error_rate: Physical error probability
        
    Returns:
        (syndromes, labels) tuple
    """
    num_stabilizers = distance * distance - 1
    
    # Generate random syndromes
    syndromes = np.random.binomial(
        1, physical_error_rate * 2,  # Detection probability
        size=(num_samples, rounds, num_stabilizers)
    ).astype(np.float32)
    
    # Generate labels (simplified: based on syndrome parity)
    labels = (syndromes.sum(axis=(1, 2)) > num_stabilizers * rounds / 4).astype(np.int32)
    
    return syndromes, labels


def benchmark_decoder(
    decoder: BaseDecoder,
    syndromes: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Benchmark a single decoder.
    
    Args:
        decoder: Decoder instance
        syndromes: Syndrome data
        labels: Ground truth labels
        
    Returns:
        Dictionary with benchmark results
    """
    num_samples = len(syndromes)
    
    # Decode and measure time
    start_time = time.time()
    predictions = decoder.decode(syndromes)
    decode_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    logical_error_rate = 1 - accuracy
    
    return {
        'decoder': decoder.__class__.__name__,
        'accuracy': float(accuracy),
        'logical_error_rate': float(logical_error_rate),
        'decode_time_seconds': float(decode_time),
        'samples_per_second': num_samples / decode_time,
        'info': decoder.get_info()
    }


def benchmark_alphaqubit(
    model_path: Optional[str],
    syndromes: np.ndarray,
    labels: np.ndarray,
    distance: int
) -> Dict[str, Any]:
    """
    Benchmark AlphaQubit decoder.
    """
    if not HAS_ALPHAQUBIT:
        return {'decoder': 'AlphaQubit', 'error': 'Not available'}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples, rounds, num_stabilizers = syndromes.shape
    num_features = 1  # Using detection events only
    grid_size = int(np.ceil(np.sqrt(num_stabilizers + 1))) - 1
    
    # Create model
    model = AlphaQubitDecoder(
        num_features=num_features,
        hidden_dim=256,
        num_stabilizers=num_stabilizers,
        grid_size=grid_size,
        num_heads=8,
        num_layers=12
    ).to(device)
    
    # Load weights if available
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Prepare data
    inputs = torch.from_numpy(syndromes[..., np.newaxis]).to(device)
    basis = torch.zeros(num_samples, dtype=torch.long, device=device)
    final_mask = torch.zeros(num_samples, num_stabilizers, device=device)
    
    # Decode
    start_time = time.time()
    with torch.no_grad():
        logits = model(inputs, basis, final_mask)
        predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
    decode_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    
    return {
        'decoder': 'AlphaQubit',
        'accuracy': float(accuracy),
        'logical_error_rate': float(1 - accuracy),
        'decode_time_seconds': float(decode_time),
        'samples_per_second': num_samples / decode_time,
        'device': str(device),
        'model_path': str(model_path) if model_path else 'random_init'
    }


def run_benchmark(
    distance: int = 5,
    rounds: Optional[int] = None,
    num_samples: int = 1000,
    physical_error_rate: float = 0.01,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full decoder benchmark.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome rounds (default: distance)
        num_samples: Number of test samples
        physical_error_rate: Physical error probability
        model_path: Path to AlphaQubit model weights
        output_file: Path to save results JSON
        
    Returns:
        Benchmark results dictionary
    """
    if rounds is None:
        rounds = distance
    
    print(f"\n{'='*60}")
    print(f"DECODER BENCHMARK")
    print(f"{'='*60}")
    print(f"Code distance: {distance}")
    print(f"Syndrome rounds: {rounds}")
    print(f"Test samples: {num_samples}")
    print(f"Physical error rate: {physical_error_rate}")
    print(f"{'='*60}\n")
    
    # Generate test data
    print("Generating synthetic test data...")
    syndromes, labels = generate_synthetic_data(
        distance, rounds, num_samples, physical_error_rate
    )
    print(f"  Syndromes shape: {syndromes.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Positive rate: {labels.mean():.2%}\n")
    
    results = {
        'config': {
            'distance': distance,
            'rounds': rounds,
            'num_samples': num_samples,
            'physical_error_rate': physical_error_rate,
        },
        'decoders': []
    }
    
    # Benchmark each decoder
    decoders_to_test = [
        ('Belief Propagation', lambda: BeliefPropagationDecoder(
            distance, rounds, physical_error_rate
        )),
        ('Union Find', lambda: UnionFindDecoder(distance, rounds)),
        ('Tensor Network', lambda: TensorNetworkDecoder(
            distance, rounds, physical_error_rate
        )),
    ]
    
    if HAS_MWPM:
        decoders_to_test.insert(0, ('MWPM', lambda: MWPMDecoder(
            distance, rounds, physical_error_rate
        )))
    
    for name, create_decoder in decoders_to_test:
        print(f"Benchmarking {name}...")
        try:
            decoder = create_decoder()
            result = benchmark_decoder(decoder, syndromes, labels)
            results['decoders'].append(result)
            print(f"  ✓ Accuracy: {result['accuracy']:.2%}")
            print(f"  ✓ LER: {result['logical_error_rate']:.4f}")
            print(f"  ✓ Speed: {result['samples_per_second']:.1f} samples/sec\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results['decoders'].append({
                'decoder': name,
                'error': str(e)
            })
    
    # Benchmark AlphaQubit
    print("Benchmarking AlphaQubit...")
    try:
        result = benchmark_alphaqubit(model_path, syndromes, labels, distance)
        results['decoders'].append(result)
        if 'error' not in result:
            print(f"  ✓ Accuracy: {result['accuracy']:.2%}")
            print(f"  ✓ LER: {result['logical_error_rate']:.4f}")
            print(f"  ✓ Speed: {result['samples_per_second']:.1f} samples/sec\n")
        else:
            print(f"  ✗ {result['error']}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        results['decoders'].append({
            'decoder': 'AlphaQubit',
            'error': str(e)
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Decoder':<25} {'Accuracy':>10} {'LER':>10} {'Speed':>15}")
    print(f"{'-'*60}")
    
    for r in results['decoders']:
        if 'error' not in r:
            print(f"{r['decoder']:<25} {r['accuracy']:>10.2%} "
                  f"{r['logical_error_rate']:>10.4f} "
                  f"{r['samples_per_second']:>12.1f}/s")
        else:
            print(f"{r['decoder']:<25} {'ERROR':>10}")
    
    print(f"{'='*60}\n")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark QEC decoders')
    parser.add_argument('--distance', '-d', type=int, default=5,
                        help='Code distance (default: 5)')
    parser.add_argument('--rounds', '-r', type=int, default=None,
                        help='Syndrome rounds (default: distance)')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                        help='Number of test samples (default: 1000)')
    parser.add_argument('--error-rate', '-p', type=float, default=0.01,
                        help='Physical error rate (default: 0.01)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to AlphaQubit model weights')
    parser.add_argument('--output', '-o', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    run_benchmark(
        distance=args.distance,
        rounds=args.rounds,
        num_samples=args.samples,
        physical_error_rate=args.error_rate,
        model_path=args.model,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
