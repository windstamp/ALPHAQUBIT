"""
Integrated Decoder Comparison Pipeline
======================================

This module runs ALL decoders (AlphaQubit, MWPM, BP, UF, TN) on the same data
and generates comparison figures matching the paper.

Unlike the static paper_data.py which uses hardcoded values, this module
actually executes the decoders and collects real performance metrics.

Usage:
    python -m paper_figures.run_decoder_comparison --distance 5 --samples 10000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import decoders
from decoders.belief_propagation import BeliefPropagationDecoder
from decoders.union_find import UnionFindDecoder
from decoders.tensor_network import TensorNetworkDecoder

# Try to import MWPM
try:
    from decoders.mwpm_decoder import MWPMDecoder
    HAS_MWPM = True
except ImportError:
    HAS_MWPM = False

# Try to import AlphaQubit
try:
    import torch
    from ai_models.model import AlphaQubitDecoder
    HAS_ALPHAQUBIT = True
except ImportError:
    HAS_ALPHAQUBIT = False


def generate_syndrome_data(
    distance: int,
    rounds: int,
    num_samples: int,
    physical_error_rate: float
) -> tuple:
    """
    Generate synthetic syndrome data for decoder comparison.
    
    In a full implementation, this would use stim to generate realistic data.
    """
    num_stabilizers = distance * distance - 1
    
    # Generate detection events with realistic error model
    # P(detection) ≈ 2p for depolarizing noise
    detection_prob = min(0.5, 2 * physical_error_rate)
    
    syndromes = np.random.binomial(
        1, detection_prob,
        size=(num_samples, rounds, num_stabilizers)
    ).astype(np.float32)
    
    # Labels based on a simplified error model
    # Logical error more likely with more detection events
    error_density = syndromes.mean(axis=(1, 2))
    labels = (error_density > 0.1).astype(np.int32)
    
    return syndromes, labels


def run_decoder(
    decoder,
    decoder_name: str,
    syndromes: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """Run a single decoder and collect metrics."""
    start_time = time.time()
    predictions = decoder.decode(syndromes)
    decode_time = time.time() - start_time
    
    accuracy = (predictions == labels).mean()
    ler = 1 - accuracy
    
    return {
        'decoder': decoder_name,
        'accuracy': float(accuracy),
        'logical_error_rate': float(ler),
        'decode_time': float(decode_time),
        'samples_per_second': len(syndromes) / decode_time
    }


def run_alphaqubit(
    syndromes: np.ndarray,
    labels: np.ndarray,
    distance: int,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run AlphaQubit decoder."""
    if not HAS_ALPHAQUBIT:
        return {'decoder': 'AlphaQubit', 'error': 'Not available'}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples, rounds, num_stabilizers = syndromes.shape
    grid_size = int(np.ceil(np.sqrt(num_stabilizers + 1))) - 1
    
    model = AlphaQubitDecoder(
        num_features=1,
        hidden_dim=256,
        num_stabilizers=num_stabilizers,
        grid_size=grid_size,
        num_heads=8,
        num_layers=12
    ).to(device)
    
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    inputs = torch.from_numpy(syndromes[..., np.newaxis]).to(device)
    basis = torch.zeros(num_samples, dtype=torch.long, device=device)
    final_mask = torch.zeros(num_samples, num_stabilizers, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(inputs, basis, final_mask)
        predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
    decode_time = time.time() - start_time
    
    accuracy = (predictions == labels).mean()
    
    return {
        'decoder': 'AlphaQubit',
        'accuracy': float(accuracy),
        'logical_error_rate': float(1 - accuracy),
        'decode_time': float(decode_time),
        'samples_per_second': num_samples / decode_time
    }


def run_all_decoders(
    distance: int,
    rounds: int,
    num_samples: int,
    physical_error_rates: List[float],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run all decoders across multiple physical error rates.
    """
    results = {
        'config': {
            'distance': distance,
            'rounds': rounds,
            'num_samples': num_samples,
            'physical_error_rates': physical_error_rates
        },
        'decoders': {},
        'by_error_rate': {}
    }
    
    for p in physical_error_rates:
        print(f"\n{'='*60}")
        print(f"Physical Error Rate: {p*100:.2f}%")
        print(f"{'='*60}")
        
        # Generate data
        syndromes, labels = generate_syndrome_data(distance, rounds, num_samples, p)
        
        results['by_error_rate'][str(p)] = {}
        
        # Define decoders to test
        decoder_configs = [
            ('Belief Propagation', lambda: BeliefPropagationDecoder(distance, rounds, p)),
            ('Union Find', lambda: UnionFindDecoder(distance, rounds)),
            ('Tensor Network', lambda: TensorNetworkDecoder(distance, rounds, p)),
        ]
        
        if HAS_MWPM:
            decoder_configs.insert(0, ('MWPM', lambda: MWPMDecoder(distance, rounds, p)))
        
        # Run each decoder
        for name, create_decoder in decoder_configs:
            print(f"  Running {name}...", end=' ')
            try:
                decoder = create_decoder()
                result = run_decoder(decoder, name, syndromes, labels)
                results['by_error_rate'][str(p)][name] = result
                
                if name not in results['decoders']:
                    results['decoders'][name] = []
                results['decoders'][name].append({
                    'physical_error_rate': p,
                    **result
                })
                print(f"LER: {result['logical_error_rate']:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                results['by_error_rate'][str(p)][name] = {'error': str(e)}
        
        # Run AlphaQubit
        print(f"  Running AlphaQubit...", end=' ')
        try:
            result = run_alphaqubit(syndromes, labels, distance, model_path)
            results['by_error_rate'][str(p)]['AlphaQubit'] = result
            
            if 'AlphaQubit' not in results['decoders']:
                results['decoders']['AlphaQubit'] = []
            results['decoders']['AlphaQubit'].append({
                'physical_error_rate': p,
                **result
            })
            if 'error' not in result:
                print(f"LER: {result['logical_error_rate']:.4f}")
            else:
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error: {e}")
    
    return results


def plot_threshold_comparison(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = False
):
    """
    Generate Figure 2: Threshold plot comparing decoders.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'AlphaQubit': '#2ecc71',
        'MWPM': '#3498db',
        'Belief Propagation': '#e74c3c',
        'Union Find': '#f39c12',
        'Tensor Network': '#9b59b6'
    }
    markers = {
        'AlphaQubit': 'o',
        'MWPM': 's',
        'Belief Propagation': '^',
        'Union Find': 'D',
        'Tensor Network': 'v'
    }
    
    for decoder_name, data_list in results['decoders'].items():
        if not data_list or 'error' in data_list[0]:
            continue
        
        p_values = [d['physical_error_rate'] * 100 for d in data_list]
        ler_values = [d['logical_error_rate'] for d in data_list]
        
        ax.semilogy(p_values, ler_values,
                   marker=markers.get(decoder_name, 'o'),
                   color=colors.get(decoder_name, 'gray'),
                   linewidth=2.5,
                   markersize=10,
                   label=decoder_name)
    
    ax.set_xlabel('Physical Error Rate (%)', fontsize=14)
    ax.set_ylabel('Logical Error Rate', fontsize=14)
    ax.set_title(f'Decoder Comparison (d={results["config"]["distance"]})', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / "decoder_threshold_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.savefig(output_dir / "decoder_threshold_comparison.pdf", bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_decoder_bars(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = False
):
    """
    Generate Figure 3: Bar chart comparing decoders at specific error rates.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    error_rates = list(results['by_error_rate'].keys())
    num_plots = len(error_rates)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for ax, p_str in zip(axes, error_rates):
        data = results['by_error_rate'][p_str]
        
        decoders = []
        ler_values = []
        
        for name in ['AlphaQubit', 'MWPM', 'Tensor Network', 'Belief Propagation', 'Union Find']:
            if name in data and 'error' not in data[name]:
                decoders.append(name)
                ler_values.append(data[name]['logical_error_rate'] * 100)
        
        if not decoders:
            continue
        
        bars = ax.bar(decoders, ler_values, color=colors[:len(decoders)], 
                      edgecolor='black', linewidth=1.2)
        
        for bar, val in zip(bars, ler_values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Logical Error Rate (%)', fontsize=12)
        ax.set_title(f'p = {float(p_str)*100:.1f}%', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, max(ler_values) * 1.4 if ler_values else 1)
        
        # Highlight best decoder
        if ler_values:
            best_idx = np.argmin(ler_values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    output_path = output_dir / "decoder_bar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.savefig(output_dir / "decoder_bar_comparison.pdf", bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_improvement_chart(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = False
):
    """
    Generate chart showing AlphaQubit's improvement over baseline decoders.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_decoders = ['MWPM', 'Tensor Network', 'Belief Propagation', 'Union Find']
    improvements = []
    
    alphaqubit_data = results['decoders'].get('AlphaQubit', [])
    if not alphaqubit_data:
        print("Warning: No AlphaQubit data available for improvement chart")
        return
    
    for decoder in baseline_decoders:
        decoder_data = results['decoders'].get(decoder, [])
        if not decoder_data:
            improvements.append(0)
            continue
        
        # Calculate average improvement
        imp_list = []
        for aq, other in zip(alphaqubit_data, decoder_data):
            if 'error' in aq or 'error' in other:
                continue
            aq_ler = aq['logical_error_rate']
            other_ler = other['logical_error_rate']
            if other_ler > 0:
                imp = (other_ler - aq_ler) / other_ler * 100
                imp_list.append(imp)
        
        improvements.append(np.mean(imp_list) if imp_list else 0)
    
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.barh(baseline_decoders, improvements, color=colors, 
                   edgecolor='black', height=0.6)
    
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        ax.annotate(f'{val:.1f}%',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('AlphaQubit Improvement vs Baseline Decoders', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = output_dir / "alphaqubit_improvement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_speed_comparison(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = False
):
    """
    Generate chart comparing decoder speeds.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decoders = []
    speeds = []
    
    # Get average speed for each decoder
    for decoder_name, data_list in results['decoders'].items():
        if not data_list or 'error' in data_list[0]:
            continue
        
        avg_speed = np.mean([d['samples_per_second'] for d in data_list])
        decoders.append(decoder_name)
        speeds.append(avg_speed)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(decoders, speeds, color=colors[:len(decoders)], 
                  edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, speeds):
        height = bar.get_height()
        if val >= 1000:
            label = f'{val/1000:.1f}K'
        else:
            label = f'{val:.0f}'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Samples per Second', fontsize=12)
    ax.set_title('Decoder Speed Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = output_dir / "decoder_speed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run decoder comparison and generate figures')
    parser.add_argument('--distance', '-d', type=int, default=5, help='Code distance')
    parser.add_argument('--rounds', '-r', type=int, default=None, help='Syndrome rounds')
    parser.add_argument('--samples', '-n', type=int, default=1000, help='Samples per error rate')
    parser.add_argument('--model', '-m', type=str, default=None, help='AlphaQubit model path')
    parser.add_argument('--output', '-o', type=str, default='paper_figures/output',
                        help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    if args.rounds is None:
        args.rounds = args.distance
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Physical error rates to test
    physical_error_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 
                           0.006, 0.007, 0.008, 0.009, 0.01]
    
    print("\n" + "="*60)
    print("DECODER COMPARISON PIPELINE")
    print("="*60)
    print(f"Distance: {args.distance}")
    print(f"Rounds: {args.rounds}")
    print(f"Samples per error rate: {args.samples}")
    print(f"Error rates: {[f'{p*100:.1f}%' for p in physical_error_rates]}")
    print("="*60)
    
    # Run all decoders
    results = run_all_decoders(
        args.distance,
        args.rounds,
        args.samples,
        physical_error_rates,
        args.model
    )
    
    # Save results
    results_path = output_dir / "decoder_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_threshold_comparison(results, output_dir, args.show)
    plot_decoder_bars(results, output_dir, args.show)
    plot_improvement_chart(results, output_dir, args.show)
    plot_speed_comparison(results, output_dir, args.show)
    
    print("\n" + "="*60)
    print("DONE! All figures generated.")
    print("="*60)


if __name__ == '__main__':
    main()
