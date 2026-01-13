#!/usr/bin/env python3
"""generate_realistic_noise_data.py - Example script for generating realistic noise data.

This script demonstrates how to use the calibration-based noise model to generate
training data with realistic, spatially-varying noise parameters.

Usage:
    # Generate data using existing calibration file
    python generate_realistic_noise_data.py --calibration configs/realistic_calibration_d5.json

    # Generate data with random calibration
    python generate_realistic_noise_data.py --distance 5 --rounds 25 --seed 42

    # Generate multi-distance pretraining dataset
    python generate_realistic_noise_data.py --mode pretrain --output-dir pretrain_data/realistic

    # Generate finetuning dataset
    python generate_realistic_noise_data.py --mode finetune --calibration configs/realistic_calibration_d5.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from my_noise_model.calibration_loader import (
    DeviceCalibration,
    generate_random_calibration,
)
from my_noise_model.realistic_noise_model import (
    RealisticNoiseModel,
    RealisticNoiseConfig,
    create_noise_model_from_calibration_file,
    create_noise_model_with_random_calibration,
)
from my_noise_model.realistic_data_generator import (
    RealisticDataGenerator,
    GeneratedDataset,
    generate_multi_distance_dataset,
    generate_finetuning_dataset,
    perturb_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_calibration_loading():
    """Demonstrate loading and inspecting calibration data."""
    print("\n" + "="*60)
    print("DEMO 1: Loading Calibration Data")
    print("="*60)
    
    calibration_path = "configs/realistic_calibration_d5.json"
    
    try:
        calibration = DeviceCalibration.from_json_file(calibration_path)
        
        print(f"\nLoaded calibration for distance {calibration.distance}")
        print(f"Number of qubits: {calibration.num_qubits}")
        print(f"Number of edges: {len(calibration.edge_calibrations)}")
        print(f"Bad qubits: {calibration.bad_qubits}")
        
        # Show statistics
        stats = calibration.get_statistics()
        print(f"\nCalibration Statistics:")
        print(f"  T1 median: {stats.get('t1_median', 'N/A'):.2f} µs")
        print(f"  T1 range: {stats.get('t1_range', ['N/A', 'N/A'])}")
        print(f"  T2 median: {stats.get('t2_median', 'N/A'):.2f} µs")
        print(f"  Readout error median: {stats.get('readout_median', 'N/A'):.4f}")
        print(f"  CZ error median: {stats.get('cz_error_median', 'N/A'):.4f}")
        print(f"  CZ error range: {stats.get('cz_error_range', ['N/A', 'N/A'])}")
        
        # Show example qubit
        q0 = calibration.get_qubit(0)
        print(f"\nExample qubit 0:")
        print(f"  T1: {q0.t1_us:.2f} µs")
        print(f"  T2: {q0.t2_us:.2f} µs")
        print(f"  Readout error: {q0.readout_error:.4f}")
        print(f"  1Q error: {q0.oneq_error:.4f}")
        
        # Show example edge
        e01 = calibration.get_edge(0, 1)
        print(f"\nExample edge (0, 1):")
        print(f"  CZ error: {e01.cz_error:.4f}")
        print(f"  CZ leakage: {e01.cz_leakage:.4f}")
        print(f"  ZZ crosstalk: {e01.zz_crosstalk:.4f}")
        
        return calibration
        
    except FileNotFoundError:
        print(f"Calibration file not found: {calibration_path}")
        print("Generating random calibration instead...")
        calibration = generate_random_calibration(distance=5, seed=42)
        return calibration


def demonstrate_random_calibration():
    """Demonstrate generating random calibration data."""
    print("\n" + "="*60)
    print("DEMO 2: Generating Random Calibration")
    print("="*60)
    
    calibration = generate_random_calibration(
        distance=5,
        seed=42,
        t1_median=73.0,
        t1_std=15.0,
        cz_error_median=0.0035,
        cz_error_std=0.0012,
        bad_qubit_fraction=0.04,
    )
    
    stats = calibration.get_statistics()
    print(f"\nGenerated random calibration:")
    print(f"  Qubits: {calibration.num_qubits}")
    print(f"  Edges: {len(calibration.edge_calibrations)}")
    print(f"  Bad qubits: {calibration.bad_qubits}")
    print(f"  T1 median: {stats.get('t1_median', 0):.2f} µs")
    print(f"  CZ error median: {stats.get('cz_error_median', 0):.4f}")
    
    return calibration


def demonstrate_noise_model(calibration: DeviceCalibration):
    """Demonstrate building a noisy circuit with calibration."""
    print("\n" + "="*60)
    print("DEMO 3: Building Noisy Circuit")
    print("="*60)
    
    noise_model = RealisticNoiseModel(
        calibration=calibration,
        distance=5,
        rounds=10,  # Shorter for demo
    )
    
    circuit = noise_model.build_noisy_circuit(basis="z")
    
    print(f"\nBuilt noisy circuit:")
    print(f"  Distance: {noise_model.config.distance}")
    print(f"  Rounds: {noise_model.config.rounds}")
    print(f"  Circuit instructions: {len(list(circuit))}")
    
    # Show error statistics
    stats = noise_model.get_error_statistics()
    print(f"\nError model statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    return noise_model


def demonstrate_sampling(noise_model: RealisticNoiseModel):
    """Demonstrate sampling from the noisy circuit."""
    print("\n" + "="*60)
    print("DEMO 4: Sampling Detection Events")
    print("="*60)
    
    num_samples = 1000
    det_events, observables = noise_model.sample(num_samples)
    
    print(f"\nSampled {num_samples} shots:")
    print(f"  Detection events shape: {det_events.shape}")
    print(f"  Observables shape: {observables.shape}")
    
    # Calculate error rate
    error_rate = observables.mean()
    print(f"  Logical error rate: {error_rate:.4f}")
    
    # Syndrome statistics
    syndromes_per_shot = det_events.sum(axis=1)
    print(f"  Avg syndromes per shot: {syndromes_per_shot.mean():.2f}")
    print(f"  Syndrome std: {syndromes_per_shot.std():.2f}")
    
    return det_events, observables


def demonstrate_data_generation():
    """Demonstrate generating a training dataset."""
    print("\n" + "="*60)
    print("DEMO 5: Generating Training Dataset")
    print("="*60)
    
    # Create generator
    generator = RealisticDataGenerator.with_random_calibration(
        distance=3,  # Small for demo
        rounds=15,
        seed=42,
    )
    
    # Generate samples
    dataset = generator.generate_samples(
        num_samples=5000,
        basis="z",
    )
    
    print(f"\nGenerated dataset:")
    print(f"  Samples: {dataset.num_samples}")
    print(f"  Distance: {dataset.distance}")
    print(f"  Rounds: {dataset.rounds}")
    print(f"  Detection events shape: {dataset.detection_events.shape}")
    print(f"  Observables shape: {dataset.observables.shape}")
    
    # Calculate statistics
    error_rate = dataset.observables.mean()
    print(f"  Logical error rate: {error_rate:.4f}")
    
    # Save to file
    output_path = "test_realistic_dataset.npz"
    generator.save_to_npz(output_path, dataset)
    print(f"\nSaved to: {output_path}")
    
    # Verify loading
    loaded = GeneratedDataset.from_npz(output_path)
    print(f"Verified loading: {loaded.num_samples} samples")
    
    # Clean up
    Path(output_path).unlink()
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic noise training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "single", "pretrain", "finetune"],
        default="demo",
        help="Generation mode (default: demo)"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pretrain_data/realistic",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (for single/finetune modes)"
    )
    parser.add_argument(
        "--distance", "-d",
        type=int,
        default=5,
        help="Surface code distance (default: 5)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=None,
        help="Number of QEC rounds (default: 5*distance)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100000,
        help="Number of samples (default: 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--basis",
        choices=["x", "z"],
        default="z",
        help="Measurement basis (default: z)"
    )
    
    args = parser.parse_args()
    
    if args.rounds is None:
        args.rounds = 5 * args.distance
    
    if args.mode == "demo":
        # Run demonstration
        print("="*60)
        print("REALISTIC NOISE MODEL DEMONSTRATION")
        print("="*60)
        
        calibration = demonstrate_calibration_loading()
        calibration_random = demonstrate_random_calibration()
        noise_model = demonstrate_noise_model(calibration)
        det_events, obs = demonstrate_sampling(noise_model)
        dataset = demonstrate_data_generation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        
    elif args.mode == "single":
        # Generate single dataset
        output_path = args.output or f"realistic_d{args.distance}_r{args.rounds}.npz"
        
        if args.calibration:
            generator = RealisticDataGenerator.from_calibration_file(
                args.calibration,
                distance=args.distance,
                rounds=args.rounds,
            )
        else:
            generator = RealisticDataGenerator.with_random_calibration(
                distance=args.distance,
                rounds=args.rounds,
                seed=args.seed,
            )
        
        dataset = generator.generate_samples(
            num_samples=args.samples,
            basis=args.basis,
        )
        
        generator.save_to_npz(output_path, dataset)
        logger.info(f"Generated {args.samples} samples to {output_path}")
        
    elif args.mode == "pretrain":
        # Generate multi-distance pretraining dataset
        distances = [3, 5, 7]
        
        files = generate_multi_distance_dataset(
            output_dir=args.output_dir,
            distances=distances,
            samples_per_config=args.samples,
            calibration_path=args.calibration,
            num_calibration_seeds=5,
            basis=args.basis,
        )
        
        logger.info(f"Generated {len(files)} datasets")
        
    elif args.mode == "finetune":
        # Generate finetuning dataset
        if not args.calibration:
            logger.error("Finetuning mode requires --calibration argument")
            sys.exit(1)
        
        output_path = args.output or f"finetune_d{args.distance}_r{args.rounds}.npz"
        
        generate_finetuning_dataset(
            calibration_path=args.calibration,
            output_path=output_path,
            distance=args.distance,
            rounds=args.rounds,
            num_samples=args.samples,
            basis=args.basis,
        )
        
        logger.info(f"Generated finetuning dataset: {output_path}")


if __name__ == "__main__":
    main()
