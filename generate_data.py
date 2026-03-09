from datetime import datetime
import argparse
import os
from pathlib import Path
import yaml
import numpy as np
from simulator.dem_generator import generate_dem_data
from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator
from simulator.experiment_loader import (
    ExperimentLoader,
    discover_experiments,
)


def generate_from_experiment(experiment_path, num_samples, basis, with_soft_info=False):
    """Generate data from experiment .dem/.stim files."""
    loader = ExperimentLoader(experiment_path, verbose=True)
    
    print(f"Loading experiment: {loader.metadata.name}")
    print(f"  Distance: {loader.metadata.distance}")
    print(f"  Rounds: {loader.metadata.rounds}")
    print(f"  Has DEM: {loader.metadata.has_dem}")
    print(f"  Has STIM: {loader.metadata.has_stim}")
    
    if with_soft_info:
        data, logicals = loader.sample_with_soft_info(num_samples)
        return data, logicals
    else:
        sampled = loader.sample(num_samples, reshape_to_rounds=True)
        return sampled.detections, sampled.observables


def main(model_type: str, num_samples: int, basis: str, experiment_path: str = None, soft: bool = False):
    # If experiment_path is provided, use ExperimentLoader to generate data from .dem/.stim files
    if experiment_path is not None:
        print(f"Generating data from experiment: {experiment_path}")
        syndromes, logicals = generate_from_experiment(
            experiment_path, num_samples, basis, with_soft_info=soft
        )
        
        # Create output directory and save
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exp_name = Path(experiment_path).name
        if soft:
            # Save as .npz with 3-channel data
            output_path = os.path.join(output_dir, f"experiment_{exp_name}_{basis}_{timestamp}.npz")
            # Parse metadata from experiment path for distance/rounds
            loader = ExperimentLoader(experiment_path, verbose=False)
            np.savez(output_path, 
                     data=syndromes, 
                     observables=logicals,  # Use 'observables' key for compatibility
                     basis=basis,
                     distance=loader.metadata.distance,
                     rounds=loader.metadata.rounds)
            print(f"Successfully generated {num_samples} samples with soft info")
            print(f"Data saved to: {output_path}")
        else:
            syndrome_path = os.path.join(output_dir, f"experiment_{exp_name}_syndromes_{basis}_{timestamp}.npy")
            logicals_path = os.path.join(output_dir, f"experiment_{exp_name}_logicals_{basis}_{timestamp}.npy")
            np.save(syndrome_path, syndromes)
            np.save(logicals_path, logicals)
            print(f"Successfully generated {num_samples} samples")
            print(f"Syndromes saved to: {syndrome_path}")
            print(f"Logical errors saved to: {logicals_path}")
        return

    # Load configuration
    # For paper-aligned, we explicitly load its documented config
    cfg_name = f"{model_type}.yaml" if model_type != "paper_aligned" else "paper_aligned.yaml"
    config_path = os.path.join("configs", cfg_name)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Generate data based on model type
    if model_type == "dem":
        syndromes, logicals = generate_dem_data(num_samples, config)
    elif model_type == "si1000":
        # ``si1000_noise_model`` expects the full configuration dictionary so that
        # parameters like distance and rounds can be overridden.  Passing only the
        # error rate (``config['p']``) previously caused an ``AttributeError`` when
        # the function attempted to access ``config.get``.  Forward the entire
        # config object instead.
        circuit = si1000_noise_model(config)
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
        import sys
        print(f"{__file__}:{sys._getframe().f_lineno}")
        print(f'syndromes.shape: {syndromes.shape}')
        print(f'logicals.shape: {logicals.shape}')
    elif model_type == "pauli_plus":
        sim = PauliPlusSimulator(config, basis)
        sampler = sim.circuit.compile_detector_sampler()  # or however your class exposes it
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    elif model_type == "paper_aligned":
        # The PauliPlus simulator in paper-aligned mode maps config 1:1 to the paper’s physical noise
        sim = PauliPlusSimulator(config, basis, mode="paper_aligned")
        sampler = sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save data as numpy arrays
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    syndrome_path = os.path.join(output_dir, f"{model_type}_syndromes_{basis}_{timestamp}.npy")
    logicals_path = os.path.join(output_dir, f"{model_type}_logicals_{basis}_{timestamp}.npy")
    np.save(syndrome_path, syndromes)
    np.save(logicals_path, logicals)

    print(f"Successfully generated {num_samples} samples")
    print(f"Syndromes saved to: {syndrome_path}")
    print(f"Logical errors saved to: {logicals_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantum error correction data")
    parser.add_argument("--model", choices=["dem", "si1000", "pauli_plus", "paper_aligned", "experiment"], 
                        default="experiment",
                        help="Type of noise model to generate. Use 'experiment' to load from .dem/.stim files")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--basis", type=str, default='z', help="basis: x or z")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment folder containing .dem/.stim files")
    parser.add_argument("--soft", action="store_true",
                        help="Include soft readout channels (3-channel output)")
    args = parser.parse_args()
    
    # If experiment path is provided or model is "experiment", use ExperimentLoader
    if args.experiment or args.model == "experiment":
        if not args.experiment:
            parser.error("--experiment path is required when using 'experiment' model")
        main("experiment", args.samples, args.basis, experiment_path=args.experiment, soft=args.soft)
    else:
        main(args.model, args.samples, args.basis)

