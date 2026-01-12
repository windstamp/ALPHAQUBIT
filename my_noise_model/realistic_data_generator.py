"""realistic_data_generator.py - Generate training data with realistic noise.

This module provides utilities for generating large-scale training datasets
using calibration-based realistic noise models. It supports:

1. Loading calibration data from experiment folders
2. Generating diverse noise scenarios by varying calibration parameters
3. Creating pretraining and finetuning datasets
4. Multi-distance and multi-round data generation

The generated data follows the format expected by the AlphaQubit training pipeline:
- Detection events (syndrome measurements)
- Observable flips (logical error indicators)
- Metadata (noise parameters, distance, rounds, etc.)

Usage:
    from my_noise_model.realistic_data_generator import RealisticDataGenerator
    
    # Create generator from calibration file
    generator = RealisticDataGenerator.from_calibration_file(
        "configs/realistic_calibration_d5.json",
        distance=5,
        rounds=25
    )
    
    # Generate training samples
    data = generator.generate_samples(num_samples=100000)
    
    # Save to file
    generator.save_to_npz("pretrain_data/realistic_d5_r25.npz", data)
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor
import logging

from .calibration_loader import (
    DeviceCalibration,
    generate_random_calibration,
)
from .realistic_noise_model import (
    RealisticNoiseModel,
    RealisticNoiseConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedDataset:
    """Container for generated dataset with metadata."""
    
    detection_events: np.ndarray  # Shape: (num_samples, num_detectors)
    observables: np.ndarray  # Shape: (num_samples, num_observables)
    
    # Metadata
    distance: int
    rounds: int
    num_samples: int
    basis: str
    
    # Noise parameters used (for reference)
    noise_params: Dict[str, Any]
    
    # Optional: per-sample metadata
    sample_metadata: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        result = {
            "detection_events": self.detection_events,
            "observables": self.observables,
            "distance": self.distance,
            "rounds": self.rounds,
            "num_samples": self.num_samples,
            "basis": self.basis,
        }
        
        # Store noise params as JSON string
        result["noise_params_json"] = json.dumps(self.noise_params)
        
        if self.sample_metadata:
            for key, val in self.sample_metadata.items():
                result[f"meta_{key}"] = val
        
        return result
    
    @classmethod
    def from_npz(cls, path: str) -> "GeneratedDataset":
        """Load dataset from NPZ file."""
        data = np.load(path, allow_pickle=True)
        
        noise_params = {}
        if "noise_params_json" in data:
            noise_params = json.loads(str(data["noise_params_json"]))
        
        # Collect sample metadata
        sample_metadata = {}
        for key in data.files:
            if key.startswith("meta_"):
                sample_metadata[key[5:]] = data[key]
        
        return cls(
            detection_events=data["detection_events"],
            observables=data["observables"],
            distance=int(data["distance"]),
            rounds=int(data["rounds"]),
            num_samples=int(data["num_samples"]),
            basis=str(data.get("basis", "z")),
            noise_params=noise_params,
            sample_metadata=sample_metadata if sample_metadata else None,
        )


class RealisticDataGenerator:
    """Generator for creating training data with realistic noise.
    
    This class provides methods for generating large-scale datasets
    using calibration-based noise models with spatial variation.
    """
    
    def __init__(
        self,
        calibration: Optional[DeviceCalibration] = None,
        config: Optional[RealisticNoiseConfig] = None,
        distance: int = 5,
        rounds: int = 25,
    ):
        """Initialize the data generator.
        
        Args:
            calibration: Device calibration data
            config: Noise configuration
            distance: Surface code distance
            rounds: Number of QEC rounds
        """
        self.calibration = calibration
        self.config = config or RealisticNoiseConfig(distance=distance, rounds=rounds)
        self.distance = distance
        self.rounds = rounds
        
        # Create noise model
        self.noise_model = RealisticNoiseModel(
            calibration=calibration,
            config=self.config,
            distance=distance,
            rounds=rounds,
        )
    
    @classmethod
    def from_calibration_file(
        cls,
        calibration_path: str,
        distance: int = 5,
        rounds: int = 25,
    ) -> "RealisticDataGenerator":
        """Create generator from a calibration JSON file."""
        calibration = DeviceCalibration.from_json_file(calibration_path)
        return cls(calibration=calibration, distance=distance, rounds=rounds)
    
    @classmethod
    def with_random_calibration(
        cls,
        distance: int = 5,
        rounds: int = 25,
        seed: int = 42,
        **calibration_kwargs,
    ) -> "RealisticDataGenerator":
        """Create generator with randomly generated calibration."""
        calibration = generate_random_calibration(
            distance=distance, seed=seed, **calibration_kwargs
        )
        return cls(calibration=calibration, distance=distance, rounds=rounds)
    
    def generate_samples(
        self,
        num_samples: int,
        basis: str = "z",
        batch_size: int = 10000,
    ) -> GeneratedDataset:
        """Generate training samples with realistic noise.
        
        Args:
            num_samples: Number of samples to generate
            basis: Measurement basis ('x' or 'z')
            batch_size: Samples per batch (for memory efficiency)
            
        Returns:
            GeneratedDataset containing detection events and observables
        """
        logger.info(f"Generating {num_samples} samples (d={self.distance}, r={self.rounds})")
        
        # Build noisy circuit
        self.noise_model.build_noisy_circuit(basis=basis)
        
        # Generate in batches
        all_det_events = []
        all_observables = []
        
        remaining = num_samples
        while remaining > 0:
            batch = min(batch_size, remaining)
            det_events, obs = self.noise_model.sample(batch)
            all_det_events.append(det_events)
            all_observables.append(obs)
            remaining -= batch
            
            if remaining > 0:
                logger.debug(f"Generated {num_samples - remaining}/{num_samples} samples")
        
        # Concatenate batches
        detection_events = np.concatenate(all_det_events, axis=0)
        observables = np.concatenate(all_observables, axis=0)
        
        # Collect noise parameters
        noise_params = self.noise_model.get_error_statistics()
        
        return GeneratedDataset(
            detection_events=detection_events,
            observables=observables,
            distance=self.distance,
            rounds=self.rounds,
            num_samples=num_samples,
            basis=basis,
            noise_params=noise_params,
        )
    
    def save_to_npz(
        self,
        path: str,
        dataset: GeneratedDataset,
        compressed: bool = True,
    ) -> None:
        """Save dataset to NPZ file.
        
        Args:
            path: Output file path
            dataset: Dataset to save
            compressed: Whether to use compression
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = dataset.to_dict()
        
        if compressed:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
        
        logger.info(f"Saved {dataset.num_samples} samples to {path}")


def generate_multi_distance_dataset(
    output_dir: str,
    distances: List[int] = [3, 5, 7],
    rounds_per_distance: Optional[Dict[int, int]] = None,
    samples_per_config: int = 100000,
    calibration_path: Optional[str] = None,
    num_calibration_seeds: int = 5,
    basis: str = "z",
) -> Dict[str, str]:
    """Generate datasets for multiple distances with diverse noise.
    
    This function creates a comprehensive pretraining dataset by:
    1. Generating data for multiple code distances
    2. Using multiple random calibration seeds for diversity
    3. Optionally using a base calibration file
    
    Args:
        output_dir: Directory to save generated files
        distances: List of code distances to generate
        rounds_per_distance: Dict mapping distance to rounds (default: 5*d)
        samples_per_config: Samples per distance/seed combination
        calibration_path: Optional base calibration file
        num_calibration_seeds: Number of random seeds to use
        basis: Measurement basis
        
    Returns:
        Dictionary mapping config names to output file paths
    """
    if rounds_per_distance is None:
        rounds_per_distance = {d: 5 * d for d in distances}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    for distance in distances:
        rounds = rounds_per_distance.get(distance, 5 * distance)
        
        for seed in range(num_calibration_seeds):
            config_name = f"d{distance}_r{rounds}_seed{seed}"
            output_path = output_dir / f"realistic_{config_name}.npz"
            
            logger.info(f"Generating {config_name}...")
            
            # Create generator with random calibration
            if calibration_path and seed == 0:
                # First seed uses provided calibration
                generator = RealisticDataGenerator.from_calibration_file(
                    calibration_path,
                    distance=distance,
                    rounds=rounds,
                )
            else:
                # Other seeds use random calibration
                generator = RealisticDataGenerator.with_random_calibration(
                    distance=distance,
                    rounds=rounds,
                    seed=42 + seed * 1000 + distance * 100,
                )
            
            # Generate and save
            dataset = generator.generate_samples(
                num_samples=samples_per_config,
                basis=basis,
            )
            generator.save_to_npz(str(output_path), dataset)
            
            generated_files[config_name] = str(output_path)
    
    # Create manifest file
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "files": generated_files,
            "distances": distances,
            "rounds_per_distance": rounds_per_distance,
            "samples_per_config": samples_per_config,
            "num_calibration_seeds": num_calibration_seeds,
            "basis": basis,
        }, f, indent=2)
    
    logger.info(f"Generated {len(generated_files)} datasets. Manifest: {manifest_path}")
    
    return generated_files


def generate_finetuning_dataset(
    calibration_path: str,
    output_path: str,
    distance: int = 5,
    rounds: int = 25,
    num_samples: int = 50000,
    basis: str = "z",
) -> str:
    """Generate a finetuning dataset using specific calibration data.
    
    Finetuning datasets should use the exact calibration parameters
    from the target experiment to maximize transfer learning.
    
    Args:
        calibration_path: Path to experiment calibration file
        output_path: Output file path
        distance: Surface code distance
        rounds: Number of QEC rounds
        num_samples: Number of samples to generate
        basis: Measurement basis
        
    Returns:
        Path to the generated file
    """
    generator = RealisticDataGenerator.from_calibration_file(
        calibration_path,
        distance=distance,
        rounds=rounds,
    )
    
    dataset = generator.generate_samples(
        num_samples=num_samples,
        basis=basis,
    )
    
    generator.save_to_npz(output_path, dataset)
    
    return output_path


# =============================================================================
# Noise variation utilities for data augmentation
# =============================================================================

def perturb_calibration(
    calibration: DeviceCalibration,
    perturbation_scale: float = 0.1,
    seed: Optional[int] = None,
) -> DeviceCalibration:
    """Create a perturbed version of calibration data for augmentation.
    
    This adds random perturbations to noise parameters to simulate
    temporal drift and measurement uncertainty.
    
    Args:
        calibration: Base calibration data
        perturbation_scale: Fractional scale of perturbations (0.1 = 10%)
        seed: Random seed
        
    Returns:
        New DeviceCalibration with perturbed parameters
    """
    rng = np.random.default_rng(seed)
    
    # Deep copy calibration
    new_qubit_cals = {}
    for qid, qcal in calibration.qubit_calibrations.items():
        # Perturb each parameter
        scale = perturbation_scale
        new_qubit_cals[qid] = type(qcal)(
            qubit_id=qcal.qubit_id,
            row=qcal.row,
            col=qcal.col,
            t1_us=qcal.t1_us * (1 + rng.normal(0, scale)),
            t2_us=qcal.t2_us * (1 + rng.normal(0, scale)),
            tphi_us=qcal.tphi_us * (1 + rng.normal(0, scale)),
            readout_error=np.clip(qcal.readout_error * (1 + rng.normal(0, scale)), 0, 0.1),
            reset_error=np.clip(qcal.reset_error * (1 + rng.normal(0, scale)), 0, 0.05),
            oneq_error=np.clip(qcal.oneq_error * (1 + rng.normal(0, scale)), 0, 0.01),
            is_bad_qubit=qcal.is_bad_qubit,
        )
    
    new_edge_cals = {}
    for edge, ecal in calibration.edge_calibrations.items():
        scale = perturbation_scale
        new_edge_cals[edge] = type(ecal)(
            qubit_a=ecal.qubit_a,
            qubit_b=ecal.qubit_b,
            cz_error=np.clip(ecal.cz_error * (1 + rng.normal(0, scale)), 0.001, 0.02),
            cz_leakage=np.clip(ecal.cz_leakage * (1 + rng.normal(0, scale)), 0, 0.005),
            zz_crosstalk=np.clip(ecal.zz_crosstalk * (1 + rng.normal(0, scale)), 0, 0.005),
            xeb_fidelity=ecal.xeb_fidelity,
            is_bad_edge=ecal.is_bad_edge,
            has_freq_collision=ecal.has_freq_collision,
        )
    
    return DeviceCalibration(
        distance=calibration.distance,
        num_qubits=calibration.num_qubits,
        seed=seed,
        qubit_calibrations=new_qubit_cals,
        edge_calibrations=new_edge_cals,
        bad_qubits=calibration.bad_qubits.copy(),
        metadata=calibration.metadata.copy(),
    )


def interpolate_calibrations(
    cal1: DeviceCalibration,
    cal2: DeviceCalibration,
    alpha: float = 0.5,
) -> DeviceCalibration:
    """Interpolate between two calibration datasets.
    
    Useful for simulating gradual parameter drift over time.
    
    Args:
        cal1: First calibration (alpha=0)
        cal2: Second calibration (alpha=1)
        alpha: Interpolation factor in [0, 1]
        
    Returns:
        Interpolated calibration
    """
    alpha = np.clip(alpha, 0, 1)
    
    new_qubit_cals = {}
    for qid in cal1.qubit_calibrations:
        q1 = cal1.qubit_calibrations[qid]
        q2 = cal2.qubit_calibrations.get(qid, q1)
        
        new_qubit_cals[qid] = type(q1)(
            qubit_id=q1.qubit_id,
            row=q1.row,
            col=q1.col,
            t1_us=(1 - alpha) * q1.t1_us + alpha * q2.t1_us,
            t2_us=(1 - alpha) * q1.t2_us + alpha * q2.t2_us,
            tphi_us=(1 - alpha) * q1.tphi_us + alpha * q2.tphi_us,
            readout_error=(1 - alpha) * q1.readout_error + alpha * q2.readout_error,
            reset_error=(1 - alpha) * q1.reset_error + alpha * q2.reset_error,
            oneq_error=(1 - alpha) * q1.oneq_error + alpha * q2.oneq_error,
            is_bad_qubit=q1.is_bad_qubit or q2.is_bad_qubit,
        )
    
    new_edge_cals = {}
    for edge in cal1.edge_calibrations:
        e1 = cal1.edge_calibrations[edge]
        e2 = cal2.edge_calibrations.get(edge, e1)
        
        new_edge_cals[edge] = type(e1)(
            qubit_a=e1.qubit_a,
            qubit_b=e1.qubit_b,
            cz_error=(1 - alpha) * e1.cz_error + alpha * e2.cz_error,
            cz_leakage=(1 - alpha) * e1.cz_leakage + alpha * e2.cz_leakage,
            zz_crosstalk=(1 - alpha) * e1.zz_crosstalk + alpha * e2.zz_crosstalk,
            xeb_fidelity=(1 - alpha) * e1.xeb_fidelity + alpha * e2.xeb_fidelity,
            is_bad_edge=e1.is_bad_edge or e2.is_bad_edge,
            has_freq_collision=e1.has_freq_collision or e2.has_freq_collision,
        )
    
    return DeviceCalibration(
        distance=cal1.distance,
        num_qubits=cal1.num_qubits,
        qubit_calibrations=new_qubit_cals,
        edge_calibrations=new_edge_cals,
        bad_qubits=list(set(cal1.bad_qubits + cal2.bad_qubits)),
        metadata={"interpolated": True, "alpha": alpha},
    )
