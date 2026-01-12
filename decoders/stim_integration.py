"""
Stim Integration Module for Decoders
====================================

This module provides unified stim-based syndrome generation and
ground truth labels for accurate decoder benchmarking.

All decoders use the same generated data for fair comparison.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False


class StimSyndromeGenerator:
    """
    Generates syndrome data from stim circuits for decoder benchmarking.
    
    This ensures all decoders are tested on identical data with
    proper ground truth labels.
    """
    
    def __init__(
        self,
        distance: int,
        rounds: Optional[int] = None,
        physical_error_rate: float = 0.01,
        basis: str = 'z',
        noise_model: str = 'si1000'
    ):
        """
        Initialize syndrome generator.
        
        Args:
            distance: Code distance d
            rounds: Number of syndrome measurement rounds (default: d)
            physical_error_rate: Physical error probability
            basis: Measurement basis ('x' or 'z')
            noise_model: Noise model type ('si1000', 'depolarizing', 'phenomenological')
        """
        if not HAS_STIM:
            raise ImportError("stim is required. Install with: pip install stim")
        
        self.distance = distance
        self.rounds = rounds if rounds is not None else distance
        self.physical_error_rate = physical_error_rate
        self.basis = basis.lower()
        self.noise_model = noise_model
        
        # Build circuit
        self.circuit = self._build_circuit()
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.sampler = self.circuit.compile_detector_sampler()
        
        # Get number of detectors and observables
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
    def _build_circuit(self) -> 'stim.Circuit':
        """Build stim circuit with specified noise model."""
        d = self.distance
        r = self.rounds
        p = self.physical_error_rate
        
        if self.noise_model == 'si1000':
            # SI1000: Standard circuit-level noise
            circuit = stim.Circuit.generated(
                f"surface_code:rotated_memory_{self.basis}",
                distance=d,
                rounds=r,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p / 10,
                before_measure_flip_probability=5 * p,
                after_reset_flip_probability=2 * p,
            )
        elif self.noise_model == 'depolarizing':
            # Simple depolarizing noise
            circuit = stim.Circuit.generated(
                f"surface_code:rotated_memory_{self.basis}",
                distance=d,
                rounds=r,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
        elif self.noise_model == 'phenomenological':
            # Phenomenological: measurement errors only
            circuit = stim.Circuit.generated(
                f"surface_code:rotated_memory_{self.basis}",
                distance=d,
                rounds=r,
                before_measure_flip_probability=p,
            )
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")
        
        return circuit
    
    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate syndrome samples with ground truth labels.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            (syndromes, labels) tuple where:
            - syndromes: shape (N, num_detectors) detection events
            - labels: shape (N,) logical error indicators
        """
        detection_events, observable_flips = self.sampler.sample(
            num_samples, separate_observables=True
        )
        
        syndromes = detection_events.astype(np.float32)
        labels = observable_flips.flatten().astype(np.int32)
        
        return syndromes, labels
    
    def sample_reshaped(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate syndrome samples reshaped to (N, R, S) format.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            (syndromes, labels) tuple where:
            - syndromes: shape (N, R, S) for R rounds, S stabilizers
            - labels: shape (N,) logical error indicators
        """
        syndromes_flat, labels = self.sample(num_samples)
        
        # Reshape to (N, R, S) assuming detectors are ordered by round
        N = num_samples
        R = self.rounds
        total_det = syndromes_flat.shape[1]
        S = total_det // R  # Stabilizers per round
        
        # Handle case where total detectors don't divide evenly
        if total_det % R != 0:
            # Pad or truncate to fit
            S = (total_det + R - 1) // R
            padded = np.zeros((N, R * S), dtype=np.float32)
            padded[:, :total_det] = syndromes_flat
            syndromes_flat = padded
        
        syndromes = syndromes_flat.reshape(N, R, S)
        
        return syndromes, labels
    
    def get_info(self) -> Dict[str, Any]:
        """Return generator information."""
        return {
            'distance': self.distance,
            'rounds': self.rounds,
            'physical_error_rate': self.physical_error_rate,
            'basis': self.basis,
            'noise_model': self.noise_model,
            'num_detectors': self.num_detectors,
            'num_observables': self.num_observables,
        }


def generate_benchmark_data(
    distance: int,
    rounds: Optional[int] = None,
    physical_error_rate: float = 0.01,
    num_samples: int = 10000,
    basis: str = 'z',
    noise_model: str = 'si1000'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to generate benchmark data.
    
    Returns:
        (syndromes, labels, info) tuple
    """
    generator = StimSyndromeGenerator(
        distance=distance,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
        basis=basis,
        noise_model=noise_model
    )
    
    syndromes, labels = generator.sample_reshaped(num_samples)
    info = generator.get_info()
    
    return syndromes, labels, info
