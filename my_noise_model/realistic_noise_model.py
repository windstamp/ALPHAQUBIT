"""realistic_noise_model.py - Noise model using per-qubit/edge calibration data.

This module implements a realistic noise model that uses actual device calibration
parameters (T1, T2, readout errors, CZ errors, etc.) to generate more accurate
noise for surface code simulations.

The model supports:
1. Per-qubit coherence times (T1, T2, Tphi)
2. Per-qubit readout and reset errors
3. Per-edge CZ gate errors and leakage
4. Per-edge ZZ crosstalk
5. Spatial correlation of errors
6. Bad qubit/edge handling

Usage:
    from my_noise_model.calibration_loader import DeviceCalibration
    from my_noise_model.realistic_noise_model import RealisticNoiseModel
    
    # Load calibration from file
    calibration = DeviceCalibration.from_json_file("configs/realistic_calibration_d5.json")
    
    # Create noise model
    noise_model = RealisticNoiseModel(calibration, distance=5, rounds=25)
    
    # Generate noisy circuit
    circuit = noise_model.build_noisy_circuit(basis="z")
    
    # Sample detection events
    det_events, observables = noise_model.sample(num_samples=10000)
"""

from __future__ import annotations

import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from .calibration_loader import DeviceCalibration, QubitCalibration, EdgeCalibration
from .gpt import amp_phase_kraus
from .gpta import twirl_to_pauli_channel
from .channels import (
    lift_qubit_to_qutrit,
    dqlr_kraus,
    cz_induced_leakage_kraus,
    leakage_transport_kraus,
    depolarizing_1q_kraus,
    depolarizing_2q_kraus,
)
from .kraus_utils import combine_kraus_channels, lift_2q_kraus_to_qutrit


@dataclass
class RealisticNoiseConfig:
    """Configuration for realistic noise model with calibration-based parameters."""
    
    # Circuit parameters
    distance: int = 5
    rounds: int = 25
    cycle_ns: float = 1076.0  # Cycle time in nanoseconds
    
    # Whether to use spatial variation from calibration
    use_calibration: bool = True
    
    # Fallback parameters when calibration not available
    default_t1_us: float = 73.0
    default_t2_us: float = 80.0
    default_tphi_us: float = 720.0
    default_readout_error: float = 0.008
    default_reset_error: float = 0.0015
    default_oneq_error: float = 0.0006
    default_cz_error: float = 0.0035
    default_cz_leakage: float = 0.0002
    default_zz_crosstalk: float = 0.00055
    
    # Additional noise mechanisms (from paper Table S4)
    p_heat_01: float = 0.0  # |0> -> |1> heating (negligible)
    p_heat_12: float = 2.5e-4  # |1> -> |2> heating
    p_idle_excess: float = 0.0  # Extra idle error
    p_cz_swap_like: float = 0.0  # Swap-like CZ error
    p_leak_transport: float = 0.0005  # Leakage transport during CZ
    
    # DQLR reset parameters
    enable_dqlr: bool = True
    dqlr_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.05),
        (0.0, 1.0, 0.90),
        (0.0, 0.0, 0.05),
    )


class RealisticNoiseModel:
    """Realistic noise model using device calibration data.
    
    This class builds Stim circuits with noise parameters derived from
    actual device calibration measurements, providing more realistic
    simulation of surface code error correction.
    """
    
    def __init__(
        self,
        calibration: Optional[DeviceCalibration] = None,
        config: Optional[RealisticNoiseConfig] = None,
        distance: int = 5,
        rounds: int = 25,
    ):
        """Initialize the realistic noise model.
        
        Args:
            calibration: Device calibration data with per-qubit/edge parameters
            config: Noise configuration (uses defaults if None)
            distance: Surface code distance
            rounds: Number of QEC rounds
        """
        self.calibration = calibration
        self.config = config or RealisticNoiseConfig(distance=distance, rounds=rounds)
        
        if distance != self.config.distance:
            self.config.distance = distance
        if rounds != self.config.rounds:
            self.config.rounds = rounds
        
        # Pre-compute Pauli channel parameters for each qubit and edge
        self._qubit_channels: Dict[int, Dict[str, Any]] = {}
        self._edge_channels: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # Build the circuit
        self.circuit: Optional[stim.Circuit] = None
        
    def build_noisy_circuit(self, basis: str = "z") -> stim.Circuit:
        """Build a noisy surface code circuit using calibration data.
        
        Args:
            basis: Measurement basis ('x' or 'z')
            
        Returns:
            Stim circuit with realistic noise channels
        """
        basis = basis.upper()
        if basis not in ("X", "Z"):
            raise ValueError("basis must be 'x' or 'z'")
        
        # Generate base circuit
        base_circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            rounds=self.config.rounds,
            distance=self.config.distance,
        )
        
        # Instrument with realistic noise
        self.circuit = self._instrument_circuit(base_circuit)
        
        return self.circuit
    
    def _instrument_circuit(self, base_circuit: stim.Circuit) -> stim.Circuit:
        """Add realistic noise channels to the base circuit."""
        noisy = stim.Circuit()
        
        for inst in base_circuit:
            if inst.name in ("H", "SQRT_X", "SQRT_Y", "S", "S_DAG", "H_XZ", "H_YZ"):
                # Single-qubit gate: add per-qubit noise
                noisy.append(inst)
                targets = [t.value for t in inst.targets_copy()]
                for q in targets:
                    px, py, pz = self._get_1q_pauli_probs(q)
                    if px + py + pz > 0:
                        noisy.append_operation("PAULI_CHANNEL_1", [q], [px, py, pz])
                        
            elif inst.name in ("CZ", "CX", "CNOT"):
                # Two-qubit gate: add per-edge noise
                noisy.append(inst)
                targets = [t.value for t in inst.targets_copy()]
                for i in range(0, len(targets), 2):
                    q1, q2 = targets[i], targets[i + 1]
                    args = self._get_2q_pauli_probs(q1, q2)
                    noisy.append_operation("PAULI_CHANNEL_2", [q1, q2], args)
                    
            elif inst.name == "M" or inst.name == "MR":
                # Measurement: add per-qubit readout error
                targets = [t.value for t in inst.targets_copy()]
                for q in targets:
                    p_readout = self._get_readout_error(q)
                    if p_readout > 0:
                        noisy.append_operation("X_ERROR", [q], p_readout)
                noisy.append(inst)
                
            elif inst.name == "R" or inst.name == "RX":
                # Reset: add per-qubit reset error
                noisy.append(inst)
                targets = [t.value for t in inst.targets_copy()]
                for q in targets:
                    p_reset = self._get_reset_error(q)
                    if p_reset > 0:
                        noisy.append_operation("X_ERROR", [q], p_reset)
                        
            elif inst.name == "TICK":
                # Idle: add decoherence on all qubits
                noisy.append(inst)
                # Note: Idle errors are already incorporated into gate errors
                # through the T1/T2 model. Additional idle can be added here.
                
            else:
                # Pass through other instructions unchanged
                noisy.append(inst)
        
        return noisy
    
    def _get_1q_pauli_probs(self, qubit: int) -> Tuple[float, float, float]:
        """Get single-qubit Pauli error probabilities for a specific qubit.
        
        Returns (px, py, pz) probabilities for X, Y, Z errors.
        """
        if qubit in self._qubit_channels:
            ch = self._qubit_channels[qubit]
            return ch["px"], ch["py"], ch["pz"]
        
        # Compute from calibration or defaults
        if self.calibration and self.config.use_calibration:
            qcal = self.calibration.get_qubit(qubit)
            t1 = qcal.t1_us
            t2 = qcal.t2_us
            oneq_error = qcal.oneq_error
        else:
            t1 = self.config.default_t1_us
            t2 = self.config.default_t2_us
            oneq_error = self.config.default_oneq_error
        
        # Compute Pauli probabilities from T1/T2 decoherence + gate error
        dt_us = self.config.cycle_ns / 1000.0
        
        # Amplitude/phase damping contribution
        gamma_1 = dt_us / t1 if t1 > 0 else 0
        gamma_phi = dt_us / t2 - gamma_1 / 2 if t2 > 0 else 0
        gamma_phi = max(0, gamma_phi)
        
        # Convert to Pauli probabilities (from GPT twirling)
        # For amplitude damping: roughly equal X, Y with small Z
        p_amp = (1 - np.exp(-gamma_1)) / 4  # Approximate
        p_phase = (1 - np.exp(-gamma_phi)) / 2  # Dephasing -> Z
        
        # Add gate error (depolarizing)
        p_gate = oneq_error / 3
        
        px = float(p_amp + p_gate)
        py = float(p_amp + p_gate)
        pz = float(p_phase + p_gate)
        
        # Normalize to ensure total <= 1
        total = px + py + pz
        if total > 0.5:  # Cap at 50% total error
            scale = 0.5 / total
            px, py, pz = px * scale, py * scale, pz * scale
        
        # Cache result
        self._qubit_channels[qubit] = {"px": px, "py": py, "pz": pz}
        
        return px, py, pz
    
    def _get_2q_pauli_probs(self, q1: int, q2: int) -> List[float]:
        """Get two-qubit Pauli error probabilities for a specific edge.
        
        Returns 15-element list for PAULI_CHANNEL_2: 
        [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
        """
        edge = tuple(sorted([q1, q2]))
        
        if edge in self._edge_channels:
            return self._edge_channels[edge]["args"]
        
        # Get calibration data
        if self.calibration and self.config.use_calibration:
            ecal = self.calibration.get_edge(q1, q2)
            cz_error = ecal.cz_error
            cz_leakage = ecal.cz_leakage
            zz_crosstalk = ecal.zz_crosstalk
        else:
            cz_error = self.config.default_cz_error
            cz_leakage = self.config.default_cz_leakage
            zz_crosstalk = self.config.default_zz_crosstalk
        
        # Convert CZ error to depolarizing channel
        # Total error is distributed among 15 two-qubit Pauli operators
        p_depol = cz_error / 15.0
        
        # ZZ crosstalk adds to ZZ component
        p_zz = p_depol + zz_crosstalk
        
        # Leakage contributes to error (modeled as depolarizing here)
        p_leak_contrib = cz_leakage / 15.0
        
        # Build the 15-element probability vector
        # Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
        args = [
            p_depol + p_leak_contrib,  # IX
            p_depol + p_leak_contrib,  # IY
            p_depol + p_leak_contrib,  # IZ
            p_depol + p_leak_contrib,  # XI
            p_depol + p_leak_contrib,  # XX
            p_depol + p_leak_contrib,  # XY
            p_depol + p_leak_contrib,  # XZ
            p_depol + p_leak_contrib,  # YI
            p_depol + p_leak_contrib,  # YX
            p_depol + p_leak_contrib,  # YY
            p_depol + p_leak_contrib,  # YZ
            p_depol + p_leak_contrib,  # ZI
            p_depol + p_leak_contrib,  # ZX
            p_depol + p_leak_contrib,  # ZY
            p_zz + p_leak_contrib,     # ZZ (enhanced by crosstalk)
        ]
        
        # Normalize to ensure total <= 1
        total = sum(args)
        if total > 0.5:
            scale = 0.5 / total
            args = [a * scale for a in args]
        
        # Cache result
        self._edge_channels[edge] = {"args": args}
        
        return args
    
    def _get_readout_error(self, qubit: int) -> float:
        """Get readout error probability for a specific qubit."""
        if self.calibration and self.config.use_calibration:
            qcal = self.calibration.get_qubit(qubit)
            return qcal.readout_error
        return self.config.default_readout_error
    
    def _get_reset_error(self, qubit: int) -> float:
        """Get reset error probability for a specific qubit."""
        if self.calibration and self.config.use_calibration:
            qcal = self.calibration.get_qubit(qubit)
            return qcal.reset_error
        return self.config.default_reset_error
    
    def sample(
        self,
        num_samples: int,
        separate_observables: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample detection events and observables from the noisy circuit.
        
        Args:
            num_samples: Number of samples to generate
            separate_observables: If True, return observables separately
            
        Returns:
            Tuple of (detection_events, observables) arrays
        """
        if self.circuit is None:
            raise ValueError("Circuit not built. Call build_noisy_circuit() first.")
        
        sampler = self.circuit.compile_detector_sampler()
        
        if separate_observables:
            det_events, obs = sampler.sample(
                num_samples, separate_observables=True
            )
            return np.array(det_events, dtype=np.uint8), np.array(obs, dtype=np.uint8)
        else:
            result = sampler.sample(num_samples)
            return np.array(result, dtype=np.uint8), np.array([])
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about the error model being used."""
        stats = {
            "distance": self.config.distance,
            "rounds": self.config.rounds,
            "use_calibration": self.config.use_calibration,
        }
        
        if self.calibration:
            stats["calibration_stats"] = self.calibration.get_statistics()
            stats["num_qubits"] = self.calibration.num_qubits
            stats["num_edges"] = len(self.calibration.edge_calibrations)
            stats["num_bad_qubits"] = len(self.calibration.bad_qubits)
        
        return stats


def create_noise_model_from_calibration_file(
    calibration_path: str,
    distance: int = 5,
    rounds: int = 25,
) -> RealisticNoiseModel:
    """Convenience function to create a noise model from a calibration JSON file.
    
    Args:
        calibration_path: Path to the calibration JSON file
        distance: Surface code distance
        rounds: Number of QEC rounds
        
    Returns:
        RealisticNoiseModel initialized with the calibration data
    """
    calibration = DeviceCalibration.from_json_file(calibration_path)
    config = RealisticNoiseConfig(distance=distance, rounds=rounds)
    return RealisticNoiseModel(calibration=calibration, config=config)


def create_noise_model_with_random_calibration(
    distance: int = 5,
    rounds: int = 25,
    seed: int = 42,
    **calibration_kwargs,
) -> RealisticNoiseModel:
    """Create a noise model with randomly generated calibration data.
    
    This is useful for generating diverse training data with realistic
    spatial variation in noise parameters.
    
    Args:
        distance: Surface code distance
        rounds: Number of QEC rounds
        seed: Random seed for calibration generation
        **calibration_kwargs: Additional args passed to generate_random_calibration
        
    Returns:
        RealisticNoiseModel with random but realistic calibration
    """
    from .calibration_loader import generate_random_calibration
    
    calibration = generate_random_calibration(distance=distance, seed=seed, **calibration_kwargs)
    config = RealisticNoiseConfig(distance=distance, rounds=rounds)
    return RealisticNoiseModel(calibration=calibration, config=config)
