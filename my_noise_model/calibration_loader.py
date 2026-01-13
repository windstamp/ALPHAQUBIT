"""calibration_loader.py - Load and parse physical calibration parameters.

This module provides utilities for loading qubit and edge calibration data
from JSON files that contain per-qubit T1, T2, readout errors, and per-edge
CZ error rates measured from real quantum processors.

The calibration data structure follows the format used in:
- configs/realistic_calibration_d5.json

Example JSON structure:
{
    "metadata": {...},
    "qubit_calibration": {
        "0": {"qubit_id": 0, "t1_us": 73.0, "t2_us": 80.0, ...},
        ...
    },
    "edge_calibration": {
        "(0, 1)": {"edge": [0, 1], "cz_error": 0.003, ...},
        ...
    }
}
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class QubitCalibration:
    """Per-qubit calibration parameters from experiment data."""
    
    qubit_id: int
    row: int = 0
    col: int = 0
    
    # Coherence times (microseconds)
    t1_us: float = 73.0  # Energy relaxation time
    t2_us: float = 80.0  # Dephasing time (T2 CPMG)
    tphi_us: float = 720.0  # Pure dephasing time
    
    # Error rates
    readout_error: float = 0.008  # Measurement error probability
    reset_error: float = 0.0015  # Reset error probability  
    oneq_error: float = 0.0006  # Single-qubit gate error
    
    # Flags
    is_bad_qubit: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QubitCalibration":
        """Create QubitCalibration from a dictionary."""
        return cls(
            qubit_id=int(data.get("qubit_id", 0)),
            row=int(data.get("row", 0)),
            col=int(data.get("col", 0)),
            t1_us=float(data.get("t1_us", 73.0)),
            t2_us=float(data.get("t2_us", 80.0)),
            tphi_us=float(data.get("tphi_us", 720.0)),
            readout_error=float(data.get("readout_error", 0.008)),
            reset_error=float(data.get("reset_error", 0.0015)),
            oneq_error=float(data.get("oneq_error", 0.0006)),
            is_bad_qubit=bool(data.get("is_bad_qubit", False)),
        )
    
    def compute_tphi_from_t1_t2(self) -> float:
        """Compute pure dephasing time from T1 and T2.
        
        Relation: 1/T2 = 1/(2*T1) + 1/T_phi
        Therefore: T_phi = 1 / (1/T2 - 1/(2*T1))
        """
        if self.t2_us <= 0 or self.t1_us <= 0:
            return self.tphi_us
        
        inv_t2 = 1.0 / self.t2_us
        inv_2t1 = 1.0 / (2.0 * self.t1_us)
        
        if inv_t2 <= inv_2t1:
            # T2 is longer than 2*T1, which is unphysical for simple model
            # Fall back to stored tphi
            return self.tphi_us
        
        return 1.0 / (inv_t2 - inv_2t1)


@dataclass  
class EdgeCalibration:
    """Per-edge (two-qubit gate) calibration parameters."""
    
    qubit_a: int
    qubit_b: int
    
    # CZ gate errors
    cz_error: float = 0.0035  # Total CZ gate error
    cz_leakage: float = 0.0002  # CZ-induced leakage probability
    zz_crosstalk: float = 0.00055  # ZZ crosstalk during CZ
    
    # Fidelity metrics
    xeb_fidelity: float = 0.9965  # Cross-entropy benchmarking fidelity
    
    # Flags
    is_bad_edge: bool = False
    has_freq_collision: bool = False
    
    @property
    def edge(self) -> Tuple[int, int]:
        """Return edge as sorted tuple."""
        return tuple(sorted([self.qubit_a, self.qubit_b]))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeCalibration":
        """Create EdgeCalibration from a dictionary."""
        edge = data.get("edge", [0, 1])
        return cls(
            qubit_a=int(edge[0] if isinstance(edge, list) else data.get("qubit_a", 0)),
            qubit_b=int(edge[1] if isinstance(edge, list) else data.get("qubit_b", 1)),
            cz_error=float(data.get("cz_error", 0.0035)),
            cz_leakage=float(data.get("cz_leakage", 0.0002)),
            zz_crosstalk=float(data.get("zz_crosstalk", 0.00055)),
            xeb_fidelity=float(data.get("xeb_fidelity", 0.9965)),
            is_bad_edge=bool(data.get("is_bad_edge", False)),
            has_freq_collision=bool(data.get("has_freq_collision", False)),
        )


@dataclass
class DeviceCalibration:
    """Full device calibration containing all qubit and edge parameters."""
    
    distance: int = 5
    num_qubits: int = 49
    seed: Optional[int] = None
    
    # Per-qubit calibration data
    qubit_calibrations: Dict[int, QubitCalibration] = field(default_factory=dict)
    
    # Per-edge calibration data  
    edge_calibrations: Dict[Tuple[int, int], EdgeCalibration] = field(default_factory=dict)
    
    # List of bad qubits to avoid
    bad_qubits: List[int] = field(default_factory=list)
    
    # Metadata statistics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json_file(cls, path: str) -> "DeviceCalibration":
        """Load device calibration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCalibration":
        """Create DeviceCalibration from a dictionary."""
        metadata = data.get("metadata", {})
        
        # Parse qubit calibrations
        qubit_calibrations = {}
        for key, qdata in data.get("qubit_calibration", {}).items():
            qid = int(key)
            qubit_calibrations[qid] = QubitCalibration.from_dict(qdata)
        
        # Parse edge calibrations
        edge_calibrations = {}
        for key, edata in data.get("edge_calibration", {}).items():
            # Key is like "(0, 1)" - parse it
            edge = EdgeCalibration.from_dict(edata)
            edge_calibrations[edge.edge] = edge
        
        return cls(
            distance=int(metadata.get("distance", 5)),
            num_qubits=len(qubit_calibrations) if qubit_calibrations else 49,
            seed=metadata.get("seed"),
            qubit_calibrations=qubit_calibrations,
            edge_calibrations=edge_calibrations,
            bad_qubits=data.get("bad_qubits", []),
            metadata=metadata,
        )
    
    def get_qubit(self, qubit_id: int) -> QubitCalibration:
        """Get calibration for a specific qubit, with fallback to defaults."""
        if qubit_id in self.qubit_calibrations:
            return self.qubit_calibrations[qubit_id]
        # Return default calibration
        return QubitCalibration(qubit_id=qubit_id)
    
    def get_edge(self, q1: int, q2: int) -> EdgeCalibration:
        """Get calibration for a specific edge, with fallback to defaults."""
        edge = tuple(sorted([q1, q2]))
        if edge in self.edge_calibrations:
            return self.edge_calibrations[edge]
        # Return default calibration
        return EdgeCalibration(qubit_a=q1, qubit_b=q2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all qubits and edges."""
        stats = {}
        
        if self.qubit_calibrations:
            t1_values = [q.t1_us for q in self.qubit_calibrations.values()]
            t2_values = [q.t2_us for q in self.qubit_calibrations.values()]
            readout_values = [q.readout_error for q in self.qubit_calibrations.values()]
            oneq_values = [q.oneq_error for q in self.qubit_calibrations.values()]
            
            stats["t1_median"] = float(np.median(t1_values))
            stats["t1_std"] = float(np.std(t1_values))
            stats["t1_range"] = [float(np.min(t1_values)), float(np.max(t1_values))]
            
            stats["t2_median"] = float(np.median(t2_values))
            stats["t2_std"] = float(np.std(t2_values))
            
            stats["readout_median"] = float(np.median(readout_values))
            stats["oneq_median"] = float(np.median(oneq_values))
        
        if self.edge_calibrations:
            cz_values = [e.cz_error for e in self.edge_calibrations.values()]
            leak_values = [e.cz_leakage for e in self.edge_calibrations.values()]
            crosstalk_values = [e.zz_crosstalk for e in self.edge_calibrations.values()]
            
            stats["cz_error_median"] = float(np.median(cz_values))
            stats["cz_error_std"] = float(np.std(cz_values))
            stats["cz_error_range"] = [float(np.min(cz_values)), float(np.max(cz_values))]
            
            stats["cz_leakage_median"] = float(np.median(leak_values))
            stats["zz_crosstalk_median"] = float(np.median(crosstalk_values))
        
        return stats


def generate_random_calibration(
    distance: int = 5,
    seed: Optional[int] = None,
    # Base parameters (from paper Table S4)
    t1_median: float = 73.0,
    t1_std: float = 15.0,
    t2_median: float = 80.0,
    t2_std: float = 20.0,
    readout_median: float = 0.008,
    readout_std: float = 0.003,
    reset_median: float = 0.0015,
    reset_std: float = 0.0005,
    oneq_median: float = 0.0006,
    oneq_std: float = 0.0002,
    cz_error_median: float = 0.0035,
    cz_error_std: float = 0.0012,
    cz_leakage_median: float = 0.0002,
    cz_leakage_std: float = 0.0001,
    zz_crosstalk_median: float = 0.00055,
    zz_crosstalk_std: float = 0.0002,
    bad_qubit_fraction: float = 0.04,
) -> DeviceCalibration:
    """Generate random calibration data mimicking real device heterogeneity.
    
    This function creates synthetic calibration data with statistical properties
    matching those observed in real superconducting quantum processors.
    
    Args:
        distance: Surface code distance (determines qubit count)
        seed: Random seed for reproducibility
        t1_median: Median T1 time in microseconds
        t1_std: Standard deviation of T1
        ... (other parameters follow paper Table S4)
        bad_qubit_fraction: Fraction of qubits marked as bad
        
    Returns:
        DeviceCalibration with randomly generated per-qubit and per-edge data
    """
    rng = np.random.default_rng(seed)
    
    # Number of qubits for rotated surface code: (2*d-1)^2 rounded
    # Actually for distance d, we have d^2 data + (d^2-1) ancilla = 2*d^2 - 1
    # But the grid is typically (2d-1) x (2d-1) with some positions empty
    # For simplicity, use d^2 + (d-1)^2 = 2*d^2 - 2*d + 1 for a rotated code
    # Or just use explicit grid: for d=5, we have 49 physical positions
    
    grid_size = 2 * distance - 1 if distance <= 5 else distance
    num_positions = grid_size * grid_size
    
    # Generate qubit calibrations
    qubit_calibrations = {}
    bad_qubits = []
    
    for i in range(num_positions):
        row = i // grid_size
        col = i % grid_size
        
        # Determine if this is a bad qubit
        is_bad = rng.random() < bad_qubit_fraction
        if is_bad:
            bad_qubits.append(i)
        
        # Generate T1, T2 with log-normal to ensure positivity
        t1 = float(np.clip(rng.normal(t1_median, t1_std), 40.0, 120.0))
        t2 = float(np.clip(rng.normal(t2_median, t2_std), 30.0, 140.0))
        
        # Ensure T2 <= 2*T1 (physical constraint)
        t2 = min(t2, 2 * t1)
        
        # Compute Tphi from T1 and T2
        if t2 < 2 * t1:
            tphi = 1.0 / (1.0/t2 - 1.0/(2*t1))
        else:
            tphi = rng.normal(720.0, 200.0)
        
        # Generate error rates (clipped to reasonable ranges)
        readout = float(np.clip(rng.normal(readout_median, readout_std), 0.002, 0.025))
        reset = float(np.clip(rng.normal(reset_median, reset_std), 0.0003, 0.005))
        oneq = float(np.clip(rng.normal(oneq_median, oneq_std), 0.0003, 0.002))
        
        # Bad qubits have worse parameters
        if is_bad:
            t1 = max(40.0, t1 * 0.6)
            t2 = max(30.0, t2 * 0.5)
            readout = min(0.025, readout * 2)
            oneq = min(0.002, oneq * 2)
        
        qubit_calibrations[i] = QubitCalibration(
            qubit_id=i,
            row=row,
            col=col,
            t1_us=t1,
            t2_us=t2,
            tphi_us=float(np.clip(tphi, 50.0, 1000.0)),
            readout_error=readout,
            reset_error=reset,
            oneq_error=oneq,
            is_bad_qubit=is_bad,
        )
    
    # Generate edge calibrations (nearest-neighbor connectivity)
    edge_calibrations = {}
    
    for i in range(num_positions):
        row = i // grid_size
        col = i % grid_size
        
        # Connect to right neighbor
        if col < grid_size - 1:
            j = i + 1
            edge = tuple(sorted([i, j]))
            if edge not in edge_calibrations:
                edge_calibrations[edge] = _generate_edge_calibration(
                    i, j, qubit_calibrations, rng,
                    cz_error_median, cz_error_std,
                    cz_leakage_median, cz_leakage_std,
                    zz_crosstalk_median, zz_crosstalk_std
                )
        
        # Connect to bottom neighbor
        if row < grid_size - 1:
            j = i + grid_size
            edge = tuple(sorted([i, j]))
            if edge not in edge_calibrations:
                edge_calibrations[edge] = _generate_edge_calibration(
                    i, j, qubit_calibrations, rng,
                    cz_error_median, cz_error_std,
                    cz_leakage_median, cz_leakage_std,
                    zz_crosstalk_median, zz_crosstalk_std
                )
    
    return DeviceCalibration(
        distance=distance,
        num_qubits=num_positions,
        seed=seed,
        qubit_calibrations=qubit_calibrations,
        edge_calibrations=edge_calibrations,
        bad_qubits=bad_qubits,
        metadata={
            "distance": distance,
            "seed": seed,
            "generated": True,
        },
    )


def _generate_edge_calibration(
    q1: int, q2: int,
    qubit_calibrations: Dict[int, QubitCalibration],
    rng: np.random.Generator,
    cz_error_median: float,
    cz_error_std: float,
    cz_leakage_median: float,
    cz_leakage_std: float,
    zz_crosstalk_median: float,
    zz_crosstalk_std: float,
) -> EdgeCalibration:
    """Generate calibration for a single edge based on connected qubit properties."""
    
    qcal1 = qubit_calibrations.get(q1)
    qcal2 = qubit_calibrations.get(q2)
    
    # CZ error depends on both qubits
    base_cz = float(np.clip(
        rng.normal(cz_error_median, cz_error_std),
        0.002, 0.008
    ))
    
    # Increase error if either qubit is bad
    is_bad = False
    if qcal1 and qcal1.is_bad_qubit:
        base_cz = min(0.008, base_cz * 1.5)
        is_bad = True
    if qcal2 and qcal2.is_bad_qubit:
        base_cz = min(0.008, base_cz * 1.5)
        is_bad = True
    
    # Leakage and crosstalk
    leakage = float(np.clip(
        rng.normal(cz_leakage_median, cz_leakage_std),
        5e-5, 8e-4
    ))
    
    crosstalk = float(np.clip(
        rng.normal(zz_crosstalk_median, zz_crosstalk_std),
        1e-4, 1.2e-3
    ))
    
    # Frequency collision (rare, ~5% of edges)
    has_freq_collision = rng.random() < 0.05
    if has_freq_collision:
        base_cz = min(0.008, base_cz * 1.3)
        crosstalk = min(0.0012, crosstalk * 1.5)
    
    # XEB fidelity is 1 - cz_error
    xeb = 1.0 - base_cz
    
    return EdgeCalibration(
        qubit_a=q1,
        qubit_b=q2,
        cz_error=base_cz,
        cz_leakage=leakage,
        zz_crosstalk=crosstalk,
        xeb_fidelity=xeb,
        is_bad_edge=is_bad,
        has_freq_collision=has_freq_collision,
    )


# ============================================================================
# Utility functions for surface code qubit mapping
# ============================================================================

def get_surface_code_qubit_map(distance: int) -> Dict[str, List[int]]:
    """Get the mapping of qubit types for a rotated surface code.
    
    Returns a dictionary with:
    - 'data': List of data qubit indices
    - 'x_ancilla': List of X-type ancilla indices
    - 'z_ancilla': List of Z-type ancilla indices
    """
    # For a distance-d rotated surface code:
    # - d^2 data qubits
    # - (d^2-1)/2 X ancillas (approximately)
    # - (d^2-1)/2 Z ancillas (approximately)
    
    # This is a simplified mapping; actual topology depends on the exact circuit
    grid_size = 2 * distance - 1
    
    data_qubits = []
    x_ancillas = []
    z_ancillas = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # Checkerboard pattern: data qubits at (even,even) and (odd,odd)
            if (i + j) % 2 == 0:
                data_qubits.append(idx)
            else:
                # X ancillas vs Z ancillas based on position
                if i % 2 == 0:
                    x_ancillas.append(idx)
                else:
                    z_ancillas.append(idx)
    
    return {
        'data': data_qubits,
        'x_ancilla': x_ancillas,
        'z_ancilla': z_ancillas,
    }
