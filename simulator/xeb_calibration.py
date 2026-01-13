"""XEB Calibration and Per-Edge Error Estimation.

This module estimates per-edge CZ error rates (p_ij) and XEB fidelities
based on Google's AlphaQubit paper (Nature 2024).

Paper References:
- Supplementary Table S2: Device parameters
- Extended Data Figure 4: CZ error distribution across edges
- Methods: XEB fidelity definition

Key Relationships:
- XEB fidelity: F_XEB = 1 - (16/15) * p_depol  (for 2Q depolarizing)
- CZ Pauli error: p_CZ ≈ 1 - F_XEB (approximate)
- Per-edge variation: log-normal distribution around median
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class XEBCalibration:
    """XEB calibration data for a surface code patch."""
    
    # Paper values (Supplementary Table S2)
    MEDIAN_CZ_XEB_FIDELITY = 0.9965  # 99.65%
    MEDIAN_CZ_PAULI_ERROR = 0.0035   # 0.35%
    
    # Distribution parameters (estimated from Extended Data Fig 4)
    CZ_ERROR_LOG_MEAN = np.log(0.0035)  # log of median
    CZ_ERROR_LOG_STD = 0.35              # ~35% relative spread
    
    # 1Q gate XEB (from paper)
    MEDIAN_1Q_XEB_FIDELITY = 0.9994  # 99.94%
    MEDIAN_1Q_PAULI_ERROR = 0.0006   # 0.06%
    
    # Readout fidelity
    MEDIAN_READOUT_FIDELITY = 0.992  # 99.2%
    MEDIAN_READOUT_ERROR = 0.008     # 0.8%


def xeb_to_pauli_error(f_xeb: float, num_qubits: int = 2) -> float:
    """Convert XEB fidelity to Pauli error probability.
    
    For depolarizing channel on n qubits:
    F_XEB = 1 - (4^n / (4^n - 1)) * p_depol
    
    Args:
        f_xeb: XEB fidelity (0 to 1)
        num_qubits: Number of qubits (1 or 2)
    
    Returns:
        Pauli error probability
    """
    dim = 4 ** num_qubits
    factor = dim / (dim - 1)
    return (1 - f_xeb) / factor * factor  # simplified: ≈ 1 - f_xeb


def pauli_error_to_xeb(p_error: float, num_qubits: int = 2) -> float:
    """Convert Pauli error probability to XEB fidelity.
    
    Args:
        p_error: Pauli error probability
        num_qubits: Number of qubits (1 or 2)
    
    Returns:
        XEB fidelity
    """
    dim = 4 ** num_qubits
    factor = dim / (dim - 1)
    return 1 - factor * p_error


class EdgeCalibrationMap:
    """Generate realistic per-edge CZ error rates based on XEB data.
    
    This creates a spatial distribution of CZ errors that matches
    the statistics observed in Google's Sycamore processor.
    """
    
    def __init__(
        self,
        distance: int,
        seed: int = 42,
        median_cz_error: float = 0.0035,
        relative_spread: float = 0.35,
    ):
        """Initialize edge calibration map.
        
        Args:
            distance: Surface code distance
            seed: Random seed for reproducibility
            median_cz_error: Median CZ Pauli error (default: 0.35%)
            relative_spread: Relative standard deviation in log space
        """
        self.distance = distance
        self.seed = seed
        self.median_cz_error = median_cz_error
        self.relative_spread = relative_spread
        
        self.rng = np.random.RandomState(seed)
        self._edge_errors: Dict[Tuple[int, int], float] = {}
        self._edge_xeb: Dict[Tuple[int, int], float] = {}
        
        self._generate_edge_map()
    
    def _get_num_edges(self) -> int:
        """Estimate number of CZ edges in rotated surface code."""
        # For distance d, approximately 2*d^2 - 2*d edges
        d = self.distance
        return 2 * d * d - 2 * d
    
    def _generate_edge_map(self):
        """Generate per-edge CZ errors from log-normal distribution."""
        num_edges = self._get_num_edges()
        
        # Log-normal distribution matching paper statistics
        log_mean = np.log(self.median_cz_error)
        log_std = self.relative_spread
        
        errors = self.rng.lognormal(log_mean, log_std, num_edges)
        
        # Clip to physical range [0.001, 0.02] (0.1% to 2%)
        errors = np.clip(errors, 0.001, 0.02)
        
        # Store as edge map (using synthetic edge indices)
        for i, err in enumerate(errors):
            # Create edge tuple (using row-major indexing for qubits)
            q1 = i
            q2 = i + 1 + (i % self.distance)  # neighbor pattern
            edge = tuple(sorted((q1, q2)))
            self._edge_errors[edge] = err
            self._edge_xeb[edge] = pauli_error_to_xeb(err, num_qubits=2)
    
    def get_edge_error(self, q1: int, q2: int) -> float:
        """Get CZ Pauli error for edge (q1, q2).
        
        If edge not in map, generates a new value deterministically.
        """
        edge = tuple(sorted((q1, q2)))
        if edge not in self._edge_errors:
            # Generate deterministically from edge indices
            local_rng = np.random.RandomState(self.seed + hash(edge) % 10000)
            log_mean = np.log(self.median_cz_error)
            err = local_rng.lognormal(log_mean, self.relative_spread)
            err = np.clip(err, 0.001, 0.02)
            self._edge_errors[edge] = err
            self._edge_xeb[edge] = pauli_error_to_xeb(err, num_qubits=2)
        return self._edge_errors[edge]
    
    def get_edge_xeb(self, q1: int, q2: int) -> float:
        """Get XEB fidelity for edge (q1, q2)."""
        self.get_edge_error(q1, q2)  # ensure generated
        edge = tuple(sorted((q1, q2)))
        return self._edge_xeb[edge]
    
    def get_statistics(self) -> Dict:
        """Get statistics of the edge error distribution."""
        errors = list(self._edge_errors.values())
        xebs = list(self._edge_xeb.values())
        
        return {
            'num_edges': len(errors),
            'cz_error_median': np.median(errors),
            'cz_error_mean': np.mean(errors),
            'cz_error_std': np.std(errors),
            'cz_error_min': np.min(errors),
            'cz_error_max': np.max(errors),
            'xeb_median': np.median(xebs),
            'xeb_mean': np.mean(xebs),
            'xeb_min': np.min(xebs),
            'xeb_max': np.max(xebs),
        }


def estimate_circuit_xeb(
    distance: int,
    rounds: int,
    cz_error: float = 0.0035,
    oneq_error: float = 0.0006,
    readout_error: float = 0.008,
) -> float:
    """Estimate overall circuit XEB fidelity.
    
    Based on error propagation through the circuit.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome rounds
        cz_error: Per-CZ Pauli error
        oneq_error: Per-1Q gate Pauli error
        readout_error: Measurement error
    
    Returns:
        Estimated circuit XEB fidelity
    """
    # Count operations per round (approximate)
    num_data_qubits = distance ** 2
    num_ancilla_qubits = (distance ** 2 - 1)
    num_cz_per_round = 2 * num_ancilla_qubits  # 4 CZ per ancilla, but paired
    num_1q_per_round = 2 * num_ancilla_qubits  # H gates
    
    # Total operations
    total_cz = num_cz_per_round * rounds
    total_1q = num_1q_per_round * rounds
    total_readout = num_ancilla_qubits * rounds + num_data_qubits
    
    # Fidelity is product of individual fidelities
    f_cz = (1 - cz_error) ** total_cz
    f_1q = (1 - oneq_error) ** total_1q
    f_readout = (1 - readout_error) ** total_readout
    
    return f_cz * f_1q * f_readout


def decompose_cz_error(
    p_cz_total: float = 0.0035,
) -> Dict[str, float]:
    """Decompose total CZ error into components (paper Table S4).
    
    The paper breaks down CZ error into:
    - Depolarizing: ~70% of total
    - Leakage: ~6% of total  
    - ZZ crosstalk: ~16% of total
    - Other: ~8%
    
    Args:
        p_cz_total: Total CZ Pauli error
    
    Returns:
        Dictionary of error components
    """
    return {
        'p_cz_depolarizing': 0.70 * p_cz_total,  # ~2.45e-3
        'p_cz_leakage': 0.06 * p_cz_total,        # ~2.1e-4
        'p_cz_zz_crosstalk': 0.16 * p_cz_total,   # ~5.6e-4
        'p_cz_other': 0.08 * p_cz_total,          # ~2.8e-4
        'p_cz_total': p_cz_total,
    }


# =============================================================================
# Convenience functions for integration with existing code
# =============================================================================

def get_paper_xeb_values() -> Dict:
    """Get XEB values from the paper for reference."""
    return {
        'cz_xeb_fidelity': 0.9965,
        'cz_pauli_error': 0.0035,
        '1q_xeb_fidelity': 0.9994,
        '1q_pauli_error': 0.0006,
        'readout_fidelity': 0.992,
        'readout_error': 0.008,
        'T1_us': 73.0,
        'T2_us': 80.0,
        'Tphi_us': 720.0,
    }


def print_xeb_report(distance: int = 5, seed: int = 42):
    """Print a report of XEB-based calibration."""
    print("=" * 60)
    print("XEB Calibration Report (Based on AlphaQubit Paper)")
    print("=" * 60)
    
    # Paper reference values
    ref = get_paper_xeb_values()
    print("\n1. PAPER REFERENCE VALUES (Table S2)")
    print("-" * 40)
    print(f"  CZ XEB Fidelity:     {ref['cz_xeb_fidelity']*100:.2f}%")
    print(f"  CZ Pauli Error:      {ref['cz_pauli_error']*100:.3f}%")
    print(f"  1Q XEB Fidelity:     {ref['1q_xeb_fidelity']*100:.2f}%")
    print(f"  1Q Pauli Error:      {ref['1q_pauli_error']*100:.3f}%")
    print(f"  Readout Fidelity:    {ref['readout_fidelity']*100:.1f}%")
    
    # Per-edge distribution
    print(f"\n2. PER-EDGE CZ ERROR DISTRIBUTION (d={distance})")
    print("-" * 40)
    edge_map = EdgeCalibrationMap(distance, seed=seed)
    stats = edge_map.get_statistics()
    print(f"  Number of edges:     {stats['num_edges']}")
    print(f"  CZ Error Median:     {stats['cz_error_median']*100:.3f}%")
    print(f"  CZ Error Mean:       {stats['cz_error_mean']*100:.3f}%")
    print(f"  CZ Error Std:        {stats['cz_error_std']*100:.3f}%")
    print(f"  CZ Error Range:      [{stats['cz_error_min']*100:.3f}%, {stats['cz_error_max']*100:.3f}%]")
    print(f"  XEB Fidelity Range:  [{stats['xeb_min']*100:.2f}%, {stats['xeb_max']*100:.2f}%]")
    
    # Error decomposition
    print("\n3. CZ ERROR DECOMPOSITION (Table S4)")
    print("-" * 40)
    decomp = decompose_cz_error(ref['cz_pauli_error'])
    print(f"  Depolarizing:        {decomp['p_cz_depolarizing']*100:.4f}% (70%)")
    print(f"  Leakage:             {decomp['p_cz_leakage']*100:.4f}% (6%)")
    print(f"  ZZ Crosstalk:        {decomp['p_cz_zz_crosstalk']*100:.4f}% (16%)")
    print(f"  Other:               {decomp['p_cz_other']*100:.4f}% (8%)")
    
    # Circuit-level estimate
    print("\n4. CIRCUIT XEB ESTIMATE")
    print("-" * 40)
    for d in [3, 5, 7]:
        for r in [5, 15, 25]:
            f_circuit = estimate_circuit_xeb(d, r)
            print(f"  d={d}, r={r:2d}: Circuit XEB ≈ {f_circuit*100:.2f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_xeb_report(distance=5)
