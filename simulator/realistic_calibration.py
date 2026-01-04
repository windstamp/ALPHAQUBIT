"""Realistic Device Calibration Generator.

This module generates a realistic device calibration file that matches
Google's Sycamore statistics as closely as possible.

Since Google didn't release exact calibration values, we:
1. Use their published statistics (median, ranges)
2. Generate realistic spatial correlations
3. Include "bad qubit" patterns seen in real devices
4. Match the distribution shapes from Extended Data figures

This gets us as close to 100% as possible without actual hardware data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class PaperCalibrationStats:
    """Statistics from AlphaQubit paper (Table S2, Extended Data)."""
    
    # T1 relaxation (µs) - Extended Data Fig 1
    t1_median: float = 73.0
    t1_std: float = 15.0
    t1_min: float = 40.0
    t1_max: float = 120.0
    
    # T2 coherence (µs)
    t2_median: float = 80.0
    t2_std: float = 20.0
    t2_min: float = 30.0
    t2_max: float = 140.0
    
    # Pure dephasing Tφ (µs)
    tphi_median: float = 720.0
    tphi_std: float = 200.0
    
    # Single-qubit gate XEB error
    oneq_error_median: float = 0.0006  # 0.06%
    oneq_error_std: float = 0.0002
    
    # CZ gate XEB error - Extended Data Fig 4
    cz_error_median: float = 0.0035  # 0.35%
    cz_error_std: float = 0.0012
    cz_error_min: float = 0.002
    cz_error_max: float = 0.008
    
    # Readout error
    readout_error_median: float = 0.008  # 0.8%
    readout_error_std: float = 0.003
    
    # Reset error
    reset_error_median: float = 0.0015  # 0.15%
    reset_error_std: float = 0.0005
    
    # Leakage parameters
    leakage_rate_median: float = 0.00025  # 0.025%
    leakage_rate_std: float = 0.0001
    
    # ZZ crosstalk
    zz_crosstalk_median: float = 0.00055  # 0.055%
    zz_crosstalk_std: float = 0.0002


class RealisticCalibrationGenerator:
    """Generate realistic device calibration matching paper statistics.
    
    Key features:
    1. Spatial correlations (nearby qubits have correlated errors)
    2. Bad qubit clusters (realistic defects)
    3. Edge effects (boundary qubits often worse)
    4. Frequency collisions (some edges have higher errors)
    """
    
    def __init__(
        self,
        distance: int,
        seed: int = 42,
        stats: Optional[PaperCalibrationStats] = None,
        include_bad_qubits: bool = True,
        bad_qubit_fraction: float = 0.05,  # 5% of qubits are "bad"
    ):
        self.distance = distance
        self.seed = seed
        self.stats = stats or PaperCalibrationStats()
        self.include_bad_qubits = include_bad_qubits
        self.bad_qubit_fraction = bad_qubit_fraction
        
        self.rng = np.random.RandomState(seed)
        
        # Surface code layout
        self.num_data_qubits = distance * distance
        self.num_ancilla_qubits = distance * distance - 1
        self.total_qubits = self.num_data_qubits + self.num_ancilla_qubits
        
        # Generated calibration data
        self.qubit_calibration: Dict[int, Dict] = {}
        self.edge_calibration: Dict[Tuple[int, int], Dict] = {}
        self.bad_qubits: List[int] = []
        
        self._generate_calibration()
    
    def _generate_spatial_correlation_field(self) -> np.ndarray:
        """Generate a spatially correlated random field.
        
        This models the fact that nearby qubits often have correlated
        error rates (e.g., due to local defects in the chip).
        """
        size = self.distance * 2  # oversample for smoothness
        
        # Generate white noise
        noise = self.rng.randn(size, size)
        
        # Apply Gaussian smoothing for spatial correlation
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(noise, sigma=1.5)
        
        # Normalize to [0, 1]
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        
        return smoothed
    
    def _select_bad_qubits(self) -> List[int]:
        """Select qubits that will have elevated error rates.
        
        In real devices, bad qubits often cluster together.
        """
        num_bad = max(1, int(self.total_qubits * self.bad_qubit_fraction))
        
        # Pick a random "defect center" and select nearby qubits
        center = self.rng.randint(0, self.total_qubits)
        center_row = center // self.distance
        center_col = center % self.distance
        
        # Score qubits by distance from defect center
        scores = []
        for q in range(self.total_qubits):
            row = q // self.distance
            col = q % self.distance
            dist = abs(row - center_row) + abs(col - center_col)
            # Add randomness so it's not perfectly circular
            score = dist + self.rng.random() * 2
            scores.append((q, score))
        
        # Select qubits with lowest scores (closest to defect)
        scores.sort(key=lambda x: x[1])
        bad_qubits = [q for q, _ in scores[:num_bad]]
        
        return bad_qubits
    
    def _generate_qubit_calibration(self):
        """Generate per-qubit calibration data."""
        
        # Generate spatial correlation field
        try:
            spatial_field = self._generate_spatial_correlation_field()
        except ImportError:
            # Fallback if scipy not available
            spatial_field = np.ones((self.distance * 2, self.distance * 2)) * 0.5
        
        for q in range(self.total_qubits):
            row = q // self.distance
            col = q % self.distance
            
            # Get spatial correlation factor
            field_row = min(row, spatial_field.shape[0] - 1)
            field_col = min(col, spatial_field.shape[1] - 1)
            spatial_factor = spatial_field[field_row, field_col]
            
            # Edge penalty (boundary qubits slightly worse)
            edge_penalty = 1.0
            if row == 0 or row == self.distance - 1:
                edge_penalty += 0.15
            if col == 0 or col == self.distance - 1:
                edge_penalty += 0.15
            
            # Bad qubit penalty
            bad_penalty = 1.0
            if self.include_bad_qubits and q in self.bad_qubits:
                bad_penalty = 2.0 + self.rng.random()  # 2-3x worse
            
            # Generate T1 with spatial correlation
            t1_base = self.stats.t1_median + (spatial_factor - 0.5) * self.stats.t1_std * 2
            t1 = self.rng.normal(t1_base / bad_penalty, self.stats.t1_std * 0.3)
            t1 = np.clip(t1, self.stats.t1_min, self.stats.t1_max)
            
            # Generate T2 (correlated with T1)
            t2_base = min(2 * t1, self.stats.t2_median + (spatial_factor - 0.5) * self.stats.t2_std)
            t2 = self.rng.normal(t2_base / bad_penalty, self.stats.t2_std * 0.3)
            t2 = np.clip(t2, self.stats.t2_min, min(2 * t1, self.stats.t2_max))
            
            # Compute Tphi from T1 and T2: 1/Tphi = 1/T2 - 1/(2*T1)
            rate_phi = 1.0/t2 - 1.0/(2.0*t1)
            tphi = 1.0/rate_phi if rate_phi > 1e-9 else 1e6
            
            # Readout error
            readout_err = self.rng.normal(
                self.stats.readout_error_median * edge_penalty * bad_penalty,
                self.stats.readout_error_std
            )
            readout_err = np.clip(readout_err, 0.001, 0.05)
            
            # Reset error
            reset_err = self.rng.normal(
                self.stats.reset_error_median * bad_penalty,
                self.stats.reset_error_std
            )
            reset_err = np.clip(reset_err, 0.0001, 0.01)
            
            # 1Q gate error
            oneq_err = self.rng.normal(
                self.stats.oneq_error_median * bad_penalty,
                self.stats.oneq_error_std
            )
            oneq_err = np.clip(oneq_err, 0.0001, 0.005)
            
            self.qubit_calibration[q] = {
                'qubit_id': q,
                'row': row,
                'col': col,
                't1_us': float(t1),
                't2_us': float(t2),
                'tphi_us': float(tphi),
                'readout_error': float(readout_err),
                'reset_error': float(reset_err),
                'oneq_error': float(oneq_err),
                'is_bad_qubit': q in self.bad_qubits,
            }
    
    def _generate_edge_calibration(self):
        """Generate per-edge (CZ) calibration data."""
        
        for q1 in range(self.total_qubits):
            r1, c1 = q1 // self.distance, q1 % self.distance
            
            for q2 in range(q1 + 1, self.total_qubits):
                r2, c2 = q2 // self.distance, q2 % self.distance
                
                # Only adjacent qubits
                if abs(r1 - r2) + abs(c1 - c2) != 1:
                    continue
                
                edge = (q1, q2)
                
                # Bad edge if either qubit is bad
                bad_edge = (q1 in self.bad_qubits) or (q2 in self.bad_qubits)
                bad_factor = 2.0 if bad_edge else 1.0
                
                # Frequency collision: some edges randomly have higher error
                # (This models real frequency crowding issues)
                freq_collision = self.rng.random() < 0.1  # 10% of edges
                collision_factor = 1.5 if freq_collision else 1.0
                
                # CZ error (log-normal for realistic heavy tail)
                log_mean = np.log(self.stats.cz_error_median * bad_factor * collision_factor)
                log_std = 0.35  # ~35% relative spread
                cz_err = self.rng.lognormal(log_mean, log_std)
                cz_err = np.clip(cz_err, self.stats.cz_error_min, self.stats.cz_error_max)
                
                # Leakage (correlated with CZ error)
                leak = self.rng.normal(
                    self.stats.leakage_rate_median * (cz_err / self.stats.cz_error_median),
                    self.stats.leakage_rate_std
                )
                leak = np.clip(leak, 0.00005, 0.002)
                
                # ZZ crosstalk
                zz = self.rng.normal(
                    self.stats.zz_crosstalk_median * collision_factor,
                    self.stats.zz_crosstalk_std
                )
                zz = np.clip(zz, 0.0001, 0.002)
                
                # XEB fidelity
                xeb_fidelity = 1.0 - cz_err
                
                self.edge_calibration[edge] = {
                    'edge': edge,
                    'qubit_a': q1,
                    'qubit_b': q2,
                    'cz_error': float(cz_err),
                    'cz_leakage': float(leak),
                    'zz_crosstalk': float(zz),
                    'xeb_fidelity': float(xeb_fidelity),
                    'is_bad_edge': bad_edge,
                    'has_freq_collision': freq_collision,
                }
    
    def _generate_calibration(self):
        """Generate complete calibration."""
        if self.include_bad_qubits:
            self.bad_qubits = self._select_bad_qubits()
        
        self._generate_qubit_calibration()
        self._generate_edge_calibration()
    
    def get_statistics(self) -> Dict:
        """Get statistics of generated calibration."""
        t1_vals = [q['t1_us'] for q in self.qubit_calibration.values()]
        t2_vals = [q['t2_us'] for q in self.qubit_calibration.values()]
        readout_vals = [q['readout_error'] for q in self.qubit_calibration.values()]
        cz_vals = [e['cz_error'] for e in self.edge_calibration.values()]
        xeb_vals = [e['xeb_fidelity'] for e in self.edge_calibration.values()]
        
        return {
            'num_qubits': len(self.qubit_calibration),
            'num_edges': len(self.edge_calibration),
            'num_bad_qubits': len(self.bad_qubits),
            't1_median': float(np.median(t1_vals)),
            't1_std': float(np.std(t1_vals)),
            't1_range': [float(np.min(t1_vals)), float(np.max(t1_vals))],
            't2_median': float(np.median(t2_vals)),
            't2_std': float(np.std(t2_vals)),
            'readout_median': float(np.median(readout_vals)),
            'cz_error_median': float(np.median(cz_vals)),
            'cz_error_std': float(np.std(cz_vals)),
            'cz_error_range': [float(np.min(cz_vals)), float(np.max(cz_vals))],
            'xeb_median': float(np.median(xeb_vals)),
            'xeb_range': [float(np.min(xeb_vals)), float(np.max(xeb_vals))],
        }
    
    def to_dict(self) -> Dict:
        """Export calibration to dictionary."""
        return {
            'metadata': {
                'distance': self.distance,
                'seed': self.seed,
                'stats': asdict(self.stats),
                'generated_stats': self.get_statistics(),
            },
            'qubit_calibration': self.qubit_calibration,
            'edge_calibration': {str(k): v for k, v in self.edge_calibration.items()},
            'bad_qubits': self.bad_qubits,
        }
    
    def save(self, path: Path):
        """Save calibration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_p_ij(self, q1: int, q2: int) -> float:
        """Get p_ij for edge (q1, q2)."""
        edge = (min(q1, q2), max(q1, q2))
        if edge in self.edge_calibration:
            return self.edge_calibration[edge]['cz_error']
        return self.stats.cz_error_median
    
    def get_xeb(self, q1: int, q2: int) -> float:
        """Get XEB fidelity for edge (q1, q2)."""
        return 1.0 - self.get_p_ij(q1, q2)


def compare_with_paper_stats(cal: RealisticCalibrationGenerator):
    """Compare generated calibration with paper statistics."""
    stats = cal.get_statistics()
    paper = cal.stats
    
    print("=" * 70)
    print("Comparison: Generated Calibration vs Paper Statistics")
    print("=" * 70)
    
    print(f"\n{'Parameter':<25} {'Paper':<15} {'Generated':<15} {'Match'}")
    print("-" * 70)
    
    checks = [
        ('T1 median (µs)', paper.t1_median, stats['t1_median']),
        ('T2 median (µs)', paper.t2_median, stats['t2_median']),
        ('Readout error', paper.readout_error_median, stats['readout_median']),
        ('CZ error median', paper.cz_error_median, stats['cz_error_median']),
        ('XEB fidelity', 1-paper.cz_error_median, stats['xeb_median']),
    ]
    
    for name, paper_val, gen_val in checks:
        rel_diff = abs(paper_val - gen_val) / paper_val * 100
        match = "✅" if rel_diff < 20 else "⚠️" if rel_diff < 50 else "❌"
        print(f"{name:<25} {paper_val:<15.4f} {gen_val:<15.4f} {match} ({rel_diff:.1f}%)")
    
    print("\n" + "=" * 70)


def print_calibration_summary(cal: RealisticCalibrationGenerator):
    """Print summary of generated calibration."""
    stats = cal.get_statistics()
    
    print("=" * 70)
    print(f"Realistic Device Calibration (d={cal.distance})")
    print("=" * 70)
    
    print(f"\nQubits: {stats['num_qubits']} total, {stats['num_bad_qubits']} bad")
    print(f"Edges:  {stats['num_edges']}")
    
    print(f"\nPer-Qubit Parameters:")
    print(f"  T1:      {stats['t1_median']:.1f} µs (std={stats['t1_std']:.1f}, range=[{stats['t1_range'][0]:.1f}, {stats['t1_range'][1]:.1f}])")
    print(f"  T2:      {stats['t2_median']:.1f} µs (std={stats['t2_std']:.1f})")
    print(f"  Readout: {stats['readout_median']*100:.2f}%")
    
    print(f"\nPer-Edge Parameters:")
    print(f"  CZ error: {stats['cz_error_median']*100:.3f}% (std={stats['cz_error_std']*100:.3f}%)")
    print(f"  CZ range: [{stats['cz_error_range'][0]*100:.3f}%, {stats['cz_error_range'][1]*100:.3f}%]")
    print(f"  XEB:      {stats['xeb_median']*100:.2f}% (range=[{stats['xeb_range'][0]*100:.2f}%, {stats['xeb_range'][1]*100:.2f}%])")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Generate calibration for d=5
    cal = RealisticCalibrationGenerator(distance=5, seed=42)
    
    print_calibration_summary(cal)
    compare_with_paper_stats(cal)
    
    # Save to file
    cal.save(Path("configs/realistic_calibration_d5.json"))
    print("\nSaved to: configs/realistic_calibration_d5.json")
