"""Parameter Estimation from Experimental Relationships.

This module estimates missing noise model parameters using:
1. Physical relationships from the paper
2. Published statistics (medians, ranges)
3. XEB fidelity data
4. Error budget decompositions

Key insight: Even without exact values, we can derive parameters
from known relationships.

Paper References:
- Table S2: Device parameters (T1, T2, XEB)
- Table S4: Pauli+ noise model
- Extended Data Fig 1: T1/T2 distributions
- Extended Data Fig 4: CZ error distribution
- Methods: Error budget decomposition
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EstimatedParameters:
    """Container for estimated noise parameters."""
    # Decoherence
    t1_us: float
    t2_us: float
    tphi_us: float
    
    # Single-qubit errors
    p_1q_gate: float
    p_1q_idle: float
    
    # Two-qubit (CZ) errors - decomposed
    p_cz_total: float
    p_cz_depolarizing: float
    p_cz_leakage: float
    p_cz_zz_crosstalk: float
    p_cz_swap_like: float
    p_cz_other: float
    
    # Readout/Reset
    p_readout: float
    p_reset: float
    
    # Leakage
    p_leakage_heating: float
    p_leakage_seepage: float
    
    # I/Q readout model
    iq_snr: float
    iq_alpha: float  # asymmetry
    iq_sigma_leak: float
    
    def to_dict(self) -> Dict:
        return {
            't1_us': self.t1_us,
            't2_us': self.t2_us,
            'tphi_us': self.tphi_us,
            'p_1q_gate': self.p_1q_gate,
            'p_1q_idle': self.p_1q_idle,
            'p_cz_total': self.p_cz_total,
            'p_cz_depolarizing': self.p_cz_depolarizing,
            'p_cz_leakage': self.p_cz_leakage,
            'p_cz_zz_crosstalk': self.p_cz_zz_crosstalk,
            'p_cz_swap_like': self.p_cz_swap_like,
            'p_cz_other': self.p_cz_other,
            'p_readout': self.p_readout,
            'p_reset': self.p_reset,
            'p_leakage_heating': self.p_leakage_heating,
            'p_leakage_seepage': self.p_leakage_seepage,
            'iq_snr': self.iq_snr,
            'iq_alpha': self.iq_alpha,
            'iq_sigma_leak': self.iq_sigma_leak,
        }


class ParameterEstimator:
    """Estimate missing parameters from known relationships.
    
    The paper provides several constraints:
    1. XEB fidelity ↔ Pauli error: F_XEB ≈ 1 - p_Pauli
    2. T2 ↔ T1, Tφ: 1/T2 = 1/(2*T1) + 1/Tφ
    3. CZ error budget: p_CZ = p_depol + p_leak + p_ZZ + p_swap + p_other
    4. Idle error from T1, Tφ, cycle time
    5. SNR from readout distributions
    """
    
    def __init__(
        self,
        cycle_ns: float = 1076.0,
        xeb_cz_fidelity: Optional[float] = None,
        t1_us: Optional[float] = None,
        t2_us: Optional[float] = None,
    ):
        """Initialize estimator with known constraints.
        
        Args:
            cycle_ns: Surface code cycle time in nanoseconds
            xeb_cz_fidelity: Measured CZ XEB fidelity (if known)
            t1_us: Measured T1 (if known)
            t2_us: Measured T2 (if known)
        """
        self.cycle_ns = cycle_ns
        self.cycle_us = cycle_ns / 1000.0
        
        # Known values from paper (Table S2)
        self.xeb_cz = xeb_cz_fidelity or 0.9965
        self.xeb_1q = 0.9994
        self.t1 = t1_us or 73.0
        self.t2 = t2_us or 80.0
    
    # =========================================================================
    # Physical Relationships
    # =========================================================================
    
    def estimate_tphi_from_t1_t2(self, t1_us: float, t2_us: float) -> float:
        """Estimate pure dephasing time Tφ from T1 and T2.
        
        Relationship: 1/T2 = 1/(2*T1) + 1/Tφ
        Therefore: 1/Tφ = 1/T2 - 1/(2*T1)
        """
        rate_2 = 1.0 / t2_us
        rate_1_half = 1.0 / (2.0 * t1_us)
        rate_phi = rate_2 - rate_1_half
        
        if rate_phi <= 0:
            # T2 limit is 2*T1, so Tφ → ∞
            return 1e6  # effectively infinite
        
        return 1.0 / rate_phi
    
    def estimate_idle_error(self, t1_us: float, tphi_us: float, dt_us: float) -> Tuple[float, float, float]:
        """Estimate idle Pauli error from decoherence.
        
        For amplitude damping + dephasing channel over time dt:
        - p_z ≈ (1 - exp(-dt/Tφ)) / 2  (pure dephasing)
        - p_x ≈ p_y ≈ (1 - exp(-dt/T1)) / 4  (amplitude damping)
        
        Returns:
            Tuple of (p_x, p_y, p_z)
        """
        # Amplitude damping contribution
        gamma_1 = 1.0 - np.exp(-dt_us / t1_us)
        p_ad = gamma_1 / 4.0  # split between X and Y
        
        # Pure dephasing contribution
        gamma_phi = 1.0 - np.exp(-dt_us / tphi_us)
        p_z_dephase = gamma_phi / 2.0
        
        # Combined (approximate, ignoring small cross-terms)
        p_x = p_ad
        p_y = p_ad
        p_z = p_z_dephase + p_ad  # Z gets contribution from both
        
        return p_x, p_y, p_z
    
    def xeb_to_pauli_error(self, f_xeb: float, num_qubits: int = 2) -> float:
        """Convert XEB fidelity to Pauli error probability.
        
        For n-qubit depolarizing channel:
        F_XEB = 1 - (4^n)/(4^n - 1) * p_depol ≈ 1 - p_depol
        """
        return 1.0 - f_xeb
    
    def pauli_to_xeb_fidelity(self, p_error: float, num_qubits: int = 2) -> float:
        """Convert Pauli error to XEB fidelity."""
        return 1.0 - p_error
    
    # =========================================================================
    # Error Budget Decomposition
    # =========================================================================
    
    def decompose_cz_error(self, p_cz_total: float) -> Dict[str, float]:
        """Decompose total CZ error into components.
        
        From paper Table S4 and error budget analysis:
        - Depolarizing: ~70% of total (incoherent rotation errors)
        - Leakage: ~6% of total (|11⟩ → |02⟩/|20⟩)
        - ZZ crosstalk: ~16% of total (residual ZZ coupling)
        - Swap-like: ~0% (typically negligible)
        - Other: ~8% (uncharacterized)
        
        Note: These fractions are approximate and can vary by edge.
        """
        return {
            'p_cz_depolarizing': 0.70 * p_cz_total,
            'p_cz_leakage': 0.06 * p_cz_total,
            'p_cz_zz_crosstalk': 0.16 * p_cz_total,
            'p_cz_swap_like': 0.00 * p_cz_total,
            'p_cz_other': 0.08 * p_cz_total,
        }
    
    def estimate_1q_excess_error(self, xeb_1q: float, idle_error: float) -> float:
        """Estimate excess 1Q gate error beyond idle decoherence.
        
        Total 1Q error = idle (from T1/Tφ) + excess (control errors)
        """
        total_1q = self.xeb_to_pauli_error(xeb_1q, num_qubits=1)
        excess = max(0, total_1q - idle_error)
        return excess
    
    # =========================================================================
    # I/Q Readout Model Estimation
    # =========================================================================
    
    def estimate_iq_snr_from_readout_error(self, p_readout: float) -> float:
        """Estimate I/Q SNR from readout error probability.
        
        For Gaussian I/Q distributions with SNR = |μ1 - μ0| / σ:
        p_error ≈ erfc(SNR / (2√2)) / 2
        
        Inverting: SNR ≈ 2√2 * erfc_inv(2 * p_error)
        
        Paper uses SNR ≈ 3.0 with p_readout ≈ 0.8%
        """
        from scipy.special import erfcinv
        
        # Clip to valid range
        p = np.clip(p_readout, 1e-6, 0.5)
        
        # Invert error function
        snr = 2.0 * np.sqrt(2.0) * erfcinv(2.0 * p)
        
        return float(snr)
    
    def estimate_readout_error_from_snr(self, snr: float) -> float:
        """Estimate readout error from I/Q SNR."""
        from scipy.special import erfc
        return 0.5 * erfc(snr / (2.0 * np.sqrt(2.0)))
    
    # =========================================================================
    # Full Parameter Estimation
    # =========================================================================
    
    def estimate_all_parameters(self) -> EstimatedParameters:
        """Estimate all noise parameters from known constraints."""
        
        # Decoherence
        t1 = self.t1
        t2 = self.t2
        tphi = self.estimate_tphi_from_t1_t2(t1, t2)
        
        # Idle error
        px, py, pz = self.estimate_idle_error(t1, tphi, self.cycle_us)
        p_idle = px + py + pz
        
        # 1Q gate error
        p_1q_total = self.xeb_to_pauli_error(self.xeb_1q, num_qubits=1)
        p_1q_excess = self.estimate_1q_excess_error(self.xeb_1q, p_idle)
        
        # CZ error decomposition
        p_cz_total = self.xeb_to_pauli_error(self.xeb_cz, num_qubits=2)
        cz_decomp = self.decompose_cz_error(p_cz_total)
        
        # Readout/Reset (from paper Table S2)
        p_readout = 0.008  # 0.8%
        p_reset = 0.0015   # 0.15%
        
        # I/Q model
        iq_snr = self.estimate_iq_snr_from_readout_error(p_readout)
        iq_alpha = 0.9     # asymmetry from paper
        iq_sigma_leak = 1.6  # leakage blob width
        
        # Leakage
        p_heat = 0.00025   # 0.025% heating per cycle
        p_seepage = 0.00005  # 0.005% seepage
        
        return EstimatedParameters(
            t1_us=t1,
            t2_us=t2,
            tphi_us=tphi,
            p_1q_gate=p_1q_total,
            p_1q_idle=p_idle,
            p_cz_total=p_cz_total,
            p_cz_depolarizing=cz_decomp['p_cz_depolarizing'],
            p_cz_leakage=cz_decomp['p_cz_leakage'],
            p_cz_zz_crosstalk=cz_decomp['p_cz_zz_crosstalk'],
            p_cz_swap_like=cz_decomp['p_cz_swap_like'],
            p_cz_other=cz_decomp['p_cz_other'],
            p_readout=p_readout,
            p_reset=p_reset,
            p_leakage_heating=p_heat,
            p_leakage_seepage=p_seepage,
            iq_snr=iq_snr,
            iq_alpha=iq_alpha,
            iq_sigma_leak=iq_sigma_leak,
        )
    
    def estimate_for_edge(
        self,
        t1_a: float,
        t1_b: float,
        t2_a: float,
        t2_b: float,
        xeb_cz: float,
    ) -> Dict[str, float]:
        """Estimate parameters for a specific edge given qubit-level data.
        
        Args:
            t1_a, t1_b: T1 times for qubits A and B
            t2_a, t2_b: T2 times for qubits A and B
            xeb_cz: XEB fidelity for this CZ gate
        
        Returns:
            Dictionary of edge-specific parameters
        """
        # Average decoherence for the pair
        t1_avg = (t1_a + t1_b) / 2.0
        t2_avg = (t2_a + t2_b) / 2.0
        tphi_avg = self.estimate_tphi_from_t1_t2(t1_avg, t2_avg)
        
        # CZ error
        p_cz = self.xeb_to_pauli_error(xeb_cz, num_qubits=2)
        cz_decomp = self.decompose_cz_error(p_cz)
        
        # Idle contribution during CZ gate (~40ns gate + alignment)
        cz_duration_us = 0.04  # ~40ns
        px, py, pz = self.estimate_idle_error(t1_avg, tphi_avg, cz_duration_us)
        p_idle_during_cz = px + py + pz
        
        return {
            't1_avg_us': t1_avg,
            't2_avg_us': t2_avg,
            'tphi_avg_us': tphi_avg,
            'p_cz_total': p_cz,
            'p_cz_idle_contribution': p_idle_during_cz,
            **cz_decomp,
        }


def estimate_missing_from_paper() -> Dict[str, float]:
    """Estimate all parameters that aren't directly specified in the paper.
    
    Paper directly specifies (Table S2, S4):
    - T1 = 73 µs (median)
    - T2_CPMG = 80 µs (median)
    - p_readout = 0.8%
    - p_reset = 0.15%
    - p_heat = 0.025%
    - XEB_CZ = 99.65%
    - XEB_1Q = 99.94%
    
    We can DERIVE:
    - Tφ from T1 and T2
    - Idle error from T1, Tφ, cycle time
    - CZ error decomposition from total
    - I/Q SNR from readout error
    """
    estimator = ParameterEstimator()
    params = estimator.estimate_all_parameters()
    
    return params.to_dict()


def print_estimation_report():
    """Print detailed parameter estimation report."""
    print("=" * 70)
    print("PARAMETER ESTIMATION FROM PAPER DATA")
    print("=" * 70)
    
    estimator = ParameterEstimator()
    
    # Known values
    print("\n1. KNOWN VALUES (from paper Table S2, S4)")
    print("-" * 50)
    print(f"  T1 (median):        {estimator.t1} µs")
    print(f"  T2_CPMG (median):   {estimator.t2} µs")
    print(f"  XEB_CZ:             {estimator.xeb_cz * 100:.2f}%")
    print(f"  XEB_1Q:             {estimator.xeb_1q * 100:.2f}%")
    print(f"  Cycle time:         {estimator.cycle_ns} ns")
    
    # Derived values
    params = estimator.estimate_all_parameters()
    
    print("\n2. DERIVED VALUES (from physical relationships)")
    print("-" * 50)
    print(f"  Tφ (pure dephasing): {params.tphi_us:.1f} µs")
    print(f"    Formula: 1/Tφ = 1/T2 - 1/(2*T1)")
    print(f"    Calculation: 1/{params.t2_us} - 1/(2×{params.t1_us}) = {1/params.tphi_us:.6f}")
    
    print(f"\n  Idle error (per cycle):")
    px, py, pz = estimator.estimate_idle_error(params.t1_us, params.tphi_us, estimator.cycle_us)
    print(f"    p_X = {px*100:.4f}%")
    print(f"    p_Y = {py*100:.4f}%")
    print(f"    p_Z = {pz*100:.4f}%")
    print(f"    Total = {params.p_1q_idle*100:.4f}%")
    
    print("\n3. CZ ERROR DECOMPOSITION")
    print("-" * 50)
    print(f"  Total CZ error:     {params.p_cz_total*100:.3f}%")
    print(f"  - Depolarizing:     {params.p_cz_depolarizing*100:.4f}% (70%)")
    print(f"  - Leakage:          {params.p_cz_leakage*100:.4f}% (6%)")
    print(f"  - ZZ crosstalk:     {params.p_cz_zz_crosstalk*100:.4f}% (16%)")
    print(f"  - Swap-like:        {params.p_cz_swap_like*100:.4f}% (0%)")
    print(f"  - Other:            {params.p_cz_other*100:.4f}% (8%)")
    
    print("\n4. I/Q READOUT MODEL")
    print("-" * 50)
    print(f"  Readout error:      {params.p_readout*100:.2f}%")
    print(f"  Estimated SNR:      {params.iq_snr:.2f}")
    print(f"  Asymmetry (α):      {params.iq_alpha}")
    print(f"  Leakage σ scale:    {params.iq_sigma_leak}")
    
    # Verify SNR → error
    p_check = estimator.estimate_readout_error_from_snr(params.iq_snr)
    print(f"  Verification: SNR={params.iq_snr:.2f} → p_readout={p_check*100:.2f}%")
    
    print("\n5. COMPLETE PARAMETER SET")
    print("-" * 50)
    for key, val in params.to_dict().items():
        if 'us' in key:
            print(f"  {key:<25} = {val:.2f} µs")
        elif val < 0.01:
            print(f"  {key:<25} = {val*100:.4f}%")
        else:
            print(f"  {key:<25} = {val:.4f}")
    
    print("\n" + "=" * 70)
    print("These parameters can be used directly in the noise model.")
    print("=" * 70)


if __name__ == "__main__":
    print_estimation_report()
