#!/usr/bin/env python3
"""Test suite for newly implemented noise channels.

Tests all the noise mechanisms from the AlphaQubit Nature 2024 paper that were
recently added to my_noise_model/channels.py.
"""

import numpy as np
import sys
sys.path.insert(0, 'c:/Users/Lenovo/software/ALPHAQUBIT')

from my_noise_model.channels import (
    readout_crosstalk_kraus,
    cosmic_ray_burst_kraus,
    measurement_induced_reset_kraus,
    leakage_seepage_kraus,
    state_dependent_amplitude_damping_kraus,
    temporal_noise_scaling,
    frequency_collision_kraus,
    spatially_correlated_crosstalk_kraus,
)


def is_valid_kraus_set(kraus_ops: list, dim: int = 3, tol: float = 1e-10) -> bool:
    """Check if Kraus operators satisfy completeness: sum(K†K) = I."""
    total = np.zeros((dim, dim), dtype=complex)
    for K in kraus_ops:
        total += K.conj().T @ K
    identity = np.eye(dim)
    return np.allclose(total, identity, atol=tol)


def test_readout_crosstalk():
    """Test readout crosstalk Kraus operators."""
    print("Testing readout_crosstalk_kraus...")
    
    # Test with various crosstalk probabilities
    for p in [0.0, 0.001, 0.01, 0.1]:
        kraus = readout_crosstalk_kraus(p)
        assert len(kraus) == 2, f"Expected 2 Kraus ops, got {len(kraus)}"
        # readout_crosstalk_kraus returns single-qubit dephasing (2x2)
        assert all(K.shape == (2, 2) for K in kraus), f"Expected 2x2 matrices, got {kraus[0].shape}"
        
        # Check completeness for qubit
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10), f"Kraus not complete for p={p}"
    
    print("  ✓ readout_crosstalk_kraus passed")


def test_cosmic_ray_burst():
    """Test cosmic ray/burst error Kraus operators."""
    print("Testing cosmic_ray_burst_kraus...")
    
    for p_burst in [0.0, 1e-5, 1e-3, 0.1]:
        kraus = cosmic_ray_burst_kraus(p_burst)
        assert len(kraus) == 5, f"Expected 5 Kraus ops (I + IXYZ), got {len(kraus)}"
        # Single qubit version: 2x2 matrices
        assert all(K.shape == (2, 2) for K in kraus), f"Expected 2x2 matrices, got {kraus[0].shape}"
        
        # Check completeness
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10), f"Kraus not complete for p_burst={p_burst}"
    
    print("  ✓ cosmic_ray_burst_kraus passed")


def test_measurement_induced_reset():
    """Test measurement-induced reset Kraus operators."""
    print("Testing measurement_induced_reset_kraus...")
    
    # Test for both measurement outcomes
    test_cases = [
        (0.001, 0.002, 0),   # After measuring 0
        (0.001, 0.002, 1),   # After measuring 1
        (0.0, 0.0, 0),       # Perfect reset
        (0.01, 0.02, 1),     # Higher errors after measuring 1
    ]
    
    for p0, p1, outcome in test_cases:
        kraus = measurement_induced_reset_kraus(p0, p1, outcome)
        assert len(kraus) == 2, f"Expected 2 Kraus ops, got {len(kraus)}"
        # Single qubit: 2x2
        assert all(K.shape == (2, 2) for K in kraus), f"Expected 2x2 matrices"
        
        # Check completeness
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10), f"Kraus not complete for ({p0}, {p1}, {outcome})"
    
    print("  ✓ measurement_induced_reset_kraus passed")


def test_leakage_seepage():
    """Test leakage seepage Kraus operators."""
    print("Testing leakage_seepage_kraus...")
    
    for p in [0.0, 5e-5, 1e-3, 0.1]:
        kraus = leakage_seepage_kraus(p)
        assert len(kraus) == 3, f"Expected 3 Kraus ops, got {len(kraus)}"
        # Two-qutrit system: 9x9 matrices
        assert all(K.shape == (9, 9) for K in kraus), f"Expected 9x9 (2-qutrit) matrices"
        
        # Check completeness for 2-qutrit system
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(9), atol=1e-10), f"Kraus not complete for p={p}"
    
    print("  ✓ leakage_seepage_kraus passed")


def test_state_dependent_amplitude_damping():
    """Test state-dependent amplitude damping Kraus operators."""
    print("Testing state_dependent_amplitude_damping_kraus...")
    
    test_cases = [
        (0.01, 0.02),   # γ₂₁ ≈ 2× γ₁₀
        (0.0, 0.0),     # No decay
        (0.001, 0.002), # Small decay
        (0.1, 0.2),     # Larger decay
    ]
    
    for gamma_10, gamma_21 in test_cases:
        kraus = state_dependent_amplitude_damping_kraus(gamma_10, gamma_21)
        assert len(kraus) == 3, f"Expected 3 Kraus ops, got {len(kraus)}"
        # Qutrit: 3x3
        assert all(K.shape == (3, 3) for K in kraus), f"Expected 3x3 qutrit matrices"
        assert is_valid_kraus_set(kraus, dim=3), f"Kraus not complete for ({gamma_10}, {gamma_21})"
    
    print("  ✓ state_dependent_amplitude_damping_kraus passed")


def test_temporal_noise_scaling():
    """Test temporal noise scaling function."""
    print("Testing temporal_noise_scaling...")
    
    base_error = 0.01
    drift_rate = 0.001
    
    # Test linear drift
    for t in [0, 10, 100, 1000]:
        scaled = temporal_noise_scaling(base_error, drift_rate, t, "linear")
        assert 0.0 <= scaled <= 1.0, f"Scaled error out of bounds: {scaled}"
        if t > 0:
            assert scaled > base_error, f"Linear drift should increase error"
    
    # Test sinusoidal drift
    scaled_sin = temporal_noise_scaling(base_error, 0.5, 15, "sinusoidal")
    assert 0.0 <= scaled_sin <= 1.0, f"Sinusoidal error out of bounds: {scaled_sin}"
    
    # Test random walk
    scaled_rw = temporal_noise_scaling(base_error, 0.01, 100, "random_walk")
    assert 0.0 <= scaled_rw <= 1.0, f"Random walk error out of bounds: {scaled_rw}"
    
    print("  ✓ temporal_noise_scaling passed")


def test_frequency_collision():
    """Test frequency collision Kraus operators."""
    print("Testing frequency_collision_kraus...")
    
    # At zero detuning, maximum error
    kraus_zero = frequency_collision_kraus(0.01, 0.0, 0.01)
    assert len(kraus_zero) == 2, f"Expected 2 Kraus ops, got {len(kraus_zero)}"
    
    # At large detuning, error should be suppressed
    kraus_far = frequency_collision_kraus(0.01, 1.0, 0.01)
    
    # Check completeness
    for kraus in [kraus_zero, kraus_far]:
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10), "Kraus not complete"
    
    print("  ✓ frequency_collision_kraus passed")


def test_spatially_correlated_crosstalk():
    """Test spatially correlated crosstalk."""
    print("Testing spatially_correlated_crosstalk_kraus...")
    
    p_base = 0.01
    
    # Adjacent qubits (distance=1) should have higher crosstalk
    kraus_near = spatially_correlated_crosstalk_kraus(p_base, 1.0, 1.0)
    
    # Far qubits (distance=5) should have lower crosstalk
    kraus_far = spatially_correlated_crosstalk_kraus(p_base, 5.0, 1.0)
    
    # Both should be valid
    for kraus in [kraus_near, kraus_far]:
        assert len(kraus) == 2
        total = sum(K.conj().T @ K for K in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10)
    
    print("  ✓ spatially_correlated_crosstalk_kraus passed")


def test_physical_behavior():
    """Test that channels have physically correct behavior."""
    print("\nTesting physical behavior...")
    
    # 1. Burst errors should increase depolarization
    print("  Testing burst error depolarization...")
    k_no_burst = cosmic_ray_burst_kraus(0.0)
    k_burst = cosmic_ray_burst_kraus(0.1)
    
    # Apply to |0⟩ state and check mixing
    rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    
    def apply_channel(rho, kraus):
        return sum(K @ rho @ K.conj().T for K in kraus)
    
    rho_no_burst = apply_channel(rho_0, k_no_burst)
    rho_burst = apply_channel(rho_0, k_burst)
    
    purity_no_burst = np.trace(rho_no_burst @ rho_no_burst).real
    purity_burst = np.trace(rho_burst @ rho_burst).real
    
    print(f"    Purity without burst: {purity_no_burst:.6f}")
    print(f"    Purity with burst:    {purity_burst:.6f}")
    assert purity_burst <= purity_no_burst + 1e-10, "Burst errors should reduce purity"
    
    # 2. T1 decay should move population from |1⟩ to |0⟩
    print("  Testing T1 decay direction...")
    k_t1 = state_dependent_amplitude_damping_kraus(0.1, 0.2)
    rho_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
    rho_after_t1 = apply_channel(rho_1, k_t1)
    
    pop_0_after = rho_after_t1[0, 0].real
    pop_1_after = rho_after_t1[1, 1].real
    
    print(f"    Starting in |1⟩: P(0)={pop_0_after:.4f}, P(1)={pop_1_after:.4f}")
    assert pop_0_after > 0, "T1 should cause |1⟩→|0⟩ decay"
    
    # 3. Leakage seepage should spread |2⟩ population
    print("  Testing leakage seepage...")
    k_seep = leakage_seepage_kraus(0.5)
    # |20⟩ state
    rho_20 = np.zeros((9, 9), dtype=complex)
    rho_20[6, 6] = 1.0  # Index for |20⟩ = 3*2 + 0 = 6
    rho_after_seep = apply_channel(rho_20, k_seep)
    
    pop_11 = rho_after_seep[4, 4].real  # Index for |11⟩ = 3*1 + 1 = 4
    print(f"    After seepage from |20⟩: P(|11⟩)={pop_11:.4f}")
    assert pop_11 > 0, "Seepage should create |11⟩ population"
    
    print("  ✓ Physical behavior tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Noise Channels from AlphaQubit Paper")
    print("=" * 60)
    print()
    
    test_readout_crosstalk()
    test_cosmic_ray_burst()
    test_measurement_induced_reset()
    test_leakage_seepage()
    test_state_dependent_amplitude_damping()
    test_temporal_noise_scaling()
    test_frequency_collision()
    test_spatially_correlated_crosstalk()
    test_physical_behavior()
    
    print()
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
