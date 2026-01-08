"""Test the corrected leakage Kraus operators."""
import numpy as np
from my_noise_model.channels import cz_induced_leakage_kraus, leakage_transport_kraus

print("=" * 60)
print("Testing Corrected Leakage Kraus Operators (Qutrit Model)")
print("=" * 60)

# Test 1: CZ-induced leakage
print("\n1. CZ-Induced Leakage (|11⟩ → |02⟩/|20⟩)")
K_cz = cz_induced_leakage_kraus(0.01)
print(f"   Number of Kraus operators: {len(K_cz)}")
print(f"   Shape: {K_cz[0].shape}")
assert K_cz[0].shape == (9, 9), f"Expected (9,9) but got {K_cz[0].shape}"

# CPTP check
S_cz = sum(k.conj().T @ k for k in K_cz)
cptp_cz = np.allclose(S_cz, np.eye(9))
print(f"   CPTP (Σ K†K = I): {cptp_cz}")
assert cptp_cz, "CZ leakage channel is not CPTP!"

# Test 2: Leakage transport
print("\n2. Leakage Transport (|12⟩ ↔ |21⟩)")
K_trans = leakage_transport_kraus(0.01)
print(f"   Number of Kraus operators: {len(K_trans)}")
print(f"   Shape: {K_trans[0].shape}")
assert K_trans[0].shape == (9, 9), f"Expected (9,9) but got {K_trans[0].shape}"

# CPTP check
S_trans = sum(k.conj().T @ k for k in K_trans)
cptp_trans = np.allclose(S_trans, np.eye(9))
print(f"   CPTP (Σ K†K = I): {cptp_trans}")
assert cptp_trans, "Leakage transport channel is not CPTP!"

# Test 3: Verify transition probabilities
print("\n3. Transition Probability Check")
p_leak = 0.02
K = cz_induced_leakage_kraus(p_leak)

# Apply to |11⟩ state
rho_11 = np.zeros((9, 9), complex)
idx = lambda i, j: 3 * i + j
rho_11[idx(1,1), idx(1,1)] = 1.0

# Apply channel
rho_out = sum(k @ rho_11 @ k.conj().T for k in K)

print(f"   Input: |11⟩")
print(f"   P(|11⟩): {np.real(rho_out[idx(1,1), idx(1,1)]):.6f} (expected {1-p_leak:.6f})")
print(f"   P(|02⟩): {np.real(rho_out[idx(0,2), idx(0,2)]):.6f} (expected {p_leak/2:.6f})")
print(f"   P(|20⟩): {np.real(rho_out[idx(2,0), idx(2,0)]):.6f} (expected {p_leak/2:.6f})")

assert np.isclose(rho_out[idx(1,1), idx(1,1)], 1-p_leak), "Wrong survival probability"
assert np.isclose(rho_out[idx(0,2), idx(0,2)], p_leak/2), "Wrong |02⟩ probability"
assert np.isclose(rho_out[idx(2,0), idx(2,0)], p_leak/2), "Wrong |20⟩ probability"

print("\n" + "=" * 60)
print("✅ All tests passed! Leakage model now uses 3-level (qutrit) system.")
print("=" * 60)
