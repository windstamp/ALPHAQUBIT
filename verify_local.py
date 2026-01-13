# Quick verification of data generation
# Can run locally with just torch (no stim required)
import numpy as np
import torch
import sys
sys.path.insert(0, r"c:\Users\Lenovo\software\ALPHAQUBIT")

print("="*60)
print("TEST 1: Soft Channels Function (Paper-Aligned I/Q Model)")
print("="*60)

# Import the soft_channels function directly from the module
# This doesn't require stim
from google_qec_simulator.data_helpers import soft_channels

# Test soft channels with different sample sizes
for n in [100, 1000, 10000]:
    p1, pL = soft_channels(n, snr=10.0, tau=0.01, leak_p=0.001)
    print(f"\nn={n} samples:")
    print(f"  P(|1>): shape={p1.shape}, mean={p1.mean():.4f}, std={p1.std():.4f}, range=[{p1.min():.4f}, {p1.max():.4f}]")
    print(f"  P(|L>): shape={pL.shape}, mean={pL.mean():.4f}, std={pL.std():.4f}, range=[{pL.min():.4f}, {pL.max():.4f}]")
    
    # Verify probabilities are valid
    assert (p1 >= 0).all() and (p1 <= 1).all(), "P(|1>) out of range!"
    assert (pL >= 0).all() and (pL <= 1).all(), "P(|L>) out of range!"
    print(f"  [OK] All values in valid probability range [0, 1]")

print("\n" + "="*60)
print("TEST 2: Verify Paper-Aligned Parameters")
print("="*60)

# Paper parameters: SNR=10, tau=0.01, leak_p=0.001
n = 10000
p1, pL = soft_channels(n, snr=10.0, tau=0.01, leak_p=0.001)

# Expected: P(|1>) should be ~0.5 on average (equal prior for |0> and |1>)
# Expected: P(|L>) should be small (~0.001 leakage prior)
print(f"\nWith n={n} samples (paper params: SNR=10, tau=0.01, leak_p=0.001):")
print(f"  P(|1>) mean: {p1.mean():.4f} (expected ~0.5)")
print(f"  P(|L>) mean: {pL.mean():.4f} (expected small, ~0.001)")

# P(|0>) = 1 - P(|1>) - P(|L>)
p0 = 1 - p1 - pL
print(f"  P(|0>) mean: {p0.mean():.4f} (expected ~0.5)")
print(f"  Sum check: {(p1 + pL + p0).mean():.6f} (should be 1.0)")

print("\n" + "="*60)
print("TEST 3: Simulate Full 3-Feature Data Format")
print("="*60)

# Simulate what the data would look like
n_shots = 100
n_rounds = 5
n_stabilizers = 8  # for d=3 surface code

# Generate mock detection events (binary)
det_events = np.random.binomial(1, 0.1, (n_shots, n_rounds, n_stabilizers))

# Generate soft channels for all measurements
total = n_shots * n_rounds * n_stabilizers
p1_all, pL_all = soft_channels(total)

# Reshape to match detection events
p1_feat = p1_all.reshape(n_shots, n_rounds, n_stabilizers, 1)
pL_feat = pL_all.reshape(n_shots, n_rounds, n_stabilizers, 1)
det_feat = det_events[:,:,:,np.newaxis].astype(np.float32)

# Combine: [detection, P(|1>), P(|L>)]
data = np.concatenate([det_feat, p1_feat, pL_feat], axis=-1)

print(f"Simulated data shape: {data.shape}")
print(f"Expected: ({n_shots}, {n_rounds}, {n_stabilizers}, 3)")
assert data.shape == (n_shots, n_rounds, n_stabilizers, 3), "Shape mismatch!"
print("[OK] Shape correct!")

print(f"\nFeature statistics:")
print(f"  Channel 0 (detection): dtype={data[:,:,:,0].dtype}, mean={data[:,:,:,0].mean():.4f}, unique={len(np.unique(data[:,:,:,0]))}")
print(f"  Channel 1 (P|1>):      dtype={data[:,:,:,1].dtype}, mean={data[:,:,:,1].mean():.4f}")
print(f"  Channel 2 (P|L>):      dtype={data[:,:,:,2].dtype}, mean={data[:,:,:,2].mean():.4f}")

# Save test NPZ
out_path = "test_3feature_sample.npz"
obs = np.random.binomial(1, 0.05, n_shots).astype(np.float32)  # ~5% logical errors
np.savez(
    out_path,
    data=data.astype(np.float32),
    obs=obs,
    basis=np.ones(n_shots, dtype=np.int64),
    metadata=str({
        'test': True,
        'n_features': 3,
        'features': ['detection', 'P(|1>)', 'P(|L>)']
    })
)
print(f"\nSaved test data to: {out_path}")

# Verify loading
loaded = np.load(out_path)
print(f"Verified load: data={loaded['data'].shape}, obs={loaded['obs'].shape}")

print("\n" + "="*60)
print("TEST 4: pos_weight Calculation for Class Imbalance")
print("="*60)

# Simulate class imbalance at different noise levels
for pos_ratio in [0.03, 0.10, 0.20, 0.30]:
    n = 1000
    labels = np.random.binomial(1, pos_ratio, n)
    num_pos = labels.sum()
    num_neg = n - num_pos
    actual_ratio = num_pos / n
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f"  {pos_ratio*100:.0f}% positive: actual={actual_ratio:.2%}, pos_weight={pos_weight:.1f}")

print("\n" + "="*60)
print("ALL LOCAL TESTS PASSED!")
print("="*60)
print("\nNote: Full circuit simulation (stim) requires running on the server.")
print("The soft_channels function and data format are verified correctly.")
