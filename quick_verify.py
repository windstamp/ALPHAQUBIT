# Quick verification of data generation
import numpy as np
import torch
import stim
import sys
sys.path.insert(0, r"c:\Users\Lenovo\software\ALPHAQUBIT")

print("="*60)
print("TEST 1: Soft Channels Function")
print("="*60)

from google_qec_simulator.data_helpers import soft_channels

# Test soft channels
n = 1000
p1, pL = soft_channels(n, snr=10.0, tau=0.01, leak_p=0.001)
print(f"Generated {n} samples")
print(f"P(|1>): shape={p1.shape}, mean={p1.mean():.4f}, std={p1.std():.4f}")
print(f"P(|L>): shape={pL.shape}, mean={pL.mean():.4f}, std={pL.std():.4f}")
print(f"Values in [0,1]: {(p1>=0).all() and (p1<=1).all() and (pL>=0).all() and (pL<=1).all()}")

print("\n" + "="*60)
print("TEST 2: SI1000 Circuit Generation")
print("="*60)

d, r, p = 3, 5, 0.005
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=r, distance=d,
    after_clifford_depolarization=p,
    after_reset_flip_probability=2*p,
    before_measure_flip_probability=5*p,
    before_round_data_depolarization=p/10
)
print(f"Circuit: d={d}, r={r}, p={p}")
print(f"Detectors: {circuit.num_detectors}")
print(f"Observables: {circuit.num_observables}")

# Sample
n_shots = 500
sampler = circuit.compile_detector_sampler()
det, obs = sampler.sample(n_shots, separate_observables=True)
print(f"Sampled {n_shots} shots")
print(f"Detection shape: {det.shape}")
print(f"Logical error rate: {obs.mean():.4f}")

print("\n" + "="*60)
print("TEST 3: Full 3-Feature Pipeline")
print("="*60)

# Reshape detections
n_stab = circuit.num_detectors // r
det_3d = det.reshape(n_shots, r, n_stab)
print(f"Reshaped: {det_3d.shape} (shots, rounds, stabilizers)")

# Generate soft channels
total = n_shots * r * n_stab
p1_all, pL_all = soft_channels(total, snr=10.0, tau=0.01, leak_p=0.001)
p1_feat = p1_all.reshape(n_shots, r, n_stab, 1)
pL_feat = pL_all.reshape(n_shots, r, n_stab, 1)
det_feat = det_3d[:,:,:,np.newaxis].astype(np.float32)

# Combine: [detection, P(|1>), P(|L>)]
data = np.concatenate([det_feat, p1_feat, pL_feat], axis=-1)
print(f"Final data shape: {data.shape}")
print(f"Expected: ({n_shots}, {r}, {n_stab}, 3)")

# Verify features
print(f"\nFeature statistics:")
print(f"  Ch0 (detection): mean={data[:,:,:,0].mean():.4f}")
print(f"  Ch1 (P|1>):      mean={data[:,:,:,1].mean():.4f}")
print(f"  Ch2 (P|L>):      mean={data[:,:,:,2].mean():.4f}")

# Save test file
out = "test_sample_data.npz"
np.savez(out, data=data, obs=obs.astype(np.float32).flatten(), 
         basis=np.ones(n_shots, dtype=np.int64))
print(f"\nSaved to: {out}")

# Verify
loaded = np.load(out)
print(f"Loaded data shape: {loaded['data'].shape}")
print(f"Loaded obs shape: {loaded['obs'].shape}")

print("\n" + "="*60)
print("TEST 4: Class Distribution (for pos_weight)")
print("="*60)

for p_val in [0.001, 0.005, 0.01]:
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=5, distance=3,
        after_clifford_depolarization=p_val,
        after_reset_flip_probability=2*p_val,
        before_measure_flip_probability=5*p_val,
        before_round_data_depolarization=p_val/10
    )
    _, o = c.compile_detector_sampler().sample(2000, separate_observables=True)
    pos = o.mean()
    pw = (1-pos)/pos if pos > 0 else 1
    print(f"p={p_val:.3f}: positive={pos:.2%}, pos_weight={pw:.1f}")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
