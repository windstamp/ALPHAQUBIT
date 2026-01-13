#!/usr/bin/env python3
"""Quick end-to-end test for the noise model."""

import numpy as np
import os
from my_noise_model import (
    DeviceCalibration,
    RealisticNoiseModel,
    RealisticDataGenerator,
    generate_random_calibration,
)

print("="*60)
print("END-TO-END TEST: Generate Training Data")
print("="*60)

# 1. Load real calibration
print("\n1. Loading calibration from experiment data...")
cal = DeviceCalibration.from_json_file("configs/realistic_calibration_d5.json")
print(f"   Loaded: {cal.num_qubits} qubits, {len(cal.edge_calibrations)} edges")

# 2. Generate training dataset
print("\n2. Generating training samples...")
generator = RealisticDataGenerator(
    calibration=cal,
    distance=5,
    rounds=25,
)
dataset = generator.generate_samples(num_samples=1000, basis="z")
print(f"   Generated: {dataset.detection_events.shape} detection events")
print(f"   Generated: {dataset.observables.shape} observables")

# 3. Compute statistics
print("\n3. Dataset statistics:")
syndrome_density = dataset.detection_events.mean()
error_rate = dataset.observables.mean()
print(f"   Syndrome density: {syndrome_density:.4f}")
print(f"   Logical error rate: {error_rate:.4f}")

# 4. Save and reload
print("\n4. Testing save/load...")
generator.save_to_npz("test_e2e_data.npz", dataset)
loaded = np.load("test_e2e_data.npz")
det_shape = loaded["detection_events"].shape
print(f"   Saved and loaded: {det_shape}")
loaded.close()  # Close the file before deleting
try:
    os.remove("test_e2e_data.npz")
except:
    pass  # Ignore cleanup errors on Windows

# 5. Test with random calibration (for pretraining diversity)
print("\n5. Testing random calibration generation...")
for seed in [42, 123, 456]:
    cal_rand = generate_random_calibration(distance=5, seed=seed)
    model = RealisticNoiseModel(calibration=cal_rand, distance=5, rounds=10)
    model.build_noisy_circuit(basis="z")
    _, obs = model.sample(100)
    print(f"   Seed {seed}: error rate = {obs.mean():.4f}")

print("\n" + "="*60)
print("END-TO-END TEST PASSED!")
print("="*60)
