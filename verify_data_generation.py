#!/usr/bin/env python3
"""
Verify data generation produces correct format with 3 features.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_soft_channels():
    """Test the soft_channels function produces correct output."""
    print("=" * 60)
    print("Testing soft_channels function")
    print("=" * 60)
    
    try:
        from google_qec_simulator.data_helpers import soft_channels
        print("✓ Successfully imported soft_channels")
    except ImportError as e:
        print(f"✗ Failed to import soft_channels: {e}")
        return False
    
    # Generate 1000 samples
    n_samples = 1000
    p1, pL = soft_channels(n_samples, snr=10.0, tau=0.01, leak_p=0.001)
    
    print(f"\nGenerated {n_samples} soft channel samples:")
    print(f"  P(|1⟩) shape: {p1.shape}, dtype: {p1.dtype}")
    print(f"  P(|L⟩) shape: {pL.shape}, dtype: {pL.dtype}")
    
    # Verify shape
    assert p1.shape == (n_samples,), f"Expected shape ({n_samples},), got {p1.shape}"
    assert pL.shape == (n_samples,), f"Expected shape ({n_samples},), got {pL.shape}"
    print("✓ Shapes correct")
    
    # Verify values are probabilities [0, 1]
    assert np.all(p1 >= 0) and np.all(p1 <= 1), "P(|1⟩) values out of range [0,1]"
    assert np.all(pL >= 0) and np.all(pL <= 1), "P(|L⟩) values out of range [0,1]"
    print("✓ Values in valid probability range [0, 1]")
    
    # Check statistics
    print(f"\nStatistics:")
    print(f"  P(|1⟩): mean={p1.mean():.4f}, std={p1.std():.4f}, min={p1.min():.4f}, max={p1.max():.4f}")
    print(f"  P(|L⟩): mean={pL.mean():.4f}, std={pL.std():.4f}, min={pL.min():.4f}, max={pL.max():.4f}")
    
    # Paper-aligned: leakage prior is 0.1%, so P(|L⟩) should be small on average
    # But after Bayesian update, some samples may have higher P(|L⟩)
    print(f"\n  Expected: P(|L⟩) mean should be small (~0.001 prior)")
    print(f"  Expected: P(|1⟩) mean should be ~0.5 (equal prior for |0⟩ and |1⟩)")
    
    return True


def test_si1000_circuit():
    """Test SI1000 circuit generation."""
    print("\n" + "=" * 60)
    print("Testing SI1000 circuit generation")
    print("=" * 60)
    
    try:
        import stim
        print("✓ Successfully imported stim")
    except ImportError as e:
        print(f"✗ Failed to import stim: {e}")
        print("  Install with: pip install stim")
        return False
    
    # Test circuit generation
    d = 3  # distance
    r = 5  # rounds
    p = 0.005  # physical error rate
    
    try:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=r,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=2 * p,
            before_measure_flip_probability=5 * p,
            before_round_data_depolarization=p / 10,
        )
        print(f"✓ Generated SI1000 circuit: d={d}, r={r}, p={p}")
        print(f"  Circuit has {circuit.num_detectors} detectors")
        print(f"  Circuit has {circuit.num_observables} observables")
    except Exception as e:
        print(f"✗ Failed to generate circuit: {e}")
        return False
    
    # Sample from circuit
    n_shots = 100
    sampler = circuit.compile_detector_sampler()
    detection_events, observables = sampler.sample(n_shots, separate_observables=True)
    
    print(f"\nSampled {n_shots} shots:")
    print(f"  Detection events shape: {detection_events.shape}")
    print(f"  Observables shape: {observables.shape}")
    print(f"  Detection rate: {detection_events.mean():.4f}")
    print(f"  Logical error rate: {observables.mean():.4f}")
    
    return True


def test_full_data_pipeline():
    """Test full data generation pipeline with 3 features."""
    print("\n" + "=" * 60)
    print("Testing full data pipeline (3 features)")
    print("=" * 60)
    
    try:
        import stim
        from google_qec_simulator.data_helpers import soft_channels
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Generate circuit and sample
    d = 3
    r = 5
    p = 0.005
    n_shots = 100
    
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=2 * p,
        before_measure_flip_probability=5 * p,
        before_round_data_depolarization=p / 10,
    )
    
    sampler = circuit.compile_detector_sampler()
    det_events, obs = sampler.sample(n_shots, separate_observables=True)
    
    # Reshape detection events
    n_detectors = circuit.num_detectors
    n_stab_per_round = n_detectors // r
    
    print(f"Circuit info:")
    print(f"  Total detectors: {n_detectors}")
    print(f"  Rounds: {r}")
    print(f"  Stabilizers per round: {n_stab_per_round}")
    
    # Reshape to (N, R, S)
    det_reshaped = det_events.reshape(n_shots, r, n_stab_per_round)
    print(f"\nReshaped detection events: {det_reshaped.shape}")
    
    # Add soft channels
    total_elements = n_shots * r * n_stab_per_round
    p1_post, pL_post = soft_channels(total_elements, snr=10.0, tau=0.01, leak_p=0.001)
    
    p1_feat = p1_post.reshape(n_shots, r, n_stab_per_round, 1)
    pL_feat = pL_post.reshape(n_shots, r, n_stab_per_round, 1)
    det_feat = det_reshaped[:, :, :, np.newaxis].astype(np.float32)
    
    # Combine features: [detection, P(|1⟩), P(|L⟩)]
    data = np.concatenate([det_feat, p1_feat, pL_feat], axis=-1)
    
    print(f"\nFinal data shape: {data.shape}")
    print(f"  Expected: ({n_shots}, {r}, {n_stab_per_round}, 3)")
    
    # Verify
    assert data.shape == (n_shots, r, n_stab_per_round, 3), \
        f"Shape mismatch! Expected ({n_shots}, {r}, {n_stab_per_round}, 3), got {data.shape}"
    print("✓ Shape correct: (shots, rounds, stabilizers, 3 features)")
    
    # Check feature channels
    print(f"\nFeature statistics:")
    print(f"  Channel 0 (detection): mean={data[:,:,:,0].mean():.4f}, unique values={len(np.unique(data[:,:,:,0]))}")
    print(f"  Channel 1 (P|1⟩):      mean={data[:,:,:,1].mean():.4f}")
    print(f"  Channel 2 (P|L⟩):      mean={data[:,:,:,2].mean():.4f}")
    
    # Save sample data
    output_path = Path("test_data_sample.npz")
    np.savez(
        output_path,
        data=data,
        obs=obs.flatten().astype(np.float32),
        basis=np.ones(n_shots, dtype=np.int64),  # Z basis
        metadata=str({
            'experiment_name': 'test_d3_r5',
            'distance': d,
            'rounds': r,
            'p': p,
            'n_shots': n_shots,
            'n_features': 3
        })
    )
    print(f"\n✓ Saved test data to: {output_path}")
    
    # Verify saved file
    loaded = np.load(output_path)
    print(f"\nVerifying saved file:")
    print(f"  data shape: {loaded['data'].shape}")
    print(f"  obs shape: {loaded['obs'].shape}")
    print(f"  basis shape: {loaded['basis'].shape}")
    
    return True


def test_class_distribution():
    """Test class distribution at different noise levels."""
    print("\n" + "=" * 60)
    print("Testing class distribution at different noise levels")
    print("=" * 60)
    
    try:
        import stim
    except ImportError:
        print("✗ stim not available")
        return False
    
    d = 3
    r = 5
    n_shots = 1000
    
    print(f"\nCode distance: {d}, Rounds: {r}, Shots: {n_shots}")
    print("-" * 50)
    
    for p in [0.001, 0.005, 0.01]:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=r,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=2 * p,
            before_measure_flip_probability=5 * p,
            before_round_data_depolarization=p / 10,
        )
        
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(n_shots, separate_observables=True)
        
        pos_ratio = obs.mean()
        neg_ratio = 1 - pos_ratio
        pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
        
        print(f"  p={p:.3f}: positive={pos_ratio:.2%}, negative={neg_ratio:.2%}, pos_weight={pos_weight:.1f}")
    
    print("\nNote: pos_weight is used in BCEWithLogitsLoss to handle class imbalance")
    return True


if __name__ == "__main__":
    print("AlphaQubit Data Generation Verification")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("soft_channels", test_soft_channels()))
    results.append(("SI1000 circuit", test_si1000_circuit()))
    results.append(("Full pipeline", test_full_data_pipeline()))
    results.append(("Class distribution", test_class_distribution()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Data generation is working correctly.")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
    
    sys.exit(0 if all_passed else 1)
