"""Test p_ij and XEB estimation."""

from simulator.device_calibration import SpatialErrorMap
from simulator.xeb_calibration import EdgeCalibrationMap, decompose_cz_error
import numpy as np

print("=" * 60)
print("Per-Edge p_ij and XEB Estimation")
print("=" * 60)

# Method 1: SpatialErrorMap (Gaussian distribution)
print("\n1. SpatialErrorMap (Gaussian distribution)")
print("-" * 40)
sm = SpatialErrorMap(distance=5, seed=42)
p_ij_values = []
for edge, data in sm.get_all_edge_errors().items():
    p_ij_values.append(data['cz_error'])

print(f"   Edges: {len(p_ij_values)}")
print(f"   p_ij median: {100*np.median(p_ij_values):.4f}%")
print(f"   p_ij mean:   {100*np.mean(p_ij_values):.4f}%")
print(f"   p_ij std:    {100*np.std(p_ij_values):.4f}%")
print(f"   p_ij range:  [{100*min(p_ij_values):.4f}%, {100*max(p_ij_values):.4f}%]")

# Method 2: EdgeCalibrationMap (log-normal, more realistic)
print("\n2. EdgeCalibrationMap (Log-normal, paper-aligned)")
print("-" * 40)
em = EdgeCalibrationMap(distance=5, seed=42)
stats = em.get_statistics()
print(f"   Edges: {stats['num_edges']}")
print(f"   p_ij median: {100*stats['cz_error_median']:.4f}%")
print(f"   p_ij mean:   {100*stats['cz_error_mean']:.4f}%")
print(f"   p_ij std:    {100*stats['cz_error_std']:.4f}%")
print(f"   p_ij range:  [{100*stats['cz_error_min']:.4f}%, {100*stats['cz_error_max']:.4f}%]")
print(f"   XEB range:   [{100*stats['xeb_min']:.2f}%, {100*stats['xeb_max']:.2f}%]")

# Show sample edges
print("\n3. Sample p_ij values (first 10 edges):")
print("-" * 40)
edges = list(sm.get_all_edge_errors().items())[:10]
for edge, data in edges:
    p_ij = data['cz_error']
    xeb = 1 - p_ij
    print(f"   Edge {edge}: p_ij = {100*p_ij:.4f}%  XEB ≈ {100*xeb:.2f}%")

# Error decomposition
print("\n4. CZ Error Decomposition (Table S4 ratios):")
print("-" * 40)
p_cz = 0.0035  # Paper median
decomp = decompose_cz_error(p_cz)
print(f"   Total CZ error:     {100*decomp['p_cz_total']:.4f}%")
print(f"   - Depolarizing:     {100*decomp['p_cz_depolarizing']:.4f}% (70%)")
print(f"   - Leakage:          {100*decomp['p_cz_leakage']:.4f}% (6%)")
print(f"   - ZZ Crosstalk:     {100*decomp['p_cz_zz_crosstalk']:.4f}% (16%)")
print(f"   - Other:            {100*decomp['p_cz_other']:.4f}% (8%)")

print("\n" + "=" * 60)
print("SUMMARY: How to use p_ij in your code")
print("=" * 60)
print("""
# In simulator/pauli_plus_simulator.py:
from simulator.device_calibration import SpatialErrorMap

spatial_map = SpatialErrorMap(distance=5, seed=42)

# Get p_ij for edge (qubit 0, qubit 1):
p_ij = spatial_map.get_p_ij(0, 1)

# Get XEB fidelity:
xeb_fidelity = 1 - p_ij

# Get all edge errors:
all_edges = spatial_map.get_all_edge_errors()
""")
