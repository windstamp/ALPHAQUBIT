#!/bin/bash
# inspect_noise_data.sh
# Detailed inspection of generated noise data to verify correctness
# Run on remote server

OUTPUT_DIR="${1:-test_output/test_pretrain}"
NUM_FILES="${2:-3}"

cd /root/work/ALPHAQUBIT

echo "=============================================="
echo " ALPHAQUBIT - Noise Data Inspection"
echo "=============================================="
echo "Checking: $OUTPUT_DIR"
echo ""

python3 << 'EOF'
import numpy as np
import os
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else "test_output/test_pretrain"
num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print(f"Inspecting noise data in: {output_dir}")
print("=" * 60)

# Find all .npz files
npz_files = sorted(Path(output_dir).rglob("*.npz"))
print(f"\nFound {len(npz_files)} .npz files")

if not npz_files:
    print("No .npz files found!")
    sys.exit(1)

# Inspect first N files in detail
for i, npz_path in enumerate(npz_files[:num_files]):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{min(num_files, len(npz_files))}] File: {npz_path.name}")
    print(f"    Path: {npz_path}")
    print(f"    Size: {npz_path.stat().st_size / 1024:.1f} KB")
    print("-" * 60)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"Arrays: {keys}")
        
        for key in keys:
            arr = data[key]
            print(f"\n  [{key}]")
            print(f"    Shape: {arr.shape}")
            print(f"    Dtype: {arr.dtype}")
            
            if arr.size == 0:
                print("    (empty array)")
                continue
            
            # Statistics
            if np.issubdtype(arr.dtype, np.number):
                print(f"    Min: {arr.min():.6f}")
                print(f"    Max: {arr.max():.6f}")
                print(f"    Mean: {arr.mean():.6f}")
                print(f"    Std: {arr.std():.6f}")
                
                # Check for NaN/Inf
                nan_count = np.isnan(arr).sum()
                inf_count = np.isinf(arr).sum()
                if nan_count > 0:
                    print(f"    WARNING: {nan_count} NaN values!")
                if inf_count > 0:
                    print(f"    WARNING: {inf_count} Inf values!")
                
                # Unique values (for binary/categorical data)
                unique = np.unique(arr)
                if len(unique) <= 10:
                    print(f"    Unique values: {unique}")
                else:
                    print(f"    Unique values: {len(unique)} distinct values")
            
            # Print sample data
            print(f"\n    Sample data (first 5 elements along each axis):")
            if arr.ndim == 1:
                print(f"    {arr[:10]}")
            elif arr.ndim == 2:
                print(f"    Shape: ({arr.shape[0]} samples, {arr.shape[1]} features)")
                for j in range(min(3, arr.shape[0])):
                    print(f"    Sample {j}: {arr[j, :min(10, arr.shape[1])]}...")
            elif arr.ndim == 3:
                print(f"    Shape: ({arr.shape[0]} samples, {arr.shape[1]} rounds, {arr.shape[2]} detectors)")
                # Print first sample, first few rounds
                for j in range(min(2, arr.shape[0])):
                    print(f"    Sample {j}, Round 0: {arr[j, 0, :min(8, arr.shape[2])]}...")
                    if arr.shape[1] > 1:
                        print(f"    Sample {j}, Round 1: {arr[j, 1, :min(8, arr.shape[2])]}...")
            elif arr.ndim == 4:
                print(f"    Shape: ({arr.shape[0]} samples, {arr.shape[1]} channels, {arr.shape[2]} rounds, {arr.shape[3]} detectors)")
                # Print first sample
                for ch in range(min(3, arr.shape[1])):
                    print(f"    Sample 0, Channel {ch}, Round 0: {arr[0, ch, 0, :min(6, arr.shape[3])]}...")
        
        data.close()
        
        # Validation checks
        print(f"\n  Validation:")
        data = np.load(npz_path)
        
        # Check syndrome data
        if 'syndromes' in data or 'detection_events' in data:
            syn_key = 'syndromes' if 'syndromes' in data else 'detection_events'
            syn = data[syn_key]
            binary_check = np.all((syn == 0) | (syn == 1))
            print(f"    ✓ {syn_key} binary (0/1): {binary_check}")
            if not binary_check and syn.dtype in [np.float32, np.float64]:
                in_range = np.all((syn >= 0) & (syn <= 1))
                print(f"    ✓ {syn_key} in [0,1] range: {in_range}")
        
        # Check logical data
        if 'logicals' in data or 'observable_flips' in data:
            log_key = 'logicals' if 'logicals' in data else 'observable_flips'
            log = data[log_key]
            binary_check = np.all((log == 0) | (log == 1))
            print(f"    ✓ {log_key} binary (0/1): {binary_check}")
        
        # Check soft readout data (3 channels: I, Q, leak)
        if 'soft_readout' in data:
            soft = data['soft_readout']
            if soft.ndim >= 2 and soft.shape[1] == 3:
                print(f"    ✓ soft_readout has 3 channels (I, Q, leak): True")
            else:
                print(f"    ⚠ soft_readout shape: {soft.shape}")
        
        data.close()
        
    except Exception as e:
        print(f"  ERROR loading file: {e}")

# Summary statistics across all files
print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL FILES")
print("=" * 60)

total_samples = 0
all_shapes = {}

for npz_path in npz_files:
    try:
        data = np.load(npz_path)
        for key in data.keys():
            shape = data[key].shape
            if key not in all_shapes:
                all_shapes[key] = []
            all_shapes[key].append(shape)
            if key in ['syndromes', 'detection_events']:
                total_samples += shape[0]
        data.close()
    except:
        pass

print(f"\nTotal samples across all files: {total_samples}")
print(f"\nArray shapes by key:")
for key, shapes in all_shapes.items():
    unique_shapes = list(set(shapes))
    print(f"  {key}: {unique_shapes}")

print("\n" + "=" * 60)
print("Inspection complete!")
EOF "$OUTPUT_DIR" "$NUM_FILES"
