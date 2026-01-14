#!/usr/bin/env python3
"""
verify_paper_alignment.py
Verify that generated noise data matches paper specifications.

Paper Requirements:
- Pre-training: 8.5M total samples
- Fine-tuning: 50K samples per experiment
- 3 soft channels (detection, P(|1⟩), P(|L⟩))
- Binary observables (logical errors)
- Shape: (samples, rounds, detectors, 3 channels)

Usage:
    python verify_paper_alignment.py test_output/test_pretrain
    python verify_paper_alignment.py pretrain_data_paper_aligned
"""

import numpy as np
import json
import sys
from pathlib import Path

# Paper specifications
PAPER_PRETRAIN_SAMPLES = 8_500_000
PAPER_FINETUNE_SAMPLES_PER_EXP = 50_000
PAPER_NUM_CHANNELS = 3  # detection, P(|1⟩), P(|L⟩)


def verify_data(output_dir: str):
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("PAPER SPECIFICATIONS:")
    print(f"  Pre-training samples: {PAPER_PRETRAIN_SAMPLES:,}")
    print(f"  Fine-tuning samples/exp: {PAPER_FINETUNE_SAMPLES_PER_EXP:,}")
    print(f"  Soft channels: {PAPER_NUM_CHANNELS}")
    print("=" * 60)

    # Check if it's a pretrain/finetune structure or flat experiments
    pretrain_dir = output_dir / "pretrain"
    finetune_dir = output_dir / "finetune"
    experiments_dir = output_dir / "experiments"
    
    # Determine structure
    if pretrain_dir.exists() or finetune_dir.exists():
        dirs_to_check = [("pretrain", pretrain_dir), ("finetune", finetune_dir)]
    elif experiments_dir.exists():
        dirs_to_check = [("experiments", experiments_dir)]
    else:
        # Flat structure - check output_dir directly
        dirs_to_check = [("data", output_dir)]

    results = {}

    for name, data_dir in dirs_to_check:
        print(f"\n{'='*60}")
        print(f"CHECKING: {name.upper()}")
        print(f"Directory: {data_dir}")
        print("=" * 60)
        
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} does not exist")
            continue
        
        npz_files = list(data_dir.rglob("*.npz"))
        print(f"\nFound {len(npz_files)} .npz files")
        
        if not npz_files:
            print("  No .npz files found")
            continue
        
        total_samples = 0
        samples_by_type = {"d3": 0, "d5": 0, "d7": 0, "bX": 0, "bZ": 0}
        shape_info = []
        issues = []
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                keys = list(data.keys())
                
                # Get main data array
                arr = None
                for key in ['data', 'syndromes', 'detection_events']:
                    if key in data:
                        arr = data[key]
                        break
                
                if arr is None:
                    issues.append(f"{npz_path.name}: No data/syndromes key (keys: {keys})")
                    data.close()
                    continue
                
                n_samples = arr.shape[0]
                total_samples += n_samples
                
                # Track by experiment type (check full path)
                full_path = str(npz_path)
                if "_d3_" in full_path or "/d3/" in full_path:
                    samples_by_type["d3"] += n_samples
                if "_d5_" in full_path or "/d5/" in full_path:
                    samples_by_type["d5"] += n_samples
                if "_d7_" in full_path or "/d7/" in full_path:
                    samples_by_type["d7"] += n_samples
                if "_bX_" in full_path or "bX" in full_path:
                    samples_by_type["bX"] += n_samples
                if "_bZ_" in full_path or "bZ" in full_path:
                    samples_by_type["bZ"] += n_samples
                
                # Check shape
                if arr.ndim == 4:
                    n, r, d, c = arr.shape
                    if c != PAPER_NUM_CHANNELS:
                        issues.append(f"{npz_path.name}: Expected {PAPER_NUM_CHANNELS} channels, got {c}")
                    shape_info.append((n, r, d, c))
                elif arr.ndim == 3:
                    shape_info.append(arr.shape + (1,))
                else:
                    shape_info.append(arr.shape)
                
                # Check observables
                if 'obs' in data:
                    obs = data['obs']
                    if not np.all((obs == 0) | (obs == 1)):
                        issues.append(f"{npz_path.name}: Observables not binary")
                
                data.close()
                
            except Exception as e:
                issues.append(f"{npz_path.name}: Error loading - {e}")
        
        print(f"\nTotal samples: {total_samples:,}")
        
        # Verification against paper targets
        if name == "pretrain":
            target = PAPER_PRETRAIN_SAMPLES
            if target > 0:
                pct = (total_samples / target) * 100
                status = "✅ PASS" if 0.9 <= (pct/100) <= 1.1 else "⚠️ WARNING" if 0.8 <= (pct/100) <= 1.2 else "❌ FAIL"
                print(f"Target: {target:,} ({pct:.1f}% achieved) {status}")
        elif name == "finetune" and npz_files:
            avg_samples = total_samples / len(npz_files)
            pct = (avg_samples / PAPER_FINETUNE_SAMPLES_PER_EXP) * 100 if PAPER_FINETUNE_SAMPLES_PER_EXP > 0 else 0
            status = "✅ PASS" if 0.9 <= (pct/100) <= 1.1 else "⚠️ WARNING" if 0.8 <= (pct/100) <= 1.2 else "❌ FAIL"
            print(f"Avg samples/file: {avg_samples:.0f} (target: {PAPER_FINETUNE_SAMPLES_PER_EXP:,}, {pct:.1f}%) {status}")
        else:
            # For experiments or flat structure
            if npz_files:
                avg_samples = total_samples / len(npz_files)
                print(f"Avg samples/file: {avg_samples:.0f}")
                print(f"Files: {len(npz_files)}")
        
        print(f"\nSamples by experiment type:")
        print(f"  d3 (distance 3): {samples_by_type['d3']:,}")
        print(f"  d5 (distance 5): {samples_by_type['d5']:,}")
        print(f"  d7 (distance 7): {samples_by_type['d7']:,}")
        print(f"  bX (X-basis): {samples_by_type['bX']:,}")
        print(f"  bZ (Z-basis): {samples_by_type['bZ']:,}")
        
        # Check shape consistency
        if shape_info:
            unique_channels = set(s[-1] if len(s) >= 4 else 1 for s in shape_info)
            print(f"\nChannel counts across files: {unique_channels}")
            if unique_channels == {PAPER_NUM_CHANNELS}:
                print(f"  ✅ All files have {PAPER_NUM_CHANNELS} channels (paper-aligned)")
            elif PAPER_NUM_CHANNELS in unique_channels:
                print(f"  ⚠️ Mixed channel counts (expected {PAPER_NUM_CHANNELS})")
            else:
                print(f"  ℹ️ Channel count: {unique_channels}")
        
        # Show sample shapes
        if shape_info:
            unique_shapes = list(set(shape_info))[:5]
            print(f"\nSample shapes (first 5 unique): {unique_shapes}")
        
        if issues:
            print(f"\n⚠️ Issues found ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        
        results[name] = {
            "total_samples": total_samples,
            "num_files": len(npz_files),
            "samples_by_type": samples_by_type,
            "issues": len(issues)
        }

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)

    total_all = sum(r.get("total_samples", 0) for r in results.values())
    total_files = sum(r.get("num_files", 0) for r in results.values())
    
    print(f"\nTotal across all directories:")
    print(f"  Samples: {total_all:,}")
    print(f"  Files: {total_files}")
    
    if total_files > 0:
        avg = total_all / total_files
        print(f"  Avg samples/file: {avg:.0f}")
    
    # Paper alignment check
    print(f"\nPaper alignment:")
    print(f"  Target pre-training: {PAPER_PRETRAIN_SAMPLES:,}")
    print(f"  Generated: {total_all:,}")
    if PAPER_PRETRAIN_SAMPLES > 0:
        pct = (total_all / PAPER_PRETRAIN_SAMPLES) * 100
        print(f"  Coverage: {pct:.2f}%")
        
        if pct < 100:
            needed = PAPER_PRETRAIN_SAMPLES - total_all
            samples_per_exp = needed // max(total_files, 1) if total_files > 0 else needed
            print(f"\n  To reach 8.5M, need {needed:,} more samples")
            print(f"  With {total_files} experiments: ~{samples_per_exp:,} more per experiment")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_paper_alignment.py <output_dir>")
        print("Example: python verify_paper_alignment.py test_output/test_pretrain")
        sys.exit(1)
    
    verify_data(sys.argv[1])
