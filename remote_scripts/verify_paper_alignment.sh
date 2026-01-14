#!/bin/bash
# verify_paper_alignment.sh
# Verify that generated noise data matches paper specifications
#
# Paper Requirements:
# - Pre-training: 8.5M total samples
# - Fine-tuning: 50K samples per experiment
# - 3 soft channels (detection, P(|1⟩), P(|L⟩))
# - Binary observables (logical errors)
# - Shape: (samples, rounds, detectors, 3 channels)

OUTPUT_DIR="${1:-pretrain_data_paper_aligned}"

cd /root/work/ALPHAQUBIT

echo "=============================================="
echo " Paper Alignment Verification"
echo "=============================================="
echo "Checking: $OUTPUT_DIR"
echo ""

python3 - "$OUTPUT_DIR" << 'EOF'
import numpy as np
import json
from pathlib import Path
import sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "pretrain_data_paper_aligned"

# Paper specifications
PAPER_PRETRAIN_SAMPLES = 8_500_000
PAPER_FINETUNE_SAMPLES_PER_EXP = 50_000
PAPER_NUM_CHANNELS = 3  # detection, P(|1⟩), P(|L⟩)

print("=" * 60)
print("PAPER SPECIFICATIONS:")
print(f"  Pre-training samples: {PAPER_PRETRAIN_SAMPLES:,}")
print(f"  Fine-tuning samples/exp: {PAPER_FINETUNE_SAMPLES_PER_EXP:,}")
print(f"  Soft channels: {PAPER_NUM_CHANNELS}")
print("=" * 60)

# Check pretrain data
pretrain_dir = Path(output_dir) / "pretrain"
finetune_dir = Path(output_dir) / "finetune"

results = {"pretrain": {}, "finetune": {}}

for name, data_dir in [("pretrain", pretrain_dir), ("finetune", finetune_dir)]:
    print(f"\n{'='*60}")
    print(f"CHECKING: {name.upper()}")
    print(f"Directory: {data_dir}")
    print("=" * 60)
    
    if not data_dir.exists():
        print(f"  WARNING: {data_dir} does not exist")
        continue
    
    npz_files = list(data_dir.rglob("*.npz"))
    print(f"\nFound {len(npz_files)} .npz files")
    
    total_samples = 0
    samples_by_type = {"d3": 0, "d5": 0, "bX": 0, "bZ": 0}
    shape_info = []
    issues = []
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            keys = list(data.keys())
            
            # Get main data array
            if 'data' in data:
                arr = data['data']
            elif 'syndromes' in data:
                arr = data['syndromes']
            elif 'detection_events' in data:
                arr = data['detection_events']
            else:
                issues.append(f"{npz_path.name}: No data/syndromes key")
                continue
            
            n_samples = arr.shape[0]
            total_samples += n_samples
            
            # Track by experiment type
            fname = npz_path.name
            if "_d3_" in fname:
                samples_by_type["d3"] += n_samples
            if "_d5_" in fname:
                samples_by_type["d5"] += n_samples
            if "_bX_" in fname:
                samples_by_type["bX"] += n_samples
            if "_bZ_" in fname:
                samples_by_type["bZ"] += n_samples
            
            # Check shape
            if arr.ndim == 4:
                n, r, d, c = arr.shape
                if c != PAPER_NUM_CHANNELS:
                    issues.append(f"{npz_path.name}: Expected {PAPER_NUM_CHANNELS} channels, got {c}")
                shape_info.append((n, r, d, c))
            elif arr.ndim == 3:
                # Might be (samples, rounds, detectors) without channel dim
                shape_info.append(arr.shape + (1,))
            
            # Check observables
            if 'obs' in data:
                obs = data['obs']
                if not np.all((obs == 0) | (obs == 1)):
                    issues.append(f"{npz_path.name}: Observables not binary")
            
            data.close()
            
        except Exception as e:
            issues.append(f"{npz_path.name}: Error loading - {e}")
    
    print(f"\nTotal samples: {total_samples:,}")
    
    # Verification
    if name == "pretrain":
        target = PAPER_PRETRAIN_SAMPLES
        pct = (total_samples / target) * 100
        status = "✅ PASS" if 0.9 <= pct <= 1.1 else "⚠️ WARNING" if 0.8 <= pct <= 1.2 else "❌ FAIL"
        print(f"Target: {target:,} ({pct:.1f}% achieved) {status}")
    else:
        files_with_target = sum(1 for _ in npz_files)
        avg_samples = total_samples / len(npz_files) if npz_files else 0
        pct = (avg_samples / PAPER_FINETUNE_SAMPLES_PER_EXP) * 100 if PAPER_FINETUNE_SAMPLES_PER_EXP > 0 else 0
        status = "✅ PASS" if 0.9 <= pct <= 1.1 else "⚠️ WARNING" if 0.8 <= pct <= 1.2 else "❌ FAIL"
        print(f"Avg samples/file: {avg_samples:.0f} (target: {PAPER_FINETUNE_SAMPLES_PER_EXP}, {pct:.1f}%) {status}")
    
    print(f"\nSamples by experiment type:")
    print(f"  d3 (distance 3): {samples_by_type['d3']:,}")
    print(f"  d5 (distance 5): {samples_by_type['d5']:,}")
    print(f"  bX (X-basis): {samples_by_type['bX']:,}")
    print(f"  bZ (Z-basis): {samples_by_type['bZ']:,}")
    
    # Check shape consistency
    if shape_info:
        unique_channels = set(s[3] for s in shape_info)
        print(f"\nChannel counts across files: {unique_channels}")
        if unique_channels == {PAPER_NUM_CHANNELS}:
            print(f"  ✅ All files have {PAPER_NUM_CHANNELS} channels (paper-aligned)")
        else:
            print(f"  ⚠️ Mixed channel counts (expected {PAPER_NUM_CHANNELS})")
    
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

pretrain_samples = results.get("pretrain", {}).get("total_samples", 0)
finetune_samples = results.get("finetune", {}).get("total_samples", 0)
finetune_files = results.get("finetune", {}).get("num_files", 0)

print(f"\nPre-training data:")
print(f"  Generated: {pretrain_samples:,} samples")
print(f"  Target: {PAPER_PRETRAIN_SAMPLES:,} samples")
print(f"  Match: {pretrain_samples/PAPER_PRETRAIN_SAMPLES*100:.1f}%")

if finetune_files > 0:
    avg_ft = finetune_samples / finetune_files
    print(f"\nFine-tuning data:")
    print(f"  Generated: {finetune_samples:,} samples across {finetune_files} files")
    print(f"  Avg per file: {avg_ft:.0f} samples")
    print(f"  Target per file: {PAPER_FINETUNE_SAMPLES_PER_EXP:,} samples")
    print(f"  Match: {avg_ft/PAPER_FINETUNE_SAMPLES_PER_EXP*100:.1f}%")

print("\n" + "=" * 60)
EOF
