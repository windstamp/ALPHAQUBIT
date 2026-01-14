#!/bin/bash
# check_generated_data.sh
# Check and analyze generated noise data
# Run on remote server

WORK_DIR="/root/work/ALPHAQUBIT"
OUTPUT_DIR="${1:-pretrain_data}"

cd "$WORK_DIR"

echo "=============================================="
echo " ALPHAQUBIT - Generated Data Analysis"
echo "=============================================="
echo "Checking: $OUTPUT_DIR"
echo ""

# 1. Directory structure
echo "=== Directory Structure ==="
if [ -d "$OUTPUT_DIR" ]; then
    tree -L 3 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type d | head -30
else
    echo "ERROR: $OUTPUT_DIR does not exist"
    exit 1
fi

# 2. File counts
echo ""
echo "=== File Counts ==="
NPZ_COUNT=$(find "$OUTPUT_DIR" -name "*.npz" | wc -l)
NPY_COUNT=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)
echo "Total .npz files: $NPZ_COUNT"
echo "Total .npy files: $NPY_COUNT"

# 3. File sizes
echo ""
echo "=== Total Size ==="
du -sh "$OUTPUT_DIR"
du -sh "$OUTPUT_DIR"/* 2>/dev/null || true

# 4. Breakdown by experiment type
echo ""
echo "=== Experiments by Type ==="
echo "surface_code_bX_d3: $(find "$OUTPUT_DIR" -path "*bX_d3*" -name "*.npz" | wc -l) files"
echo "surface_code_bX_d5: $(find "$OUTPUT_DIR" -path "*bX_d5*" -name "*.npz" | wc -l) files"
echo "surface_code_bZ_d3: $(find "$OUTPUT_DIR" -path "*bZ_d3*" -name "*.npz" | wc -l) files"
echo "surface_code_bZ_d5: $(find "$OUTPUT_DIR" -path "*bZ_d5*" -name "*.npz" | wc -l) files"

# 5. Sample files
echo ""
echo "=== Sample Files (first 10) ==="
find "$OUTPUT_DIR" -name "*.npz" | head -10

# 6. Manifest summary
echo ""
echo "=== Manifest Summary ==="
if [ -f "$OUTPUT_DIR/MANIFEST.json" ]; then
    echo "Manifest exists: $OUTPUT_DIR/MANIFEST.json"
    echo "Created at: $(cat "$OUTPUT_DIR/MANIFEST.json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('created_at','N/A'))")"
    echo "Total items: $(cat "$OUTPUT_DIR/MANIFEST.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('items',[])))")"
else
    echo "No MANIFEST.json found"
fi

# 7. Inspect a sample .npz file
echo ""
echo "=== Sample .npz File Contents ==="
SAMPLE_FILE=$(find "$OUTPUT_DIR" -name "*.npz" | head -1)
if [ -n "$SAMPLE_FILE" ]; then
    echo "Inspecting: $SAMPLE_FILE"
    python3 << EOF
import numpy as np
import os

f = np.load("$SAMPLE_FILE", allow_pickle=True)
print(f"Arrays in file: {list(f.keys())}")
for key in f.keys():
    arr = f[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim > 0 and arr.size > 0:
        print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
f.close()
EOF
else
    echo "No .npz files found to inspect"
fi

# 8. Check for any errors in log
echo ""
echo "=== Recent Logs ==="
if ls pipeline_*.log 1>/dev/null 2>&1; then
    LATEST_LOG=$(ls -t pipeline_*.log | head -1)
    echo "Latest log: $LATEST_LOG"
    echo "Last 20 lines:"
    tail -20 "$LATEST_LOG"
else
    echo "No pipeline logs found"
fi

echo ""
echo "=============================================="
echo " Analysis Complete"
echo "=============================================="
