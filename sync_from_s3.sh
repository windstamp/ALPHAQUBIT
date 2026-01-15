#!/bin/bash
# AlphaQubit Server Sync Script
# Run this on the server to pull latest code from S3

S3_BUCKET="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/"
LOCAL_DIR="/root/work/ALPHAQUBIT"

echo "[$(date '+%H:%M:%S')] Pulling latest code from S3..."

# Pull specific Python files
for f in train_parallel_models.py train_8npu_simple.py train_full_pipeline.py; do
    rclone copy "${S3_BUCKET}${f}" "$LOCAL_DIR/" --s3-no-check-bucket 2>/dev/null && \
        echo "  Downloaded: $f" || echo "  Not found: $f"
done

# Pull shell scripts
for f in run_training_pipeline.sh; do
    rclone copy "${S3_BUCKET}${f}" "$LOCAL_DIR/" --s3-no-check-bucket 2>/dev/null && \
        echo "  Downloaded: $f" || echo "  Not found: $f"
done

echo "[$(date '+%H:%M:%S')] Sync complete!"

# Show what's running
echo ""
echo "=== Current Training Status ==="
if pgrep -f "train_parallel_models.py" > /dev/null; then
    echo "✓ train_parallel_models.py is RUNNING"
    tail -5 parallel_training.log 2>/dev/null
else
    echo "✗ train_parallel_models.py is NOT running"
fi

if pgrep -f "train_8npu_simple.py" > /dev/null; then
    echo "✓ train_8npu_simple.py is RUNNING"
else
    echo "✗ train_8npu_simple.py is NOT running"
fi
