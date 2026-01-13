#!/bin/bash
# =============================================================================
# Auto-Sync: S3 → Server
# Run this script on the remote server (in background)
# It watches S3 and pulls changes automatically
# =============================================================================
#
# Usage: 
#   ./auto_sync_from_s3.sh &          # Run in background
#   nohup ./auto_sync_from_s3.sh &    # Run in background (persists after logout)
#
# =============================================================================

S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT"
LOCAL_PATH="/root/work/ALPHAQUBIT"
SYNC_INTERVAL=30  # seconds between sync checks

echo "=============================================="
echo "  Auto-Sync: S3 → Server"
echo "=============================================="
echo "☁️  S3 Source: ${S3_REMOTE}"
echo "📁 Local Path: ${LOCAL_PATH}"
echo "⏱️  Check interval: ${SYNC_INTERVAL}s"
echo "=============================================="
echo ""
echo "Running... (Press Ctrl+C to stop)"
echo ""

# Create local directory if not exists
mkdir -p ${LOCAL_PATH}

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for updates..."
    
    rclone sync ${S3_REMOTE} ${LOCAL_PATH} \
        --exclude "__pycache__/**" \
        --exclude "*.pyc" \
        --exclude ".git/**" \
        --exclude "*.egg-info/**" \
        --exclude "backup_*/**" \
        --quiet
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Sync complete"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Sync failed"
    fi
    
    sleep ${SYNC_INTERVAL}
done
