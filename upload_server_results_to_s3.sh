#!/bin/bash
# =============================================================================
# Upload Server Results to S3 (Run on Remote Server)
# =============================================================================
#
# Usage:
#   bash upload_server_results_to_s3.sh                    # Upload all results
#   bash upload_server_results_to_s3.sh --list             # List local results
#   bash upload_server_results_to_s3.sh --include-models   # Include model weights
#
# =============================================================================

set -e

# Configuration - MODIFY THESE AS NEEDED
S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
PROJECT_NAME="ALPHAQUBIT"
SERVER_RESULTS_DIR="${HOME}/ALPHAQUBIT/server_results"
S3_RESULTS_PATH="${S3_REMOTE}/${PROJECT_NAME}/server_results"

# Parse arguments
INCLUDE_MODELS=false
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-models)
            INCLUDE_MODELS=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--list] [--include-models]"
            echo ""
            echo "Options:"
            echo "  --list            List local results without uploading"
            echo "  --include-models  Include model weights (.pth files)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Upload Server Results to S3"
echo "=============================================="
echo "Source: ${SERVER_RESULTS_DIR}"
echo "S3 Destination: ${S3_RESULTS_PATH}"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "❌ rclone not found! Install with: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

# Check if results directory exists
if [ ! -d "${SERVER_RESULTS_DIR}" ]; then
    echo "❌ Results directory not found: ${SERVER_RESULTS_DIR}"
    echo ""
    echo "Looking for results in common locations..."
    
    # Search for results
    for dir in "${HOME}/ALPHAQUBIT" "${HOME}/alphaqubit" "/workspace/ALPHAQUBIT" "."; do
        if [ -d "${dir}" ]; then
            echo "  Checking ${dir}..."
            find "${dir}" -maxdepth 3 -type d -name "*results*" 2>/dev/null || true
            find "${dir}" -maxdepth 3 -type f -name "*.json" 2>/dev/null | head -5 || true
        fi
    done
    exit 1
fi

# List mode
if [ "$LIST_ONLY" = true ]; then
    echo "📋 Local Results:"
    echo "----------------------------------------"
    ls -la "${SERVER_RESULTS_DIR}"
    echo ""
    echo "📊 Summary:"
    echo "  JSON files: $(find ${SERVER_RESULTS_DIR} -name '*.json' | wc -l)"
    echo "  Log files:  $(find ${SERVER_RESULTS_DIR} -name '*.log' -o -name '*.txt' | wc -l)"
    echo "  Plot files: $(find ${SERVER_RESULTS_DIR} -name '*.png' -o -name '*.pdf' | wc -l)"
    echo "  Model files: $(find ${SERVER_RESULTS_DIR} -name '*.pth' -o -name '*.pt' | wc -l)"
    exit 0
fi

# Build exclude arguments
EXCLUDE_ARGS=""
if [ "$INCLUDE_MODELS" = false ]; then
    EXCLUDE_ARGS="--exclude *.pth --exclude *.pt --exclude *.bin --exclude *.safetensors --exclude *.ckpt --exclude *.h5"
    echo "⚠️  Excluding model files (use --include-models to include)"
else
    echo "📦 Including model files (this may take a while)"
fi

# Create timestamp for this upload
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo ""
echo "🕐 Upload timestamp: ${TIMESTAMP}"
echo ""

# Upload to S3
echo "🚀 Uploading to S3..."
echo "Command: rclone copy ${SERVER_RESULTS_DIR} ${S3_RESULTS_PATH} --progress ${EXCLUDE_ARGS}"
echo ""

rclone copy "${SERVER_RESULTS_DIR}" "${S3_RESULTS_PATH}" \
    --progress \
    ${EXCLUDE_ARGS} \
    --transfers 8 \
    --checkers 16

echo ""
echo "✅ Upload complete!"
echo ""

# Verify upload
echo "📋 Verifying S3 contents..."
rclone ls "${S3_RESULTS_PATH}" | head -20
echo ""

# Show S3 summary
echo "📊 S3 Summary:"
echo "  Total files: $(rclone ls ${S3_RESULTS_PATH} | wc -l)"
echo "  Total size: $(rclone size ${S3_RESULTS_PATH} 2>/dev/null | grep 'Total size' || echo 'calculating...')"
echo ""
echo "=============================================="
echo "  Upload Complete!"
echo "=============================================="
echo ""
echo "Next steps on local machine:"
echo "  1. Run: .\\pull_results_from_s3.ps1"
echo "  2. Or:  python pull_from_s3.py"
