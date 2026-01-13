#!/bin/bash
# =============================================================================
# Pull AlphaQubit Training Results from S3 to Local
# Only pulls analysis files (results, logs, plots) - NOT model weights
# =============================================================================
#
# Usage:
#   ./pull_results_from_s3.sh                    # Pull latest results
#   ./pull_results_from_s3.sh --list             # List available results
#   ./pull_results_from_s3.sh 20241230_123456    # Pull specific timestamp
#   ./pull_results_from_s3.sh --all              # Pull all results
#   ./pull_results_from_s3.sh --include-models   # Include model files (large!)
#
# =============================================================================

set -e

# Configuration
S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
PROJECT_NAME="ALPHAQUBIT"
S3_RESULTS_PATH="${S3_REMOTE}/${PROJECT_NAME}/training_results"
LOCAL_RESULTS_DIR="./s3_results"

# Flag for including models
INCLUDE_MODELS=false

# Exclusion patterns (large model files)
EXCLUDE_ARGS=(
    "--exclude" "*.pth"
    "--exclude" "*.bin"
    "--exclude" "*.safetensors"
    "--exclude" "*.pt"
    "--exclude" "*.ckpt"
    "--exclude" "*.h5"
    "--exclude" "*.npy"
    "--exclude" "*.npz"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================="
echo "  Pull AlphaQubit Results from S3"
echo -e "==============================================${NC}"
echo "S3 Source: ${S3_RESULTS_PATH}"
echo "Local Destination: ${LOCAL_RESULTS_DIR}"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}❌ rclone not found!${NC}"
    echo "Install rclone first: https://rclone.org/downloads/"
    exit 1
fi

# Function: List available results
list_results() {
    echo -e "${BLUE}📋 Available results on S3:${NC}"
    rclone lsd "${S3_RESULTS_PATH}" 2>/dev/null | while read line; do
        folder=$(echo "$line" | awk '{print $NF}')
        echo "   📁 ${folder}"
    done
    
    if [ $? -ne 0 ]; then
        echo "   No results found"
    fi
}

# Function: Get latest timestamp folder
get_latest() {
    rclone lsd "${S3_RESULTS_PATH}" 2>/dev/null | tail -1 | awk '{print $NF}'
}

# Function: Pull results
pull_results() {
    local timestamp="$1"
    local dest="${LOCAL_RESULTS_DIR}/${timestamp}"
    
    mkdir -p "$dest"
    
    echo -e "${YELLOW}📥 Downloading: ${timestamp}${NC}"
    echo "   Destination: ${dest}"
    
    if [ "$INCLUDE_MODELS" = false ]; then
        echo -e "   Mode: ${BLUE}Analysis files only (excluding model weights)${NC}"
        rclone copy "${S3_RESULTS_PATH}/${timestamp}" "$dest" \
            --progress \
            --transfers 4 \
            "${EXCLUDE_ARGS[@]}"
    else
        echo -e "   Mode: ${RED}All files including models${NC}"
        rclone copy "${S3_RESULTS_PATH}/${timestamp}" "$dest" \
            --progress \
            --transfers 4
    fi
    
    echo ""
    echo -e "${GREEN}✅ Download complete!${NC}"
    echo "   Results saved to: ${dest}"
}

# Main logic
case "${1:-}" in
    --help|-h)
        echo "Usage:"
        echo "  $0                      Pull latest results (analysis files only)"
        echo "  $0 --list               List available results"
        echo "  $0 <timestamp>          Pull specific timestamp"
        echo "  $0 --all                Pull all results"
        echo "  $0 --include-models     Also download model weights (large!)"
        echo ""
        echo "By default, only downloads: .json, .txt, .log, .png, .jpg, .pdf, .md, .csv"
        echo "Excludes: .pth, .bin, .safetensors, .pt, .ckpt, .h5, .npy, .npz"
        exit 0
        ;;
    --list)
        list_results
        exit 0
        ;;
    --include-models)
        INCLUDE_MODELS=true
        shift
        ;;
esac

# Handle remaining arguments after flags
case "${1:-}" in
    --all)
        echo -e "${YELLOW}📥 Downloading ALL results (analysis files only)...${NC}"
        mkdir -p "${LOCAL_RESULTS_DIR}"
        
        if [ "$INCLUDE_MODELS" = false ]; then
            rclone copy "${S3_RESULTS_PATH}" "${LOCAL_RESULTS_DIR}" \
                --progress \
                --transfers 4 \
                "${EXCLUDE_ARGS[@]}"
        else
            rclone copy "${S3_RESULTS_PATH}" "${LOCAL_RESULTS_DIR}" \
                --progress \
                --transfers 4
        fi
        echo -e "${GREEN}✅ All results downloaded to: ${LOCAL_RESULTS_DIR}${NC}"
        ;;
    "")
        # Pull latest
        latest=$(get_latest)
        if [ -z "$latest" ]; then
            echo -e "${RED}❌ No results found on S3${NC}"
            exit 1
        fi
        echo "Latest: ${latest}"
        pull_results "$latest"
        ;;
    *)
        # Pull specific timestamp
        pull_results "$1"
        ;;
esac

echo ""
echo -e "${GREEN}=============================================="
echo "  Done!"
echo -e "==============================================${NC}"
