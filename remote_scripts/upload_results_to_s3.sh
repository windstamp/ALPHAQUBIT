#!/bin/bash
# =============================================================================
# Upload AlphaQubit Training Results to S3 (using rclone)
# Run this script on the remote server
# =============================================================================
#
# Usage:
#   ./upload_results_to_s3.sh                    # Upload all training results
#   ./upload_results_to_s3.sh --latest           # Upload only latest results
#   ./upload_results_to_s3.sh --list             # List what's on S3
#   ./upload_results_to_s3.sh /path/to/folder    # Upload specific folder
#
# =============================================================================

set -e

# Configuration
S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
PROJECT_NAME="ALPHAQUBIT"
PROJECT_ROOT="/root/work/${PROJECT_NAME}"
S3_RESULTS_PATH="${S3_REMOTE}/${PROJECT_NAME}/training_results"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================="
echo "  Upload AlphaQubit Results to S3"
echo -e "==============================================${NC}"
echo "Project: ${PROJECT_ROOT}"
echo "S3 Destination: ${S3_RESULTS_PATH}"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}❌ rclone not found!${NC}"
    echo "Install: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

cd "${PROJECT_ROOT}" || { echo -e "${RED}❌ Project directory not found${NC}"; exit 1; }

# Function: Upload a directory
upload_dir() {
    local src="$1"
    local dest="$2"
    
    if [ -d "$src" ]; then
        local size=$(du -sh "$src" 2>/dev/null | cut -f1)
        echo -e "${YELLOW}📤 Uploading ${src} (${size})...${NC}"
        rclone copy "$src" "${S3_RESULTS_PATH}/${TIMESTAMP}/${dest}" \
            --progress \
            --transfers 4 \
            --exclude "__pycache__/**" \
            --exclude "*.pyc" \
            --exclude ".git/**"
        echo -e "${GREEN}✓ ${dest} uploaded${NC}"
    else
        echo -e "   Skipping ${src} (not found)"
    fi
}

# Function: Upload a file
upload_file() {
    local src="$1"
    
    if [ -f "$src" ]; then
        echo -e "${YELLOW}📤 Uploading ${src}...${NC}"
        rclone copy "$src" "${S3_RESULTS_PATH}/${TIMESTAMP}/" --progress
        echo -e "${GREEN}✓ $(basename $src) uploaded${NC}"
    fi
}

# Function: List remote results
list_remote() {
    echo -e "${BLUE}📋 Results on S3:${NC}"
    rclone lsd "${S3_RESULTS_PATH}" 2>/dev/null || echo "   No results found"
}

# Function: Upload all training results
upload_all() {
    echo -e "${YELLOW}📦 Uploading all training results...${NC}"
    echo ""
    
    # Test results
    upload_dir "test_results_v2" "test_results_v2"
    
    # Fine-tuned models
    upload_dir "finetuned_models_v2" "finetuned_models_v2"
    
    # Pretrained models
    upload_dir "pretrained_models" "pretrained_models"
    
    # Benchmark results
    upload_dir "benchmark_results" "benchmark_results"
    upload_dir "npu_benchmark_results" "npu_benchmark_results"
    
    # Research reports
    upload_dir "research_report" "research_report"
    
    # Output files (plots, etc.)
    upload_dir "output" "output"
    
    # Individual files
    upload_file "pipeline_results.json"
    upload_file "pipeline_log.txt"
    upload_file "benchmark_results.json"
    
    # Model checkpoints (.pth files)
    for pth_file in *.pth; do
        if [ -f "$pth_file" ]; then
            upload_file "$pth_file"
        fi
    done
}

# Main logic
case "${1:-}" in
    --help|-h)
        echo "Usage:"
        echo "  $0              Upload all training results"
        echo "  $0 --latest     Upload only most recent results"
        echo "  $0 --list       List remote results"
        echo "  $0 /path        Upload specific directory"
        exit 0
        ;;
    --list)
        list_remote
        exit 0
        ;;
    --latest)
        echo -e "${YELLOW}📦 Uploading latest results only...${NC}"
        upload_dir "test_results_v2" "test_results_v2"
        upload_file "pipeline_results.json"
        upload_file "pipeline_log.txt"
        ;;
    "")
        upload_all
        ;;
    *)
        if [ -d "$1" ]; then
            folder_name=$(basename "$1")
            echo -e "${YELLOW}📦 Uploading ${1}...${NC}"
            rclone copy "$1" "${S3_RESULTS_PATH}/${TIMESTAMP}/${folder_name}" \
                --progress \
                --transfers 4 \
                --exclude "__pycache__/**" \
                --exclude "*.pyc"
        elif [ -f "$1" ]; then
            upload_file "$1"
        else
            echo -e "${RED}❌ Not found: $1${NC}"
            exit 1
        fi
        ;;
esac

echo ""
echo -e "${GREEN}=============================================="
echo "  ✅ Upload Complete!"
echo "==============================================${NC}"
echo "Results at: ${S3_RESULTS_PATH}/${TIMESTAMP}/"
echo ""
echo "To download locally:"
echo "  rclone copy ${S3_RESULTS_PATH}/${TIMESTAMP} ./results -P"
