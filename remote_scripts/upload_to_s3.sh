#!/bin/bash
# =============================================================================
# Upload AlphaQubit Results to S3
# Run this script on the remote server
# =============================================================================

set -e

# Configuration - MODIFY THESE
S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
S3_PREFIX="${S3_PREFIX:-alphaqubit/results}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Timestamp for unique folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${TIMESTAMP}"

echo "=============================================="
echo "  Upload AlphaQubit Results to S3"
echo "=============================================="
echo "S3 Destination: ${S3_PATH}"
echo ""

# Navigate to project directory
cd ~/work/ALPHAQUBIT

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Installing..."
    pip install awscli
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo ""
    echo "❌ AWS credentials not configured!"
    echo ""
    echo "Please configure AWS credentials first:"
    echo "  Option 1: Run 'aws configure'"
    echo "  Option 2: Export environment variables:"
    echo "    export AWS_ACCESS_KEY_ID=your_access_key"
    echo "    export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "    export AWS_REGION=us-east-1"
    echo ""
    exit 1
fi

echo "✓ AWS credentials OK"
echo ""

# Upload directories
echo "Uploading results to S3..."
echo ""

# 1. Test results
if [ -d "test_results_v2" ]; then
    echo "📤 Uploading test_results_v2/..."
    aws s3 cp test_results_v2/ ${S3_PATH}/test_results_v2/ --recursive
    echo "✓ test_results_v2 uploaded"
fi

# 2. Fine-tuned models
if [ -d "finetuned_models_v2" ]; then
    echo "📤 Uploading finetuned_models_v2/..."
    aws s3 cp finetuned_models_v2/ ${S3_PATH}/finetuned_models_v2/ --recursive
    echo "✓ finetuned_models_v2 uploaded"
fi

# 3. Pretrained models
if [ -d "pretrained_models" ]; then
    echo "📤 Uploading pretrained_models/..."
    aws s3 cp pretrained_models/ ${S3_PATH}/pretrained_models/ --recursive
    echo "✓ pretrained_models uploaded"
fi

# 4. Pipeline results
if [ -f "pipeline_results.json" ]; then
    echo "📤 Uploading pipeline_results.json..."
    aws s3 cp pipeline_results.json ${S3_PATH}/
    echo "✓ pipeline_results.json uploaded"
fi

# 5. Pipeline log
if [ -f "pipeline_log.txt" ]; then
    echo "📤 Uploading pipeline_log.txt..."
    aws s3 cp pipeline_log.txt ${S3_PATH}/
    echo "✓ pipeline_log.txt uploaded"
fi

# 6. Benchmark results (if exists)
if [ -d "benchmark_results" ]; then
    echo "📤 Uploading benchmark_results/..."
    aws s3 cp benchmark_results/ ${S3_PATH}/benchmark_results/ --recursive
    echo "✓ benchmark_results uploaded"
fi

# 7. NPU benchmark results (if exists)
if [ -d "npu_benchmark_results" ]; then
    echo "📤 Uploading npu_benchmark_results/..."
    aws s3 cp npu_benchmark_results/ ${S3_PATH}/npu_benchmark_results/ --recursive
    echo "✓ npu_benchmark_results uploaded"
fi

# 8. Research report (if exists)
if [ -d "research_report" ]; then
    echo "📤 Uploading research_report/..."
    aws s3 cp research_report/ ${S3_PATH}/research_report/ --recursive
    echo "✓ research_report uploaded"
fi

# 9. Output directory (if exists)
if [ -d "output" ]; then
    echo "📤 Uploading output/..."
    aws s3 cp output/ ${S3_PATH}/output/ --recursive --exclude "*.npy"
    echo "✓ output uploaded"
fi

echo ""
echo "=============================================="
echo "  ✅ Upload Complete!"
echo "=============================================="
echo ""
echo "S3 Location: ${S3_PATH}"
echo ""
echo "To download results locally:"
echo "  aws s3 sync ${S3_PATH} ./server_results/"
echo ""
echo "To list uploaded files:"
echo "  aws s3 ls ${S3_PATH}/ --recursive"
echo "=============================================="
