#!/bin/bash
# =============================================================================
# Quick S3 Upload - One-liner for AlphaQubit Results
# =============================================================================
#
# Usage:
#   # Set your S3 bucket first
#   export S3_BUCKET=your-bucket-name
#   
#   # Then run
#   ./quick_s3_upload.sh
#
# Or run with bucket name:
#   S3_BUCKET=my-bucket ./quick_s3_upload.sh
#
# =============================================================================

S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd ~/work/ALPHAQUBIT

echo "Uploading to s3://${S3_BUCKET}/alphaqubit/${TIMESTAMP}/"

# Upload all results in one command
aws s3 sync . s3://${S3_BUCKET}/alphaqubit/${TIMESTAMP}/ \
    --exclude "*" \
    --include "test_results_v2/*" \
    --include "finetuned_models_v2/*" \
    --include "pretrained_models/*" \
    --include "pipeline_results.json" \
    --include "pipeline_log.txt" \
    --include "benchmark_results/*" \
    --include "npu_benchmark_results/*" \
    --include "research_report/*" \
    --include "output/*.png" \
    --include "output/*.json" \
    --include "output/*.md"

echo ""
echo "✅ Done! Results at: s3://${S3_BUCKET}/alphaqubit/${TIMESTAMP}/"
