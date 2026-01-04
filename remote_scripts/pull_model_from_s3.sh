#!/bin/bash
# =============================================================================
# Pull Qwen2.5-Coder Model from S3 to Remote Server
# Run this script on the remote server
# =============================================================================

set -e

S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
MODEL_NAME="Qwen2.5-Coder-0.5B-Instruct"
LOCAL_PATH="${HOME}/models/${MODEL_NAME}"

echo "=============================================="
echo "  Pull Model from S3"
echo "=============================================="
echo "☁️  S3 Source: ${S3_REMOTE}/models/${MODEL_NAME}"
echo "📁 Local Path: ${LOCAL_PATH}"
echo "=============================================="
echo ""

# Create local directory
mkdir -p ${LOCAL_PATH}

echo "📥 Downloading model..."
rclone copy "${S3_REMOTE}/models/${MODEL_NAME}" "${LOCAL_PATH}" -P \
    --exclude ".cache/**"

echo ""
echo "✅ Model downloaded successfully!"
echo "📁 Location: ${LOCAL_PATH}"
echo ""

# Verify files
echo "📋 Downloaded files:"
ls -lh ${LOCAL_PATH}

echo ""
echo "✅ Done! Model is ready at: ${LOCAL_PATH}"
