#!/bin/bash
# =============================================================================
# AlphaQubit 快速测试脚本
# 用于验证环境和流程是否正确
# =============================================================================

set -e

echo "=============================================="
echo "AlphaQubit 快速测试"
echo "=============================================="

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphaqubit

# 进入工作目录
cd ~/alphaqubit/ALPHAQUBIT

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results_quick_test_${TIMESTAMP}"

echo "结果目录: ${RESULTS_DIR}"
echo ""

# 运行快速测试
echo "开始快速测试..."
python run_server_pipeline.py \
    --quick-test \
    --output-dir ${RESULTS_DIR}

echo ""
echo "=============================================="
echo "快速测试完成!"
echo "=============================================="
echo "结果保存在: ${RESULTS_DIR}"
echo ""
echo "查看结果:"
echo "  ls -la ${RESULTS_DIR}/"
echo "  cat ${RESULTS_DIR}/pipeline_metrics.json"
echo "=============================================="
