#!/bin/bash
# =============================================================================
# AlphaQubit 完整训练运行脚本
# 在NPU服务器上后台运行完整的论文复现流程
# =============================================================================

set -e

echo "=============================================="
echo "AlphaQubit 完整训练启动"
echo "=============================================="

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphaqubit

# 进入工作目录
cd ~/alphaqubit/ALPHAQUBIT

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results_full_${TIMESTAMP}"
LOG_FILE="./training_${TIMESTAMP}.log"

echo "结果目录: ${RESULTS_DIR}"
echo "日志文件: ${LOG_FILE}"
echo ""

# 检查NPU状态
echo "检查NPU状态..."
if command -v npu-smi &> /dev/null; then
    npu-smi info
fi
echo ""

# 使用nohup在后台运行
echo "启动训练 (后台运行)..."
nohup python run_server_pipeline.py \
    --output-dir ${RESULTS_DIR} \
    --device auto \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo $PID > training.pid

echo ""
echo "=============================================="
echo "训练已在后台启动!"
echo "=============================================="
echo ""
echo "进程ID: ${PID}"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "常用命令:"
echo "  查看日志:     tail -f ${LOG_FILE}"
echo "  查看进度:     grep -E 'Stage|完成' ${LOG_FILE}"
echo "  检查进程:     ps aux | grep ${PID}"
echo "  停止训练:     kill ${PID}"
echo ""
echo "训练完成后，结果将保存在: ${RESULTS_DIR}"
echo "=============================================="
