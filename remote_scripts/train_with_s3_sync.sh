#!/bin/bash
# =============================================================================
# 一键训练脚本：从 S3 拉取代码 → 训练 → 上传结果
# =============================================================================
#
# 使用方法:
#   ./train_with_s3_sync.sh              # 完整训练
#   ./train_with_s3_sync.sh --quick      # 快速测试
#   ./train_with_s3_sync.sh --watch      # 持续监控模式
#
# =============================================================================

set -e

S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT"
LOCAL_PATH="/root/work/ALPHAQUBIT"
QUICK_MODE=false
WATCH_MODE=false

# 解析参数
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --watch)
            WATCH_MODE=true
            ;;
    esac
done

echo "=============================================="
echo "  AlphaQubit 训练流程 (S3 同步)"
echo "=============================================="
echo "S3: ${S3_REMOTE}"
echo "本地: ${LOCAL_PATH}"
echo "=============================================="

# 函数：从 S3 拉取最新代码
pull_from_s3() {
    echo ""
    echo "📥 从 S3 拉取最新代码..."
    rclone sync ${S3_REMOTE} ${LOCAL_PATH} \
        --exclude "pretrain_data/**" \
        --exclude "backup_*/**" \
        --exclude "*.npy" \
        --exclude "__pycache__/**" \
        --progress
    echo "✅ 代码同步完成"
}

# 函数：上传结果到 S3
push_results_to_s3() {
    echo ""
    echo "📤 上传结果到 S3..."
    
    # 上传测试结果
    if [ -d "${LOCAL_PATH}/test_results_v2" ]; then
        rclone copy ${LOCAL_PATH}/test_results_v2 ${S3_REMOTE}/test_results_v2 --progress
    fi
    
    # 上传模型
    if [ -d "${LOCAL_PATH}/finetuned_models_v2" ]; then
        rclone copy ${LOCAL_PATH}/finetuned_models_v2 ${S3_REMOTE}/finetuned_models_v2 --progress
    fi
    
    if [ -d "${LOCAL_PATH}/pretrained_models" ]; then
        rclone copy ${LOCAL_PATH}/pretrained_models ${S3_REMOTE}/pretrained_models --progress
    fi
    
    # 上传日志和结果
    if [ -f "${LOCAL_PATH}/pipeline_results.json" ]; then
        rclone copy ${LOCAL_PATH}/pipeline_results.json ${S3_REMOTE}/ --progress
    fi
    
    if [ -f "${LOCAL_PATH}/pipeline_log.txt" ]; then
        rclone copy ${LOCAL_PATH}/pipeline_log.txt ${S3_REMOTE}/ --progress
    fi
    
    echo "✅ 结果上传完成"
}

# 函数：运行训练
run_training() {
    echo ""
    echo "🚀 开始训练..."
    cd ${LOCAL_PATH}
    
    if [ "$QUICK_MODE" = true ]; then
        echo "⚡ 快速测试模式"
        python run_full_pipeline.py --npu --quick-test
    else
        echo "📊 完整训练模式"
        python run_full_pipeline.py --npu
    fi
}

# 主流程
if [ "$WATCH_MODE" = true ]; then
    # 监控模式：持续检查 S3 更新并自动同步
    echo "👀 监控模式：每 60 秒检查 S3 更新"
    while true; do
        echo "[$(date '+%H:%M:%S')] 检查 S3 更新..."
        rclone sync ${S3_REMOTE} ${LOCAL_PATH} \
            --exclude "pretrain_data/**" \
            --exclude "backup_*/**" \
            --exclude "*.npy" \
            --exclude "__pycache__/**" \
            --quiet
        echo "[$(date '+%H:%M:%S')] ✓ 同步完成"
        sleep 60
    done
else
    # 训练模式
    pull_from_s3
    run_training
    push_results_to_s3
    
    echo ""
    echo "=============================================="
    echo "  ✅ 全部完成！"
    echo "=============================================="
    echo "结果已上传到 S3，可在本地拉取："
    echo "  rclone copy ${S3_REMOTE}/test_results_v2 ./results/"
    echo "=============================================="
fi
