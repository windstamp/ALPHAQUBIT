#!/bin/bash
# =============================================================================
# 下载训练结果到本地脚本
# 在远程服务器上运行，打包结果供下载
# =============================================================================

echo "=============================================="
echo "打包训练结果"
echo "=============================================="

cd ~/alphaqubit/ALPHAQUBIT

# 查找最新的结果目录
LATEST_RESULTS=$(ls -td results_* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULTS" ]; then
    echo "未找到结果目录!"
    exit 1
fi

echo "找到结果目录: ${LATEST_RESULTS}"
echo ""

# 创建打包文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="alphaqubit_results_${TIMESTAMP}.tar.gz"

# 打包结果 (排除大的数据文件，只保留模型和指标)
echo "打包中..."
tar -czvf ${ARCHIVE_NAME} \
    --exclude='*.npy' \
    --exclude='data/' \
    ${LATEST_RESULTS}/

echo ""
echo "=============================================="
echo "打包完成!"
echo "=============================================="
echo ""
echo "打包文件: ${ARCHIVE_NAME}"
echo "文件大小: $(du -h ${ARCHIVE_NAME} | cut -f1)"
echo ""
echo "在本地电脑上运行以下命令下载:"
echo "  scp user@server:~/alphaqubit/ALPHAQUBIT/${ARCHIVE_NAME} ."
echo ""
echo "如果需要包含数据文件，运行:"
echo "  scp -r user@server:~/alphaqubit/ALPHAQUBIT/${LATEST_RESULTS} ."
echo "=============================================="
