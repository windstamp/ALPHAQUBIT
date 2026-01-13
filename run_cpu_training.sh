#!/bin/bash
# AlphaQubit CPU 并行训练 - 一键运行脚本
# 运行: bash run_cpu_training.sh

set -e

echo "=============================================="
echo "  AlphaQubit CPU 并行训练"
echo "  $(date)"
echo "=============================================="

# 进入工作目录
cd ~/work/ALPHAQUBIT

# 设置 CPU 优化环境变量
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

# 显示 CPU 信息
echo ""
echo "CPU 信息:"
nproc
echo ""

# 选择运行模式
MODE=${1:-"full"}

case $MODE in
    "quick")
        echo "快速测试模式"
        nohup python run_pipeline_cpu_parallel.py --quick-test > pipeline_cpu.log 2>&1 &
        ;;
    "finetune")
        echo "仅 Fine-tuning (跳到步骤3)"
        nohup python run_pipeline_cpu_parallel.py --skip-to 3 > pipeline_cpu.log 2>&1 &
        ;;
    "full")
        echo "完整流程"
        nohup python run_pipeline_cpu_parallel.py > pipeline_cpu.log 2>&1 &
        ;;
    *)
        echo "用法: bash run_cpu_training.sh [quick|finetune|full]"
        exit 1
        ;;
esac

PID=$!
echo ""
echo "=============================================="
echo "  训练已在后台启动!"
echo "  PID: $PID"
echo "=============================================="
echo ""
echo "监控命令:"
echo "  tail -f pipeline_cpu.log          # 查看日志"
echo "  tail -f pipeline_cpu_parallel.log # 详细日志"
echo "  htop                              # CPU 使用"
echo "  ps aux | grep python              # 进程状态"
echo ""
echo "停止训练:"
echo "  kill $PID"
echo "  pkill -f run_pipeline_cpu_parallel"
echo ""

# 显示初始日志
sleep 2
echo "初始日志:"
echo "----------------------------------------"
tail -20 pipeline_cpu.log 2>/dev/null || echo "(等待日志生成...)"
