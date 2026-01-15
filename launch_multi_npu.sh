#!/bin/bash
# Multi-NPU Training Launch Script for Ascend 910B
# This script properly sets up HCCL and launches distributed training

set -e

# Configuration
WORLD_SIZE=${WORLD_SIZE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# Script arguments (pass through)
SCRIPT_ARGS="$@"

echo "========================================"
echo "Multi-NPU Training Launch Script"
echo "========================================"
echo "World Size: $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Script Args: $SCRIPT_ARGS"
echo "========================================"

# Kill any existing training processes
echo "Cleaning up existing processes..."
pkill -9 -f "train_single_npu.py" 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p logs

# Set HCCL environment variables
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# Launch workers
for ((RANK=0; RANK<$WORLD_SIZE; RANK++)); do
    export RANK=$RANK
    export LOCAL_RANK=$RANK
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    LOG_FILE="logs/rank_${RANK}.log"
    
    echo "Launching rank $RANK on NPU $RANK..."
    
    if [ $RANK -eq 0 ]; then
        # Run rank 0 in foreground to see output
        python train_single_npu.py --rank $RANK --world-size $WORLD_SIZE $SCRIPT_ARGS 2>&1 | tee $LOG_FILE &
        RANK0_PID=$!
    else
        # Run other ranks in background
        python train_single_npu.py --rank $RANK --world-size $WORLD_SIZE $SCRIPT_ARGS > $LOG_FILE 2>&1 &
    fi
done

echo "All workers launched. Waiting for rank 0 (PID: $RANK0_PID)..."
wait $RANK0_PID

echo "Training complete!"
