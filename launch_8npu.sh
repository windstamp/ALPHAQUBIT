#!/bin/bash
# Multi-NPU Training Launch Script for Ascend 910B
# Launches 8 separate Python processes, one per NPU

set -e

WORLD_SIZE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Pass all arguments to the training script
ARGS="$@"

echo "========================================"
echo "Launching 8-NPU Distributed Training"
echo "========================================"

# Kill any existing training
pkill -9 -f "train_single_npu.py" 2>/dev/null || true
sleep 2

# Create directories
mkdir -p logs checkpoints

# Export common env vars
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export HCCL_CONNECT_TIMEOUT=1200

# Launch all 8 workers
for RANK in 0 1 2 3 4 5 6 7; do
    export RANK=$RANK
    export LOCAL_RANK=$RANK
    
    echo "Starting rank $RANK on NPU $RANK..."
    
    python train_single_npu.py \
        --rank $RANK \
        --world-size $WORLD_SIZE \
        $ARGS > logs/rank_${RANK}.log 2>&1 &
    
    PIDS[$RANK]=$!
    echo "  PID: ${PIDS[$RANK]}"
done

echo ""
echo "All 8 workers launched!"
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitoring rank 0 output..."
echo "========================================"

# Wait a moment for processes to start
sleep 5

# Tail rank 0 log
tail -f logs/rank_0.log
