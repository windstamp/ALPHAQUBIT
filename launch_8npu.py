#!/usr/bin/env python3
"""
Multi-NPU Launcher for AlphaQubit Training

This script launches 8 separate Python processes, one per NPU,
using subprocess to avoid multiprocessing issues with Ascend NPUs.

Usage:
    python launch_8npu.py --mode pretrain --data-dir output --filter-distance 3
"""

import argparse
import os
import subprocess
import sys
import time
import signal

def main():
    parser = argparse.ArgumentParser(description="Launch 8-NPU Training")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--data-dir", default="output")
    parser.add_argument("--filter-distance", type=int, default=None)
    parser.add_argument("--filter-rounds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    
    world_size = args.world_size
    
    print("=" * 70)
    print(f"Launching {world_size}-NPU Distributed Training")
    print("=" * 70)
    
    # Kill any existing training processes
    print("Cleaning up existing processes...")
    os.system("pkill -9 -f 'train_single_npu.py' 2>/dev/null || true")
    time.sleep(2)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"
    env["WORLD_SIZE"] = str(world_size)
    env["HCCL_CONNECT_TIMEOUT"] = "1200"
    
    # Build command args
    cmd_args = [
        f"--mode={args.mode}",
        f"--data-dir={args.data_dir}",
        f"--max-rounds={args.max_rounds}",
        f"--epochs={args.epochs}",
        f"--batch-size={args.batch_size}",
        f"--lr={args.lr}",
        f"--hidden-dim={args.hidden_dim}",
        f"--num-layers={args.num_layers}",
        f"--num-heads={args.num_heads}",
        f"--save-dir={args.save_dir}",
        f"--world-size={world_size}",
    ]
    
    if args.filter_distance:
        cmd_args.append(f"--filter-distance={args.filter_distance}")
    if args.filter_rounds:
        cmd_args.append(f"--filter-rounds={args.filter_rounds}")
    if args.checkpoint:
        cmd_args.append(f"--checkpoint={args.checkpoint}")
    if args.max_files:
        cmd_args.append(f"--max-files={args.max_files}")
    
    # Launch all workers
    processes = []
    log_files = []
    
    for rank in range(world_size):
        rank_env = env.copy()
        rank_env["RANK"] = str(rank)
        rank_env["LOCAL_RANK"] = str(rank)
        
        log_path = f"logs/rank_{rank}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)
        
        cmd = [
            sys.executable,
            "train_single_npu.py",
            f"--rank={rank}",
        ] + cmd_args
        
        print(f"Starting rank {rank} on NPU {rank}...")
        
        proc = subprocess.Popen(
            cmd,
            env=rank_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        processes.append(proc)
        print(f"  PID: {proc.pid}")
    
    print()
    print(f"All {world_size} workers launched!")
    print(f"PIDs: {[p.pid for p in processes]}")
    print()
    print("Monitoring rank 0 output (Ctrl+C to stop monitoring, training continues)...")
    print("=" * 70)
    
    # Give processes time to start
    time.sleep(5)
    
    # Monitor rank 0 log
    try:
        with open("logs/rank_0.log", "r") as f:
            while True:
                # Check if rank 0 is still running
                if processes[0].poll() is not None:
                    # Process ended, read remaining output
                    remaining = f.read()
                    if remaining:
                        print(remaining, end="")
                    break
                
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.5)
                    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
        print(f"Check logs/rank_*.log for progress")
        print(f"To kill training: pkill -f train_single_npu.py")
    
    # Wait for all processes
    print("\nWaiting for all processes to complete...")
    
    exit_codes = []
    for i, proc in enumerate(processes):
        code = proc.wait()
        exit_codes.append(code)
        if code != 0:
            print(f"Rank {i} exited with code {code}")
    
    # Close log files
    for f in log_files:
        f.close()
    
    if all(code == 0 for code in exit_codes):
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Training finished with errors. Check logs/rank_*.log")
        print("=" * 70)
        
    return max(exit_codes) if exit_codes else 0


if __name__ == "__main__":
    sys.exit(main())
