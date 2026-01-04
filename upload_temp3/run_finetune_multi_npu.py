#!/usr/bin/env python3
"""
Multi-NPU Parallel Fine-tuning for AlphaQubit
============================================

This script runs fine-tuning on ALL 8 NPUs in parallel.
Each NPU processes different experiments simultaneously.

Usage:
    python run_finetune_multi_npu.py --npu-count 8
"""

import os
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import subprocess
import sys
import queue
import threading


def find_npz_files(data_dir, pattern='*.npz'):
    """Find all NPZ files in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    npz_files = list(data_path.glob(pattern))
    npz_files.sort()
    return npz_files


def run_finetune_on_npu(npz_file, args, npu_id):
    """Run fine-tuning for a single NPZ file on a specific NPU."""
    exp_name = npz_file.stem.replace('samples_', '')
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'ai_models', 'fine_tune_npz.py'),
        '--data', str(npz_file),
        '--output-dir', args.output_dir,
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--patience', str(args.patience),
        '--hidden-dim', str(args.hidden_dim),
        '--num-heads', str(args.num_heads),
        '--num-layers', str(args.num_layers),
        '--num-workers', str(args.num_workers),
        '--npu',  # Always use NPU
    ]
    
    if args.pretrained:
        cmd.extend(['--pretrained', args.pretrained])
    
    if args.amp:
        cmd.append('--amp')
    
    if args.mla:
        cmd.append('--mla')
    
    # Set specific NPU device
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = str(npu_id)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'experiment': exp_name,
                'npu_id': npu_id,
                'status': 'success',
                'elapsed_time': elapsed,
            }
        else:
            return {
                'experiment': exp_name,
                'npu_id': npu_id,
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': result.stderr[-500:] if result.stderr else result.stdout[-500:]
            }
    
    except subprocess.TimeoutExpired:
        return {
            'experiment': exp_name,
            'npu_id': npu_id,
            'status': 'timeout',
            'elapsed_time': args.timeout,
            'error': f'Timeout after {args.timeout}s'
        }
    except Exception as e:
        return {
            'experiment': exp_name,
            'npu_id': npu_id,
            'status': 'exception',
            'elapsed_time': time.time() - start_time,
            'error': str(e)
        }


def worker(npu_id, task_queue, results, args, progress_lock, progress_counter):
    """Worker function for each NPU."""
    while True:
        try:
            npz_file = task_queue.get_nowait()
        except queue.Empty:
            break
        
        result = run_finetune_on_npu(npz_file, args, npu_id)
        results.append(result)
        
        with progress_lock:
            progress_counter[0] += 1
            total = progress_counter[1]
            current = progress_counter[0]
            exp_name = result['experiment']
            status = result['status']
            elapsed = result['elapsed_time']
            print(f"[{current}/{total}] NPU-{npu_id}: {exp_name} - {status} ({elapsed:.1f}s)")
        
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Multi-NPU parallel fine-tuning for AlphaQubit")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='google_finetune_data/finetune',
                        help='Directory containing fine-tuning NPZ files')
    parser.add_argument('--pretrained', type=str, default='alphaqubit_pauli_plus.pth',
                        help='Path to pretrained model weights')
    parser.add_argument('--output-dir', type=str, default='finetuned_models',
                        help='Output directory for fine-tuned models')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (paper: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout per experiment in seconds')
    
    # NPU arguments
    parser.add_argument('--npu-count', type=int, default=8, help='Number of NPUs to use')
    parser.add_argument('--npu-list', type=str, default=None, 
                        help='Comma-separated list of NPU IDs to use (e.g., "0,1,2,3")')
    
    # Filtering arguments
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments that already have fine-tuned models')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process experiments matching this pattern')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of experiments to process')
    
    # Model architecture
    parser.add_argument('--mla', action='store_true',
                        help='Use MLA model instead of standard transformer')
    
    args = parser.parse_args()
    
    # Determine NPU list
    if args.npu_list:
        npu_ids = [int(x.strip()) for x in args.npu_list.split(',')]
    else:
        npu_ids = list(range(args.npu_count))
    
    print("=" * 80)
    print("MULTI-NPU PARALLEL FINE-TUNING")
    print("=" * 80)
    print(f"NPUs to use: {npu_ids}")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all NPZ files
    print(f"Searching for NPZ files in {args.data_dir}...")
    npz_files = find_npz_files(args.data_dir)
    
    # Filter if requested
    if args.filter:
        npz_files = [f for f in npz_files if args.filter in f.stem]
        print(f"Filtered to {len(npz_files)} experiments matching '{args.filter}'")
    
    # Skip existing if requested
    if args.skip_existing:
        filtered = []
        for npz_file in npz_files:
            exp_name = npz_file.stem.replace('samples_', '')
            model_path = os.path.join(args.output_dir, f"finetuned_{exp_name}.pth")
            if not os.path.exists(model_path):
                filtered.append(npz_file)
        
        skipped = len(npz_files) - len(filtered)
        if skipped > 0:
            print(f"Skipping {skipped} existing models")
        npz_files = filtered
    
    # Limit if requested
    if args.limit:
        npz_files = npz_files[:args.limit]
        print(f"Limited to {args.limit} experiments")
    
    if not npz_files:
        print("No experiments to process!")
        return
    
    total_experiments = len(npz_files)
    print(f"\nTotal experiments to fine-tune: {total_experiments}")
    print(f"Using {len(npu_ids)} NPUs in parallel")
    print(f"Estimated time: {total_experiments * 3 / len(npu_ids) / 60:.1f} minutes")
    print()
    
    # Create task queue
    task_queue = queue.Queue()
    for npz_file in npz_files:
        task_queue.put(npz_file)
    
    # Results and progress tracking
    results = []
    progress_lock = threading.Lock()
    progress_counter = [0, total_experiments]  # [current, total]
    
    # Start timing
    total_start = time.time()
    
    # Launch workers for each NPU
    threads = []
    for npu_id in npu_ids:
        t = threading.Thread(
            target=worker,
            args=(npu_id, task_queue, results, args, progress_lock, progress_counter)
        )
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Calculate total time
    total_elapsed = time.time() - total_start
    
    # Summary
    print()
    print("=" * 80)
    print("FINE-TUNING SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Average time per experiment: {total_elapsed/len(results):.1f}s")
    print(f"Speedup with {len(npu_ids)} NPUs: ~{len(npu_ids)}x")
    
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  - NPU-{r['npu_id']}: {r['experiment']}: {r.get('error', 'unknown')[:100]}")
    
    # Save summary
    summary = {
        'total_experiments': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_time_seconds': total_elapsed,
        'npus_used': npu_ids,
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'finetune_multi_npu_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"\n✓ All fine-tuning jobs completed!")


if __name__ == '__main__':
    main()
