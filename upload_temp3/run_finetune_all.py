"""
Batch fine-tune AlphaQubit decoder on all Google QEC v3.5 experimental data.
Run all 118 experiments with a single command.
Supports NPU acceleration and parallel processing.
"""

import os
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys


def find_npz_files(data_dir, pattern='*.npz'):
    """Find all NPZ files in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    npz_files = list(data_path.glob(pattern))
    npz_files.sort()
    return npz_files


def run_finetune(npz_file, args, gpu_id=None):
    """Run fine-tuning for a single NPZ file."""
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
    ]
    
    if args.pretrained:
        cmd.extend(['--pretrained', args.pretrained])
    
    if args.npu:
        cmd.append('--npu')
    
    if args.amp:
        cmd.append('--amp')
    
    if args.mla:
        cmd.append('--mla')
    
    # Set GPU if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n{'='*80}")
    print(f"Fine-tuning [{len([f for f in os.listdir(args.output_dir) if f.endswith('.pth')]) + 1}]: {exp_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Use Popen to stream output in real-time instead of capturing it
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print in real-time
            output_lines.append(line)
        
        # Wait for completion
        returncode = process.wait()
        elapsed = time.time() - start_time
        
        if returncode == 0:
            print(f"\n✓ Completed {exp_name} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            return {
                'experiment': exp_name,
                'status': 'success',
                'elapsed_time': elapsed,
                'output': ''.join(output_lines)
            }
        else:
            print(f"\n✗ Failed {exp_name} (exit code {returncode})")
            return {
                'experiment': exp_name,
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': ''.join(output_lines[-50:])  # Last 50 lines for debugging
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Exception in {exp_name}: {e}")
        return {
            'experiment': exp_name,
            'status': 'exception',
            'elapsed_time': elapsed,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Batch fine-tune AlphaQubit on all Google QEC experiments")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='google_finetune_data/finetune',
                        help='Directory containing fine-tuning NPZ files')
    parser.add_argument('--pretrained', type=str, default='alphaqubit_pauli_plus.pth',
                        help='Path to pretrained model weights (REQUIRED for paper-aligned results)')
    parser.add_argument('--output-dir', type=str, default='finetuned_models',
                        help='Output directory for fine-tuned models')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (paper: 1e-5 for finetuning)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    
    # Device and parallelization arguments
    parser.add_argument('--npu', action='store_true', help='Use NPU for training')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of experiments to run in parallel (use with caution)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments that already have fine-tuned models')
    
    # Filtering arguments
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process experiments matching this pattern (e.g., "d3" or "bX")')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of experiments to process (for testing)')
    
    # Model architecture selection
    parser.add_argument('--mla', action='store_true',
                        help='Use MLA (Multi-head Latent Attention) model instead of standard transformer')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all NPZ files
    print(f"Searching for NPZ files in {args.data_dir}...")
    npz_files = find_npz_files(args.data_dir)
    
    # Filter if requested
    if args.filter:
        npz_files = [f for f in npz_files if args.filter in f.stem]
        print(f"Filtered to {len(npz_files)} experiments matching '{args.filter}'")
    
    # Limit if requested
    if args.limit:
        npz_files = npz_files[:args.limit]
        print(f"Limited to {args.limit} experiments")
    
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
    
    if not npz_files:
        print("No experiments to process!")
        return
    
    print(f"\nFound {len(npz_files)} experiments to fine-tune")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel jobs: {args.parallel}")
    print()
    
    # Start time
    total_start = time.time()
    
    # Run fine-tuning
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        print(f"Running {args.parallel} experiments in parallel...")
        print("WARNING: Progress bars may overlap with parallel execution")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_finetune, npz_file, args, i % args.parallel): (i, npz_file)
                for i, npz_file in enumerate(npz_files)
            }
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                print(f"\n{'='*80}")
                print(f"Progress: {completed}/{len(npz_files)} experiments completed ({completed*100/len(npz_files):.1f}%)")
                print(f"{'='*80}\n")
    else:
        # Sequential processing
        print("Running experiments sequentially...")
        for i, npz_file in enumerate(npz_files, 1):
            print(f"\n{'='*80}")
            print(f"Overall Progress: {i}/{len(npz_files)} ({i*100/len(npz_files):.1f}%)")
            estimated_remaining = ((time.time() - total_start) / i) * (len(npz_files) - i)
            print(f"Estimated time remaining: {estimated_remaining/3600:.1f} hours")
            print(f"{'='*80}")
            
            result = run_finetune(npz_file, args)
            results.append(result)
    
    # Calculate total time
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*80}")
    print("FINE-TUNING SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Average time per experiment: {total_elapsed/len(results):.1f}s")
    
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  - {r['experiment']}: {r.get('error', 'unknown error')[:100]}")
    
    # Save summary
    summary = {
        'total_experiments': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_time_seconds': total_elapsed,
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'finetune_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"\n✓ All fine-tuning jobs completed!")


if __name__ == '__main__':
    main()
