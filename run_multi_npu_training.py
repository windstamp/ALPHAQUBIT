#!/usr/bin/env python3
"""
Multi-NPU Distributed Training for AlphaQubit (Memory Optimized)
================================================================

This script provides robust multi-NPU training with:
1. Proper memory management to avoid OOM
2. Gradient accumulation for large effective batch sizes
3. Memory-efficient data loading
4. Automatic memory cleanup between experiments
5. Process isolation for stability

Key Memory Optimization Features:
- Chunked data loading (don't load full dataset into memory)
- Gradient checkpointing
- Mixed precision training (FP16/BF16)
- Explicit garbage collection
- NPU memory cache clearing
- Process-level isolation

Usage:
    # Single NPU (for debugging)
    python run_multi_npu_training.py --npu-ids 0 --data-dir google_finetune_data/finetune
    
    # 4 NPUs
    python run_multi_npu_training.py --npu-ids 0,1,2,3 --data-dir google_finetune_data/finetune
    
    # All 8 NPUs with memory optimization
    python run_multi_npu_training.py --npu-ids 0,1,2,3,4,5,6,7 --memory-efficient
"""

import os
import sys
import argparse
import json
import time
import gc
import queue
import threading
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with memory-optimized defaults."""
    # Model
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 12
    
    # Training (paper-aligned)
    batch_size: int = 64  # Reduced from 128 for memory
    gradient_accumulation_steps: int = 2  # Effective batch = 128
    epochs: int = 30
    lr: float = 1e-5
    weight_decay: float = 1e-3
    patience: int = 5
    
    # Memory optimization
    num_workers: int = 2  # Reduced for memory
    pin_memory: bool = True
    use_amp: bool = True  # Mixed precision
    gradient_checkpointing: bool = True
    max_memory_gb: float = 28.0  # Leave some headroom on 32GB NPU
    
    # NPU specific
    npu_memory_fraction: float = 0.9  # Use 90% of NPU memory
    clear_cache_frequency: int = 10  # Clear cache every N batches
    
    # Timeout
    timeout_seconds: int = 3600  # 1 hour per experiment


def setup_npu_memory_config(npu_id: int, config: TrainingConfig):
    """Configure NPU memory settings for optimal performance."""
    try:
        import torch
        import torch_npu
        
        # Set visible device
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(npu_id)
        
        # Memory optimization settings
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Disable some verbose logging
        os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
        os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'  # ERROR only
        
        # Initialize NPU
        torch.npu.set_device(0)  # Local device 0 (mapped via ASCEND_RT_VISIBLE_DEVICES)
        
        # Set memory fraction
        if hasattr(torch.npu, 'set_per_process_memory_fraction'):
            torch.npu.set_per_process_memory_fraction(config.npu_memory_fraction)
        
        return True
    except Exception as e:
        print(f"[NPU-{npu_id}] Memory config failed: {e}")
        return False


def cleanup_npu_memory(npu_id: int):
    """Clean up NPU memory after training."""
    try:
        import torch
        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
            torch.npu.synchronize()
        gc.collect()
    except Exception as e:
        print(f"[NPU-{npu_id}] Cleanup warning: {e}")


# =============================================================================
# Memory-Efficient Dataset
# =============================================================================

class MemoryEfficientNPZDataset:
    """Memory-efficient dataset that loads data in chunks."""
    
    def __init__(self, npz_path: str, chunk_size: int = 10000):
        import numpy as np
        import torch
        
        self.npz_path = npz_path
        self.chunk_size = chunk_size
        
        # Load metadata only first
        with np.load(npz_path, mmap_mode='r') as data:
            self.n_samples = data['data'].shape[0]
            self.n_rounds = data['data'].shape[1]
            self.n_detectors = data['data'].shape[2]
            self.n_features = data['data'].shape[3]
            try:
                self.metadata = json.loads(str(data['metadata']))
            except:
                self.metadata = {}
        
        # Calculate grid size
        import math
        d = int(math.sqrt(self.n_detectors + 1))
        if d * d != self.n_detectors + 1:
            d = int(math.sqrt(self.n_detectors)) + 1
        self.grid_size = d
        
        # Create final mask
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:self.n_detectors+1],
            dtype=torch.long
        )
        if len(self.final_mask) < self.n_detectors:
            padding = torch.zeros(self.n_detectors - len(self.final_mask), dtype=torch.long)
            self.final_mask = torch.cat([self.final_mask, padding])
        elif len(self.final_mask) > self.n_detectors:
            self.final_mask = self.final_mask[:self.n_detectors]
    
    def load_chunk(self, start_idx: int, end_idx: int):
        """Load a chunk of data into memory."""
        import numpy as np
        import torch
        
        with np.load(self.npz_path, mmap_mode='r') as data:
            detection_events = torch.from_numpy(
                data['data'][start_idx:end_idx].copy()
            ).float()
            observables = torch.from_numpy(
                data['obs'][start_idx:end_idx].copy()
            ).float()
            basis_ids = torch.from_numpy(
                data['basis'][start_idx:end_idx].copy()
            ).long()
        
        # Convert observables to binary labels
        labels = torch.zeros_like(observables)
        labels[observables == 49.0] = 1.0
        labels[(observables != 48.0) & (observables != 49.0) & (observables != 10.0)] = observables[
            (observables != 48.0) & (observables != 49.0) & (observables != 10.0)
        ]
        
        return detection_events, basis_ids, labels
    
    def __len__(self):
        return self.n_samples


# =============================================================================
# Training Function (runs in separate process)
# =============================================================================

def train_single_experiment(
    npz_path: str,
    npu_id: int,
    config: TrainingConfig,
    pretrained_path: Optional[str],
    output_dir: str,
    use_mla: bool = False
) -> Dict:
    """Train a single experiment on a specific NPU.
    
    This function is designed to run in a separate process for isolation.
    """
    import numpy as np
    import torch
    
    result = {
        'npz_path': npz_path,
        'npu_id': npu_id,
        'status': 'unknown',
        'error': None,
        'metrics': {}
    }
    
    start_time = time.time()
    
    try:
        # Import torch_npu
        import torch_npu
        
        # Setup NPU
        if not setup_npu_memory_config(npu_id, config):
            result['status'] = 'npu_init_failed'
            result['error'] = 'Failed to initialize NPU'
            return result
        
        device = torch.device('npu:0')
        
        # Import model
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        if use_mla:
            from ai_models.model_mla import AlphaQubitDecoder
        else:
            from ai_models.model import AlphaQubitDecoder
        
        # Load dataset (memory efficient)
        dataset = MemoryEfficientNPZDataset(npz_path)
        exp_name = dataset.metadata.get('experiment_name', Path(npz_path).stem.replace('samples_', ''))
        
        print(f"[NPU-{npu_id}] Training {exp_name}: {dataset.n_samples} samples")
        
        # Create model
        model = AlphaQubitDecoder(
            num_features=dataset.n_features,
            hidden_dim=config.hidden_dim,
            num_stabilizers=dataset.n_detectors,
            grid_size=dataset.grid_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        ).to(device)
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
        
        # Setup training
        # Calculate class weights
        _, _, all_labels = dataset.load_chunk(0, min(10000, dataset.n_samples))
        num_pos = all_labels.sum().item()
        num_neg = len(all_labels) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)
        del all_labels
        gc.collect()
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=0
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Split indices
        n_train = int(0.9 * dataset.n_samples)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, dataset.n_samples))
        
        for epoch in range(1, config.epochs + 1):
            # Training
            model.train()
            np.random.shuffle(train_indices)
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # Process in chunks
            for chunk_start in range(0, n_train, config.batch_size * config.gradient_accumulation_steps):
                chunk_end = min(chunk_start + config.batch_size * config.gradient_accumulation_steps, n_train)
                chunk_indices = train_indices[chunk_start:chunk_end]
                
                # Load chunk
                x_chunk, basis_chunk, labels_chunk = dataset.load_chunk(
                    min(chunk_indices), max(chunk_indices) + 1
                )
                
                # Reindex within chunk
                chunk_min = min(chunk_indices)
                local_indices = [i - chunk_min for i in chunk_indices]
                
                optimizer.zero_grad()
                
                # Mini-batches within chunk
                for i in range(0, len(local_indices), config.batch_size):
                    batch_local = local_indices[i:i + config.batch_size]
                    
                    x = x_chunk[batch_local].to(device)
                    basis = basis_chunk[batch_local].to(device)
                    labels = labels_chunk[batch_local].to(device)
                    final_mask = dataset.final_mask.unsqueeze(0).expand(len(batch_local), -1).to(device)
                    
                    # Forward pass
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits = model(x, basis, final_mask)
                            loss = criterion(logits.squeeze(), labels) / config.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        logits = model(x, basis, final_mask)
                        loss = criterion(logits.squeeze(), labels) / config.gradient_accumulation_steps
                        loss.backward()
                    
                    # Stats
                    total_loss += loss.item() * config.gradient_accumulation_steps * len(batch_local)
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                    total_correct += (preds == labels).sum().item()
                    total_samples += len(batch_local)
                    
                    # Clean up batch tensors
                    del x, basis, labels, final_mask, logits, loss, preds
                
                # Optimizer step after gradient accumulation
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Clean up chunk
                del x_chunk, basis_chunk, labels_chunk
                
                # Periodic memory cleanup
                if (chunk_start // (config.batch_size * config.gradient_accumulation_steps)) % config.clear_cache_frequency == 0:
                    if hasattr(torch, 'npu'):
                        torch.npu.empty_cache()
                    gc.collect()
            
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for chunk_start in range(0, len(val_indices), config.batch_size * 4):
                    chunk_end = min(chunk_start + config.batch_size * 4, len(val_indices))
                    chunk_idx = val_indices[chunk_start:chunk_end]
                    
                    x_chunk, basis_chunk, labels_chunk = dataset.load_chunk(
                        min(chunk_idx), max(chunk_idx) + 1
                    )
                    
                    chunk_min = min(chunk_idx)
                    local_indices = [i - chunk_min for i in chunk_idx]
                    
                    x = x_chunk[local_indices].to(device)
                    basis = basis_chunk[local_indices].to(device)
                    labels = labels_chunk[local_indices].to(device)
                    final_mask = dataset.final_mask.unsqueeze(0).expand(len(local_indices), -1).to(device)
                    
                    logits = model(x, basis, final_mask)
                    loss = criterion(logits.squeeze(), labels)
                    
                    val_loss += loss.item() * len(local_indices)
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_samples += len(local_indices)
                    
                    del x, basis, labels, final_mask, logits, loss, preds, x_chunk, basis_chunk, labels_chunk
            
            val_loss = val_loss / val_samples
            val_acc = val_correct / val_samples
            
            scheduler.step()
            
            print(f"[NPU-{npu_id}] {exp_name} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(output_dir, f"finetuned_{exp_name}.pth")
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"[NPU-{npu_id}] {exp_name} Early stopping at epoch {epoch}")
                    break
            
            # Clear cache after each epoch
            if hasattr(torch, 'npu'):
                torch.npu.empty_cache()
            gc.collect()
        
        result['status'] = 'success'
        result['metrics'] = {
            'best_val_loss': best_val_loss,
            'final_val_acc': val_acc,
            'epochs_trained': epoch
        }
        
    except Exception as e:
        import traceback
        result['status'] = 'error'
        result['error'] = f"{str(e)}\n{traceback.format_exc()}"
    
    finally:
        # Cleanup
        cleanup_npu_memory(npu_id)
        result['elapsed_time'] = time.time() - start_time
    
    return result


def worker_process(
    npu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: TrainingConfig,
    pretrained_path: Optional[str],
    output_dir: str,
    use_mla: bool
):
    """Worker process for a single NPU."""
    while True:
        try:
            npz_path = task_queue.get(timeout=1)
            if npz_path is None:  # Poison pill
                break
            
            result = train_single_experiment(
                npz_path=npz_path,
                npu_id=npu_id,
                config=config,
                pretrained_path=pretrained_path,
                output_dir=output_dir,
                use_mla=use_mla
            )
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put({
                'npz_path': npz_path if 'npz_path' in dir() else 'unknown',
                'npu_id': npu_id,
                'status': 'worker_error',
                'error': str(e)
            })


# =============================================================================
# Main
# =============================================================================

def find_npz_files(data_dir: str, pattern: str = '*.npz') -> List[Path]:
    """Find all NPZ files in directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(data_path.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-NPU Training for AlphaQubit (Memory Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing NPZ training files')
    parser.add_argument('--pretrained', type=str, default='alphaqubit_pauli_plus.pth',
                        help='Path to pretrained model')
    parser.add_argument('--output-dir', type=str, default='finetuned_models',
                        help='Output directory')
    
    # NPU arguments
    parser.add_argument('--npu-ids', type=str, default='0',
                        help='Comma-separated NPU IDs (e.g., "0,1,2,3")')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size per NPU (default: 64)')
    parser.add_argument('--grad-accum', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Max epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    
    # Memory optimization
    parser.add_argument('--memory-efficient', action='store_true',
                        help='Enable all memory optimizations')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (default: 2)')
    
    # Model
    parser.add_argument('--mla', action='store_true',
                        help='Use MLA architecture')
    
    # Filtering
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter experiments by pattern')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if model already exists')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of experiments')
    
    args = parser.parse_args()
    
    # Parse NPU IDs
    npu_ids = [int(x.strip()) for x in args.npu_ids.split(',')]
    
    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        use_amp=not args.no_amp
    )
    
    if args.memory_efficient:
        config.batch_size = 32
        config.gradient_accumulation_steps = 4
        config.num_workers = 1
        config.gradient_checkpointing = True
    
    # Print config
    print("=" * 70)
    print("MULTI-NPU TRAINING (MEMORY OPTIMIZED)")
    print("=" * 70)
    print(f"NPUs: {npu_ids}")
    print(f"Batch size: {config.batch_size} × {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Mixed precision: {config.use_amp}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find NPZ files
    npz_files = find_npz_files(args.data_dir)
    
    if args.filter:
        npz_files = [f for f in npz_files if args.filter in f.stem]
    
    if args.skip_existing:
        filtered = []
        for f in npz_files:
            exp_name = f.stem.replace('samples_', '')
            if not os.path.exists(os.path.join(args.output_dir, f"finetuned_{exp_name}.pth")):
                filtered.append(f)
        print(f"Skipping {len(npz_files) - len(filtered)} existing models")
        npz_files = filtered
    
    if args.limit:
        npz_files = npz_files[:args.limit]
    
    if not npz_files:
        print("No experiments to process!")
        return
    
    print(f"Experiments to train: {len(npz_files)}")
    print(f"Estimated time: {len(npz_files) * 5 / len(npu_ids):.0f} minutes")
    print()
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add tasks
    for npz_file in npz_files:
        task_queue.put(str(npz_file))
    
    # Add poison pills
    for _ in npu_ids:
        task_queue.put(None)
    
    # Start workers
    processes = []
    for npu_id in npu_ids:
        p = mp.Process(
            target=worker_process,
            args=(npu_id, task_queue, result_queue, config, args.pretrained, args.output_dir, args.mla)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts to avoid initialization conflicts
    
    # Collect results
    results = []
    start_time = time.time()
    
    while len(results) < len(npz_files):
        try:
            result = result_queue.get(timeout=config.timeout_seconds)
            results.append(result)
            
            exp_name = Path(result['npz_path']).stem.replace('samples_', '')
            status = result['status']
            elapsed = result.get('elapsed_time', 0)
            
            print(f"[{len(results)}/{len(npz_files)}] NPU-{result['npu_id']}: {exp_name} - {status} ({elapsed:.0f}s)")
            
            if result['status'] == 'error':
                print(f"  Error: {result.get('error', 'unknown')[:200]}")
                
        except queue.Empty:
            # Check if workers are still alive
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                print("All workers finished")
                break
    
    # Wait for processes
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {len(results)}, Success: {len(successful)}, Failed: {len(failed)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Throughput: {len(successful) / (total_time/3600):.1f} experiments/hour")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed[:10]:
            print(f"  - {Path(r['npz_path']).stem}: {r.get('error', 'unknown')[:100]}")
    
    # Save summary
    summary = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_time_seconds': total_time,
        'npu_ids': npu_ids,
        'config': config.__dict__,
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'multi_npu_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
