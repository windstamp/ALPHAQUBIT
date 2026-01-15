#!/usr/bin/env python3
"""
AlphaQubit Parallel Model Training - One Model Per NPU

This script trains multiple independent models in parallel,
with each model running on a separate NPU.
For training 100+ models efficiently on 8 NPUs.

Data format expected:
  - 'data': syndrome data, shape (N, T, H*W, C) or similar
  - 'obs': observable/label data, shape (N, 1)
  - 'basis': 'X' or 'Z' string

Usage:
    python train_parallel_models.py --data-dir output --filter-distance 3 --num-models 100
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from datetime import datetime
import threading
import queue
import math
import json

# NPU setup
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False
    print("Warning: torch_npu not available, will use CPU")


def setup_npu(device_id):
    """Setup NPU device"""
    if HAS_NPU:
        torch.npu.set_device(device_id)
        device = torch.device(f'npu:{device_id}')
        torch.npu.config.allow_internal_format = False
    else:
        device = torch.device('cpu')
    return device


class SurfaceCodeDataset(Dataset):
    """Dataset for surface code experiment data
    
    Expected NPZ format:
        - 'data': shape (N, T, spatial, channels) - syndrome measurements
        - 'obs': shape (N, 1) - observable labels
        - 'basis': 'X' or 'Z' string
    
    Handles variable time steps by padding to max_rounds.
    """
    
    def __init__(self, data_files, max_samples=None, max_rounds=25, verbose=True):
        """Load data from NPZ files with padding for variable time steps"""
        all_data = []
        all_labels = []
        
        if verbose:
            print(f"Loading {len(data_files)} files...")
        
        loaded = 0
        skipped = 0
        
        for f in data_files:
            try:
                npz = np.load(f, allow_pickle=True)
                keys = list(npz.keys())
                
                # Get syndrome data - key is 'data'
                if 'data' not in keys:
                    if verbose:
                        print(f"Skip {os.path.basename(f)}: no 'data' key, keys={keys}")
                    skipped += 1
                    continue
                
                # Get labels - key is 'obs'
                if 'obs' not in keys:
                    if verbose:
                        print(f"Skip {os.path.basename(f)}: no 'obs' key, keys={keys}")
                    skipped += 1
                    continue
                
                data = np.array(npz['data'], dtype=np.float32)
                labels = np.array(npz['obs'], dtype=np.float32)
                
                # Data shape: (N, T, spatial, channels) e.g. (72033, 3, 8, 3)
                N, T, spatial, channels = data.shape
                
                # Pad time dimension to max_rounds
                if T < max_rounds:
                    pad_width = ((0, 0), (0, max_rounds - T), (0, 0), (0, 0))
                    data = np.pad(data, pad_width, mode='constant', constant_values=0)
                elif T > max_rounds:
                    data = data[:, :max_rounds, :, :]  # Truncate if too long
                
                # Ensure labels is 2D
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                
                all_data.append(data)
                all_labels.append(labels)
                loaded += 1
                
                if verbose and loaded <= 5:
                    print(f"  Loaded {os.path.basename(f)}: orig_T={T}, padded shape={data.shape}")
                
            except Exception as e:
                if verbose:
                    print(f"Error loading {os.path.basename(f)}: {e}")
                skipped += 1
        
        if len(all_data) == 0:
            print(f"ERROR: No data loaded! Loaded {loaded}, skipped {skipped}")
            self.data = np.array([])
            self.labels = np.array([])
            return
        
        # Concatenate all data (now all have same shape)
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        if verbose:
            print(f"Loaded {loaded} files, skipped {skipped}")
            print(f"Total samples: {len(self.data)}, shape: {self.data.shape}")
            print(f"Labels shape: {self.labels.shape}")
        
        # Limit samples if needed
        if max_samples and len(self.data) > max_samples:
            idx = np.random.choice(len(self.data), max_samples, replace=False)
            self.data = self.data[idx]
            self.labels = self.labels[idx]
            if verbose:
                print(f"Limited to {max_samples} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


def find_data_files(data_dir, filter_distance=None, verbose=True):
    """Find all NPZ files in data directory"""
    patterns = [
        os.path.join(data_dir, "**", "*.npz"),
        os.path.join(data_dir, "*.npz"),
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    all_files = list(set(all_files))
    
    if verbose:
        print(f"Found {len(all_files)} NPZ files in {data_dir}")
    
    # Filter by distance if specified
    if filter_distance is not None:
        filtered = []
        for f in all_files:
            basename = os.path.basename(f)
            if f"_d{filter_distance}_" in basename:
                filtered.append(f)
        
        if len(filtered) > 0:
            all_files = filtered
            if verbose:
                print(f"Filtered to {len(all_files)} files with distance={filter_distance}")
    
    return sorted(all_files)


# ==============================================================================
# Model Definition
# ==============================================================================

class TransformerEncoderLayerManual(nn.Module):
    """Manual Transformer Encoder Layer (avoids CPU fallback on NPU)"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        B, T, D = x.shape
        
        # Self-attention
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        x = self.norm1(x + self.dropout(out))
        
        # Feed-forward
        ff_out = self.ff2(self.dropout(F.gelu(self.ff1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class AlphaQubitModelSimple(nn.Module):
    """Simplified AlphaQubit model for surface code data
    
    Input: (B, T, spatial, C) where T=time steps, spatial=flattened spatial dims
    Output: (B, 1) logits for binary classification
    """
    
    def __init__(self, input_dim=24, hidden_dim=256, num_heads=8, 
                 num_layers=6, num_classes=1, max_seq_len=50):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection: flatten spatial*channels -> hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerManual(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (B, T, spatial, C) e.g. (B, 3, 8, 3)
        B = x.shape[0]
        
        # Flatten last two dims: (B, T, spatial*C)
        if len(x.shape) == 4:
            x = x.view(B, x.shape[1], -1)
        elif len(x.shape) == 3:
            pass  # Already (B, T, features)
        
        T = x.shape[1]
        
        # Project to hidden dim
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Add positional encoding
        seq_len = min(T, self.pos_encoding.size(1))
        x = x[:, :seq_len] + self.pos_encoding[:, :seq_len]
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling and output
        x = x.mean(dim=1)  # (B, hidden_dim)
        logits = self.output_head(x)
        
        return logits


# ==============================================================================
# Training Functions
# ==============================================================================

def train_one_model(npu_id, model_id, data_files, args, result_queue):
    """Train a single model on a single NPU"""
    
    try:
        # Setup device
        device = setup_npu(npu_id)
        print(f"[NPU {npu_id}] Model {model_id}: Starting on {device}")
        
        # Create dataset with padding to max_rounds=25
        dataset = SurfaceCodeDataset(
            data_files, 
            max_samples=args.max_samples_per_model,
            max_rounds=25,  # Pad all sequences to 25 time steps
            verbose=True
        )
        
        if len(dataset) == 0:
            print(f"[NPU {npu_id}] ERROR: No data loaded!")
            result_queue.put((model_id, npu_id, False, "No data loaded"))
            return
        
        print(f"[NPU {npu_id}] Dataset size: {len(dataset)}")
        
        # Determine input dimension from data
        sample_x, sample_y = dataset[0]
        print(f"[NPU {npu_id}] Sample shape: x={sample_x.shape}, y={sample_y.shape}")
        
        # Calculate input_dim = spatial * channels
        if len(sample_x.shape) == 3:
            # Shape: (T, spatial, C)
            input_dim = sample_x.shape[1] * sample_x.shape[2]
        elif len(sample_x.shape) == 2:
            # Shape: (T, features)
            input_dim = sample_x.shape[1]
        else:
            input_dim = 24  # Default
        
        print(f"[NPU {npu_id}] Input dimension: {input_dim}")
        
        # Create dataloader
        sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        # Create model
        model = AlphaQubitModelSimple(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_classes=1,
            max_seq_len=50
        ).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[NPU {npu_id}] Model parameters: {num_params:,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_loss = float('inf')
        best_acc = 0.0
        start_time = time.time()
        last_save_time = start_time
        
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                
                # Ensure y has correct shape
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()
                
                num_batches += 1
                
                # Progress every 100 batches
                if batch_idx > 0 and batch_idx % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    acc = correct / total * 100
                    print(f"[NPU {npu_id}] Model {model_id} Epoch {epoch+1} Batch {batch_idx}: "
                          f"loss={avg_loss:.4f}, acc={acc:.1f}%")
            
            scheduler.step()
            
            # Epoch stats
            avg_loss = epoch_loss / max(num_batches, 1)
            acc = correct / total * 100 if total > 0 else 0
            elapsed = time.time() - start_time
            
            print(f"[NPU {npu_id}] Model {model_id} Epoch {epoch+1}/{args.epochs}: "
                  f"loss={avg_loss:.4f}, acc={acc:.1f}%, time={elapsed/60:.1f}min")
            
            # Save checkpoint every 2 hours
            if time.time() - last_save_time > 7200:
                save_path = os.path.join(args.output_dir, f"model_{model_id}_epoch{epoch+1}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'acc': acc,
                }, save_path)
                print(f"[NPU {npu_id}] Saved checkpoint: {save_path}")
                last_save_time = time.time()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = acc
                save_path = os.path.join(args.output_dir, f"model_{model_id}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'acc': acc,
                }, save_path)
        
        # Final save
        final_path = os.path.join(args.output_dir, f"model_{model_id}_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': args.epochs,
            'loss': avg_loss,
            'acc': acc,
        }, final_path)
        
        total_time = time.time() - start_time
        print(f"[NPU {npu_id}] Model {model_id} COMPLETED in {total_time/3600:.2f} hours, "
              f"best_loss={best_loss:.4f}, best_acc={best_acc:.1f}%")
        
        result_queue.put((model_id, npu_id, True, 
                         f"Completed in {total_time/3600:.2f}h, loss={best_loss:.4f}, acc={best_acc:.1f}%"))
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[NPU {npu_id}] Model {model_id} ERROR: {error_msg}")
        result_queue.put((model_id, npu_id, False, error_msg))


def run_parallel_training(args):
    """Run multiple models in parallel, one per NPU"""
    
    # Find data files
    data_files = find_data_files(args.data_dir, args.filter_distance)
    
    if len(data_files) == 0:
        print("ERROR: No data files found!")
        print(f"Searched in: {args.data_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"=" * 60)
    print(f"AlphaQubit Parallel Model Training")
    print(f"=" * 60)
    print(f"Total models to train: {args.num_models}")
    print(f"NPUs available: {args.num_npus}")
    print(f"Data files: {len(data_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"=" * 60)
    
    # Result tracking
    result_queue = queue.Queue()
    completed_models = []
    failed_models = []
    
    # Model queue
    model_queue = list(range(args.start_model, args.start_model + args.num_models))
    active_threads = {}
    
    start_time = time.time()
    
    while model_queue or active_threads:
        # Check for completed models
        while not result_queue.empty():
            model_id, npu_id, success, msg = result_queue.get()
            
            if npu_id in active_threads:
                del active_threads[npu_id]
            
            if success:
                completed_models.append(model_id)
                print(f"\n>>> Model {model_id} COMPLETED on NPU {npu_id}: {msg}")
            else:
                failed_models.append(model_id)
                print(f"\n>>> Model {model_id} FAILED on NPU {npu_id}: {msg}")
            
            print(f">>> Progress: {len(completed_models)}/{args.num_models} completed, "
                  f"{len(failed_models)} failed, {len(model_queue)} pending")
        
        # Start new models on free NPUs
        for npu_id in range(args.num_npus):
            if npu_id not in active_threads and model_queue:
                model_id = model_queue.pop(0)
                
                thread = threading.Thread(
                    target=train_one_model,
                    args=(npu_id, model_id, data_files, args, result_queue)
                )
                thread.start()
                active_threads[npu_id] = (thread, model_id)
                
                print(f"\n>>> Started Model {model_id} on NPU {npu_id}")
                time.sleep(2)
        
        time.sleep(10)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"=" * 60)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Models completed: {len(completed_models)}")
    print(f"Models failed: {len(failed_models)}")
    if failed_models:
        print(f"Failed model IDs: {failed_models}")
    print(f"Output directory: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="AlphaQubit Parallel Model Training")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='output',
                        help='Directory containing training data (default: output)')
    parser.add_argument('--filter-distance', type=int, default=None,
                        help='Filter data by code distance (e.g., 3, 5, 7)')
    parser.add_argument('--max-samples-per-model', type=int, default=None,
                        help='Maximum samples per model')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs per model')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Parallel training arguments
    parser.add_argument('--num-models', type=int, default=100,
                        help='Total number of models to train')
    parser.add_argument('--num-npus', type=int, default=8,
                        help='Number of NPUs to use')
    parser.add_argument('--start-model', type=int, default=0,
                        help='Starting model ID (for resuming)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Output directory for trained models')
    
    args = parser.parse_args()
    
    print(f"Arguments: {vars(args)}")
    
    run_parallel_training(args)


if __name__ == "__main__":
    main()
