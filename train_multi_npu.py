#!/usr/bin/env python3
"""
Multi-NPU Training for AlphaQubit using Distributed Data Parallel (DDP)

This script properly utilizes ALL 8 NPUs using PyTorch's DistributedDataParallel.
For Ascend NPUs, this requires proper HCCL initialization.

Usage (launch with torchrun):
    torchrun --nproc_per_node=8 train_multi_npu.py --mode pretrain --data-dir output --filter-distance 3

Or launch manually:
    python train_multi_npu.py --local_rank 0  # On each NPU
"""

import argparse
import math
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

# Paper-aligned hyperparameters
PAPER_BATCH_SIZE = 256  # Per-device batch size
PAPER_LR = 1e-4
PAPER_PRETRAIN_EPOCHS = 100
PAPER_FINETUNE_EPOCHS = 50


def setup_distributed(backend='hccl'):
    """Initialize distributed training environment for Ascend NPUs."""
    # Get rank and world size from environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # Set the NPU device
    try:
        import torch_npu
        torch.npu.set_device(local_rank)
        device = torch.device(f'npu:{local_rank}')
        
        # Initialize process group with HCCL backend for NPU
        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='hccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
        print(f"[Rank {rank}] Using NPU:{local_rank}, World size: {world_size}")
    except ImportError:
        # Fallback to CUDA/CPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            if world_size > 1 and not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
            print(f"[Rank {rank}] Using CUDA:{local_rank}")
        else:
            device = torch.device('cpu')
            print(f"[Rank {rank}] Using CPU")
    
    return device, local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class PaddedSoftReadoutDataset(Dataset):
    """Dataset that pads sequences to a maximum round length."""
    
    def __init__(self, npz_path: str, max_rounds: int = 25, basis_id: Optional[int] = None):
        data = np.load(npz_path, allow_pickle=True)
        
        # Get data
        if 'data' in data:
            det_events = data['data']
        elif 'detection_events' in data:
            det_events = data['detection_events']
        else:
            raise KeyError(f"No data in {npz_path}")
        
        # Get labels
        if 'observables' in data:
            labels = data['observables']
        elif 'labels' in data:
            labels = data['labels']
        elif 'obs' in data:
            labels = data['obs']
        else:
            raise KeyError(f"No labels in {npz_path}")
        
        # Infer basis from filename
        if basis_id is None:
            if '_bX_' in str(npz_path) or 'basis_x' in str(npz_path).lower():
                basis_id = 0
            else:
                basis_id = 1
        
        N = det_events.shape[0]
        
        # Handle shapes
        if det_events.ndim == 4:
            x = det_events.astype(np.float32)
            _, R, S, F = x.shape
        elif det_events.ndim == 3:
            x = det_events[:, :, :, np.newaxis].astype(np.float32)
            _, R, S, F = x.shape
        else:
            raise ValueError(f"Unexpected shape: {det_events.shape}")
        
        # Infer distance
        d_sq = S + 1
        distance = int(math.sqrt(d_sq))
        
        # Add basis channel if needed
        if F == 3:
            basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=np.float32)
            x = np.concatenate([x, basis_feat], axis=-1)
            F = 4
        
        # Pad rounds to max_rounds
        if R < max_rounds:
            pad_rounds = max_rounds - R
            padding = np.zeros((N, pad_rounds, S, F), dtype=np.float32)
            x = np.concatenate([x, padding], axis=1)
        elif R > max_rounds:
            x = x[:, :max_rounds, :, :]
        
        # Create attention mask (1 for real data, 0 for padding)
        mask = np.ones((N, max_rounds), dtype=np.float32)
        mask[:, R:] = 0
        
        # Pad stabilizers to perfect square
        _, _, S, F = x.shape
        d = math.isqrt(S + 1)
        if d * d != S + 1:
            d += 1
            pad = d * d - 1 - S
            x = np.concatenate([x, np.zeros((N, max_rounds, pad, F), dtype=np.float32)], axis=2)
            S = d * d - 1
        
        self.X = torch.tensor(x, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(labels.flatten().astype(np.float32), dtype=torch.float32)
        self.basis = torch.full((N,), basis_id, dtype=torch.long)
        
        self.distance = distance
        self.rounds = R
        self.max_rounds = max_rounds
        self.num_stabilizers = S
        self.n_features = F
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.mask[idx], self.basis[idx]), self.y[idx]


def discover_npz_files(data_dir: str, filter_distance: int = None, filter_rounds: int = None) -> List[str]:
    """Find .npz files with optional filtering."""
    data_path = Path(data_dir)
    npz_files = list(data_path.glob("*.npz"))
    
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            npz_files.extend(subdir.glob("*.npz"))
    
    npz_files = sorted([str(f) for f in npz_files])
    
    if filter_distance:
        pattern = f"_d{filter_distance}_"
        npz_files = [f for f in npz_files if pattern in f]
    
    if filter_rounds:
        pattern = f"_r{filter_rounds:02d}_"
        npz_files = [f for f in npz_files if pattern in f]
    
    return npz_files


def build_padded_dataset(npz_files: List[str], max_rounds: int = 25, rank: int = 0) -> ConcatDataset:
    """Build dataset with padding for variable-length sequences."""
    datasets = []
    
    for npz_path in npz_files:
        try:
            ds = PaddedSoftReadoutDataset(npz_path, max_rounds=max_rounds)
            if len(ds) > 0:
                datasets.append(ds)
                if rank == 0:
                    print(f"  {Path(npz_path).name}: N={len(ds)}, R={ds.rounds}→{max_rounds}")
        except Exception as e:
            if rank == 0:
                print(f"Warning: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets")
    
    return ConcatDataset(datasets)


class AlphaQubitTransformer(nn.Module):
    """AlphaQubit decoder with spatial + temporal attention."""
    
    def __init__(
        self,
        num_features: int = 4,
        hidden_dim: int = 256,
        num_stabilizers: int = 24,
        max_rounds: int = 25,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_stabilizers = num_stabilizers
        self.max_rounds = max_rounds
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_stabilizers, hidden_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_rounds, 1, hidden_dim) * 0.02)
        
        # Basis embedding
        self.basis_embed = nn.Embedding(2, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask=None, basis=None):
        B, R, S, F = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.spatial_pos[:, :, :S, :]
        x = x + self.temporal_pos[:, :R, :, :]
        
        # Add basis embedding
        if basis is not None:
            basis_emb = self.basis_embed(basis)
            x = x + basis_emb.unsqueeze(1).unsqueeze(2)
        
        # Reshape for transformer: (B, R*S, hidden)
        x = x.reshape(B, R * S, -1)
        
        # Create attention mask if needed
        if mask is not None:
            # Expand mask from (B, R) to (B, R*S)
            attn_mask = mask.unsqueeze(-1).expand(-1, -1, S).reshape(B, R * S)
            attn_mask = attn_mask.bool()
        else:
            attn_mask = None
        
        # Apply transformer layers
        for layer in self.layers:
            if attn_mask is not None:
                x = layer(x, src_key_padding_mask=~attn_mask)
            else:
                x = layer(x)
        
        x = self.norm(x)
        
        # Pool over sequence
        if mask is not None:
            attn_mask_expanded = attn_mask.unsqueeze(-1).float()
            x = (x * attn_mask_expanded).sum(dim=1) / attn_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Classify
        logits = self.classifier(x).squeeze(-1)
        
        return logits


def train_epoch(model, loader, optimizer, device, rank):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, ((x, mask, basis), y) in enumerate(loader):
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x, mask, basis)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for (x, mask, basis), y in loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        logits = model(x, mask, basis)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
    
    return total_loss / total_samples, total_correct / total_samples


def train(model, train_loader, val_loader, train_sampler, epochs, lr, device, save_dir, prefix, rank, world_size):
    """Full training loop with DDP."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    history = []
    
    save_path = Path(save_dir)
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{epochs}", flush=True)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, rank)
        
        # Only evaluate on rank 0 to avoid duplicate work
        if rank == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
        else:
            val_loss, val_acc = 0, 0
        
        scheduler.step()
        
        if rank == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {lr_now:.2e}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr_now,
            })
            
            # Get model state dict (unwrap DDP)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_state, save_path / f"{prefix}_best.pth")
                print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path / f"{prefix}_epoch{epoch}.pth")
        
        # Sync all processes
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_state, save_path / f"{prefix}_final.pth")
        
        with open(save_path / f"{prefix}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Multi-NPU AlphaQubit Training")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--data-dir", default="output")
    parser.add_argument("--filter-distance", type=int, default=None)
    parser.add_argument("--filter-rounds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=PAPER_LR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Set default epochs
    if args.epochs is None:
        args.epochs = PAPER_PRETRAIN_EPOCHS if args.mode == "pretrain" else PAPER_FINETUNE_EPOCHS
    
    # Initialize distributed training
    device, local_rank, world_size, rank = setup_distributed()
    
    torch.manual_seed(42 + rank)
    
    # Only print on rank 0
    if rank == 0:
        print("\n" + "="*70)
        print(f"AlphaQubit Multi-NPU Training - {args.mode.upper()}")
        print("="*70)
        print(f"  World size (NPUs): {world_size}")
        print(f"  Per-device batch: {args.batch_size}")
        print(f"  Effective batch:  {args.batch_size * world_size}")
        print(f"  Data: {args.data_dir}")
        print(f"  Filter: distance={args.filter_distance}, rounds={args.filter_rounds}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}")
        print("="*70 + "\n")
    
    # Find data files
    npz_files = discover_npz_files(args.data_dir, args.filter_distance, args.filter_rounds)
    
    if not npz_files:
        if rank == 0:
            print("ERROR: No .npz files found")
        sys.exit(1)
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    if rank == 0:
        print(f"Found {len(npz_files)} .npz files\n")
        print("Loading datasets...")
    
    # Build dataset
    dataset = build_padded_dataset(npz_files, max_rounds=args.max_rounds, rank=rank)
    
    if rank == 0:
        print(f"\nTotal samples: {len(dataset):,}")
    
    # Split dataset
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    split = int(0.9 * n)
    
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds = torch.utils.data.Subset(dataset, indices[split:])
    
    if rank == 0:
        print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    
    # Create distributed sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Batches per epoch: {len(train_loader)}\n")
    
    # Get dimensions
    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    
    if rank == 0:
        print(f"Model config: max_rounds={args.max_rounds}, S={S}, F={F}")
        print(f"  Hidden: {args.hidden_dim}, Layers: {args.num_layers}, Heads: {args.num_heads}")
    
    # Create model
    model = AlphaQubitTransformer(
        num_features=F,
        hidden_dim=args.hidden_dim,
        num_stabilizers=S,
        max_rounds=args.max_rounds,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  Parameters: {n_params:,}\n")
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        if rank == 0:
            print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Move model to device
    model = model.to(device)
    
    # Wrap with DDP for multi-NPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            print(f"Using DistributedDataParallel across {world_size} NPUs\n")
    
    # Train
    prefix = f"{args.mode}_d{args.filter_distance or 'all'}"
    if args.filter_rounds:
        prefix += f"_r{args.filter_rounds}"
    
    if rank == 0:
        print("Starting training...\n")
    
    model, history = train(
        model, train_loader, val_loader, train_sampler,
        args.epochs, args.lr, device, args.save_dir, prefix, rank, world_size
    )
    
    # Summary
    if rank == 0:
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best val loss: {min(h['val_loss'] for h in history):.4f}")
        print(f"Best val acc:  {max(h['val_acc'] for h in history):.4f}")
        print(f"Checkpoints saved to: {args.save_dir}/")
        print("="*70)
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
