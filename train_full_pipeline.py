#!/usr/bin/env python3
"""
Full AlphaQubit Training Pipeline

This script implements a complete training pipeline following the paper:
1. Pre-training on all available data (mixed rounds with padding)
2. Fine-tuning on specific experiments
3. Evaluation and checkpoint saving

Paper-aligned hyperparameters:
- Pre-training: 8.5M samples, batch=256, lr=1e-4, epochs=100
- Fine-tuning: 50K samples per experiment

Usage:
    # Full pre-training on all d3 data
    python train_full_pipeline.py --mode pretrain --data-dir output --filter-distance 3 --npu

    # Fine-tuning on specific rounds
    python train_full_pipeline.py --mode finetune --data-dir output --filter-distance 3 --filter-rounds 25 --checkpoint checkpoints/pretrain_best.pth --npu
"""

import argparse
import math
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Paper-aligned hyperparameters
PAPER_BATCH_SIZE = 256
PAPER_LR = 1e-4
PAPER_PRETRAIN_EPOCHS = 100
PAPER_FINETUNE_EPOCHS = 50


class PaddedSoftReadoutDataset(Dataset):
    """
    Dataset that pads sequences to a maximum round length.
    This allows batching data with different numbers of rounds.
    """
    
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
        mask[:, R:] = 0  # Mask padded rounds
        
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
        
        print(f"  {Path(npz_path).name}: N={N}, R={R}→{max_rounds}, S={S}, F={F}")
    
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


def build_padded_dataset(npz_files: List[str], max_rounds: int = 25) -> ConcatDataset:
    """Build dataset with padding for variable-length sequences."""
    datasets = []
    
    for npz_path in npz_files:
        try:
            ds = PaddedSoftReadoutDataset(npz_path, max_rounds=max_rounds)
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"Warning: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets")
    
    return ConcatDataset(datasets)


class AlphaQubitTransformer(nn.Module):
    """
    AlphaQubit decoder with proper handling of:
    - Variable-length sequences (with masking)
    - Spatial attention over stabilizers
    - Temporal attention over rounds
    """
    
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
        
        # Spatial transformer (over stabilizers)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_layers // 2)
        
        # Temporal transformer (over rounds)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers // 2)
        
        # Output head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, mask=None, basis=None):
        """
        Args:
            x: (B, R, S, F) input tensor
            mask: (B, R) attention mask for rounds
            basis: (B,) basis labels (unused)
        """
        B, R, S, F = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, R, S, H)
        
        # Add positional embeddings
        x = x + self.spatial_pos[:, :, :S, :]
        x = x + self.temporal_pos[:, :R, :, :]
        
        # Spatial attention (per round)
        x_flat = x.view(B * R, S, -1)  # (B*R, S, H)
        x_flat = self.spatial_transformer(x_flat)
        x = x_flat.view(B, R, S, -1)
        
        # Pool over stabilizers
        x = x.mean(dim=2)  # (B, R, H)
        
        # Temporal attention with mask
        if mask is not None:
            # Create attention mask: True means masked (ignored)
            attn_mask = (mask == 0)  # (B, R)
            # For TransformerEncoder, we need to expand to (B*num_heads, R, R) or use src_key_padding_mask
            x = self.temporal_transformer(x, src_key_padding_mask=attn_mask)
        else:
            x = self.temporal_transformer(x)
        
        # Pool over rounds (only non-masked)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, R, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Classify
        logits = self.classifier(x).squeeze(-1)
        
        return logits


def train_epoch(model, loader, optimizer, device, scaler=None, accum_steps=1):
    """Train one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, ((x, mask, basis), y) in enumerate(loader):
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, mask, basis)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(x, mask, basis)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
        
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item() * accum_steps:.4f}", flush=True)
    
    # Handle remaining gradients
    if len(loader) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
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


def train(model, train_loader, val_loader, epochs, lr, device, save_dir, prefix="model", accum_steps=1):
    """Full training loop with checkpointing and gradient accumulation support."""
    model = model.to(device)
    
    if accum_steps > 1:
        print(f"Using gradient accumulation with {accum_steps} steps (effective batch = batch_size × {accum_steps})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    history = []
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, accum_steps=accum_steps)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
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
        
        # Get state dict
        model_state = model.state_dict()
        
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
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, save_path / f"{prefix}_epoch{epoch}.pth")
    
    # Save final model and history
    model_state = model.state_dict()
    torch.save(model_state, save_path / f"{prefix}_final.pth")
    
    with open(save_path / f"{prefix}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Full AlphaQubit Training Pipeline")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "eval"], default="pretrain")
    parser.add_argument("--data-dir", default="output")
    parser.add_argument("--filter-distance", type=int, default=None)
    parser.add_argument("--filter-rounds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=25, help="Pad/truncate sequences to this length")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PAPER_LR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--checkpoint", default=None, help="Load from checkpoint")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps (simulates larger batch)")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    
    # Set epochs based on mode
    if args.epochs is None:
        args.epochs = PAPER_PRETRAIN_EPOCHS if args.mode == "pretrain" else PAPER_FINETUNE_EPOCHS
    
    torch.manual_seed(42)
    
    # Device setup
    if args.npu:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = torch.device("npu:0")
                print(f"Using NPU: {torch.npu.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("NPU not available, using CPU")
        except ImportError:
            device = torch.device("cpu")
            print("torch_npu not installed, using CPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Compute effective batch size with gradient accumulation
    effective_batch = args.batch_size * args.accum_steps
    
    # Banner
    print("\n" + "="*70)
    print(f"AlphaQubit Training Pipeline - {args.mode.upper()}")
    print("="*70)
    print(f"  Data: {args.data_dir}")
    print(f"  Filter: distance={args.filter_distance}, rounds={args.filter_rounds}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, Accum: {args.accum_steps}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  LR: {args.lr}")
    print("="*70 + "\n")
    
    # Find data
    npz_files = discover_npz_files(args.data_dir, args.filter_distance, args.filter_rounds)
    
    if not npz_files:
        print("ERROR: No .npz files found")
        sys.exit(1)
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    print(f"Found {len(npz_files)} .npz files\n")
    
    # Build dataset
    print("Loading datasets with padding...")
    dataset = build_padded_dataset(npz_files, max_rounds=args.max_rounds)
    print(f"\nTotal samples: {len(dataset):,}")
    
    # Split
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    split = int(0.9 * n)
    
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds = torch.utils.data.Subset(dataset, indices[split:])
    
    # Use workers for data loading
    num_workers = 4
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # Get dimensions from first sample
    (x0, _, _), _ = dataset[0]
    # x0 shape is (max_rounds, S, F)
    R, S, F = x0.shape
    
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
    print(f"  Parameters: {n_params:,}")
    print(f"  Gradient accum steps: {args.accum_steps}\n")
    
    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Train
    prefix = f"{args.mode}_d{args.filter_distance or 'all'}"
    if args.filter_rounds:
        prefix += f"_r{args.filter_rounds}"
    
    print("Starting training...\n")
    model, history = train(model, train_loader, val_loader, args.epochs, args.lr, 
                          device, args.save_dir, prefix=prefix, accum_steps=args.accum_steps)
    
    # Summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best val loss: {min(h['val_loss'] for h in history):.4f}")
    print(f"Best val acc:  {max(h['val_acc'] for h in history):.4f}")
    print(f"Checkpoints saved to: {args.save_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
