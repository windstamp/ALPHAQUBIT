#!/usr/bin/env python3
"""
Train AlphaQubit from soft readout .npz files (3-channel format).

This script is designed for the Google QEC experiment-aligned data format:
- Shape: (N, rounds, detectors, 3) 
- Channels: detection events, P(|1⟩), P(|L⟩)

Usage:
    python train_soft_readout.py --data-dir output/ --epochs 100 --batch-size 256 --npu
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Paper-aligned hyperparameters
PAPER_BATCH_SIZE = 256
PAPER_LR = 1e-4
PAPER_EPOCHS = 100


class SoftReadoutDataset(Dataset):
    """
    Dataset for soft readout .npz files with 3-channel format.
    
    Expected .npz format:
        - data: (N, rounds, detectors, 3) - soft readout channels
        - observables: (N,) or (N, 1) - logical error labels
    """
    
    def __init__(self, npz_path: str, basis_id: Optional[int] = None):
        """
        Args:
            npz_path: Path to .npz file
            basis_id: 0 for X basis, 1 for Z basis. If None, inferred from filename.
        """
        data = np.load(npz_path, allow_pickle=True)
        
        # Get data - expect 3-channel format
        if 'data' in data:
            det_events = data['data']
        elif 'detection_events' in data:
            det_events = data['detection_events']
        else:
            raise KeyError(f"No data or detection_events in {npz_path}")
        
        # Get labels (observables)
        if 'observables' in data:
            labels = data['observables']
        elif 'labels' in data:
            labels = data['labels']
        elif 'obs' in data:
            labels = data['obs']  # Legacy key from older generation
        else:
            raise KeyError(f"No observables or labels in {npz_path}")
        
        # Get/infer metadata
        distance = int(data.get('distance', 0))
        rounds = int(data.get('rounds', 0))
        basis = str(data.get('basis', 'z'))
        
        # Infer basis from filename if not set
        if basis_id is None:
            if '_X_' in str(npz_path) or 'basis_x' in str(npz_path).lower():
                basis_id = 0
            else:
                basis_id = 1 if basis.lower() == 'z' else 0
        
        N = det_events.shape[0]
        
        # Handle different input shapes
        if det_events.ndim == 4:
            # Already in (N, R, S, F) format - ideal case
            x = det_events.astype(np.float32)
            _, R, S, F = x.shape
        elif det_events.ndim == 3:
            # Shape is (N, R, S) - add feature dim
            x = det_events[:, :, :, np.newaxis].astype(np.float32)
            _, R, S, F = x.shape
        elif det_events.ndim == 2:
            # Shape is (N, total) - need to infer R, S
            total_det = det_events.shape[1]
            if distance == 0:
                for d in [3, 5, 7]:
                    s = d * d - 1
                    if total_det % s == 0:
                        distance = d
                        break
            num_stabilizers = distance * distance - 1
            inferred_rounds = total_det // num_stabilizers
            det_events = det_events.reshape(N, inferred_rounds, num_stabilizers)
            x = det_events[:, :, :, np.newaxis].astype(np.float32)
            _, R, S, F = x.shape
        else:
            raise ValueError(f"Unexpected data shape: {det_events.shape}")
        
        # Infer distance if not set
        if distance == 0:
            d_sq = S + 1
            distance = int(math.sqrt(d_sq))
        
        # Add basis channel if we only have 3 channels (paper has 3 + basis = 4)
        if F == 3:
            # Add basis as 4th channel
            basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=np.float32)
            x = np.concatenate([x, basis_feat], axis=-1)
            F = 4
        
        # Pad stabilizers to make S+1 a perfect square
        d = math.isqrt(S + 1)
        if d * d != S + 1:
            d += 1
            pad = d * d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=np.float32)], axis=2)
            S = d * d - 1
        
        self.X = torch.tensor(x, dtype=torch.float32)
        
        # Labels
        labels = labels.flatten().astype(np.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        
        # Basis
        self.basis = torch.full((N,), basis_id, dtype=torch.long)
        
        # Final mask for stabilizer types
        self.final_mask = torch.zeros(N, S, dtype=torch.long)
        for idx, (r, c) in enumerate([(i // d, i % d) for i in range(1, S + 1)]):
            self.final_mask[:, idx] = 1 if (r + c) % 2 == 0 else 2
        
        self.distance = distance
        self.rounds = R
        self.num_stabilizers = S
        self.n_features = F
        
        print(f"  Loaded {Path(npz_path).name}: N={N}, R={R}, S={S}, F={F}, d={distance}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.basis[idx], self.final_mask[idx]), self.y[idx]


def discover_npz_files(data_dir: str) -> List[str]:
    """Find all .npz files in data directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    npz_files = []
    
    # Direct .npz files
    npz_files.extend(data_path.glob("*.npz"))
    
    # Subdirectories
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            npz_files.extend(subdir.glob("*.npz"))
    
    return sorted([str(f) for f in npz_files])


def build_dataset(npz_files: List[str]) -> ConcatDataset:
    """Build concatenated dataset from multiple .npz files."""
    datasets = []
    
    for npz_path in npz_files:
        try:
            ds = SoftReadoutDataset(npz_path)
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {npz_path}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    return ConcatDataset(datasets)


def build_datasets_by_shape(npz_files: List[str]) -> dict:
    """
    Build datasets grouped by shape (R, S, F).
    Returns dict: {(R, S, F): ConcatDataset}
    """
    shape_groups = {}  # (R, S, F) -> list of datasets
    
    for npz_path in npz_files:
        try:
            ds = SoftReadoutDataset(npz_path)
            if len(ds) > 0:
                # Get shape from first sample
                (x0, _, _), _ = ds[0]
                shape_key = tuple(x0.shape)  # (R, S, F)
                
                if shape_key not in shape_groups:
                    shape_groups[shape_key] = []
                shape_groups[shape_key].append(ds)
        except Exception as e:
            print(f"Warning: Could not load {npz_path}: {e}")
            continue
    
    if not shape_groups:
        raise ValueError("No valid datasets loaded")
    
    # Concatenate datasets within each group
    result = {}
    for shape_key, ds_list in shape_groups.items():
        result[shape_key] = ConcatDataset(ds_list)
        print(f"  Shape {shape_key}: {len(result[shape_key]):,} samples from {len(ds_list)} files")
    
    return result


class AlphaQubitDecoder(nn.Module):
    """
    AlphaQubit decoder with Transformer architecture.
    
    Processes soft readout data through:
    1. Input projection
    2. Spatial (grid) self-attention  
    3. Temporal (rounds) attention
    4. Classification head
    """
    
    def __init__(
        self,
        num_features: int = 4,
        hidden_dim: int = 256,
        num_stabilizers: int = 24,
        grid_size: int = 5,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_stabilizers = num_stabilizers
        self.grid_size = grid_size
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_stabilizers, hidden_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal attention (across rounds)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, basis=None, final_mask=None):
        """
        Args:
            x: (B, R, S, F) soft readout data
            basis: (B,) basis index (unused for now)
            final_mask: (B, S) stabilizer type mask
        
        Returns:
            logits: (B,) prediction logits
        """
        B, R, S, F = x.shape
        
        # Reshape for processing: (B, R, S, F) -> (B*R, S, F)
        x = x.view(B * R, S, F)
        
        # Input projection: (B*R, S, F) -> (B*R, S, H)
        x = self.input_proj(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :S, :]
        
        # Spatial self-attention
        x = self.transformer(x)  # (B*R, S, H)
        
        # Pool over stabilizers: (B*R, S, H) -> (B*R, H)
        x = x.mean(dim=1)
        
        # Reshape for temporal: (B*R, H) -> (B, R, H)
        x = x.view(B, R, -1)
        
        # Temporal attention
        x_attn, _ = self.temporal_attn(x, x, x)  # (B, R, H)
        x = x + x_attn
        
        # Pool over rounds: (B, R, H) -> (B, H)
        x = x.mean(dim=1)
        
        # Classify
        logits = self.classifier(x).squeeze(-1)  # (B,)
        
        return logits


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, ((x, basis, mask), y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        basis = basis.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x, basis, mask)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for (x, basis, mask), y in loader:
        x = x.to(device)
        y = y.to(device)
        basis = basis.to(device)
        mask = mask.to(device)
        
        logits = model(x, basis, mask)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
    
    return total_loss / total_samples, total_correct / total_samples


def train(model, train_loader, val_loader, epochs, lr, device, save_dir="."):
    """Full training loop."""
    
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs/NPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, save_path / "alphaqubit_best.pth")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train AlphaQubit from soft readout .npz files")
    parser.add_argument("--data-dir", default="output",
                        help="Directory containing .npz files")
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS,
                        help=f"Number of epochs (paper: {PAPER_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE,
                        help=f"Batch size (paper: {PAPER_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=PAPER_LR,
                        help=f"Learning rate (paper: {PAPER_LR})")
    parser.add_argument("--npu", action="store_true", help="Use NPU (Ascend)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--save-dir", default=".", help="Directory to save checkpoints")
    parser.add_argument("--max-files", type=int, default=None, 
                        help="Maximum number of .npz files to load (for testing)")
    parser.add_argument("--filter-distance", type=int, default=None,
                        help="Only use files with this distance (3 or 5)")
    parser.add_argument("--filter-rounds", type=int, default=None,
                        help="Only use files with this number of rounds")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    # Device setup
    if args.npu:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = torch.device("npu:0")
                print(f"Using NPU: {torch.npu.get_device_name(0)}")
            else:
                print("NPU not available, falling back to CPU")
                device = torch.device("cpu")
        except ImportError:
            print("torch_npu not installed, falling back to CPU")
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Find data files
    print(f"\n{'='*60}")
    print(f"Searching for .npz files in {args.data_dir}...")
    npz_files = discover_npz_files(args.data_dir)
    
    if not npz_files:
        print(f"ERROR: No .npz files found in {args.data_dir}")
        sys.exit(1)
    
    # Filter by distance/rounds if specified
    if args.filter_distance:
        # Filter by distance in filename (e.g., _d3_ or _d5_)
        pattern = f"_d{args.filter_distance}_"
        npz_files = [f for f in npz_files if pattern in f]
        print(f"Filtered to distance={args.filter_distance}: {len(npz_files)} files")
    
    if args.filter_rounds:
        # Filter by rounds in filename (e.g., _r25_)
        pattern = f"_r{args.filter_rounds:02d}_"
        npz_files = [f for f in npz_files if pattern in f]
        print(f"Filtered to rounds={args.filter_rounds}: {len(npz_files)} files")
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
        print(f"Limited to {args.max_files} files for testing")
    
    if not npz_files:
        print("ERROR: No files remain after filtering")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Build datasets grouped by shape
    print(f"\n{'='*60}")
    print("Loading datasets (grouped by shape)...")
    shape_datasets = build_datasets_by_shape(npz_files)
    
    # Select the largest dataset group for training
    # Or if all shapes are the same, use that
    if len(shape_datasets) == 1:
        shape_key = list(shape_datasets.keys())[0]
        dataset = shape_datasets[shape_key]
        print(f"\nAll files have same shape: {shape_key}")
    else:
        # Multiple shapes - pick the one with most samples
        print(f"\nWARNING: Found {len(shape_datasets)} different shapes!")
        print("Selecting shape with most samples for training...")
        shape_key = max(shape_datasets.keys(), key=lambda k: len(shape_datasets[k]))
        dataset = shape_datasets[shape_key]
        print(f"Selected shape {shape_key} with {len(dataset):,} samples")
        print("TIP: Use --filter-distance and --filter-rounds to select specific data")
    
    print(f"Total training samples: {len(dataset):,}")
    
    # Split into train/val (90/10)
    n = len(dataset)
    indices = torch.randperm(n)
    split = int(0.9 * n)
    
    train_ds = torch.utils.data.Subset(dataset, indices[:split].tolist())
    val_ds = torch.utils.data.Subset(dataset, indices[split:].tolist())
    
    # Use num_workers=0 to avoid multiprocessing issues on NPU
    num_workers = 0 if args.npu else 4
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")
    
    # Get dimensions from shape key
    R, S, F = shape_key
    d = int(math.sqrt(S + 1))
    grid_size = d
    
    print(f"\n{'='*60}")
    print(f"Model configuration:")
    print(f"  Input: R={R} rounds, S={S} stabilizers, F={F} features")
    print(f"  Grid: {d}x{d}")
    print(f"  Hidden: {args.hidden_dim}, Layers: {args.num_layers}, Heads: {args.num_heads}")
    
    # Create model
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=args.hidden_dim,
        num_stabilizers=S,
        grid_size=grid_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"  Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    model = train(model, train_loader, val_loader, args.epochs, args.lr, device, args.save_dir)
    
    # Save final model
    final_path = Path(args.save_dir) / "alphaqubit_final.pth"
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state, final_path)
    print(f"\n✓ Saved final model: {final_path}")


if __name__ == "__main__":
    main()
