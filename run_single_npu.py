#!/usr/bin/env python3
"""
AlphaQubit Multi-NPU Training with DataParallel (No HCCL)

Uses all 8 NPUs via DataParallel - avoids HCCL distributed training complexity.
DataParallel replicates model to all NPUs and splits batches automatically.

Usage:
    python run_single_npu.py --quick-test
    python run_single_npu.py --samples 50000 --epochs 50
"""

import os
import sys
import argparse
import json
import math
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Device Setup (All NPUs with DataParallel)
# ============================================================================

def setup_device():
    """Setup all NPUs with DataParallel or fallback to CUDA/CPU."""
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            npu_count = torch.npu.device_count()
            logger.info(f"Found {npu_count} NPUs available")
            device = torch.device('npu:0')  # Primary device
            return device, npu_count
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPUs available")
        device = torch.device('cuda:0')
        return device, gpu_count
    
    logger.info("Using CPU")
    return torch.device('cpu'), 1


def wrap_model_dataparallel(model, device, device_count):
    """Wrap model with DataParallel if multiple devices available."""
    model = model.to(device)
    
    if device_count > 1:
        if device.type == 'npu':
            device_ids = list(range(device_count))
            model = nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel on {device_count} NPUs")
        elif device.type == 'cuda':
            device_ids = list(range(device_count))
            model = nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel on {device_count} GPUs")
    
    return model


# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_data(
    num_samples: int,
    distance: int,
    rounds: int,
    error_rate: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic QEC training data."""
    if seed is not None:
        np.random.seed(seed)
    
    num_stabilizers = distance * distance - 1
    detection_prob = 2 * error_rate * (1 - error_rate)
    
    detection_events = np.random.binomial(
        1, detection_prob, 
        size=(num_samples, rounds, num_stabilizers)
    ).astype(np.float32)
    
    p_th = 0.01
    ler_per_round = (error_rate / p_th) ** ((distance + 1) / 2) * 0.01
    total_ler = 1 - (1 - ler_per_round) ** rounds
    total_ler = np.clip(total_ler, 0.01, 0.5)
    
    logical_errors = np.random.binomial(
        1, total_ler, size=(num_samples,)
    ).astype(np.float32)
    
    soft_readout = np.clip(
        detection_events + np.random.normal(0, 0.1, detection_events.shape),
        0, 1
    ).astype(np.float32)
    
    basis_id = np.zeros_like(detection_events)
    features = np.stack([detection_events, soft_readout, basis_id], axis=-1)
    
    return features, logical_errors


def generate_pretraining_data(
    output_dir: str,
    samples_per_config: int = 10000,
    distances: List[int] = [3, 5],
    error_rates: List[float] = [0.01]
) -> List[str]:
    """Generate pretraining data - single distance to avoid padding issues."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Use single distance to avoid dimension mismatch
    d = max(distances)
    r = d * 5  # rounds = 5*d
    
    for p in error_rates:
        for seed in range(3):
            filename = f"pretrain_d{d}_r{r}_p{p:.4f}_seed{seed}.npz"
            filepath = output_path / filename
            
            logger.info(f"Generating {filename}...")
            
            features, labels = generate_synthetic_data(
                num_samples=samples_per_config,
                distance=d,
                rounds=r,
                error_rate=p,
                seed=seed * 1000 + d * 100 + int(p * 10000)
            )
            
            np.savez_compressed(
                filepath,
                data=features,
                labels=labels,
                distance=d,
                rounds=r,
                error_rate=p
            )
            
            generated_files.append(str(filepath))
            logger.info(f"  Saved {len(labels)} samples to {filepath}")
    
    manifest = {
        "files": generated_files,
        "total_samples": len(generated_files) * samples_per_config,
        "distance": d,
        "rounds": r,
        "generated_at": datetime.now().isoformat()
    }
    
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Generated {len(generated_files)} files")
    
    return generated_files


# ============================================================================
# Dataset
# ============================================================================

class NPZDataset(Dataset):
    """Simple dataset from NPZ files."""
    
    def __init__(self, npz_files: List[str]):
        data_list = []
        labels_list = []
        
        for npz_file in npz_files:
            with np.load(npz_file) as data:
                data_list.append(data['data'])
                labels_list.append(data['labels'])
        
        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        
        self.n_samples, self.n_rounds, self.n_stabilizers, self.n_features = self.data.shape
        
        d = int(math.sqrt(self.n_stabilizers + 1))
        if d * d != self.n_stabilizers + 1:
            d = int(math.sqrt(self.n_stabilizers)) + 1
        self.grid_size = d
        
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 
             for r in range(d) for c in range(d)][1:self.n_stabilizers+1],
            dtype=torch.long
        )
        
        if len(self.final_mask) < self.n_stabilizers:
            padding = torch.zeros(self.n_stabilizers - len(self.final_mask), dtype=torch.long)
            self.final_mask = torch.cat([self.final_mask, padding])
        
        logger.info(f"Loaded {self.n_samples} samples")
        logger.info(f"  Shape: R={self.n_rounds}, S={self.n_stabilizers}, F={self.n_features}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        basis = torch.tensor(0, dtype=torch.long)
        return (data, basis, self.final_mask), label


# ============================================================================
# Model (Simplified)
# ============================================================================

class StabilizerEmbedder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
        super().__init__()
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
        self.final_on_emb = nn.Embedding(1, hidden_dim)
        self.final_off_emb = nn.Embedding(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = torch.zeros(B, S, self.index_embedding.embedding_dim, device=x.device)
        
        for i, proj in enumerate(self.feature_projs):
            h = h + proj(x[..., i:i+1])

        idx = torch.arange(S, device=x.device)
        h = h + self.index_embedding(idx)

        h = h + (final_mask == 1).unsqueeze(-1).float() * \
              self.final_on_emb(torch.zeros((), dtype=torch.long, device=x.device))
        h = h + (final_mask == 2).unsqueeze(-1).float() * \
              self.final_off_emb(torch.zeros((), dtype=torch.long, device=x.device))
        
        return self.norm(h)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x


class AlphaQubitModel(nn.Module):
    def __init__(
        self,
        num_features: int = 3,
        hidden_dim: int = 256,
        num_stabilizers: int = 24,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, inputs: torch.Tensor, basis: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        B, R, S, F = inputs.shape
        
        state = torch.zeros(B, S, self.hidden_dim, device=inputs.device)
        
        for r in range(R):
            x = inputs[:, r]
            emb = self.embedder(x, final_mask if r == R - 1 else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            
            for layer in self.layers:
                state = layer(state)
        
        state = self.norm(state)
        pooled = state.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    save_dir: str = "results/single_npu"
):
    """Train the model on single device."""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, ((data, basis, mask), labels) in enumerate(train_loader):
            data = data.to(device)
            basis = basis.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(data, basis, mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for (data, basis, mask), labels in val_loader:
                data = data.to(device)
                basis = basis.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                
                logits = model(data, basis, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * data.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_samples += data.size(0)
        
        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                   f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
                   f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path / "best_model.pth")
            logger.info(f"  New best! Saved model with acc={val_acc:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc
            }, save_path / f"checkpoint_epoch{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), save_path / "final_model.pth")
    logger.info(f"Training complete! Best acc: {best_acc:.4f}")
    
    return best_acc


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AlphaQubit Single-NPU Training')
    
    parser.add_argument('--data-dir', type=str, default='pretrain_data/single',
                        help='Directory for training data')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Samples per configuration')
    parser.add_argument('--skip-datagen', action='store_true')
    
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=12)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--output-dir', type=str, default='results/single_npu')
    parser.add_argument('--quick-test', action='store_true')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.samples = 5000
        args.epochs = 10
        args.num_layers = 4
        logger.info("Quick test mode")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    device = setup_device()
    
    # Generate data
    if not args.skip_datagen:
        logger.info("=" * 60)
        logger.info("Step 1: Generating training data")
        logger.info("=" * 60)
        
        data_files = generate_pretraining_data(
            output_dir=args.data_dir,
            samples_per_config=args.samples,
            distances=[5],  # Single distance to avoid padding
            error_rates=[0.005, 0.01, 0.015] if not args.quick_test else [0.01]
        )
    else:
        data_dir = Path(args.data_dir)
        data_files = [str(f) for f in data_dir.glob("*.npz")]
        if not data_files:
            raise FileNotFoundError(f"No .npz files in {args.data_dir}")
    
    # Load dataset
    logger.info("=" * 60)
    logger.info("Step 2: Loading data")
    logger.info("=" * 60)
    
    dataset = NPZDataset(data_files)
    
    # Split
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    split = int(0.9 * n)
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("=" * 60)
    logger.info("Step 3: Training model")
    logger.info("=" * 60)
    
    model = AlphaQubitModel(
        num_features=dataset.n_features,
        hidden_dim=args.hidden_dim,
        num_stabilizers=dataset.n_stabilizers,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Train
    best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.output_dir
    )
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best accuracy: {best_acc:.4f}")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
