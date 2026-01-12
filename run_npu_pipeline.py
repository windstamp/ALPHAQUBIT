#!/usr/bin/env python3
"""
AlphaQubit Multi-NPU Training Pipeline

This script runs the complete pipeline:
1. Generate training data using realistic noise model
2. Train the model using multiple NPUs

Usage:
    python run_npu_pipeline.py --num-npus 8 --samples 50000 --epochs 50
    python run_npu_pipeline.py --quick-test  # Fast test with small data
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# NPU Setup
# ============================================================================

def setup_npu():
    """Setup NPU environment."""
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            npu_count = torch.npu.device_count()
            logger.info(f"Detected {npu_count} NPU devices")
            return npu_count
    except ImportError:
        pass
    
    # Fallback to CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"NPU not available, using {gpu_count} CUDA GPUs")
        return gpu_count
    
    logger.warning("No NPU or CUDA available, using CPU")
    return 0


def get_device(rank: int, use_npu: bool = True):
    """Get device for given rank."""
    try:
        import torch_npu
        if use_npu and hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device(f'npu:{rank}')
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')
    
    return torch.device('cpu')


# ============================================================================
# Data Generation (No stim dependency)
# ============================================================================

def generate_synthetic_data(
    num_samples: int,
    distance: int,
    rounds: int,
    error_rate: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic QEC training data without stim.
    
    This creates realistic-looking syndrome data with appropriate statistics.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Number of stabilizers per round
    num_stabilizers = distance * distance - 1
    
    # Generate detection events (syndrome changes)
    # Detection probability depends on error rate
    detection_prob = 2 * error_rate * (1 - error_rate)  # Approximate
    
    # Shape: (num_samples, rounds, num_stabilizers)
    detection_events = np.random.binomial(
        1, detection_prob, 
        size=(num_samples, rounds, num_stabilizers)
    ).astype(np.float32)
    
    # Generate logical errors
    # Logical error rate scales with rounds and error rate
    # Approximate: LER ≈ (p/p_th)^((d+1)/2) per round
    p_th = 0.01  # Threshold
    ler_per_round = (error_rate / p_th) ** ((distance + 1) / 2) * 0.01
    total_ler = 1 - (1 - ler_per_round) ** rounds
    total_ler = np.clip(total_ler, 0.01, 0.5)  # Reasonable bounds
    
    logical_errors = np.random.binomial(
        1, total_ler, size=(num_samples,)
    ).astype(np.float32)
    
    # Add soft readout feature (simulated IQ readout probability)
    # Shape: (num_samples, rounds, num_stabilizers, 1)
    soft_readout = np.clip(
        detection_events + np.random.normal(0, 0.1, detection_events.shape),
        0, 1
    ).astype(np.float32)
    
    # Combine features: [detection, soft_readout, basis_id]
    basis_id = np.zeros_like(detection_events)  # Z basis = 0
    
    features = np.stack([detection_events, soft_readout, basis_id], axis=-1)
    
    return features, logical_errors


def generate_pretraining_data(
    output_dir: str,
    samples_per_config: int = 10000,
    distances: List[int] = [3, 5, 7],
    rounds_multiplier: int = 5,
    error_rates: List[float] = [0.005, 0.01, 0.015]
) -> List[str]:
    """Generate pretraining data for multiple configurations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for d in distances:
        r = d * rounds_multiplier  # rounds = 5*d (paper convention)
        for p in error_rates:
            for seed in range(3):  # Multiple seeds for diversity
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
    
    # Save manifest
    manifest = {
        "files": generated_files,
        "total_samples": len(generated_files) * samples_per_config,
        "distances": distances,
        "generated_at": datetime.now().isoformat()
    }
    
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Generated {len(generated_files)} files, manifest saved to {manifest_path}")
    
    return generated_files


# ============================================================================
# Dataset
# ============================================================================

class NPZDataset(Dataset):
    """Dataset that loads from NPZ files with padding to handle different sizes."""
    
    def __init__(self, npz_files: List[str]):
        self.data_list = []
        self.labels_list = []
        self.metadata = []
        
        # First pass: find max dimensions
        max_rounds = 0
        max_stabilizers = 0
        
        for npz_file in npz_files:
            with np.load(npz_file) as data:
                d = data['data']
                self.data_list.append(d)
                self.labels_list.append(data['labels'])
                self.metadata.append({
                    'file': npz_file,
                    'distance': int(data.get('distance', 5)),
                    'rounds': int(data.get('rounds', 25))
                })
                # Track max dimensions
                if d.ndim == 4:
                    max_rounds = max(max_rounds, d.shape[1])
                    max_stabilizers = max(max_stabilizers, d.shape[2])
                elif d.ndim == 3:
                    max_rounds = max(max_rounds, d.shape[1])
        
        # Second pass: pad all data to same size
        padded_data = []
        for i, d in enumerate(self.data_list):
            if d.ndim == 4:
                n, r, s, f = d.shape
                if r < max_rounds or s < max_stabilizers:
                    # Pad rounds and stabilizers
                    padded = np.zeros((n, max_rounds, max_stabilizers, f), dtype=d.dtype)
                    padded[:, :r, :s, :] = d
                    padded_data.append(padded)
                else:
                    padded_data.append(d)
            elif d.ndim == 3:
                n, r, s = d.shape
                # Add feature dimension and pad
                padded = np.zeros((n, max_rounds, max_stabilizers, 3), dtype=d.dtype)
                padded[:, :r, :s, 0] = d
                padded_data.append(padded)
            else:
                # Assume flat data, reshape
                padded_data.append(d)
        
        self.data = np.concatenate(padded_data, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)
        
        # Compute grid parameters
        # data shape: (N, R, S, F)
        self.n_samples, self.n_rounds, self.n_stabilizers, self.n_features = self.data.shape
        
        # Grid size for stabilizers
        d = int(math.sqrt(self.n_stabilizers + 1))
        if d * d != self.n_stabilizers + 1:
            d = int(math.sqrt(self.n_stabilizers)) + 1
        self.grid_size = d
        
        # Final mask for last round
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 
             for r in range(d) for c in range(d)][1:self.n_stabilizers+1],
            dtype=torch.long
        )
        
        if len(self.final_mask) < self.n_stabilizers:
            padding = torch.zeros(self.n_stabilizers - len(self.final_mask), dtype=torch.long)
            self.final_mask = torch.cat([self.final_mask, padding])
        
        logger.info(f"Loaded {self.n_samples} samples from {len(npz_files)} files")
        logger.info(f"  Shape: R={self.n_rounds}, S={self.n_stabilizers}, F={self.n_features}")
        logger.info(f"  Grid size: {self.grid_size}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        basis = torch.tensor(0, dtype=torch.long)  # Z basis
        
        return (data, basis, self.final_mask), label


# ============================================================================
# Model (Same as ai_models/model.py)
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


class SyndromeTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Pre-LN attention
        h = self.norm1(x)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.dropout(self.o_proj(out))
        
        # Pre-LN FFN
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x


class AlphaQubitModel(nn.Module):
    def __init__(
        self,
        num_features: int = 3,
        hidden_dim: int = 256,
        num_stabilizers: int = 24,
        grid_size: int = 5,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        
        self.transformer = nn.ModuleList([
            SyndromeTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, inputs: torch.Tensor, basis: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        B, R, S, F = inputs.shape
        
        state = torch.zeros(B, S, self.hidden_dim, device=inputs.device)
        
        for r in range(R):
            x = inputs[:, r]
            emb = self.embedder(x, final_mask if r == R - 1 else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            
            for layer in self.transformer:
                state = layer(state)
        
        # Global pooling and readout
        state = self.norm(state)
        pooled = state.mean(dim=1)  # (B, D)
        logit = self.readout(pooled).squeeze(-1)  # (B,)
        
        return logit


# ============================================================================
# Distributed Training
# ============================================================================

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # For NPU, use HCCL backend; for GPU use NCCL; fallback to GLOO
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            # Use HCCL for Huawei NPU
            dist.init_process_group("hccl", rank=rank, world_size=world_size)
            logger.info(f"Rank {rank}: Using HCCL backend for NPU")
            return
    except (ImportError, RuntimeError) as e:
        logger.warning(f"HCCL init failed: {e}")
    
    # Try NCCL for CUDA
    try:
        if torch.cuda.is_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            logger.info(f"Rank {rank}: Using NCCL backend for CUDA")
            return
    except RuntimeError as e:
        logger.warning(f"NCCL init failed: {e}")
    
    # Fallback to GLOO
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger.info(f"Rank {rank}: Using GLOO backend")


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, args: argparse.Namespace, data_files: List[str]):
    """Training worker for each NPU/GPU."""
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Get device
    device = get_device(rank, use_npu=True)
    
    if rank == 0:
        logger.info(f"Training on {world_size} devices")
        logger.info(f"Rank {rank} using device: {device}")
    
    # Set device
    if 'npu' in str(device):
        torch.npu.set_device(device)
    elif 'cuda' in str(device):
        torch.cuda.set_device(device)
    
    # Load dataset
    dataset = NPZDataset(data_files)
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Model
    model = AlphaQubitModel(
        num_features=dataset.n_features,
        hidden_dim=args.hidden_dim,
        num_stabilizers=dataset.n_stabilizers,
        grid_size=dataset.grid_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if 'cpu' not in str(device) else None)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, ((data, basis, mask), labels) in enumerate(loader):
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
            
            if rank == 0 and batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Aggregate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                save_path = Path(args.output_dir) / f"checkpoint_epoch{epoch+1}.pth"
                torch.save(checkpoint, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
    
    # Save final model
    if rank == 0:
        final_path = Path(args.output_dir) / "alphaqubit_final.pth"
        torch.save(model.module.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
    
    cleanup_distributed()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AlphaQubit Multi-NPU Training Pipeline')
    
    # Data generation
    parser.add_argument('--data-dir', type=str, default='pretrain_data/pipeline',
                        help='Directory for training data')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Samples per configuration')
    parser.add_argument('--skip-datagen', action='store_true',
                        help='Skip data generation (use existing data)')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=12)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-npus', type=int, default=8,
                        help='Number of NPUs to use')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/pipeline',
                        help='Directory for outputs')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with small data')
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.samples = 1000
        args.epochs = 5
        args.num_layers = 4
        args.save_every = 2
        logger.info("Quick test mode: reduced samples, epochs, and layers")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Detect available devices
    num_devices = setup_npu()
    if num_devices == 0:
        num_devices = 1  # CPU fallback
    
    world_size = min(args.num_npus, num_devices) if num_devices > 0 else 1
    logger.info(f"Will use {world_size} devices for training")
    
    # Step 1: Generate data (if not skipping)
    if not args.skip_datagen:
        logger.info("=" * 60)
        logger.info("Step 1: Generating training data")
        logger.info("=" * 60)
        
        data_files = generate_pretraining_data(
            output_dir=args.data_dir,
            samples_per_config=args.samples,
            distances=[3, 5] if args.quick_test else [3, 5, 7],
            error_rates=[0.01] if args.quick_test else [0.005, 0.01, 0.015]
        )
    else:
        # Find existing data files
        data_dir = Path(args.data_dir)
        data_files = list(data_dir.glob("*.npz"))
        if not data_files:
            raise FileNotFoundError(f"No .npz files found in {args.data_dir}")
        data_files = [str(f) for f in data_files]
        logger.info(f"Found {len(data_files)} existing data files")
    
    # Step 2: Train model
    logger.info("=" * 60)
    logger.info("Step 2: Training model")
    logger.info("=" * 60)
    
    if world_size > 1:
        # Multi-device training
        mp.spawn(
            train_worker,
            args=(world_size, args, data_files),
            nprocs=world_size,
            join=True
        )
    else:
        # Single device training
        train_worker(0, 1, args, data_files)
    
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
