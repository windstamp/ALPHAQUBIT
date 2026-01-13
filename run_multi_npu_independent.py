#!/usr/bin/env python3
"""
AlphaQubit Multi-NPU Independent Training

Each NPU trains its own model independently - NO communication between NPUs.
- NPU 0: d=3, p=0.001-0.003
- NPU 1: d=3, p=0.004-0.006
- NPU 2: d=3, p=0.007-0.010
- NPU 3: d=5, p=0.001-0.003
- NPU 4: d=5, p=0.004-0.006
- NPU 5: d=5, p=0.007-0.010
- NPU 6: d=7, p=0.001-0.005
- NPU 7: d=7, p=0.006-0.010

Benefits:
- No HCCL, no DDP, no communication overhead
- Each NPU is fully independent - no timeout issues
- 8x throughput compared to single NPU
- Can ensemble models for better accuracy

Usage:
    python run_multi_npu_independent.py --quick-test
    python run_multi_npu_independent.py --samples 50000 --epochs 50
"""

import os
import sys

# CRITICAL: Set spawn method BEFORE any other imports that might create locks
import multiprocessing as mp
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import argparse
import json
import math
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(npu_id: int, log_dir: str = "logs"):
    """Setup logger for each NPU process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"NPU{npu_id}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(f"{log_dir}/npu{npu_id}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(f'%(asctime)s [NPU{npu_id}] %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# NPU Training Configuration
# ============================================================================

def get_npu_config(npu_id: int, quick_test: bool = False) -> Dict[str, Any]:
    """
    Get training configuration for each NPU.
    Each NPU trains on different distance/error_rate combinations.
    """
    if quick_test:
        # Quick test: all NPUs train same small config
        return {
            "npu_id": npu_id,
            "distance": 3,
            "error_rates": [0.005, 0.01],
            "samples_per_rate": 2000,
            "epochs": 5,
            "description": f"NPU{npu_id} quick test"
        }
    
    # Full training: distribute work across NPUs
    configs = [
        # NPU 0-2: d=3 with different error rates
        {"distance": 3, "error_rates": [0.001, 0.002, 0.003], "description": "d=3 low noise"},
        {"distance": 3, "error_rates": [0.004, 0.005, 0.006], "description": "d=3 mid noise"},
        {"distance": 3, "error_rates": [0.007, 0.008, 0.009, 0.010], "description": "d=3 high noise"},
        
        # NPU 3-5: d=5 with different error rates
        {"distance": 5, "error_rates": [0.001, 0.002, 0.003], "description": "d=5 low noise"},
        {"distance": 5, "error_rates": [0.004, 0.005, 0.006], "description": "d=5 mid noise"},
        {"distance": 5, "error_rates": [0.007, 0.008, 0.009, 0.010], "description": "d=5 high noise"},
        
        # NPU 6-7: d=7 with different error rates
        {"distance": 7, "error_rates": [0.001, 0.002, 0.003, 0.004, 0.005], "description": "d=7 low noise"},
        {"distance": 7, "error_rates": [0.006, 0.007, 0.008, 0.009, 0.010], "description": "d=7 high noise"},
    ]
    
    config = configs[npu_id % len(configs)]
    config["npu_id"] = npu_id
    config["samples_per_rate"] = 50000
    config["epochs"] = 50
    
    return config


# ============================================================================
# Data Generation (uses stim if available, else synthetic)
# ============================================================================

def generate_training_data(
    distance: int,
    error_rates: List[float],
    samples_per_rate: int,
    output_dir: str,
    npu_id: int,
    logger: logging.Logger
) -> List[str]:
    """Generate training data for this NPU."""
    
    output_path = Path(output_dir) / f"npu{npu_id}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    rounds = distance * 3  # rounds = 3*d
    num_stabilizers = distance * distance - 1
    
    for error_rate in error_rates:
        filename = f"train_d{distance}_r{rounds}_p{error_rate:.4f}.npz"
        filepath = output_path / filename
        
        logger.info(f"Generating {samples_per_rate} samples: d={distance}, p={error_rate:.4f}")
        
        try:
            # Try using stim for realistic data
            import stim
            data, labels = generate_stim_data(
                distance=distance,
                rounds=rounds,
                error_rate=error_rate,
                num_samples=samples_per_rate,
                seed=npu_id * 10000 + int(error_rate * 100000)
            )
        except ImportError:
            # Fallback to synthetic data
            logger.warning("stim not available, using synthetic data")
            data, labels = generate_synthetic_data(
                distance=distance,
                rounds=rounds,
                error_rate=error_rate,
                num_samples=samples_per_rate,
                seed=npu_id * 10000 + int(error_rate * 100000)
            )
        
        np.savez_compressed(
            filepath,
            data=data,
            labels=labels,
            distance=distance,
            rounds=rounds,
            error_rate=error_rate
        )
        
        generated_files.append(str(filepath))
        logger.info(f"  Saved {len(labels)} samples to {filepath}")
    
    return generated_files


def generate_stim_data(
    distance: int,
    rounds: int,
    error_rate: float,
    num_samples: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data using stim surface code simulation."""
    import stim
    
    # Generate rotated surface code circuit
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=error_rate,
        before_measure_flip_probability=error_rate,
        after_reset_flip_probability=error_rate,
    )
    
    # Sample detection events and observables
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observables = sampler.sample(
        num_samples, separate_observables=True
    )
    
    # Convert to features: (samples, rounds, stabilizers, features)
    num_stabilizers = distance * distance - 1
    num_detectors = detection_events.shape[1]
    
    # Reshape detection events to (samples, rounds, stabilizers)
    detectors_per_round = num_stabilizers
    actual_rounds = num_detectors // detectors_per_round
    
    if actual_rounds * detectors_per_round != num_detectors:
        # Pad or truncate
        target_detectors = rounds * num_stabilizers
        if num_detectors < target_detectors:
            padding = np.zeros((num_samples, target_detectors - num_detectors), dtype=detection_events.dtype)
            detection_events = np.concatenate([detection_events, padding], axis=1)
        else:
            detection_events = detection_events[:, :target_detectors]
    
    detection_events = detection_events.reshape(num_samples, rounds, num_stabilizers)
    
    # Create 3-feature tensor: (detection, soft_readout, basis_id)
    detection_float = detection_events.astype(np.float32)
    soft_readout = np.clip(
        detection_float + np.random.normal(0, 0.05, detection_float.shape),
        0, 1
    ).astype(np.float32)
    basis_id = np.zeros_like(detection_float)  # Z-basis = 0
    
    features = np.stack([detection_float, soft_readout, basis_id], axis=-1)
    labels = observables[:, 0].astype(np.float32)
    
    return features, labels


def generate_synthetic_data(
    distance: int,
    rounds: int,
    error_rate: float,
    num_samples: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic QEC training data (fallback)."""
    np.random.seed(seed)
    
    num_stabilizers = distance * distance - 1
    detection_prob = 2 * error_rate * (1 - error_rate)
    
    # Detection events
    detection_events = np.random.binomial(
        1, detection_prob, 
        size=(num_samples, rounds, num_stabilizers)
    ).astype(np.float32)
    
    # Logical error probability (approximation)
    p_th = 0.01
    ler_per_round = (error_rate / p_th) ** ((distance + 1) / 2) * 0.01
    total_ler = 1 - (1 - ler_per_round) ** rounds
    total_ler = np.clip(total_ler, 0.01, 0.5)
    
    logical_errors = np.random.binomial(
        1, total_ler, size=(num_samples,)
    ).astype(np.float32)
    
    # Create features
    soft_readout = np.clip(
        detection_events + np.random.normal(0, 0.1, detection_events.shape),
        0, 1
    ).astype(np.float32)
    basis_id = np.zeros_like(detection_events)
    
    features = np.stack([detection_events, soft_readout, basis_id], axis=-1)
    
    return features, logical_errors


# ============================================================================
# Dataset (Module-level for pickling)
# ============================================================================

class NPZDataset:
    """Simple dataset from NPZ files - module level for multiprocessing."""
    
    def __init__(self, npz_files: List[str]):
        import torch
        
        data_list = []
        labels_list = []
        
        for f in npz_files:
            with np.load(f) as npz:
                data_list.append(npz['data'])
                labels_list.append(npz['labels'])
        
        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        
        self.n_samples, self.n_rounds, self.n_stabilizers, self.n_features = self.data.shape
        
        # Create stabilizer type mask
        d = int(math.sqrt(self.n_stabilizers + 1))
        self.grid_size = d
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 
             for r in range(d) for c in range(d)][1:self.n_stabilizers+1],
            dtype=torch.long
        )
        if len(self.final_mask) < self.n_stabilizers:
            padding = torch.zeros(self.n_stabilizers - len(self.final_mask), dtype=torch.long)
            self.final_mask = torch.cat([self.final_mask, padding])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        import torch
        data = torch.from_numpy(self.data[idx].copy()).float()
        label = torch.tensor(self.labels[idx]).float()
        basis = torch.tensor(0, dtype=torch.long)
        return (data, basis, self.final_mask), label


def create_dataset(npz_files: List[str], logger: logging.Logger):
    """Create PyTorch dataset from NPZ files."""
    dataset = NPZDataset(npz_files)
    logger.info(f"Dataset: {dataset.n_samples} samples, shape=({dataset.n_rounds}, {dataset.n_stabilizers}, {dataset.n_features})")
    return dataset


# ============================================================================
# Model
# ============================================================================

def create_model(num_features: int, num_stabilizers: int, hidden_dim: int, num_heads: int, num_layers: int):
    """Create AlphaQubit model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class StabilizerEmbedder(nn.Module):
        def __init__(self, num_features, hidden_dim, num_stabilizers):
            super().__init__()
            self.feature_projs = nn.ModuleList([
                nn.Linear(1, hidden_dim) for _ in range(num_features)
            ])
            self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
            self.final_on_emb = nn.Embedding(1, hidden_dim)
            self.final_off_emb = nn.Embedding(1, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x, final_mask):
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
        def __init__(self, hidden_dim, num_heads, dropout=0.1):
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
        
        def forward(self, x):
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
        def __init__(self, num_features, hidden_dim, num_stabilizers, num_heads, num_layers, dropout=0.1):
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
        
        def forward(self, inputs, basis, final_mask):
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
    
    return AlphaQubitModel(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_stabilizers=num_stabilizers,
        num_heads=num_heads,
        num_layers=num_layers
    )


# ============================================================================
# Training Loop
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    save_dir: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Train model and return results."""
    import torch
    import torch.nn as nn
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
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
            
            if batch_idx % 50 == 0:
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
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                   f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path / "best_model.pth")
            logger.info(f"  New best! Saved model with acc={val_acc:.4f}")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, save_path / f"checkpoint_epoch{epoch+1}.pth")
    
    # Save final
    torch.save(model.state_dict(), save_path / "final_model.pth")
    
    # Save history
    with open(save_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return {"best_acc": best_acc, "history": history}


# ============================================================================
# NPU Worker Process
# ============================================================================

def npu_worker(
    npu_id: int,
    config: Dict[str, Any],
    args: argparse.Namespace,
    result_queue: mp.Queue
):
    """
    Worker function that runs on a single NPU.
    Each NPU trains its own independent model.
    """
    import torch
    
    # Setup logger for this NPU
    logger = setup_logger(npu_id, log_dir=f"{args.output_dir}/logs")
    
    logger.info("=" * 60)
    logger.info(f"NPU {npu_id} Starting: {config['description']}")
    logger.info(f"Distance: {config['distance']}, Error rates: {config['error_rates']}")
    logger.info("=" * 60)
    
    try:
        # Setup NPU device
        import torch_npu
        device = torch.device(f'npu:{npu_id}')
        torch.npu.set_device(device)
        logger.info(f"Using device: {device}")
        
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{npu_id % torch.cuda.device_count()}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        logger.info(f"Using device: {device}")
    
    # Clear memory
    try:
        torch.npu.empty_cache()
    except:
        pass
    
    # Step 1: Generate data
    logger.info("Step 1: Generating training data...")
    data_files = generate_training_data(
        distance=config['distance'],
        error_rates=config['error_rates'],
        samples_per_rate=config['samples_per_rate'],
        output_dir=f"{args.output_dir}/data",
        npu_id=npu_id,
        logger=logger
    )
    
    # Step 2: Create dataset
    logger.info("Step 2: Creating dataset...")
    from torch.utils.data import DataLoader
    
    dataset = create_dataset(data_files, logger)
    
    # Split train/val
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    split = int(0.9 * n)
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])
    
    # Use num_workers=0 to avoid nested multiprocessing issues
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Batch size: {batch_size}")
    
    # Step 3: Create model
    logger.info("Step 3: Creating model...")
    model = create_model(
        num_features=dataset.n_features,
        num_stabilizers=dataset.n_stabilizers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Step 4: Train
    logger.info("Step 4: Training...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['epochs'],
        lr=args.lr,
        save_dir=f"{args.output_dir}/models/npu{npu_id}",
        logger=logger
    )
    
    logger.info("=" * 60)
    logger.info(f"NPU {npu_id} Complete! Best accuracy: {results['best_acc']:.4f}")
    logger.info("=" * 60)
    
    # Send results back
    result_queue.put({
        "npu_id": npu_id,
        "config": config,
        "best_acc": results['best_acc'],
        "status": "success"
    })


def npu_worker_wrapper(args_tuple):
    """Wrapper for multiprocessing."""
    npu_id, config, args, result_queue = args_tuple
    try:
        npu_worker(npu_id, config, args, result_queue)
    except Exception as e:
        import traceback
        result_queue.put({
            "npu_id": npu_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AlphaQubit Multi-NPU Independent Training')
    
    parser.add_argument('--num-npus', type=int, default=8, help='Number of NPUs to use')
    parser.add_argument('--samples', type=int, default=50000, help='Samples per error rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per NPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    parser.add_argument('--hidden-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    
    parser.add_argument('--output-dir', type=str, default='results/multi_npu_independent')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.samples = 2000
        args.epochs = 5
        args.num_layers = 4
        args.batch_size = 64
    
    # Setup output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Main logger
    logger = setup_logger(99, log_dir=f"{args.output_dir}/logs")
    logger.info("=" * 70)
    logger.info("AlphaQubit Multi-NPU Independent Training")
    logger.info("=" * 70)
    logger.info(f"Using {args.num_npus} NPUs")
    logger.info(f"Samples per rate: {args.samples}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Model: hidden={args.hidden_dim}, heads={args.num_heads}, layers={args.num_layers}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # Get configs for each NPU
    configs = [get_npu_config(i, args.quick_test) for i in range(args.num_npus)]
    
    # Override samples/epochs from args
    for config in configs:
        config['samples_per_rate'] = args.samples
        config['epochs'] = args.epochs
    
    # Print NPU assignments
    logger.info("\nNPU Assignments:")
    for config in configs:
        logger.info(f"  NPU {config['npu_id']}: d={config['distance']}, "
                   f"error_rates={config['error_rates']}, "
                   f"samples={config['samples_per_rate']}")
    logger.info("")
    
    # Use spawn method for NPU compatibility - MUST be before creating Queue
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create result queue AFTER setting start method
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    # Start worker processes
    logger.info("Starting NPU workers...")
    processes = []
    
    for i, config in enumerate(configs):
        p = ctx.Process(
            target=npu_worker_wrapper,
            args=((i, config, args, result_queue),)
        )
        p.start()
        processes.append(p)
        logger.info(f"  Started NPU {i} worker (PID: {p.pid})")
    
    # Wait for all processes to complete
    logger.info("\nWaiting for NPU workers to complete...")
    
    results = []
    for _ in range(args.num_npus):
        result = result_queue.get()
        results.append(result)
        if result['status'] == 'success':
            logger.info(f"NPU {result['npu_id']} completed: acc={result['best_acc']:.4f}")
        else:
            logger.error(f"NPU {result['npu_id']} failed: {result.get('error', 'Unknown error')}")
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 70)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    logger.info(f"Successful: {len(successful)}/{args.num_npus}")
    
    if successful:
        logger.info("\nResults by NPU:")
        for r in sorted(successful, key=lambda x: x['npu_id']):
            cfg = r['config']
            logger.info(f"  NPU {r['npu_id']}: d={cfg['distance']}, "
                       f"error_rates={cfg['error_rates']}, "
                       f"accuracy={r['best_acc']:.4f}")
        
        avg_acc = sum(r['best_acc'] for r in successful) / len(successful)
        logger.info(f"\nAverage accuracy: {avg_acc:.4f}")
    
    if failed:
        logger.info("\nFailed NPUs:")
        for r in failed:
            logger.info(f"  NPU {r['npu_id']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        "num_npus": args.num_npus,
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "args": vars(args),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{args.output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nSummary saved to {args.output_dir}/summary.json")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
