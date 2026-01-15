#!/usr/bin/env python3
"""
Simple 8-NPU Training for AlphaQubit

Uses manual data parallelism with 8 NPUs without HCCL distributed training.
Each batch is split across 8 NPUs and gradients are averaged manually.

This avoids HCCL initialization issues by running everything in a single process.

Usage:
    python train_8npu_simple.py --mode pretrain --data-dir output --filter-distance 3
"""

import argparse
import math
import os
import sys
import json
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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

# Number of NPUs
NUM_NPUS = 8

# Email configuration
NOTIFICATION_EMAIL = "xudayj@chinamobile.com"


def send_email_notification(subject: str, body: str, to_email: str = NOTIFICATION_EMAIL):
    """Send email notification when training completes."""
    try:
        # Try multiple SMTP servers
        smtp_servers = [
            ("smtp.gmail.com", 587),
            ("smtp.qq.com", 587),
            ("smtp.163.com", 25),
            ("localhost", 25),
        ]
        
        hostname = socket.gethostname()
        
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = f"AlphaQubit Training <noreply@{hostname}>"
        msg['To'] = to_email
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Try to send via local sendmail first (most likely to work on server)
        try:
            import subprocess
            sendmail_path = "/usr/sbin/sendmail"
            if os.path.exists(sendmail_path):
                p = subprocess.Popen([sendmail_path, "-t", "-oi"], stdin=subprocess.PIPE)
                p.communicate(msg.as_bytes())
                if p.returncode == 0:
                    print(f"✓ Email notification sent to {to_email}")
                    return True
        except Exception as e:
            pass
        
        # Try mail command
        try:
            import subprocess
            result = subprocess.run(
                ["mail", "-s", subject, to_email],
                input=body.encode(),
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✓ Email notification sent to {to_email}")
                return True
        except Exception as e:
            pass
        
        print(f"Note: Email notification not sent (no mail service configured)")
        print(f"  Recipient: {to_email}")
        print(f"  Subject: {subject}")
        return False
        
    except Exception as e:
        print(f"Email notification failed: {e}")
        return False


class PaddedSoftReadoutDataset(Dataset):
    """Dataset that pads sequences to a maximum round length."""
    
    def __init__(self, npz_path: str, max_rounds: int = 25, basis_id: Optional[int] = None):
        data = np.load(npz_path, allow_pickle=True)
        
        if 'data' in data:
            det_events = data['data']
        elif 'detection_events' in data:
            det_events = data['detection_events']
        else:
            raise KeyError(f"No data in {npz_path}")
        
        if 'observables' in data:
            labels = data['observables']
        elif 'labels' in data:
            labels = data['labels']
        elif 'obs' in data:
            labels = data['obs']
        else:
            raise KeyError(f"No labels in {npz_path}")
        
        if basis_id is None:
            if '_bX_' in str(npz_path) or 'basis_x' in str(npz_path).lower():
                basis_id = 0
            else:
                basis_id = 1
        
        N = det_events.shape[0]
        
        if det_events.ndim == 4:
            x = det_events.astype(np.float32)
            _, R, S, F = x.shape
        elif det_events.ndim == 3:
            x = det_events[:, :, :, np.newaxis].astype(np.float32)
            _, R, S, F = x.shape
        else:
            raise ValueError(f"Unexpected shape: {det_events.shape}")
        
        d_sq = S + 1
        distance = int(math.sqrt(d_sq))
        
        if F == 3:
            basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=np.float32)
            x = np.concatenate([x, basis_feat], axis=-1)
            F = 4
        
        if R < max_rounds:
            pad_rounds = max_rounds - R
            padding = np.zeros((N, pad_rounds, S, F), dtype=np.float32)
            x = np.concatenate([x, padding], axis=1)
        elif R > max_rounds:
            x = x[:, :max_rounds, :, :]
        
        mask = np.ones((N, max_rounds), dtype=np.float32)
        mask[:, R:] = 0
        
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


def build_padded_dataset(npz_files: List[str], max_rounds: int = 25) -> ConcatDataset:
    """Build dataset with padding for variable-length sequences."""
    datasets = []
    
    for npz_path in npz_files:
        try:
            ds = PaddedSoftReadoutDataset(npz_path, max_rounds=max_rounds)
            if len(ds) > 0:
                datasets.append(ds)
                print(f"  {Path(npz_path).name}: N={len(ds)}, R={ds.rounds}→{max_rounds}")
        except Exception as e:
            print(f"Warning: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets")
    
    return ConcatDataset(datasets)


class TransformerEncoderLayerManual(nn.Module):
    """
    Manual implementation of TransformerEncoderLayer to avoid NPU fallback.
    Standard nn.TransformerEncoderLayer causes 'aten::_transformer_encoder_layer_fwd' 
    fallback to CPU on Ascend NPUs.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head attention
        # Note: src_mask in MultiheadAttention is different from TransformerEncoderLayer
        # We need to handle masks carefully if they are provided
        
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


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
        
        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_stabilizers, hidden_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_rounds, 1, hidden_dim) * 0.02)
        self.basis_embed = nn.Embedding(2, hidden_dim)
        
        # Use manual transformer layers to avoid NPU fallback
        self.layers = nn.ModuleList([
            TransformerEncoderLayerManual(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask=None, basis=None):
        B, R, S, F = x.shape
        
        x = self.input_proj(x)
        x = x + self.spatial_pos[:, :, :S, :]
        x = x + self.temporal_pos[:, :R, :, :]
        
        if basis is not None:
            basis_emb = self.basis_embed(basis)
            x = x + basis_emb.unsqueeze(1).unsqueeze(2)
        
        x = x.reshape(B, R * S, -1)
        
        # Mask handling
        key_padding_mask = None
        if mask is not None:
            # Flatten mask: (B, R) -> (B, R*S)
            # Expand mask to all stabilizers in a round
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, S).reshape(B, R * S)
            # In MultiheadAttention, key_padding_mask=True means IGNORE
            # Our mask is 1=Valid, 0=Padding
            # So key_padding_mask should be True where mask is 0
            key_padding_mask = (mask_expanded == 0)
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        x = self.norm(x)
        
        if mask is not None:
            # Recreate boolean mask for averaging
            valid_mask = ~key_padding_mask
            valid_mask_float = valid_mask.float().unsqueeze(-1)
            x = (x * valid_mask_float).sum(dim=1) / valid_mask_float.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        logits = self.classifier(x).squeeze(-1)
        return logits


class MultiNPUModel:
    """
    Simple multi-NPU data parallel model.
    
    Replicates model across all NPUs and runs forward/backward passes
    sequentially but efficiently. Avoids complex stream operations that
    can cause TBE errors on Ascend NPUs.
    """
    
    def __init__(self, model: nn.Module, num_npus: int = 8):
        self.num_npus = num_npus
        self.devices = [torch.device(f'npu:{i}') for i in range(num_npus)]
        
        # Create a copy of the model on each NPU
        self.models = []
        for i, dev in enumerate(self.devices):
            torch.npu.set_device(i)
            if i == 0:
                # Original model on NPU 0
                model_copy = model.to(dev)
            else:
                # Clone model to other NPUs
                import copy
                model_copy = copy.deepcopy(model).to(dev)
            self.models.append(model_copy)
        
        # Reset to device 0
        torch.npu.set_device(0)
        print(f"Model replicated to {num_npus} NPUs")
    
    def train(self):
        for m in self.models:
            m.train()
    
    def eval(self):
        for m in self.models:
            m.eval()
    
    def parameters(self):
        """Return parameters of master model (NPU 0)."""
        return self.models[0].parameters()
    
    def state_dict(self):
        """Return state dict of master model."""
        return self.models[0].state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict to all models."""
        for i, m in enumerate(self.models):
            torch.npu.set_device(i)
            m.load_state_dict(state_dict)
        torch.npu.set_device(0)
    
    def sync_models(self):
        """Sync all models to master (NPU 0)."""
        master_state = self.models[0].state_dict()
        for i in range(1, self.num_npus):
            torch.npu.set_device(i)
            # Move state dict to target device
            target_state = {k: v.to(self.devices[i]) for k, v in master_state.items()}
            self.models[i].load_state_dict(target_state)
        torch.npu.set_device(0)
    
    def forward_backward(self, x_batch, mask_batch, basis_batch, y_batch, criterion):
        """
        Sequential forward and backward pass across all NPUs.
        
        Each NPU processes its chunk, then gradients are averaged.
        Returns: average loss
        """
        batch_size = x_batch.size(0)
        chunk_size = batch_size // self.num_npus
        
        losses = []
        
        # Forward + backward on each NPU sequentially
        for i in range(self.num_npus):
            start = i * chunk_size
            end = start + chunk_size
            
            torch.npu.set_device(i)
            
            # Move data to this NPU
            x_i = x_batch[start:end].to(self.devices[i])
            mask_i = mask_batch[start:end].to(self.devices[i])
            basis_i = basis_batch[start:end].to(self.devices[i])
            y_i = y_batch[start:end].to(self.devices[i])
            
            # Forward + backward
            logits = self.models[i](x_i, mask_i, basis_i)
            loss = criterion(logits, y_i)
            loss.backward()
            losses.append(loss.item())
        
        # Average gradients to master (NPU 0)
        torch.npu.set_device(0)
        with torch.no_grad():
            for name, param in self.models[0].named_parameters():
                if param.grad is not None:
                    grad_sum = param.grad.clone()
                    for j in range(1, self.num_npus):
                        other_grad = dict(self.models[j].named_parameters())[name].grad
                        grad_sum += other_grad.to(self.devices[0])
                    param.grad = grad_sum / self.num_npus
        
        return sum(losses) / len(losses)
    
    def forward_only(self, x_batch, mask_batch, basis_batch, y_batch, criterion):
        """Sequential forward pass for evaluation."""
        batch_size = x_batch.size(0)
        
        if batch_size < self.num_npus:
            # Small batch: use single NPU
            torch.npu.set_device(0)
            x = x_batch.to(self.devices[0])
            mask = mask_batch.to(self.devices[0])
            basis = basis_batch.to(self.devices[0])
            y = y_batch.to(self.devices[0])
            
            with torch.no_grad():
                logits = self.models[0](x, mask, basis)
                loss = criterion(logits, y)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == y).sum().item()
            return loss.item() * len(y), correct, len(y)
        
        chunk_size = batch_size // self.num_npus
        remainder = batch_size % self.num_npus
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(self.num_npus):
                start = i * chunk_size
                end = start + chunk_size + (remainder if i == self.num_npus - 1 else 0)
                
                if start >= batch_size:
                    break
                
                torch.npu.set_device(i)
                
                x_i = x_batch[start:end].to(self.devices[i])
                mask_i = mask_batch[start:end].to(self.devices[i])
                basis_i = basis_batch[start:end].to(self.devices[i])
                y_i = y_batch[start:end].to(self.devices[i])
                
                logits = self.models[i](x_i, mask_i, basis_i)
                loss = criterion(logits, y_i)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == y_i).sum().item()
                
                total_loss += loss.item() * len(y_i)
                total_correct += correct
                total_samples += len(y_i)
        
        torch.npu.set_device(0)
        return total_loss, total_correct, total_samples


def train_epoch(multi_model, loader, optimizer, criterion):
    """Train one epoch across 8 NPUs."""
    multi_model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, ((x, mask, basis), y) in enumerate(loader):
        optimizer.zero_grad()
        
        loss = multi_model.forward_backward(x, mask, basis, y, criterion)
        
        # Clip gradients (master model)
        torch.nn.utils.clip_grad_norm_(multi_model.parameters(), 1.0)
        
        optimizer.step()
        
        # Sync models after optimizer step
        multi_model.sync_models()
        
        # Track metrics (approximate from loss)
        total_loss += loss * len(y)
        total_samples += len(y)
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss:.4f}", flush=True)
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(multi_model, loader, criterion):
    """Evaluate model across NPUs."""
    multi_model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for (x, mask, basis), y in loader:
        loss, correct, n = multi_model.forward_only(x, mask, basis, y, criterion)
        total_loss += loss
        total_correct += correct
        total_samples += n
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Simple 8-NPU AlphaQubit Training")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--data-dir", default="output")
    parser.add_argument("--filter-distance", type=int, default=None)
    parser.add_argument("--filter-rounds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PAPER_LR)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file to load model from")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint (restore epoch and optimizer)")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-npus", type=int, default=8)
    args = parser.parse_args()
    
    if args.epochs is None:
        args.epochs = PAPER_PRETRAIN_EPOCHS if args.mode == "pretrain" else PAPER_FINETUNE_EPOCHS
    
    # Total batch size across all NPUs
    total_batch = args.batch_size * args.num_npus
    
    print("=" * 70)
    print(f"AlphaQubit Simple Multi-NPU Training - {args.mode.upper()}")
    print("=" * 70)
    print(f"  Number of NPUs: {args.num_npus}")
    print(f"  Per-NPU batch:  {args.batch_size}")
    print(f"  Total batch:    {total_batch}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 70)
    print()
    
    # Initialize NPUs
    print("Initializing NPUs...")
    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        
        # Test all NPUs
        for i in range(args.num_npus):
            torch.npu.set_device(i)
            name = torch.npu.get_device_name(i)
            print(f"  NPU {i}: {name}")
        
        torch.npu.set_device(0)
        print(f"\nAll {args.num_npus} NPUs ready!")
        
    except ImportError as e:
        print(f"ERROR: torch_npu not available: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: NPU initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    torch.manual_seed(42)
    
    # Find data files
    print(f"\nSearching for data in: {args.data_dir}")
    npz_files = discover_npz_files(args.data_dir, args.filter_distance, args.filter_rounds)
    
    if not npz_files:
        print("ERROR: No .npz files found")
        sys.exit(1)
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    print(f"Found {len(npz_files)} .npz files\n")
    
    # Build dataset
    print("Loading datasets...")
    dataset = build_padded_dataset(npz_files, max_rounds=args.max_rounds)
    print(f"\nTotal samples: {len(dataset):,}")
    
    # Split dataset
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    split = int(0.9 * n)
    
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds = torch.utils.data.Subset(dataset, indices[split:])
    
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    
    # Create data loaders with total batch size
    # Use multiple workers to utilize CPU resources for data loading
    num_workers = min(32, os.cpu_count() or 8)
    print(f"Using {num_workers} CPU workers for data loading")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=total_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Pin memory for faster host-to-device transfer
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=total_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Get dimensions
    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    
    print(f"\nModel config: max_rounds={args.max_rounds}, S={S}, F={F}")
    print(f"  Hidden: {args.hidden_dim}, Layers: {args.num_layers}, Heads: {args.num_heads}")
    
    # Create model on CPU first
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
    
    # Load checkpoint if provided
    start_epoch = 1
    loaded_optimizer_state = None
    loaded_history = []
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            if args.resume:
                # Restore training state
                start_epoch = state.get('epoch', 0) + 1
                loaded_optimizer_state = state.get('optimizer_state_dict', None)
                loaded_history = state.get('history', [])
                print(f"Resuming training from epoch {start_epoch}")
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Create multi-NPU model
    print(f"\nReplicating model to {args.num_npus} NPUs...")
    multi_model = MultiNPUModel(model, num_npus=args.num_npus)
    
    # Optimizer and scheduler (on master model)
    optimizer = torch.optim.AdamW(multi_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Restore optimizer state if resuming
    if loaded_optimizer_state is not None:
        try:
            optimizer.load_state_dict(loaded_optimizer_state)
            print("Restored optimizer state")
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=start_epoch-2 if start_epoch > 1 else -1)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    history = loaded_history.copy() if loaded_history else []
    
    # Restore best val loss from history
    if history:
        best_val_loss = min(h.get('val_loss', float('inf')) for h in history)
        print(f"Restored best val loss: {best_val_loss:.4f}")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{args.mode}_d{args.filter_distance or 'all'}"
    if args.filter_rounds:
        prefix += f"_r{args.filter_rounds}"
    
    # Save training configuration
    config_path = save_path / f"{prefix}_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 70)
    print(f"Starting training from epoch {start_epoch}...")
    print("=" * 70 + "\n")
    
    last_checkpoint_time = datetime.now()
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = datetime.now()
            
            print(f"Epoch {epoch}/{args.epochs} [LR: {optimizer.param_groups[0]['lr']:.2e}]")
            
            # Train
            train_loss = train_epoch(multi_model, train_loader, optimizer, criterion)
            
            # Evaluate
            val_loss, val_acc = evaluate(multi_model, val_loader, criterion)
            
            scheduler.step()
            
            # Memory stats
            if torch.cuda.is_available(): # For NPU via torch_npu which often mocks cuda
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                print(f"  Mem: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            # Record history
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr'],
                "time": epoch_time,
                "timestamp": datetime.now().isoformat()
            }
            history.append(record)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = save_path / f"{prefix}_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": multi_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "args": vars(args),
                    "history": history
                }, ckpt_path)
                print(f"  ✓ Best model saved: {ckpt_path}")
            
            # Save latest checkpoint (EVERY EPOCH)
            latest_path = save_path / f"{prefix}_latest.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": multi_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
                "history": history
            }, latest_path)

            # Save checkpoint every 10 epochs or every 2 hours
            time_since_last_ckpt = (datetime.now() - last_checkpoint_time).total_seconds()
            
            if epoch % 10 == 0 or time_since_last_ckpt > 7200:
                ckpt_path = save_path / f"{prefix}_epoch{epoch}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": multi_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "args": vars(args),
                    "history": history
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
                last_checkpoint_time = datetime.now()
            
            # Save history
            history_path = save_path / f"{prefix}_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print()
            
    except Exception as e:
        print(f"\nERROR: Training crashed at epoch {epoch}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save emergency checkpoint
        try:
            ckpt_path = save_path / f"{prefix}_CRASH_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": multi_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "error": str(e)
            }, ckpt_path)
            print(f"Emergency checkpoint saved: {ckpt_path}")
        except:
            print("Failed to save emergency checkpoint")
            
        # Send error email
        send_email_notification(
            f"[AlphaQubit] Training CRASHED - {args.mode}",
            f"Training crashed at epoch {epoch}\n\nError: {e}\n\nServer: {socket.gethostname()}"
        )
        raise e
    
    # Calculate total training time
    total_time = sum(h['time'] for h in history)
    total_hours = total_time / 3600
    
    print("=" * 70)
    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val acc:  {history[-1]['val_acc']*100:.2f}%")
    print(f"Total time: {total_hours:.2f} hours")
    print(f"Checkpoints saved to: {save_path}")
    print("=" * 70)
    
    # Send email notification
    subject = f"[AlphaQubit] Training Completed - {args.mode} d{args.filter_distance or 'all'}"
    body = f"""
AlphaQubit Training Completed!
==============================

Mode: {args.mode}
Distance filter: {args.filter_distance or 'all'}
NPUs used: {args.num_npus}
Epochs: {args.epochs}
Batch size: {args.batch_size} x {args.num_npus} = {args.batch_size * args.num_npus}

Results:
--------
Best validation loss: {best_val_loss:.4f}
Final validation accuracy: {history[-1]['val_acc']*100:.2f}%
Total training time: {total_hours:.2f} hours

Checkpoints saved to: {save_path}
- Best model: {prefix}_best.pth
- History: {prefix}_history.json

Server: {socket.gethostname()}
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    send_email_notification(subject, body)


if __name__ == "__main__":
    main()
