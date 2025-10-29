"""
Fine-tune AlphaQubit decoder on Google QEC v3.5 experimental data (NPZ format).
Supports NPU acceleration via --npu flag.
"""

import os
import argparse
import json
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import the model
import sys
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_models.model_mla import AlphaQubitDecoder
except ImportError:
    from model_mla import AlphaQubitDecoder


class NPZDataset(Dataset):
    """Dataset for NPZ files from Google experimental data."""
    
    def __init__(self, npz_path: str):
        """
        Load NPZ file containing:
        - data: (shots, rounds, detectors, features) detection events
        - obs: (shots,) observable outcomes  
        - basis: (shots,) measurement basis (0=X, 1=Z)
        - metadata: JSON string with experiment info
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        print(f"Loading {npz_path}...")
        data = np.load(npz_path)
        
        # Load arrays
        self.detection_events = torch.from_numpy(data['data']).float()  # (shots, rounds, detectors, features)
        self.observables = torch.from_numpy(data['obs']).float()  # (shots,)
        self.basis_ids = torch.from_numpy(data['basis']).long()  # (shots,)
        
        # Parse metadata
        try:
            self.metadata = json.loads(str(data['metadata']))
        except:
            self.metadata = {}
        
        self.n_shots, self.n_rounds, self.n_detectors, self.n_features = self.detection_events.shape
        
        # Calculate grid size for final_mask
        # Assuming square grid: (grid_size)^2 = n_detectors + 1
        import math
        d = int(math.sqrt(self.n_detectors + 1))
        if d * d != self.n_detectors + 1:
            d = int(math.sqrt(self.n_detectors)) + 1
        self.grid_size = d
        
        # Create final_mask (checkerboard pattern)
        # Skip the first (dummy) stabilizer
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:self.n_detectors+1],
            dtype=torch.long
        )
        
        # Pad if necessary
        if len(self.final_mask) < self.n_detectors:
            padding = torch.zeros(self.n_detectors - len(self.final_mask), dtype=torch.long)
            self.final_mask = torch.cat([self.final_mask, padding])
        elif len(self.final_mask) > self.n_detectors:
            self.final_mask = self.final_mask[:self.n_detectors]
        
        print(f"  Loaded {self.n_shots} shots, {self.n_rounds} rounds, {self.n_detectors} detectors")
        print(f"  Grid size: {self.grid_size}, final_mask shape: {self.final_mask.shape}")
        print(f"  Metadata: {self.metadata.get('experiment_name', 'unknown')}")
    
    def __len__(self):
        return self.n_shots
    
    def __getitem__(self, idx):
        """
        Returns:
            x: (rounds, detectors, features) detection events
            basis_id: scalar basis identifier
            final_mask: (detectors,) mask for final round
            obs: scalar observable outcome (label)
        """
        x = self.detection_events[idx]  # (rounds, detectors, features)
        basis_id = self.basis_ids[idx]
        obs = self.observables[idx]
        
        # Convert observable to binary label (0 or 1)
        # Google format: 48.0='0', 49.0='1', 10.0=newline
        if obs == 48.0 or obs == 10.0:
            label = 0.0
        elif obs == 49.0:
            label = 1.0
        else:
            # Fallback: treat as binary already
            label = float(obs)
        
        return x, basis_id, self.final_mask.clone(), torch.tensor(label)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    xs, basis_ids, final_masks, labels = zip(*batch)
    
    # Stack everything
    x_batch = torch.stack(xs, dim=0)  # (B, R, D, F)
    basis_batch = torch.stack(basis_ids, dim=0)  # (B,)
    mask_batch = torch.stack(final_masks, dim=0)  # (B, D)
    label_batch = torch.stack(labels, dim=0)  # (B,)
    
    return x_batch, basis_batch, mask_batch, label_batch


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for x, basis, final_mask, labels in pbar:
        x = x.to(device)
        basis = basis.to(device)
        final_mask = final_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, basis, final_mask)
                loss = criterion(logits.squeeze(), labels)
        else:
            logits = model(x, basis, final_mask)
            loss = criterion(logits.squeeze(), labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Statistics
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += x.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_correct/total_samples:.4f}'})
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, basis, final_mask, labels in tqdm(dataloader, desc="Validating"):
            x = x.to(device)
            basis = basis.to(device)
            final_mask = final_mask.to(device)
            labels = labels.to(device)
            
            logits = model(x, basis, final_mask)
            loss = criterion(logits.squeeze(), labels)
            
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def fine_tune(npz_path, args):
    """Fine-tune model on a single NPZ file."""
    
    # Setup device
    if args.npu:
        try:
            import torch_npu
            if hasattr(torch, 'npu') and torch.npu.is_available():
                device = torch.device('npu:0')
                print(f"Using NPU device: {device}")
            else:
                print("Warning: NPU requested but not available, falling back to CUDA/CPU")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            print("Warning: torch_npu not installed, falling back to CUDA/CPU")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Load dataset
    dataset = NPZDataset(npz_path)
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn
    )
    
    # Create model
    # The model expects (B, R, S, F) where S = detectors
    model = AlphaQubitDecoder(
        num_features=dataset.n_features,
        hidden_dim=args.hidden_dim,
        num_stabilizers=dataset.n_detectors,
        grid_size=int(np.sqrt(dataset.n_detectors)) + 1,  # Approximate grid size
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    # Load pretrained weights if available
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained}")
        try:
            state_dict = torch.load(args.pretrained, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained weights successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Mixed precision training for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and args.amp else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Device: {device}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'─'*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            exp_name = dataset.metadata.get('experiment_name', Path(npz_path).stem.replace('samples_', ''))
            model_path = os.path.join(args.output_dir, f"finetuned_{exp_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"   ✓ New best model! (improved by {improvement:.4f}) → {model_path}")
        else:
            patience_counter += 1
            print(f"   No improvement (patience: {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping triggered after {epoch} epochs")
                break
    
    # Cleanup
    del model, optimizer, train_loader, val_loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'npu':
        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
    
    exp_name = dataset.metadata.get('experiment_name', Path(npz_path).stem.replace('samples_', ''))
    print(f"\n{'='*60}")
    print(f"✓ Fine-tuning completed for: {exp_name}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final model: {model_path}")
    print(f"{'='*60}\n")
    
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AlphaQubit on Google QEC data (NPZ format)")
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to NPZ file')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('--output-dir', type=str, default='finetuned_models', help='Output directory for fine-tuned models')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision (CUDA only)')
    
    # Device arguments
    parser.add_argument('--npu', action='store_true', help='Use NPU for training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune
    fine_tune(args.data, args)


if __name__ == '__main__':
    main()
