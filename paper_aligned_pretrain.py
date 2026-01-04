#!/usr/bin/env python3
"""
Paper-Aligned Pre-Training Script for AlphaQubit
================================================

This script follows the EXACT specifications from the AlphaQubit Nature paper
to generate pre-training data and train the model.

Paper specifications:
- 8.5M synthetic samples
- SI1000 noise model
- Physical error rates: p ∈ {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01}
- Code distances: d ∈ {3, 5, 7}
- Rounds: variable (1, 5, 10, 25)
- Batch size: 256
- Learning rate: 1e-4
- Epochs: 100
- Optimizer: AdamW with weight_decay=1e-4
- Scheduler: Cosine Annealing

Usage:
    python paper_aligned_pretrain.py --samples 8500000 --epochs 100
    
For quick test:
    python paper_aligned_pretrain.py --quick-test
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm

# Import stim for circuit simulation
try:
    import stim
except ImportError:
    print("ERROR: stim not installed. Run: pip install stim")
    sys.exit(1)

# =============================================================================
# Paper Constants
# =============================================================================

PAPER_P_GRID = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
PAPER_CODE_DISTANCES = [3, 5, 7]
PAPER_ROUNDS = [1, 5, 10, 25]
PAPER_TOTAL_SAMPLES = 8_500_000
PAPER_BATCH_SIZE = 256
PAPER_LR = 1e-4
PAPER_EPOCHS = 100
PAPER_WEIGHT_DECAY = 1e-4

# =============================================================================
# SI1000 Noise Model (simplified version matching paper)
# =============================================================================

def create_si1000_circuit(d: int, rounds: int, p: float, basis: str = "Z") -> stim.Circuit:
    """Create an SI1000 surface code circuit.
    
    SI1000 noise model parameters (per Google/Stim standard):
    - meas_bitflip: 5p (before measurement)
    - reset_bitflip: 2p (after reset)  
    - twoq_depol: p (after 2Q Clifford gates)
    - oneq_depol: p/10 (after 1Q Clifford gates)
    - idle: p/10 (before round data depolarization)
    
    Args:
        d: Code distance (3, 5, or 7)
        rounds: Number of QEC rounds
        p: Physical error rate
        basis: Measurement basis ("X" or "Z")
    
    Returns:
        stim.Circuit with SI1000 noise
    """
    # Generate standard surface code circuit with SI1000 noise
    # IMPORTANT: These parameters must match the paper exactly!
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_{}".format(basis.lower()),
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,          # p for 2Q gates (DEPOLARIZE2)
        after_reset_flip_probability=2 * p,       # 2p for reset (SI1000 spec)
        before_measure_flip_probability=5 * p,    # 5p for measurement (SI1000 spec)
        before_round_data_depolarization=p / 10,  # p/10 for idle (SI1000 spec)
    )
    return circuit


def sample_circuit(circuit: stim.Circuit, n_shots: int) -> tuple:
    """Sample detection events and observables from a circuit.
    
    Returns:
        (detection_events, observables) tuple of numpy arrays
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, observables = sampler.sample(n_shots, separate_observables=True)
    return detection_events.astype(np.float32), observables.astype(np.float32).flatten()


# =============================================================================
# Dataset
# =============================================================================

# Import soft channels generator for paper-aligned I/Q readout model
try:
    from google_qec_simulator.data_helpers import soft_channels
    HAS_SOFT_CHANNELS = True
except ImportError:
    HAS_SOFT_CHANNELS = False
    print("Warning: soft_channels not available, using 2-feature mode (detection + basis)")


class PretrainDataset(Dataset):
    """Dataset for pre-training data with paper-aligned soft channels.
    
    Paper specifies 3 input features per stabilizer:
    1. Detection event (binary)
    2. P(|1⟩) posterior from I/Q readout model
    3. P(|L⟩) posterior (leakage probability)
    """
    
    def __init__(self, syndromes: np.ndarray, labels: np.ndarray, basis_id: int, grid_size: int,
                 use_soft_channels: bool = True, snr: float = 10.0, tau: float = 0.01):
        """
        Args:
            syndromes: (N, S) or (N, R, S) detection events
            labels: (N,) observable outcomes
            basis_id: 0 for X, 1 for Z
            grid_size: d-1 where d is code distance
            use_soft_channels: Whether to generate soft I/Q readout features
            snr: Signal-to-noise ratio for soft channels
            tau: Amplitude damping time constant
        """
        x = np.asarray(syndromes, dtype=np.float32)
        
        # Reshape to (N, R, S, F) format
        if x.ndim == 2:
            x = x[:, None, :, None]  # (N, 1, S, 1)
        elif x.ndim == 3:
            x = x[:, :, :, None]  # (N, R, S, 1)
        
        N, R, S, _ = x.shape
        
        # Paper-aligned: Add soft channels (P(|1⟩) and P(|L⟩) posteriors)
        if use_soft_channels and HAS_SOFT_CHANNELS:
            # Generate soft channel posteriors for all elements
            total_elements = N * R * S
            p1_post, pL_post = soft_channels(total_elements, snr=snr, tau=tau)
            
            # Reshape to match input shape
            p1_feat = p1_post.reshape(N, R, S, 1)
            pL_feat = pL_post.reshape(N, R, S, 1)
            
            # Concatenate: [detection, P(|1⟩), P(|L⟩)]
            x = np.concatenate([x, p1_feat, pL_feat], axis=-1)
        else:
            # Fallback: just add basis feature (not paper-aligned!)
            basis_feat = np.full((N, R, S, 1), float(basis_id), dtype=np.float32)
            x = np.concatenate([x, basis_feat], axis=-1)
        
        F = x.shape[-1]
        
        # Pad to square grid if needed
        d = math.isqrt(S + 1)
        if d * d != S + 1:
            d += 1
            pad = d * d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=np.float32)], axis=2)
            S = x.shape[2]
        
        # Create final mask (checkerboard pattern)
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
            dtype=torch.int8
        )
        
        self.inputs = torch.from_numpy(x)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.basis_tensor = torch.tensor(basis_id, dtype=torch.int8)
        self.grid_size = grid_size
        self.num_features = F
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.inputs[idx], self.basis_tensor, self.final_mask), self.labels[idx]


# =============================================================================
# Model Import
# =============================================================================

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ai_models.model import AlphaQubitDecoder as AlphaQubitDecoderTransformer
    from ai_models.model_mla import AlphaQubitDecoder as AlphaQubitDecoderMLA
except ImportError:
    from model import AlphaQubitDecoder as AlphaQubitDecoderTransformer
    from model_mla import AlphaQubitDecoder as AlphaQubitDecoderMLA


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for (xb, basis, mask), yb in pbar:
        xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
        
        optimizer.zero_grad()
        outputs = model(xb, basis, mask)
        loss = criterion(outputs, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(yb)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += len(yb)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (xb, basis, mask), yb in loader:
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            outputs = model(xb, basis, mask)
            loss = criterion(outputs, yb)
            
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    
    return total_loss / total, correct / total


def generate_balanced_pretraining_data(
    total_samples: int,
    p_grid: list,
    distances: list,
    rounds_list: list,
    output_dir: Path,
    verbose: bool = True
):
    """Generate pre-training data with balanced noise distribution.
    
    This is the KEY to paper alignment - we must train on data from ALL noise levels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate samples per configuration
    n_configs = len(p_grid) * len(distances) * len(rounds_list) * 2  # *2 for X and Z basis
    samples_per_config = total_samples // n_configs
    
    if verbose:
        print(f"Generating {total_samples:,} samples across {n_configs} configurations")
        print(f"  - {samples_per_config:,} samples per (p, d, r, basis) configuration")
        print(f"  - Physical error rates: {p_grid}")
        print(f"  - Code distances: {distances}")
        print(f"  - Rounds: {rounds_list}")
    
    all_datasets = []
    manifest = []
    
    for p in tqdm(p_grid, desc="Noise levels"):
        for d in distances:
            for r in rounds_list:
                for basis in ["X", "Z"]:
                    basis_id = 0 if basis == "X" else 1
                    
                    # Generate circuit and sample
                    try:
                        circuit = create_si1000_circuit(d, r, p, basis)
                        syndromes, labels = sample_circuit(circuit, samples_per_config)
                        
                        # Calculate positive ratio for monitoring
                        pos_ratio = labels.mean()
                        
                        # Create dataset
                        grid_size = d - 1  # For surface code
                        dataset = PretrainDataset(syndromes, labels, basis_id, grid_size)
                        all_datasets.append(dataset)
                        
                        manifest.append({
                            'p': p,
                            'd': d,
                            'r': r,
                            'basis': basis,
                            'samples': len(dataset),
                            'positive_ratio': float(pos_ratio),
                        })
                        
                        if verbose and len(manifest) % 20 == 0:
                            print(f"  Generated: p={p}, d={d}, r={r}, basis={basis}, "
                                  f"pos_ratio={pos_ratio:.2%}")
                    
                    except Exception as e:
                        print(f"WARNING: Failed to generate p={p}, d={d}, r={r}, basis={basis}: {e}")
                        continue
    
    # Save manifest
    manifest_path = output_dir / "pretrain_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'total_samples': sum(m['samples'] for m in manifest),
            'configs': manifest,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    if verbose:
        total = sum(m['samples'] for m in manifest)
        avg_pos = np.mean([m['positive_ratio'] for m in manifest])
        print(f"\nGenerated {total:,} total samples")
        print(f"Average positive ratio across all configs: {avg_pos:.2%}")
        print(f"Manifest saved to: {manifest_path}")
    
    return ConcatDataset(all_datasets), manifest


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned AlphaQubit pre-training")
    
    # Data generation
    parser.add_argument("--samples", type=int, default=PAPER_TOTAL_SAMPLES,
                        help=f"Total samples (paper: {PAPER_TOTAL_SAMPLES:,})")
    parser.add_argument("--p-grid", type=str, default=",".join(map(str, PAPER_P_GRID)),
                        help="Comma-separated physical error rates")
    parser.add_argument("--distances", type=str, default=",".join(map(str, PAPER_CODE_DISTANCES)),
                        help="Comma-separated code distances")
    parser.add_argument("--rounds", type=str, default=",".join(map(str, PAPER_ROUNDS)),
                        help="Comma-separated round numbers")
    
    # Training
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS,
                        help=f"Training epochs (paper: {PAPER_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE,
                        help=f"Batch size (paper: {PAPER_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=PAPER_LR,
                        help=f"Learning rate (paper: {PAPER_LR})")
    parser.add_argument("--weight-decay", type=float, default=PAPER_WEIGHT_DECAY,
                        help=f"Weight decay (paper: {PAPER_WEIGHT_DECAY})")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--mla", action="store_true", help="Use MLA model variant")
    
    # Output
    parser.add_argument("--output", type=str, default="alphaqubit_paper_aligned.pth",
                        help="Output model path")
    parser.add_argument("--data-dir", type=str, default="pretrain_data",
                        help="Directory for generated data")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with minimal samples")
    
    args = parser.parse_args()
    
    # Parse grid parameters
    p_grid = [float(x) for x in args.p_grid.split(",")]
    distances = [int(x) for x in args.distances.split(",")]
    rounds_list = [int(x) for x in args.rounds.split(",")]
    
    # Quick test overrides
    if args.quick_test:
        args.samples = 10000
        args.epochs = 2
        p_grid = [0.001, 0.005, 0.01]
        distances = [3]
        rounds_list = [5]
        print("=== QUICK TEST MODE ===")
    
    print("=" * 60)
    print("PAPER-ALIGNED ALPHAQUBIT PRE-TRAINING")
    print("=" * 60)
    print(f"Samples: {args.samples:,}")
    print(f"P-grid: {p_grid}")
    print(f"Distances: {distances}")
    print(f"Rounds: {rounds_list}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("\n1. Generating pre-training data...")
    data_dir = Path(args.data_dir)
    dataset, manifest = generate_balanced_pretraining_data(
        args.samples, p_grid, distances, rounds_list, data_dir
    )
    
    print(f"\n   Total dataset size: {len(dataset):,}")
    
    # Paper-aligned: 95/5 train/val split (from configs/paper_aligned.yaml)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"   Train: {train_size:,}, Val: {val_size:,} (95/5 split)")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    
    # Get a sample to determine dimensions
    (sample_x, _, sample_mask), _ = dataset[0]
    R, S, F = sample_x.shape
    print(f"   Input shape: (R={R}, S={S}, F={F})")
    print(f"   Features: {F} ({'detection + P(|1⟩) + P(|L⟩)' if F == 3 else 'detection + basis'})")
    
    # Create model
    print("\n2. Creating model...")
    ModelClass = AlphaQubitDecoderMLA if args.mla else AlphaQubitDecoderTransformer
    
    # Grid size from first config
    grid_size = distances[0] - 1
    
    model = ModelClass(
        num_features=F,
        hidden_dim=args.hidden_dim,
        num_stabilizers=S,
        grid_size=grid_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    
    # Training setup with class-weighted loss (paper-aligned)
    # Compute class weights from manifest
    all_pos_ratios = [m['positive_ratio'] for m in manifest]
    avg_pos_ratio = np.mean(all_pos_ratios)
    if avg_pos_ratio > 0 and avg_pos_ratio < 1:
        pos_weight = torch.tensor([(1 - avg_pos_ratio) / avg_pos_ratio], device=device)
        print(f"   Using pos_weight = {pos_weight.item():.2f} (avg positive ratio: {avg_pos_ratio:.2%})")
    else:
        pos_weight = torch.tensor([1.0], device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # Training loop
    print("\n3. Training...")
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"   -> Saved best model (val_acc={val_acc:.4f})")
    
    # Final save
    torch.save(model.state_dict(), args.output)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")
    print()
    print("Next steps:")
    print(f"  1. Fine-tune: python run_finetune_all.py --pretrained {args.output}")
    print(f"  2. Test: python run_decode_all.py --model-dir finetuned_models")


if __name__ == "__main__":
    main()
