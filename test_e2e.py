#!/usr/bin/env python3
"""
End-to-end test using real simulated data.
Tests the complete training pipeline with actual NPZ files.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '.')


class NPZDataset(Dataset):
    """Dataset that loads from NPZ file format used in pipeline."""
    
    def __init__(self, npz_path: str, basis_id: int = 0):
        data = np.load(npz_path)
        
        # data shape: (N, R, S, F) or similar
        x = data['data'].astype(np.float32)
        y = data['observables'].astype(np.float32)
        
        # Handle various shapes
        if x.ndim == 2:  # (N, S)
            x = x[:, None, :, None]
        elif x.ndim == 3:  # (N, R, S)
            x = x[..., None]
        
        self.N, self.R, self.S, self.F = x.shape
        
        # Add basis feature
        basis_feat = np.full((self.N, self.R, self.S, 1), basis_id, dtype=np.float32)
        x = np.concatenate([x, basis_feat], axis=-1)
        self.F += 1
        
        # Pad stabilizers to make S+1 a perfect square
        d = int(math.ceil(math.sqrt(self.S + 1)))
        if d * d != self.S + 1:
            d_new = d + 1 if d * d < self.S + 1 else d
            target_S = d_new * d_new - 1
            if target_S > self.S:
                pad = target_S - self.S
                x = np.concatenate([x, np.zeros((self.N, self.R, pad, self.F), dtype=np.float32)], axis=2)
                self.S = target_S
        
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)
        self.basis = torch.full((self.N,), basis_id, dtype=torch.long)
        
        # Compute final mask
        d = int(math.sqrt(self.S + 1))
        self.final_mask = torch.zeros(self.N, self.S, dtype=torch.long)
        for idx in range(self.S):
            r, c = (idx + 1) // d, (idx + 1) % d
            self.final_mask[:, idx] = 1 if (r + c) % 2 == 0 else 2
        
        print(f"Loaded {npz_path}:")
        print(f"  Samples: {self.N}, Rounds: {self.R}, Stabilizers: {self.S}, Features: {self.F}")
        print(f"  Grid size: {d}, Labels: {self.y.sum().item():.0f} positive / {self.N} total")
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return (self.X[idx], self.basis[idx], self.final_mask[idx]), self.y[idx]


def test_real_data_training():
    """Test training with real NPZ data."""
    print("=" * 60)
    print("  End-to-End Test: Training with Real Data")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    # Load real data
    npz_path = "simulated_data/samples_surface_code_bX_d5_r01_center_5_5.npz"
    if not os.path.exists(npz_path):
        print(f"  ⚠ Data file not found: {npz_path}")
        return False
    
    try:
        dataset = NPZDataset(npz_path, basis_id=0)
        
        # Use subset for quick test
        subset_size = min(500, len(dataset))
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        
        # Split into train/val
        train_size = int(0.8 * subset_size)
        train_data = torch.utils.data.Subset(subset, range(train_size))
        val_data = torch.utils.data.Subset(subset, range(train_size, subset_size))
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        # Get dimensions
        (sample_x, _, _), _ = dataset[0]
        R, S, F = sample_x.shape
        grid_size = int(math.sqrt(S + 1))
        
        print(f"\n  Creating model: hidden=128, heads=4, layers=4")
        print(f"  Input: R={R}, S={S}, F={F}, grid={grid_size}")
        
        # Create smaller model for CPU testing
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=128,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=4,
            num_layers=4
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Training loop
        print(f"\n  Training for 5 epochs on {train_size} samples...")
        
        for epoch in range(5):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for (xb, basis, mask), yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb, basis, mask)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == yb).sum().item()
                train_total += xb.size(0)
            
            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for (xb, basis, mask), yb in val_loader:
                    logits = model(xb, basis, mask)
                    loss = loss_fn(logits, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == yb).sum().item()
                    val_total += xb.size(0)
            
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            
            print(f"  Epoch {epoch+1}/5: train_loss={train_loss/train_total:.4f}, "
                  f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")
        
        print(f"\n  ✓ Training completed successfully!")
        print(f"  ✓ Final validation accuracy: {val_acc:.1f}%")
        
        # Test inference speed
        model.eval()
        import time
        (xb, basis, mask), _ = next(iter(val_loader))
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(xb, basis, mask)
        elapsed = time.time() - start
        
        print(f"  ✓ Inference: {elapsed*1000/10:.1f}ms per batch (batch_size=32)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """Test model checkpoint save and load."""
    print("\n" + "=" * 60)
    print("  Test: Model Save/Load")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    import tempfile
    
    try:
        # Create model
        model = AlphaQubitDecoder(
            num_features=2,
            hidden_dim=64,
            num_stabilizers=8,
            grid_size=3,
            num_heads=4,
            num_layers=2
        )
        
        # Create test input
        B, R, S, F = 2, 5, 8, 2
        inputs = torch.randn(B, R, S, F)
        basis = torch.randint(0, 2, (B,))
        final_mask = torch.randint(1, 3, (B, S))
        
        # Get output before save
        model.eval()
        with torch.no_grad():
            output_before = model(inputs, basis, final_mask)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'num_features': 2,
                'hidden_dim': 64,
                'num_stabilizers': 8,
                'grid_size': 3,
                'num_heads': 4,
                'num_layers': 2
            }
        }, checkpoint_path)
        
        print(f"  ✓ Model saved to {checkpoint_path}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = checkpoint['config']
        
        model_loaded = AlphaQubitDecoder(**config)
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        model_loaded.eval()
        
        # Get output after load
        with torch.no_grad():
            output_after = model_loaded(inputs, basis, final_mask)
        
        # Compare outputs
        diff = (output_before - output_after).abs().max().item()
        assert diff < 1e-6, f"Output mismatch: max diff = {diff}"
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Output match: max diff = {diff:.2e}")
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_consistency():
    """Test that model produces consistent results regardless of batch arrangement."""
    print("\n" + "=" * 60)
    print("  Test: Batch Consistency")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    try:
        model = AlphaQubitDecoder(
            num_features=2,
            hidden_dim=64,
            num_stabilizers=8,
            grid_size=3,
            num_heads=4,
            num_layers=2
        )
        model.eval()
        
        # Create test inputs
        B, R, S, F = 4, 5, 8, 2
        inputs = torch.randn(B, R, S, F)
        basis = torch.randint(0, 2, (B,))
        final_mask = torch.randint(1, 3, (B, S))
        
        # Run as batch
        with torch.no_grad():
            batch_output = model(inputs, basis, final_mask)
        
        # Run individually and compare
        individual_outputs = []
        for i in range(B):
            with torch.no_grad():
                out = model(
                    inputs[i:i+1], 
                    basis[i:i+1], 
                    final_mask[i:i+1]
                )
                individual_outputs.append(out.item())
        
        individual_outputs = torch.tensor(individual_outputs)
        
        # Compare
        diff = (batch_output - individual_outputs).abs().max().item()
        
        print(f"  Batch output:      {batch_output.tolist()}")
        print(f"  Individual output: {individual_outputs.tolist()}")
        print(f"  Max difference:    {diff:.2e}")
        
        # Allow small numerical differences
        assert diff < 1e-5, f"Batch inconsistency: max diff = {diff}"
        
        print(f"  ✓ Batch consistency verified")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  AlphaQubit End-to-End Tests")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test with real data
    results["Real Data Training"] = test_real_data_training()
    
    # Test model persistence
    results["Model Save/Load"] = test_model_save_load()
    
    # Test batch consistency
    results["Batch Consistency"] = test_batch_consistency()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
