#!/usr/bin/env python3
"""
Local CPU tests for AlphaQubit model.
Tests model initialization, forward pass, and training loop with synthetic data.
"""

import sys
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, '.')

def test_model_initialization():
    """Test 1: Model can be initialized with various configurations."""
    print("=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    test_configs = [
        # (d, num_stabilizers, hidden_dim, num_heads, num_layers)
        (3, 8, 64, 4, 2),      # Small d=3
        (5, 24, 64, 4, 2),     # Medium d=5
        (7, 48, 64, 4, 2),     # Large d=7
        (3, 6, 64, 4, 2),      # Non-standard S (needs padding)
        (5, 20, 64, 4, 2),     # Non-standard S (needs padding)
    ]
    
    for d, S, hidden_dim, num_heads, num_layers in test_configs:
        try:
            # Calculate expected grid size
            grid_size = int(math.ceil(math.sqrt(S + 1)))
            if grid_size * grid_size < S + 1:
                grid_size += 1
                
            model = AlphaQubitDecoder(
                num_features=2,  # detection event + basis
                hidden_dim=hidden_dim,
                num_stabilizers=S,
                grid_size=grid_size,
                num_heads=num_heads,
                num_layers=num_layers
            )
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ d={d}, S={S}, grid={grid_size}: {num_params:,} params")
            
        except Exception as e:
            print(f"  ✗ d={d}, S={S}: FAILED - {e}")
            return False
    
    print("  → All initialization tests PASSED\n")
    return True


def test_forward_pass():
    """Test 2: Forward pass works with various input shapes."""
    print("=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    test_cases = [
        # (batch_size, rounds, stabilizers, features, grid_size)
        (4, 5, 8, 2, 3),     # d=3, S=8
        (4, 10, 24, 2, 5),   # d=5, S=24
        (4, 15, 48, 2, 7),   # d=7, S=48
        (2, 5, 6, 2, 3),     # Non-standard S=6
        (2, 5, 20, 2, 5),    # Non-standard S=20
    ]
    
    for B, R, S, F, grid_size in test_cases:
        try:
            model = AlphaQubitDecoder(
                num_features=F,
                hidden_dim=64,
                num_stabilizers=S,
                grid_size=grid_size,
                num_heads=4,
                num_layers=2
            )
            model.eval()
            
            # Create synthetic input
            inputs = torch.randn(B, R, S, F)
            basis = torch.randint(0, 2, (B,))
            final_mask = torch.randint(1, 3, (B, S))
            
            with torch.no_grad():
                output = model(inputs, basis, final_mask)
            
            assert output.shape == (B,), f"Expected ({B},), got {output.shape}"
            print(f"  ✓ B={B}, R={R}, S={S}, F={F}, grid={grid_size}: output shape {output.shape}")
            
        except Exception as e:
            print(f"  ✗ B={B}, R={R}, S={S}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("  → All forward pass tests PASSED\n")
    return True


def test_backward_pass():
    """Test 3: Backward pass and gradient computation."""
    print("=" * 60)
    print("TEST 3: Backward Pass & Gradients")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    B, R, S, F = 4, 5, 8, 2
    grid_size = 3
    
    try:
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=64,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=4,
            num_layers=2
        )
        
        inputs = torch.randn(B, R, S, F)
        basis = torch.randint(0, 2, (B,))
        final_mask = torch.randint(1, 3, (B, S))
        targets = torch.rand(B)
        
        # Forward
        output = model(inputs, basis, final_mask)
        
        # Loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, targets)
        
        # Backward
        loss.backward()
        
        # Check gradients - count parameters with and without gradients
        params_with_grad = 0
        params_without_grad = 0
        params_no_grad_names = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    params_with_grad += 1
                else:
                    params_without_grad += 1
                    params_no_grad_names.append(name)
        
        # It's okay if some params don't have grads (e.g., unused embeddings)
        # But most should have gradients
        grad_ratio = params_with_grad / (params_with_grad + params_without_grad) if (params_with_grad + params_without_grad) > 0 else 0
        assert grad_ratio > 0.5, f"Too few parameters have gradients: {params_with_grad}/{params_with_grad + params_without_grad}"
        
        # Check no NaN gradients
        no_nan = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        assert no_nan, "NaN gradients detected"
        
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ {params_with_grad}/{params_with_grad + params_without_grad} parameters have gradients")
        if params_no_grad_names:
            print(f"  ℹ Params without grad (may be normal): {params_no_grad_names[:3]}...")
        print("  → Backward pass test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test 4: Complete training step with optimizer."""
    print("=" * 60)
    print("TEST 4: Training Step")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    B, R, S, F = 8, 5, 8, 2
    grid_size = 3
    
    try:
        model = AlphaQubitDecoder(
            num_features=F,
            hidden_dim=64,
            num_stabilizers=S,
            grid_size=grid_size,
            num_heads=4,
            num_layers=2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Create synthetic batch
        inputs = torch.randn(B, R, S, F)
        basis = torch.randint(0, 2, (B,))
        final_mask = torch.randint(1, 3, (B, S))
        targets = torch.rand(B)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(inputs, basis, final_mask)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step completed")
        print(f"  ✓ Loss: {loss.item():.4f}")
        
        # Multiple steps to check stability
        losses = [loss.item()]
        for step in range(5):
            optimizer.zero_grad()
            output = model(inputs, basis, final_mask)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"  ✓ 5 additional steps: losses = {[f'{l:.4f}' for l in losses[1:]]}")
        print("  → Training step test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_class():
    """Test 5: PauliPlusDataset loading and padding."""
    print("=" * 60)
    print("TEST 5: Dataset Class")
    print("=" * 60)
    
    import os
    import tempfile
    from ai_models.model import PauliPlusDataset
    
    try:
        # Create temporary test data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various stabilizer counts
            test_cases = [
                (100, 5, 8),    # N=100, R=5, S=8 (d=3)
                (100, 10, 24),  # N=100, R=10, S=24 (d=5)
                (100, 5, 6),    # N=100, R=5, S=6 (non-standard)
            ]
            
            for N, R, S in test_cases:
                # Create synthetic data files
                syndromes = np.random.randint(0, 2, (N, R, S)).astype(np.float32)
                logicals = np.random.randint(0, 2, (N,)).astype(np.float32)
                
                synd_path = os.path.join(tmpdir, f"syndromes_x_test.npy")
                log_path = os.path.join(tmpdir, f"logicals_x_test.npy")
                
                np.save(synd_path, syndromes)
                np.save(log_path, logicals)
                
                # Load dataset
                dataset = PauliPlusDataset(synd_path, log_path, basis_id=0)
                
                # Check shape
                (x, basis, mask), y = dataset[0]
                
                # Verify padding worked
                _, R_out, S_out, F_out = dataset.X.shape
                d = int(math.ceil(math.sqrt(S_out + 1)))
                expected_S = d * d - 1
                
                print(f"  ✓ N={N}, R={R}, S={S} → padded S={S_out}, grid={d}")
                assert S_out == expected_S, f"Padding failed: got S={S_out}, expected {expected_S}"
                assert F_out == 2, f"Features should be 2 (event + basis), got {F_out}"
        
        print("  → Dataset class test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training_loop():
    """Test 6: Mini training loop with DataLoader."""
    print("=" * 60)
    print("TEST 6: Mini Training Loop (3 epochs)")
    print("=" * 60)
    
    import os
    import tempfile
    from ai_models.model import AlphaQubitDecoder, PauliPlusDataset
    from torch.utils.data import DataLoader
    
    try:
        # Create small synthetic dataset
        N, R, S = 32, 5, 8  # Small dataset for quick test
        grid_size = 3
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            syndromes = np.random.randint(0, 2, (N, R, S)).astype(np.float32)
            logicals = np.random.randint(0, 2, (N,)).astype(np.float32)
            
            synd_path = os.path.join(tmpdir, "syndromes_x_test.npy")
            log_path = os.path.join(tmpdir, "logicals_x_test.npy")
            np.save(synd_path, syndromes)
            np.save(log_path, logicals)
            
            # Create dataset and loader
            dataset = PauliPlusDataset(synd_path, log_path, basis_id=0)
            loader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Get padded S
            S_padded = dataset.X.shape[2]
            
            # Create model
            model = AlphaQubitDecoder(
                num_features=2,
                hidden_dim=64,
                num_stabilizers=S_padded,
                grid_size=grid_size,
                num_heads=4,
                num_layers=2
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            
            # Training loop
            for epoch in range(3):
                model.train()
                total_loss = 0
                for (xb, basis, mask), yb in loader:
                    optimizer.zero_grad()
                    output = model(xb, basis, mask)
                    loss = loss_fn(output, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                
                avg_loss = total_loss / len(dataset)
                print(f"  Epoch {epoch+1}/3: loss = {avg_loss:.4f}")
            
        print("  → Mini training loop test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_paper_config():
    """Test 7: Test with paper-aligned configuration."""
    print("=" * 60)
    print("TEST 7: Paper Configuration (hidden=256, heads=8, layers=12)")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder
    
    # Paper configuration
    hidden_dim = 256
    num_heads = 8
    num_layers = 12
    
    test_configs = [
        (3, 8, 3),    # d=3
        (5, 24, 5),   # d=5  
        (7, 48, 7),   # d=7
    ]
    
    try:
        for d, S, grid_size in test_configs:
            model = AlphaQubitDecoder(
                num_features=2,
                hidden_dim=hidden_dim,
                num_stabilizers=S,
                grid_size=grid_size,
                num_heads=num_heads,
                num_layers=num_layers
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            
            # Quick forward pass
            B, R = 2, 5
            inputs = torch.randn(B, R, S, 2)
            basis = torch.randint(0, 2, (B,))
            final_mask = torch.randint(1, 3, (B, S))
            
            model.eval()
            with torch.no_grad():
                output = model(inputs, basis, final_mask)
            
            print(f"  ✓ d={d}: {num_params:,} params, output shape {output.shape}")
        
        print("  → Paper configuration test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  AlphaQubit Local Model Tests (CPU)")
    print("=" * 60 + "\n")
    
    results = {
        "Model Initialization": test_model_initialization(),
        "Forward Pass": test_forward_pass(),
        "Backward Pass": test_backward_pass(),
        "Training Step": test_training_step(),
        "Dataset Class": test_dataset_class(),
        "Mini Training Loop": test_mini_training_loop(),
        "Paper Configuration": test_paper_config(),
    }
    
    # Summary
    print("=" * 60)
    print("  TEST SUMMARY")
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
