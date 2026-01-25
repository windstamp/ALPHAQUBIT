#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test training script for AlphaQubitDecoder with simulated random data.

Usage:
    python ai_models/test_train.py --num_samples 1000 --epochs 10 --batch_size 32
"""

import argparse
import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from model import AlphaQubitDecoder


class RandomSyndromeDataset(Dataset):
    """Generate random syndrome data for testing."""
    
    def __init__(self, num_samples, num_rounds, num_stabilizers, num_features, grid_size):
        super().__init__()
        self.num_samples = num_samples
        self.num_rounds = num_rounds
        self.num_stabilizers = num_stabilizers
        self.num_features = num_features
        self.grid_size = grid_size
        
        self.syndromes = torch.randint(0, 2, (num_samples, num_rounds, num_stabilizers, num_features), 
                                       dtype=torch.float32)
        
        self.basis = torch.randint(0, 2, (num_samples,), dtype=torch.long)
        
        d = grid_size + 1
        final_mask_pattern = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
            dtype=torch.long,
        )
        self.final_mask = final_mask_pattern.unsqueeze(0).expand(num_samples, -1)
        
        self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (self.syndromes[idx], self.basis[idx], self.final_mask[idx]), self.labels[idx]


def register_hooks(model, hook_dict):
    """Register forward hooks to capture layer inputs/outputs."""
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input_shapes = [tuple(x.shape) if hasattr(x, 'shape') else str(type(x)) for x in input]
                input_dtypes = [str(x.dtype) if hasattr(x, 'dtype') else str(type(x)) for x in input]
            else:
                input_shapes = [tuple(input.shape) if hasattr(input, 'shape') else str(type(input))]
                input_dtypes = [str(input.dtype) if hasattr(input, 'dtype') else str(type(input))]
            
            if isinstance(output, tuple):
                output_shapes = [tuple(x.shape) if hasattr(x, 'shape') else str(type(x)) for x in output]
                output_dtypes = [str(x.dtype) if hasattr(x, 'dtype') else str(type(x)) for x in output]
            else:
                output_shapes = [tuple(output.shape) if hasattr(output, 'shape') else str(type(output))]
                output_dtypes = [str(output.dtype) if hasattr(output, 'dtype') else str(type(output))]
            
            hook_dict[name] = {
                'input_shapes': input_shapes,
                'input_dtypes': input_dtypes,
                'output_shapes': output_shapes,
                'output_dtypes': output_dtypes
            }
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    return hooks


def train_model(model, train_loader, val_loader, epochs, lr, device, save_path, weight_decay=1e-4, enable_profiling=False, profile_dir="./profiling_logs"):
    """Training loop with validation."""
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Train Samples: {len(train_loader.dataset)}")
    print(f"  Val Samples: {len(val_loader.dataset)}")
    if enable_profiling:
        print(f"  Profiling: Enabled (output to {profile_dir})")
    print(f"{'='*60}\n")
    
    hook_dict = {}
    hooks = None
    if enable_profiling:
        os.makedirs(profile_dir, exist_ok=True)
        hooks = register_hooks(model, hook_dict)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        use_profiler = enable_profiling and epoch == 1
        profiler = None
        
        if use_profiler:
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.__enter__()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, ((inputs, basis, final_mask), labels) in enumerate(pbar):
            inputs = inputs.to(device)
            basis = basis.to(device)
            final_mask = final_mask.to(device)
            labels = labels.to(device)
            
            logits = model(inputs, basis, final_mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
            
            if use_profiler and batch_idx >= 2:
                break
        
        if use_profiler and profiler is not None:
            profiler.__exit__(None, None, None)
            
            trace_file = os.path.join(profile_dir, f"train_trace_epoch{epoch}.json")
            profiler.export_chrome_trace(trace_file)
            print(f"\nProfiler trace saved to: {trace_file}")
            
            print("\nTop 20 CPU operations:")
            print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            
            if torch.cuda.is_available():
                print("\nTop 20 CUDA operations:")
                print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            
            if hook_dict:
                hook_file = os.path.join(profile_dir, f"layer_shapes_epoch{epoch}.json")
                with open(hook_file, 'w') as f:
                    json.dump(hook_dict, f, indent=2)
                print(f"Layer shapes saved to: {hook_file}\n")
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for (inputs, basis, final_mask), labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  "):
                inputs = inputs.to(device)
                basis = basis.to(device)
                final_mask = final_mask.to(device)
                labels = labels.to(device)
                
                logits = model(inputs, basis, final_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, save_path)
            print(f"Saved best model to {save_path} (val_loss: {avg_val_loss:.4f})\n")
    
    if hooks:
        for hook in hooks:
            hook.remove()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaQubitDecoder with random data")
    
    parser.add_argument("--num_features", type=int, default=2, help="Number of input features")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_stabilizers", type=int, default=48, help="Number of stabilizers")
    parser.add_argument("--grid_size", type=int, default=6, help="Grid size (d-1 where d×d is the grid)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of syndrome rounds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda", "npu"],
                       help="Device to use for training")
    parser.add_argument("--save_path", type=str, default="alphaqubit_test.pth",
                       help="Path to save the trained model")
    parser.add_argument("--profile", action="store_true",
                       help="Enable profiling to capture operator execution details")
    parser.add_argument("--profile_dir", type=str, default="./profiling_logs",
                       help="Directory to save profiling results")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"AlphaQubit Test Training Script")
    print(f"{'='*60}")
    print(f"\nModel Configuration:")
    print(f"  Features: {args.num_features}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Stabilizers: {args.num_stabilizers}")
    print(f"  Grid Size: {args.grid_size}")
    print(f"  Attention Heads: {args.num_heads}")
    print(f"  Transformer Layers: {args.num_layers}")
    
    print(f"\nData Configuration:")
    print(f"  Total Samples: {args.num_samples}")
    print(f"  Syndrome Rounds: {args.num_rounds}")
    print(f"  Train Split: {args.train_split}")
    print(f"  Batch Size: {args.batch_size}\n")
    
    print("Generating random syndrome dataset...")
    dataset = RandomSyndromeDataset(
        num_samples=args.num_samples,
        num_rounds=args.num_rounds,
        num_stabilizers=args.num_stabilizers,
        num_features=args.num_features,
        grid_size=args.grid_size
    )
    
    train_size = int(args.num_samples * args.train_split)
    val_size = args.num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset created: {train_size} train, {val_size} val samples")
    
    print("\nInitializing AlphaQubitDecoder...")
    model = AlphaQubitDecoder(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_stabilizers=args.num_stabilizers,
        grid_size=args.grid_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_path,
        weight_decay=args.weight_decay,
        enable_profiling=args.profile,
        profile_dir=args.profile_dir
    )


if __name__ == "__main__":
    main()
