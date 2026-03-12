#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test evaluation script for AlphaQubitDecoder with simulated random data.

Usage:
    python ai_models/test_eval.py --checkpoint alphaqubit_test.pth --num_samples 200
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import numpy as np

from model import AlphaQubitDecoder
from profiling_utils import register_hooks, print_profiler_summary


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


def evaluate_model(model, data_loader, device, verbose=True, enable_profiling=False, profile_dir="./profiling_logs"):
    """Evaluate model on test data."""
    
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating model...")
    if enable_profiling:
        print(f"Profiling enabled (output to {profile_dir})")
    print(f"{'='*60}\n")
    
    hook_dict = {}
    hooks = None
    if enable_profiling:
        os.makedirs(profile_dir, exist_ok=True)
        hooks = register_hooks(model, hook_dict)
    
    profiler = None
    if enable_profiling:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.__enter__()
    
    with torch.no_grad():
        for batch_idx, ((inputs, basis, final_mask), labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            inputs = inputs.to(device)
            basis = basis.to(device)
            final_mask = final_mask.to(device)
            labels = labels.to(device)
            
            logits = model(inputs, basis, final_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_logits.extend(logits.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if enable_profiling and batch_idx >= 2:
                break
    
    if enable_profiling and profiler is not None:
        profiler.__exit__(None, None, None)
        
        trace_file = os.path.join(profile_dir, "test_eval_trace.json")
        profiler.export_chrome_trace(trace_file)
        print(f"\nProfiler trace saved to: {trace_file}")
        
        print_profiler_summary(profiler)
        
        if hook_dict:
            hook_file = os.path.join(profile_dir, "layer_shapes_test_eval.json")
            with open(hook_file, 'w') as f:
                json.dump(hook_dict, f, indent=2)
            print(f"Layer shapes saved to: {hook_file}\n")
    
    if hooks:
        for hook in hooks:
            hook.remove()
    
    all_logits = np.array(all_logits)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = (all_preds == all_labels).mean()
    
    true_positives = ((all_preds == 1) & (all_labels == 1)).sum()
    false_positives = ((all_preds == 1) & (all_labels == 0)).sum()
    false_negatives = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"  Test Loss:     {avg_loss:.4f}")
    print(f"  Accuracy:      {accuracy:.4f} ({int(accuracy * len(all_labels))}/{len(all_labels)})")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1 Score:      {f1_score:.4f}")
    print(f"{'='*60}\n")
    
    if verbose:
        print(f"Prediction Distribution:")
        print(f"  Predicted 0: {(all_preds == 0).sum()} samples")
        print(f"  Predicted 1: {(all_preds == 1).sum()} samples")
        print(f"\nLabel Distribution:")
        print(f"  Label 0: {(all_labels == 0).sum()} samples")
        print(f"  Label 1: {(all_labels == 1).sum()} samples")
        print(f"\nLogits Statistics:")
        print(f"  Mean: {all_logits.mean():.4f}")
        print(f"  Std:  {all_logits.std():.4f}")
        print(f"  Min:  {all_logits.min():.4f}")
        print(f"  Max:  {all_logits.max():.4f}")
        print()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': all_preds,
        'labels': all_labels,
        'logits': all_logits
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaQubitDecoder with random data")
    
    parser.add_argument("--num_features", type=int, default=2, help="Number of input features")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_stabilizers", type=int, default=48, help="Number of stabilizers")
    parser.add_argument("--grid_size", type=int, default=6, help="Grid size (d-1 where d×d is the grid)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of syndrome rounds")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for test data generation")
    
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda", "npu"],
                       help="Device to use for evaluation")
    
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")
    parser.add_argument("--save_predictions", type=str, default=None,
                       help="Path to save predictions as .npz file")
    parser.add_argument("--profile", action="store_true",
                       help="Enable profiling to capture operator execution details")
    parser.add_argument("--profile_dir", type=str, default="./profiling_logs",
                       help="Directory to save profiling results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("\nPlease run training first:")
        print("  python ai_models/test_train.py --epochs 10 --num_samples 1000")
        return
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"AlphaQubit Test Evaluation Script")
    print(f"{'='*60}")
    print(f"\nModel Configuration:")
    print(f"  Features: {args.num_features}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Stabilizers: {args.num_stabilizers}")
    print(f"  Grid Size: {args.grid_size}")
    print(f"  Attention Heads: {args.num_heads}")
    print(f"  Transformer Layers: {args.num_layers}")
    
    print(f"\nEvaluation Configuration:")
    print(f"  Test Samples: {args.num_samples}")
    print(f"  Syndrome Rounds: {args.num_rounds}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}\n")
    
    print("Initializing AlphaQubitDecoder...")
    model = AlphaQubitDecoder(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_stabilizers=args.num_stabilizers,
        grid_size=args.grid_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"  Training val_loss: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"  Training val_acc:  {checkpoint['val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}\n")
    
    print("Generating random test dataset...")
    test_dataset = RandomSyndromeDataset(
        num_samples=args.num_samples,
        num_rounds=args.num_rounds,
        num_stabilizers=args.num_stabilizers,
        num_features=args.num_features,
        grid_size=args.grid_size
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset created: {args.num_samples} samples")
    
    results = evaluate_model(model, test_loader, device, verbose=args.verbose, 
                           enable_profiling=args.profile, profile_dir=args.profile_dir)
    
    if args.save_predictions:
        np.savez(
            args.save_predictions,
            predictions=results['predictions'],
            labels=results['labels'],
            logits=results['logits'],
            accuracy=results['accuracy'],
            precision=results['precision'],
            recall=results['recall'],
            f1_score=results['f1_score']
        )
        print(f"Predictions saved to {args.save_predictions}")


if __name__ == "__main__":
    main()
