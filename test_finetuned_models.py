"""
Test fine-tuned AlphaQubit models on Google QEC v3.5 test data.
Evaluates all fine-tuned models and generates comprehensive results.
"""

import os
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from fine_tune_npz
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_models'))

try:
    from ai_models.fine_tune_npz import NPZDataset, collate_fn
    from ai_models.model_mla import AlphaQubitDecoder
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_models'))
    from fine_tune_npz import NPZDataset, collate_fn
    from model_mla import AlphaQubitDecoder


def load_model(model_path, dataset, device):
    """Load a fine-tuned model."""
    model = AlphaQubitDecoder(
        num_features=dataset.n_features,
        hidden_dim=256,  # Match fine-tuning settings
        num_stabilizers=dataset.n_detectors,
        grid_size=int(np.sqrt(dataset.n_detectors)) + 1,
        num_heads=8,
        num_layers=12
    ).to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for x, basis, final_mask, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            basis = basis.to(device)
            final_mask = final_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(x, basis, final_mask)
            loss = criterion(logits.squeeze(), labels)
            
            # Get predictions
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()
            
            # Store results
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.squeeze().cpu().numpy())
            total_loss += loss.item() * x.size(0)
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    
    # Logical error rate (LER) = fraction of incorrect predictions
    ler = 1.0 - accuracy
    
    # True positives, false positives, etc.
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_loss = total_loss / len(all_labels)
    
    metrics = {
        'accuracy': float(accuracy),
        'logical_error_rate': float(ler),
        'avg_loss': float(avg_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_samples': int(len(all_labels)),
        'positive_samples': int(np.sum(all_labels)),
        'negative_samples': int(len(all_labels) - np.sum(all_labels))
    }
    
    return metrics, all_preds, all_logits


def test_experiment(model_path, test_npz_path, args, device):
    """Test a single fine-tuned model on its test set."""
    exp_name = Path(model_path).stem.replace('finetuned_', '')
    
    print(f"\n{'='*80}")
    print(f"Testing: {exp_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Load test dataset
        print(f"Loading test data: {test_npz_path}")
        dataset = NPZDataset(test_npz_path)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_fn
        )
        
        # Load model
        print(f"Loading model: {model_path}")
        model = load_model(model_path, dataset, device)
        
        # Evaluate
        print("Evaluating...")
        metrics, predictions, logits = evaluate_model(model, dataloader, device)
        
        elapsed = time.time() - start_time
        
        # Print results
        print(f"\nResults for {exp_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Logical Error Rate: {metrics['logical_error_rate']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        
        # Save predictions if requested
        if args.save_predictions:
            pred_dir = os.path.join(args.results_dir, 'predictions')
            os.makedirs(pred_dir, exist_ok=True)
            
            pred_path = os.path.join(pred_dir, f"{exp_name}_predictions.npz")
            np.savez_compressed(
                pred_path,
                predictions=predictions,
                logits=logits,
                metadata=json.dumps(dataset.metadata)
            )
            print(f"  Saved predictions to {pred_path}")
        
        # Cleanup
        del model, dataloader, dataset
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        result = {
            'experiment': exp_name,
            'status': 'success',
            'metrics': metrics,
            'elapsed_time': elapsed,
            'model_path': str(model_path),
            'test_data_path': str(test_npz_path)
        }
        
        return result
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error testing {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'experiment': exp_name,
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed
        }


def generate_summary_report(results, output_path):
    """Generate a summary report from all test results."""
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    # Overall statistics
    summary = {
        'total_experiments': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if successful:
        # Aggregate metrics
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in successful])
        avg_ler = np.mean([r['metrics']['logical_error_rate'] for r in successful])
        avg_f1 = np.mean([r['metrics']['f1_score'] for r in successful])
        
        summary['average_metrics'] = {
            'accuracy': float(avg_accuracy),
            'logical_error_rate': float(avg_ler),
            'f1_score': float(avg_f1)
        }
        
        # Best and worst performing experiments
        best_exp = min(successful, key=lambda r: r['metrics']['logical_error_rate'])
        worst_exp = max(successful, key=lambda r: r['metrics']['logical_error_rate'])
        
        summary['best_experiment'] = {
            'name': best_exp['experiment'],
            'logical_error_rate': best_exp['metrics']['logical_error_rate'],
            'accuracy': best_exp['metrics']['accuracy']
        }
        
        summary['worst_experiment'] = {
            'name': worst_exp['experiment'],
            'logical_error_rate': worst_exp['metrics']['logical_error_rate'],
            'accuracy': worst_exp['metrics']['accuracy']
        }
        
        # Group by code type
        by_code_type = defaultdict(list)
        for r in successful:
            exp_name = r['experiment']
            if 'surface_code' in exp_name:
                code_type = 'surface_code'
            elif 'repetition_code' in exp_name:
                code_type = 'repetition_code'
            else:
                code_type = 'unknown'
            by_code_type[code_type].append(r)
        
        summary['by_code_type'] = {}
        for code_type, exps in by_code_type.items():
            summary['by_code_type'][code_type] = {
                'count': len(exps),
                'avg_accuracy': float(np.mean([e['metrics']['accuracy'] for e in exps])),
                'avg_ler': float(np.mean([e['metrics']['logical_error_rate'] for e in exps]))
            }
    
    # Full results
    summary['results'] = results
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nTotal experiments tested: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    
    if successful:
        print(f"\nAverage Metrics:")
        print(f"  Accuracy: {summary['average_metrics']['accuracy']:.4f}")
        print(f"  Logical Error Rate: {summary['average_metrics']['logical_error_rate']:.4f}")
        print(f"  F1 Score: {summary['average_metrics']['f1_score']:.4f}")
        
        print(f"\nBest Experiment: {summary['best_experiment']['name']}")
        print(f"  LER: {summary['best_experiment']['logical_error_rate']:.4f}")
        
        print(f"\nWorst Experiment: {summary['worst_experiment']['name']}")
        print(f"  LER: {summary['worst_experiment']['logical_error_rate']:.4f}")
        
        print(f"\nBy Code Type:")
        for code_type, stats in summary['by_code_type'].items():
            print(f"  {code_type}: {stats['count']} experiments, avg LER = {stats['avg_ler']:.4f}")
    
    print(f"\n✓ Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned AlphaQubit models")
    
    # Data arguments
    parser.add_argument('--model-dir', type=str, default='finetuned_models',
                        help='Directory containing fine-tuned models')
    parser.add_argument('--test-dir', type=str, default='google_finetune_data/test',
                        help='Directory containing test NPZ files')
    parser.add_argument('--results-dir', type=str, default='test_results',
                        help='Directory to save test results')
    
    # Testing arguments
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save model predictions for each experiment')
    
    # Device arguments
    parser.add_argument('--npu', action='store_true', help='Use NPU for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Specific device to use (e.g., "cuda:0", "npu:0")')
    
    # Filtering arguments
    parser.add_argument('--filter', type=str, default=None,
                        help='Only test experiments matching this pattern')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of experiments to test')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif args.npu:
        try:
            import torch_npu
            if hasattr(torch, 'npu') and torch.npu.is_available():
                device = torch.device('npu:0')
            else:
                print("Warning: NPU requested but not available")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            print("Warning: torch_npu not installed")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Find all fine-tuned models
    model_dir = Path(args.model_dir)
    test_dir = Path(args.test_dir)
    
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    model_files = list(model_dir.glob('finetuned_*.pth'))
    model_files.sort()
    
    print(f"\nFound {len(model_files)} fine-tuned models")
    
    # Match models with test data
    test_pairs = []
    for model_path in model_files:
        exp_name = model_path.stem.replace('finetuned_', '')
        test_npz = test_dir / f"samples_{exp_name}.npz"
        
        if test_npz.exists():
            test_pairs.append((model_path, test_npz))
        else:
            print(f"Warning: No test data found for {exp_name}")
    
    print(f"Found {len(test_pairs)} model-test pairs")
    
    # Filter if requested
    if args.filter:
        test_pairs = [(m, t) for m, t in test_pairs if args.filter in m.stem]
        print(f"Filtered to {len(test_pairs)} pairs matching '{args.filter}'")
    
    # Limit if requested
    if args.limit:
        test_pairs = test_pairs[:args.limit]
        print(f"Limited to {args.limit} pairs")
    
    if not test_pairs:
        print("No experiments to test!")
        return
    
    print(f"\nTesting {len(test_pairs)} experiments...\n")
    
    # Test all experiments
    total_start = time.time()
    results = []
    
    for model_path, test_npz in test_pairs:
        result = test_experiment(model_path, test_npz, args, device)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Generate summary report
    summary_path = os.path.join(args.results_dir, 'test_summary.json')
    generate_summary_report(results, summary_path)
    
    print(f"\nTotal testing time: {total_elapsed/60:.1f} minutes")
    print(f"\n✓ All testing completed!")


if __name__ == '__main__':
    main()
