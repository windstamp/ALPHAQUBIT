#!/usr/bin/env python3
"""
Generate test_summary.json from existing prediction files.
This reconstructs the summary when predictions exist but the original summary wasn't saved.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

def calculate_metrics_from_predictions(pred_file):
    """Load predictions and calculate metrics."""
    try:
        data = np.load(pred_file)
        predictions = data['predictions']
        
        # Try to get labels from metadata
        if 'metadata' in data.files:
            metadata = json.loads(str(data['metadata']))
        else:
            metadata = {}
        
        # We need to load the original test data to get labels
        # The prediction file name format: {experiment}_predictions.npz
        exp_name = pred_file.stem.replace('_predictions', '')
        
        # Try to find the corresponding test NPZ file
        test_npz_candidates = [
            Path(f'google_finetune_data/test/samples_{exp_name}.npz'),
            Path(f'test_data/samples_{exp_name}.npz'),
        ]
        
        labels = None
        for test_npz in test_npz_candidates:
            if test_npz.exists():
                test_data = np.load(test_npz)
                if 'y' in test_data.files:
                    labels = test_data['y']
                elif 'labels' in test_data.files:
                    labels = test_data['labels']
                break
        
        if labels is None:
            print(f"Warning: Could not find labels for {exp_name}, skipping")
            return None, exp_name
        
        # Ensure same length
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        
        # Calculate metrics
        accuracy = np.mean(predictions == labels)
        ler = 1.0 - accuracy
        
        # Confusion matrix
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'logical_error_rate': float(ler),
            'avg_loss': 0.0,  # Not available from predictions alone
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': int(len(labels)),
            'positive_samples': int(np.sum(labels)),
            'negative_samples': int(len(labels) - np.sum(labels))
        }
        
        return metrics, exp_name
        
    except Exception as e:
        print(f"Error processing {pred_file}: {e}")
        return None, None


def main():
    predictions_dir = Path('test_results/predictions')
    
    if not predictions_dir.exists():
        print(f"Error: {predictions_dir} not found")
        return 1
    
    pred_files = sorted(predictions_dir.glob('*_predictions.npz'))
    
    print(f"Found {len(pred_files)} prediction files")
    print("Generating summary from predictions...")
    
    results = []
    
    for pred_file in pred_files:
        print(f"Processing {pred_file.name}...", end=' ')
        metrics, exp_name = calculate_metrics_from_predictions(pred_file)
        
        if metrics:
            result = {
                'experiment': exp_name,
                'status': 'success',
                'metrics': metrics,
                'elapsed_time': 0.0,  # Not available
                'model_path': f'finetuned_models/finetuned_{exp_name}.pth',
                'test_data_path': f'google_finetune_data/test/samples_{exp_name}.npz'
            }
            results.append(result)
            print(f"✓ LER={metrics['logical_error_rate']:.4f}")
        else:
            print("✗ Failed")
    
    if not results:
        print("No successful results found")
        return 1
    
    # Generate summary
    successful = [r for r in results if r['status'] == 'success']
    
    avg_accuracy = np.mean([r['metrics']['accuracy'] for r in successful])
    avg_ler = np.mean([r['metrics']['logical_error_rate'] for r in successful])
    avg_f1 = np.mean([r['metrics']['f1_score'] for r in successful])
    
    best_exp = min(successful, key=lambda r: r['metrics']['logical_error_rate'])
    worst_exp = max(successful, key=lambda r: r['metrics']['logical_error_rate'])
    
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
    
    summary = {
        'total_experiments': len(results),
        'successful': len(successful),
        'failed': 0,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'average_metrics': {
            'accuracy': float(avg_accuracy),
            'logical_error_rate': float(avg_ler),
            'f1_score': float(avg_f1)
        },
        'best_experiment': {
            'name': best_exp['experiment'],
            'logical_error_rate': best_exp['metrics']['logical_error_rate'],
            'accuracy': best_exp['metrics']['accuracy']
        },
        'worst_experiment': {
            'name': worst_exp['experiment'],
            'logical_error_rate': worst_exp['metrics']['logical_error_rate'],
            'accuracy': worst_exp['metrics']['accuracy']
        },
        'by_code_type': {}
    }
    
    for code_type, exps in by_code_type.items():
        summary['by_code_type'][code_type] = {
            'count': len(exps),
            'avg_accuracy': float(np.mean([e['metrics']['accuracy'] for e in exps])),
            'avg_ler': float(np.mean([e['metrics']['logical_error_rate'] for e in exps]))
        }
    
    summary['results'] = results
    
    # Save summary
    output_path = Path('test_results/test_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(successful)}")
    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Average LER: {avg_ler:.4f}")
    print(f"Best experiment: {best_exp['experiment']} (LER={best_exp['metrics']['logical_error_rate']:.4f})")
    print(f"Worst experiment: {worst_exp['experiment']} (LER={worst_exp['metrics']['logical_error_rate']:.4f})")
    print(f"\nBy code type:")
    for code_type, stats in summary['by_code_type'].items():
        print(f"  {code_type}: {stats['count']} experiments, avg LER = {stats['avg_ler']:.4f}")
    
    print(f"\n{'='*80}")
    print("Now run: python3 run_evaluation_and_plot.py")
    print("(It will skip re-evaluation and generate plots from this summary)")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
