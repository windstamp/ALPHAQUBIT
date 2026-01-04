"""
Deep Investigation Script: Analyze S3 results to understand d5 performance gap
"""
import json
import numpy as np
from pathlib import Path

# Load results
results_path = Path("s3_results/20251230_220521/test_results_v2/test_summary.json")
with open(results_path) as f:
    data = json.load(f)

results = data['results']

print("=" * 80)
print("INVESTIGATION: Why is d5 performance so much worse than d3?")
print("=" * 80)

# 1. Compare d3 vs d5 at same noise level (r01)
print("\n### 1. COMPARISON: d3 vs d5 at r01 (low noise)")
print("-" * 60)

d3_r01 = [r for r in results if '_d3_r01_' in r['experiment']]
d5_r01 = [r for r in results if '_d5_r01_' in r['experiment']]

print(f"\nd3_r01 experiments: {len(d3_r01)}")
for r in d3_r01:
    metrics = r['metrics']
    print(f"  {r['experiment']}")
    print(f"    LER: {metrics['logical_error_rate']:.4f}")
    print(f"    Acc: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1_score']:.4f}")
    print(f"    True Pos: {metrics['true_positives']}, False Pos: {metrics['false_positives']}")
    print(f"    True Neg: {metrics['true_negatives']}, False Neg: {metrics['false_negatives']}")
    print(f"    Positive samples: {metrics['positive_samples']}, Negative: {metrics['negative_samples']}")
    pos_ratio = metrics['positive_samples'] / metrics['total_samples']
    print(f"    Label ratio (positive): {pos_ratio:.2%}")

print(f"\nd5_r01 experiments: {len(d5_r01)}")
for r in d5_r01:
    metrics = r['metrics']
    print(f"  {r['experiment']}")
    print(f"    LER: {metrics['logical_error_rate']:.4f}")
    print(f"    Acc: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1_score']:.4f}")
    print(f"    True Pos: {metrics['true_positives']}, False Pos: {metrics['false_positives']}")
    print(f"    True Neg: {metrics['true_negatives']}, False Neg: {metrics['false_negatives']}")
    print(f"    Positive samples: {metrics['positive_samples']}, Negative: {metrics['negative_samples']}")
    pos_ratio = metrics['positive_samples'] / metrics['total_samples']
    print(f"    Label ratio (positive): {pos_ratio:.2%}")

# 2. Check if model is just predicting all 0s
print("\n### 2. PREDICTION BIAS ANALYSIS")
print("-" * 60)

# Check precision/recall pattern
for r in results[:20]:  # Check first 20
    metrics = r['metrics']
    tp, fp, tn, fn = metrics['true_positives'], metrics['false_positives'], \
                      metrics['true_negatives'], metrics['false_negatives']
    total_pred_pos = tp + fp
    total_pred_neg = tn + fn
    total_actual_pos = tp + fn
    total_actual_neg = tn + fp
    
    # If model predicts all 0s, precision=0, recall=0, TP=0, FP=0
    if tp == 0 and fp == 0:
        print(f"⚠️ {r['experiment']}: Model predicts ALL ZEROS! (TP=0, FP=0)")
    elif total_pred_pos < total_actual_pos * 0.1:
        print(f"⚠️ {r['experiment']}: Model predicts mostly zeros ({total_pred_pos} vs {total_actual_pos} actual positives)")

# 3. Summary statistics by distance
print("\n### 3. SUMMARY BY CODE DISTANCE")
print("-" * 60)

by_distance = {}
for r in results:
    exp = r['experiment']
    for d in [3, 5, 7, 25]:
        if f'_d{d}_' in exp:
            if d not in by_distance:
                by_distance[d] = []
            by_distance[d].append(r['metrics'])
            break

for d in sorted(by_distance.keys()):
    metrics_list = by_distance[d]
    lers = [m['logical_error_rate'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    
    # Count how many have zero TP (predicting all 0s)
    zero_tp_count = sum(1 for m in metrics_list if m['true_positives'] == 0)
    
    print(f"\nDistance d={d}: ({len(metrics_list)} experiments)")
    print(f"  LER: mean={np.mean(lers):.4f}, std={np.std(lers):.4f}, min={np.min(lers):.4f}, max={np.max(lers):.4f}")
    print(f"  Precision: mean={np.mean(precisions):.4f}")
    print(f"  Recall: mean={np.mean(recalls):.4f}")
    print(f"  Models predicting all zeros: {zero_tp_count}/{len(metrics_list)}")

# 4. Check label imbalance
print("\n### 4. LABEL IMBALANCE ANALYSIS")
print("-" * 60)

for r in results:
    metrics = r['metrics']
    pos_ratio = metrics['positive_samples'] / metrics['total_samples']
    if pos_ratio < 0.05 or pos_ratio > 0.95:
        print(f"Imbalanced: {r['experiment']} - positive ratio: {pos_ratio:.2%}")
