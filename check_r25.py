import json
d=json.load(open('s3_results/20251230_220521/test_results_v2/test_summary.json'))
r25=[r for r in d['results'] if '_r25_' in r['experiment']]
print("High-noise (r25) experiments:")
for r in r25[:5]:
    m = r['metrics']
    print(f"{r['experiment']}:")
    print(f"  TP={m['true_positives']}, FP={m['false_positives']}, TN={m['true_negatives']}, FN={m['false_negatives']}")
    print(f"  Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")
    print(f"  Positive ratio: {m['positive_samples']/m['total_samples']:.2%}")
