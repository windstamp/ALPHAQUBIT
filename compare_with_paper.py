import json

# Load results
with open('s3_results/20251230_220521/test_results_v2/test_summary.json') as f:
    d = json.load(f)

# Paper reference values
paper_ref = {
    'surface_code_bX_d3_r01_center_5_7': 0.025,
    'surface_code_bX_d3_r01_center_3_5': 0.028,
    'surface_code_bX_d5_r01_center_5_5': 0.015,
    'surface_code_bX_d5_r25_center_5_5': 0.085,
}

print("=" * 70)
print("COMPARISON: Our Results vs Paper Reference")
print("=" * 70)

# Check key experiments
for r in d['results']:
    exp = r['experiment']
    ler = r['metrics']['logical_error_rate']
    
    # Check if this is a paper-comparable experiment
    for paper_exp, paper_ler in paper_ref.items():
        if paper_exp in exp or exp in paper_exp:
            gap = (ler - paper_ler) / paper_ler * 100
            print(f"\n{exp}")
            print(f"  Our LER:   {ler:.4f} ({ler*100:.2f}%)")
            print(f"  Paper LER: {paper_ler:.4f} ({paper_ler*100:.2f}%)")
            print(f"  Gap:       {gap:+.1f}%")

# Also show best low-noise results
print("\n" + "=" * 70)
print("ALL LOW-NOISE (r01) RESULTS:")
print("=" * 70)
r01_results = [r for r in d['results'] if '_r01_' in r['experiment']]
for r in sorted(r01_results, key=lambda x: x['metrics']['logical_error_rate']):
    exp = r['experiment']
    ler = r['metrics']['logical_error_rate']
    print(f"  {exp}: {ler:.4f} ({ler*100:.2f}%)")

# High noise results
print("\n" + "=" * 70)
print("HIGH-NOISE (r25) RESULTS:")
print("=" * 70)
r25_results = [r for r in d['results'] if '_r25_' in r['experiment']]
for r in sorted(r25_results, key=lambda x: x['metrics']['logical_error_rate']):
    exp = r['experiment']
    ler = r['metrics']['logical_error_rate']
    # Paper says d5_r25 should be ~8.5%
    print(f"  {exp}: {ler:.4f} ({ler*100:.2f}%)")
