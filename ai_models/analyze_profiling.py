#!/usr/bin/env python3
"""
Analyze profiling results from test_train.py and test_eval.py.

Usage:
    python ai_models/analyze_profiling.py --profile_dir ./profiling_logs
"""

import argparse
import json
import os
from pathlib import Path


def analyze_layer_shapes(json_file):
    """Analyze layer shapes from hooks."""
    print(f"\n{'='*80}")
    print(f"Layer Shapes Analysis: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"{'Layer Name':<50} {'Op Type':<35} {'Input Shape':<30} {'Output Shape':<30}")
    print("-" * 145)
    
    for name, info in data.items():
        op_type = info.get('op_type', 'N/A')
        # Simplify op_type display by showing just the class name
        if '.' in op_type:
            op_type_short = op_type.split('.')[-1]
        else:
            op_type_short = op_type
        
        input_shapes = info.get('input_shapes', [])
        output_shapes = info.get('output_shapes', [])
        input_dtypes = info.get('input_dtypes', [])
        output_dtypes = info.get('output_dtypes', [])
        
        input_str = str(input_shapes[0]) if input_shapes else "N/A"
        output_str = str(output_shapes[0]) if output_shapes else "N/A"
        
        print(f"{name:<50} {op_type_short:<35} {input_str:<30} {output_str:<30}")
        
        if len(input_dtypes) > 0 and input_dtypes[0] != str(type(None)):
            print(f"{'':>50} {'':>35} dtype: {input_dtypes[0]:<23} dtype: {output_dtypes[0] if output_dtypes else 'N/A'}")


def analyze_chrome_trace(json_file):
    """Analyze Chrome trace JSON."""
    print(f"\n{'='*80}")
    print(f"Chrome Trace Analysis: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'traceEvents' not in data:
        print("No trace events found in the file.")
        return
    
    events = data['traceEvents']
    
    op_times = {}
    for event in events:
        if event.get('cat') == 'kernel' or event.get('cat') == 'cpu_op':
            name = event.get('name', 'Unknown')
            dur = event.get('dur', 0)
            
            if name not in op_times:
                op_times[name] = {'count': 0, 'total_time': 0, 'avg_time': 0}
            
            op_times[name]['count'] += 1
            op_times[name]['total_time'] += dur
    
    for name, stats in op_times.items():
        stats['avg_time'] = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print(f"{'Operator Name':<50} {'Count':<10} {'Total Time (us)':<20} {'Avg Time (us)':<20}")
    print("-" * 100)
    
    for name, stats in sorted_ops[:30]:
        print(f"{name:<50} {stats['count']:<10} {stats['total_time']:<20.2f} {stats['avg_time']:<20.2f}")


def extract_operator_info(json_file):
    """Extract detailed operator information including tensor shapes."""
    print(f"\n{'='*80}")
    print(f"Detailed Operator Information: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'traceEvents' not in data:
        print("No trace events found.")
        return
    
    events = data['traceEvents']
    
    operators = []
    for event in events:
        if 'args' in event and 'Input Dims' in event.get('args', {}):
            op_info = {
                'name': event.get('name', 'Unknown'),
                'input_dims': event['args'].get('Input Dims', []),
                'input_type': event['args'].get('Input type', []),
                'duration': event.get('dur', 0)
            }
            operators.append(op_info)
    
    if not operators:
        print("No operator shape information found in trace.")
        return
    
    print(f"{'Operator':<40} {'Input Shapes':<50} {'Duration (us)':<15}")
    print("-" * 105)
    
    for op in operators[:50]:
        input_dims_str = str(op['input_dims'])[:48]
        print(f"{op['name']:<40} {input_dims_str:<50} {op['duration']:<15.2f}")
        if op['input_type']:
            print(f"{'':>40} Type: {str(op['input_type'])}")


def main():
    parser = argparse.ArgumentParser(description="Analyze profiling results")
    parser.add_argument("--profile_dir", type=str, default="./profiling_logs",
                       help="Directory containing profiling results")
    parser.add_argument("--layer_shapes", action="store_true",
                       help="Analyze layer shapes only")
    parser.add_argument("--chrome_trace", action="store_true",
                       help="Analyze Chrome trace only")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed operator information")
    
    args = parser.parse_args()
    
    profile_path = Path(args.profile_dir)
    
    if not profile_path.exists():
        print(f"Error: Profile directory not found: {args.profile_dir}")
        print("\nPlease run training/evaluation with --profile flag first:")
        print("  python ai_models/test_train.py --num_samples 1000 --epochs 2 --batch_size 32 --profile")
        print("  python ai_models/test_eval.py --checkpoint alphaqubit_test.pth --num_samples 200 --profile")
        return
    
    layer_shape_files = list(profile_path.glob("layer_shapes_*.json"))
    trace_files = list(profile_path.glob("*_trace*.json"))
    
    if not args.chrome_trace and (not args.layer_shapes or args.layer_shapes):
        for layer_file in sorted(layer_shape_files):
            analyze_layer_shapes(layer_file)
    
    if not args.layer_shapes and (not args.chrome_trace or args.chrome_trace):
        for trace_file in sorted(trace_files):
            analyze_chrome_trace(trace_file)
            
            if args.detailed:
                extract_operator_info(trace_file)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
    
    print("Tips:")
    print("1. View Chrome traces in chrome://tracing")
    print(f"   Load files: {', '.join([f.name for f in trace_files])}")
    print("2. Layer shapes show tensor dimensions at each layer")
    print("3. Use --detailed flag for more operator information")


if __name__ == "__main__":
    main()
