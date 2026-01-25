# Profiling and Operator Analysis Guide

This guide explains how to capture and analyze computational graphs and operator execution information for the AlphaQubitDecoder model.

## Features

The profiling system captures:
- **Operator execution times** (CPU and CUDA)
- **Tensor shapes and dtypes** at each layer
- **Memory usage** information
- **Call stacks** for debugging
- **Chrome trace** for visualization

## Quick Start

### 1. Training with Profiling

```bash
python ai_models/test_train.py \
  --num_samples 1000 \
  --epochs 2 \
  --batch_size 32 \
  --profile \
  --profile_dir ./profiling_logs
```

This will:
- Profile the first epoch only (to avoid excessive overhead)
- Save results to `./profiling_logs/`
- Generate:
  - `train_trace_epoch1.json` - Chrome trace file
  - `layer_shapes_epoch1.json` - Layer input/output shapes

### 2. Evaluation with Profiling

```bash
python ai_models/test_eval.py \
  --checkpoint alphaqubit_test.pth \
  --num_samples 200 \
  --profile \
  --profile_dir ./profiling_logs
```

This will:
- Profile the first 3 batches
- Generate:
  - `eval_trace.json` - Chrome trace file
  - `layer_shapes_eval.json` - Layer shapes

### 3. Analyze Results

```bash
# Analyze all profiling results
python ai_models/analyze_profiling.py --profile_dir ./profiling_logs

# Analyze layer shapes only
python ai_models/analyze_profiling.py --profile_dir ./profiling_logs --layer_shapes

# Analyze Chrome traces only
python ai_models/analyze_profiling.py --profile_dir ./profiling_logs --chrome_trace

# Show detailed operator information
python ai_models/analyze_profiling.py --profile_dir ./profiling_logs --detailed
```

## Output Files

### 1. Chrome Trace Files (`*_trace*.json`)

These can be visualized in Chrome browser:
1. Open Chrome and navigate to `chrome://tracing`
2. Click "Load" and select the JSON file
3. Explore the timeline visualization

**What you can see:**
- Operator execution timeline
- GPU kernel launches
- CPU operations
- Memory allocation events

### 2. Layer Shapes Files (`layer_shapes_*.json`)

JSON format showing tensor shapes at each layer:

```json
{
  "embedder.feature_projs.0": {
    "input_shapes": [[32, 48, 1]],
    "input_dtypes": ["torch.float32"],
    "output_shapes": [[32, 48, 256]],
    "output_dtypes": ["torch.float32"]
  },
  ...
}
```

## Console Output

When profiling is enabled, you'll see tables like:

```
Top 20 CPU operations:
---------------------------------  ------------  ------------  ------------
Name                               Self CPU %    Self CPU      Total CPU
---------------------------------  ------------  ------------  ------------
aten::addmm                        45.23%        123.45ms      145.67ms
aten::bmm                          23.12%        63.21ms       78.90ms
aten::copy_                        12.34%        33.75ms       33.75ms
...
```

## Example Workflow

```bash
# Step 1: Train with profiling
python ai_models/test_train.py \
  --num_samples 1000 \
  --epochs 2 \
  --batch_size 32 \
  --profile

# Step 2: Evaluate with profiling
python ai_models/test_eval.py \
  --checkpoint alphaqubit_test.pth \
  --num_samples 200 \
  --profile

# Step 3: Analyze results
python ai_models/analyze_profiling.py --detailed

# Step 4: View in Chrome
# Open chrome://tracing and load profiling_logs/train_trace_epoch1.json
```

## Performance Impact

Profiling adds overhead:
- **Training**: Only first epoch is profiled
- **Evaluation**: Only first 3 batches are profiled
- **Memory**: Additional ~100-500MB for trace data
- **Time**: 10-30% slower when profiling is active

## Interpreting Results

### Layer Shapes Analysis

Shows data flow through the model:
```
Layer Name                         Input Shape              Output Shape
--------------------------------------------------------------------------------
embedder.feature_projs.0          (32, 48, 1)              (32, 48, 256)
transformer.layers.0              (32, 48, 256)            (32, 48, 256)
readout.data_conv                 (32, 256, 6, 6)          (32, 256, 7, 7)
```

### Operator Analysis

Key operators to look for:
- `aten::addmm` - Matrix multiplication (Linear layers)
- `aten::bmm` - Batch matrix multiplication (Attention)
- `aten::softmax` - Softmax operations
- `aten::layer_norm` - Layer normalization
- `aten::conv2d` - 2D convolutions

### Memory Usage

In Chrome trace:
- Look for `[memory]` events
- Check peak memory usage
- Identify memory allocation hotspots

## Troubleshooting

### Issue: "No profiling data found"

Solution: Ensure you ran with `--profile` flag

### Issue: "Chrome trace file too large"

Solution: Reduce batch size or number of samples

### Issue: "CUDA profiling not working"

Solution: Ensure CUDA is available and enabled in PyTorch

## Advanced Usage

### Export Computational Graph

To capture the model graph with torch.fx:

```python
import torch.fx as fx

# After creating the model
symbolic_traced = fx.symbolic_trace(model)
print(symbolic_traced.graph)

# Save graph
with open('model_graph.txt', 'w') as f:
    f.write(str(symbolic_traced.graph))
```

### Custom Profiling Scope

Modify the code to profile specific sections:

```python
from torch.profiler import record_function

with record_function("custom_section"):
    output = model(input)
```

## Command Reference

### Training Options
```
--profile              Enable profiling
--profile_dir DIR      Output directory for profiling data (default: ./profiling_logs)
```

### Evaluation Options
```
--profile              Enable profiling
--profile_dir DIR      Output directory for profiling data (default: ./profiling_logs)
```

### Analysis Options
```
--profile_dir DIR      Input directory with profiling data
--layer_shapes         Analyze only layer shapes
--chrome_trace         Analyze only Chrome traces
--detailed             Show detailed operator information
```

## Additional Resources

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [Chrome Tracing Viewer](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
