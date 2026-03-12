"""Shared profiling utilities for AlphaQubit training and inference scripts.

Functions
---------
register_hooks
    Attach forward hooks to every leaf module of a model to record
    operator types, input shapes/dtypes, output shapes/dtypes, and increment
    an operator call counter.
print_profiler_summary
    Print a ranked table of the top CPU (and optionally CUDA) operations
    captured by a :class:`torch.profiler.profile` instance.
print_hook_dict_summary
    Print a human-readable summary of the hook_dict and operator_counter
    populated by :func:`register_hooks`.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def register_hooks(
    model: "torch.nn.Module",
    hook_dict: dict,
    operator_counter: "defaultdict[str, int]",
) -> list:
    """Register forward hooks on all leaf modules to capture tensor metadata.

    Each leaf module (a module with no children) gets a hook that records:

    * ``op_type``       – fully-qualified class name of the module
    * ``input_shapes``  – list of shapes (or type strings) for each input tensor
    * ``input_dtypes``  – matching list of dtype strings
    * ``output_shapes`` – same for outputs
    * ``output_dtypes`` – same for outputs

    The results are written into *hook_dict* in-place, keyed by the
    ``named_modules`` name of each leaf.  *operator_counter* is incremented
    by one for the fully-qualified type of each module on every forward call,
    so it reflects how many times each operator type was invoked.

    Parameters
    ----------
    model:
        The model to instrument.
    hook_dict:
        Dictionary populated in-place during the forward pass.
    operator_counter:
        A ``defaultdict(int)`` incremented in-place on every forward call.

    Returns
    -------
    list
        List of hook handles; call ``h.remove()`` on each when done.
    """

    def make_hook(name: str, module_type: str):
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
                'op_type': module_type,
                'input_shapes': input_shapes,
                'input_dtypes': input_dtypes,
                'output_shapes': output_shapes,
                'output_dtypes': output_dtypes,
            }

            module_class = module.__class__.__name__
            # operator_counter[module_class] += 1
            operator_counter[module_type] += 1
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module_type = type(module).__module__ + '.' + type(module).__qualname__
            hook = module.register_forward_hook(make_hook(name, module_type))
            hooks.append(hook)
    return hooks


def print_profiler_summary(profiler, row_limit: int = 20, show_details: bool = True) -> None:
    """Print a ranked table of operations captured by *profiler*, plus an
    operator list summary modelled on ``profile_operators.py``.

    Prints the top *row_limit* operations sorted by CPU time, and — when a
    CUDA device is available — also sorted by CUDA time.  Then appends:

    * Optional per-event detail block (``show_details=True``): input shapes,
      call count, and CPU time for every event with non-zero CPU time.
    * Unique operator list (events that have both CPU and CUDA time > 0),
      sorted alphabetically.
    * Top-10 operators by CPU time with count, CPU time, and CUDA time.
    * Totals: unique operator count and total call count.

    Parameters
    ----------
    profiler:
        A :class:`torch.profiler.profile` instance after its context has
        exited (i.e. ``profiler.__exit__`` has been called).
    row_limit:
        Number of rows to display in the key_averages table (default 20).
    show_details:
        When *True*, print per-event detail block before the unique operator
        list (default *False*).
    """
    # ------------------------------------------------------------------
    # Existing: key_averages tables (CPU and optionally CUDA)
    # ------------------------------------------------------------------
    print(f"\nTop {row_limit} CPU operations:")
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit))
    if torch.cuda.is_available():
        print(f"\nTop {row_limit} CUDA operations:")
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))

    # ------------------------------------------------------------------
    # Supplemental: op list logic from profile_operators.py
    # ------------------------------------------------------------------
    print("=" * 80)
    print("All operator calls (with duplicates):")
    print("=" * 80)

    if show_details:
        cuda_col = torch.cuda.is_available()
        header = f"{'Operator':<50} {'Count':>8} {'CPU(ms)':>12}"
        if cuda_col:
            header += f" {'CUDA(ms)':>12}"
        header += "  Input shapes"
        print(f"\n{header}")
        print("-" * (len(header) + 20))
        for event in profiler.key_averages():
            if event.cpu_time_total > 0:
                cuda_t = getattr(event, 'cuda_time_total', 0) or getattr(event, 'device_time_total', 0)
                shapes_str = str(event.input_shapes) if event.input_shapes else ""
                row = f"{event.key:<50} {event.count:>8} {event.cpu_time_total / 1000:>12.2f}"
                if cuda_col:
                    row += f" {cuda_t / 1000:>12.2f}"
                row += f"  {shapes_str}"
                print(row)

    unique_ops: set[str] = set()
    op_stats: list[dict] = []
    for event in profiler.key_averages():
        # Always require non-zero CPU time; on CUDA-enabled machines also
        # require non-zero CUDA time so purely-CPU bookkeeping ops are excluded.
        if event.cpu_time_total <= 0:
            continue
        cuda_time = getattr(event, 'cuda_time_total', 0) or getattr(event, 'device_time_total', 0)
        if torch.cuda.is_available() and cuda_time <= 0:
            continue
        unique_ops.add(event.key)
        op_stats.append({
            'name': event.key,
            'count': event.count,
            'cpu_time': event.cpu_time_total / 1000,
            'cuda_time': cuda_time / 1000,
        })

    print("\n" + "=" * 80)
    print("Unique operators list:")
    print("=" * 80)
    for op in sorted(unique_ops):
        print(f"{op}")

    print("\n" + "=" * 80)
    print("Top 10 operators by CPU time:")
    print("=" * 80)
    for i, stat in enumerate(sorted(op_stats, key=lambda x: x['cpu_time'], reverse=True)[:10], 1):
        print(f"{i}. {stat['name']}")
        print(f"   Count: {stat['count']}, CPU: {stat['cpu_time']:.2f}ms, CUDA: {stat['cuda_time']:.2f}ms")

    print(f"\nTotal unique operators: {len(unique_ops)}")
    print(f"Total operator calls: {sum(stat['count'] for stat in op_stats)}")


def print_hook_dict_summary(
    hook_dict: dict,
    operator_counter: "defaultdict[str, int] | None" = None,
    show_details: bool = True,
    max_display: int = 20,
) -> None:
    """Print a human-readable summary of hook captures and operator statistics.

    Parameters
    ----------
    hook_dict:
        Dictionary produced by :func:`register_hooks` (keyed by module name).
        Because each key is the unique module path, this has exactly one entry
        per leaf module in the model (the last forward pass overwrites earlier
        ones).  Statistics derived from *hook_dict* therefore represent
        **module counts** — i.e. how many leaf modules of each type exist in
        the model architecture.
    operator_counter:
        Optional ``defaultdict(int)`` produced by :func:`register_hooks`.
        Unlike *hook_dict*, this is incremented on **every** forward call, so
        its values reflect **invocation counts** across all profiled batches
        (e.g. 46 Linear modules × 3 batches = 138 invocations).  When *None*,
        the operator statistics section falls back to counting from *hook_dict*,
        which gives module counts consistent with
        ``analyze_profiling.py``/the saved JSON.
    show_details:
        When *True*, print per-module tensor shape/dtype information for up to
        *max_display* modules.
    max_display:
        Maximum number of module entries to show when *show_details* is *True*.
    """
    # ------------------------------------------------------------------
    # Layer shapes table  (mirrors analyze_layer_shapes format)
    # ------------------------------------------------------------------

    display_count = min(max_display, len(hook_dict)) if show_details else 0

    if display_count > 0:
        print(f"\n{'='*80}")
        print("Layer Shapes (leaf modules):")
        print(f"{'='*80}\n")

        print(f"{'Layer Name':<50} {'Op Type':<35} {'Input Shape':<30} {'Output Shape':<30}")
        print("-" * 145)

        for name, info in list(hook_dict.items())[:display_count]:
            op_type = info.get('op_type', 'N/A')
            op_type_short = op_type.split('.')[-1] if '.' in op_type else op_type

            input_shapes  = info.get('input_shapes',  [])
            output_shapes = info.get('output_shapes', [])
            input_dtypes  = info.get('input_dtypes',  [])
            output_dtypes = info.get('output_dtypes', [])

            input_str  = str(input_shapes[0])  if input_shapes  else "N/A"
            output_str = str(output_shapes[0]) if output_shapes else "N/A"

            print(f"{name:<50} {op_type_short:<35} {input_str:<30} {output_str:<30}")

            if input_dtypes and input_dtypes[0] != str(type(None)):
                dtype_out = output_dtypes[0] if output_dtypes else 'N/A'
                print(f"{'':>50} {'':>35} dtype: {input_dtypes[0]:<23} dtype: {dtype_out}")

        if len(hook_dict) > display_count:
            print(f"... ({len(hook_dict)} total leaf modules)")

    # ------------------------------------------------------------------
    # Unique operator types  (mirrors analyze_unique_operators format)
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Unique Operator Types Statistics:")
    print(f"{'='*80}\n")

    # Build unique short-name list and per-module counts from hook_dict.
    # op_type_count: full-path → number of leaf modules of that type in the
    # model.  This matches what analyze_profiling.py reads from the saved JSON
    # and is always derived from hook_dict regardless of operator_counter.
    op_type_count: dict[str, int] = {}
    unique_ops_short: list[str] = []
    for _info in hook_dict.values():
        op_type = _info.get('op_type', 'N/A')
        op_type_short = op_type.split('.')[-1] if '.' in op_type else op_type
        op_type_count[op_type] = op_type_count.get(op_type, 0) + 1
        if op_type_short not in unique_ops_short:
            unique_ops_short.append(op_type_short)

    unique_ops_short.sort()
    print(f"Total unique operator types: {len(unique_ops_short)}\n")
    print(f"Unique operator list (sorted):")
    for op_type_short in unique_ops_short:
        print(f"{op_type_short}")

    # ------------------------------------------------------------------
    # Full-path operator statistics.
    # When operator_counter is provided, show invocation counts (total calls
    # across all profiled batches).  When None, fall back to op_type_count
    # derived from hook_dict (module counts, consistent with analyze_profiling.py).
    # ------------------------------------------------------------------
    if operator_counter is not None:
        # operator_counter counts every forward() call across batches, so
        # "138 calls" means 46 modules × 3 batches, not 138 distinct modules.
        print(f"\n{'='*80}")
        print("Operator invocation statistics (all batches, sorted by call count):")
        print(f"{'='*80}")
        sorted_ops = sorted(operator_counter.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_ops:
            print(f"{op_type}: {count} calls")
    else:
        # op_type_count is derived from hook_dict, consistent with the saved
        # JSON and analyze_profiling.py: one entry per leaf module.
        print(f"\n{'='*80}")
        print("Operator statistics (sorted by module count):")
        print(f"{'='*80}")
        sorted_ops = sorted(op_type_count.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_ops:
            print(f"{op_type}: {count}")

    print(f"\nTotal leaf modules captured: {len(hook_dict)}")
