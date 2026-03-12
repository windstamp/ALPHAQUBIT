"""Shared profiling utilities for AlphaQubit training and inference scripts.

Functions
---------
register_hooks
    Attach forward hooks to every leaf module of a model to record
    operator types, input shapes/dtypes, and output shapes/dtypes.
print_profiler_summary
    Print a ranked table of the top CPU (and optionally CUDA) operations
    captured by a :class:`torch.profiler.profile` instance.
"""

from __future__ import annotations

import torch


def register_hooks(model: "torch.nn.Module", hook_dict: dict) -> list:
    """Register forward hooks on all leaf modules to capture tensor metadata.

    Each leaf module (a module with no children) gets a hook that records:

    * ``op_type``     – fully-qualified class name of the module
    * ``input_shapes``  – list of shapes (or type strings) for each input tensor
    * ``input_dtypes``  – matching list of dtype strings
    * ``output_shapes`` – same for outputs
    * ``output_dtypes`` – same for outputs

    The results are written into *hook_dict* in-place, keyed by the
    ``named_modules`` name of each leaf.

    Parameters
    ----------
    model:
        The model to instrument.
    hook_dict:
        Dictionary populated in-place during the forward pass.

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
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module_type = type(module).__module__ + '.' + type(module).__qualname__
            hook = module.register_forward_hook(make_hook(name, module_type))
            hooks.append(hook)
    return hooks


def print_profiler_summary(profiler, row_limit: int = 20) -> None:
    """Print a ranked table of operations captured by *profiler*.

    Prints the top *row_limit* operations sorted by CPU time, and — when a
    CUDA device is available — also sorted by CUDA time.

    Parameters
    ----------
    profiler:
        A :class:`torch.profiler.profile` instance after its context has
        exited (i.e. ``profiler.__exit__`` has been called).
    row_limit:
        Number of rows to display per table (default 20).
    """
    print(f"\nTop {row_limit} CPU operations:")
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit))
    if torch.cuda.is_available():
        print(f"\nTop {row_limit} CUDA operations:")
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
