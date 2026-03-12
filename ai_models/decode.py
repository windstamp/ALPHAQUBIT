#!/usr/bin/env python3
"""Decode detection-event syndromes with a trained AlphaQubit model.

This utility mirrors the command advertised in the project README.  It loads
``*.npy``/``*.npz`` bundles containing detection events, feeds them through a
trained :class:`~ai_models.model_mla.AlphaQubitDecoder` checkpoint, and writes
summary metrics to ``results/``.

The loader is intentionally forgiving:

* ``.npz`` files are expected to contain a ``data`` array of shape ``(N, …)``
  plus optional label arrays such as ``obs`` or ``label``.
* ``.npy`` files may store raw syndrome tensors or a pickled dictionary with
  ``data``/``syndromes`` and optional ``logical`` keys.

Only a *single* measurement basis is supported per file.  If it cannot be
inferred from the filename or payload, provide ``--basis x`` or ``--basis z``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from ai_models.profiling_utils import register_hooks, print_profiler_summary, print_hook_dict_summary

# ---------------------------------------------------------------------
#  NPU Support - Import torch_npu if available
# ---------------------------------------------------------------------
try:
    import torch_npu
    NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, 'npu') else False
except ImportError:
    NPU_AVAILABLE = False


def get_device(device_str: str = 'auto') -> torch.device:
    """Get the best available device."""
    if device_str == 'auto':
        if NPU_AVAILABLE:
            return torch.device('npu:0')
        elif torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    elif device_str == 'npu':
        if not NPU_AVAILABLE:
            print("Warning: NPU requested but not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device('npu:0')
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device('cuda:0')
    else:
        return torch.device(device_str)


# Import both model implementations - standard transformer is default
from ai_models.model import AlphaQubitDecoder as AlphaQubitDecoderTransformer
from ai_models.model_mla import AlphaQubitDecoder as AlphaQubitDecoderMLA


LabelArray = Optional[np.ndarray]
BasisArray = Optional[np.ndarray]


@dataclass
class LoadedSyndromes:
    """Container for data parsed from ``--data``."""

    syndromes: np.ndarray
    labels: LabelArray
    basis: BasisArray


class InferenceDataset(Dataset):
    """Minimal dataset yielding (inputs, basis, mask) tuples for decoding."""

    def __init__(
        self,
        inputs: torch.Tensor,
        basis: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if basis.ndim != 1:
            raise ValueError("Basis tensor must be 1D (one label per sample)")
        if inputs.shape[0] != basis.shape[0]:
            raise ValueError("Mismatch between number of samples and basis vector")

        self.inputs = inputs
        self.basis = basis
        self.final_mask = final_mask.to(torch.int8)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.basis[idx], self.final_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved AlphaQubit checkpoint (.pth)")
    parser.add_argument("--data", type=Path, required=True, help="Syndrome data (.npz/.npy) to decode")
    parser.add_argument("--basis", type=str, default=None, help="Override measurement basis: 'x' or 'z'")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for decoding")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run on (default: auto)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width of the decoder (must match training)")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads (must match training)")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers (must match training)")
    parser.add_argument("--output", type=Path, default=None, help="Where to store aggregated metrics (JSON)")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional path to save per-shot probabilities (.npy)")
    parser.add_argument(
        "--cudaq",
        action="store_true",
        help="Enable CUDA-Q execution so inference can be dispatched to a quantum backend via NVQLink",
    )
    parser.add_argument(
        "--cudaq-target",
        type=str,
        default=None,
        help="Optional CUDA-Q target name (defaults to NVQLink-enabled target when --cudaq is set)",
    )
    parser.add_argument(
        "--mla",
        action="store_true",
        help="Use MLA (Multi-head Latent Attention) model instead of standard transformer",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional path to a separate file containing ground truth labels (.npy/.npz) for accuracy evaluation",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling to capture operator execution details",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="./profiling_logs",
        help="Directory to save profiling results",
    )
    return parser.parse_args()


def load_syndrome_file(path: Path) -> LoadedSyndromes:
    """Load a ``.npy``/``.npz`` bundle and return its contents."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "data" not in data:
                raise KeyError(f"Expected 'data' array in {path}")
            syndromes = np.asarray(data["data"])
            labels = _pick_first_existing(data, ["obs", "label", "labels", "logical", "logical_error"])
            basis = _pick_first_existing(data, ["basis", "basis_id", "bases", "basis_ids"])
            return LoadedSyndromes(syndromes, labels, basis)

    array = np.load(path, allow_pickle=True)
    if isinstance(array, np.ndarray) and array.dtype == object:
        # Could be a pickled dictionary wrapped in an object array.
        if array.shape == ():
            array = array.item()
        elif array.ndim == 1 and array.size == 1:
            array = array[0]

    if isinstance(array, dict):
        syndromes = np.asarray(array.get("data") or array.get("syndromes") or array.get("detections"))
        if syndromes is None:
            raise KeyError(f"Could not locate syndrome array inside {path}")
        labels = array.get("obs") or array.get("label") or array.get("labels") or array.get("logical")
        basis = array.get("basis") or array.get("basis_id")
        return LoadedSyndromes(np.asarray(syndromes), _as_optional_array(labels), _as_optional_array(basis))

    return LoadedSyndromes(np.asarray(array), None, None)


def _pick_first_existing(store: Iterable[str], keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in store:
            return np.asarray(store[key])
    return None


def _as_optional_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr


def infer_basis(basis_arg: Optional[str], loaded_basis: BasisArray, data_path: Path, num_samples: int) -> torch.Tensor:
    """Resolve the measurement basis for each sample."""

    if basis_arg is not None:
        basis_id = _basis_str_to_id(basis_arg)
        return torch.full((num_samples,), basis_id, dtype=torch.int8)

    if loaded_basis is not None:
        arr = np.asarray(loaded_basis)
        if arr.ndim == 0:
            basis_id = _validate_basis_id(int(arr))
            return torch.full((num_samples,), basis_id, dtype=torch.int8)
        if arr.ndim == 1 and arr.shape[0] == num_samples:
            vec = np.vectorize(_validate_basis_id, otypes=[np.int8])(arr)
            return torch.from_numpy(vec.astype(np.int8))
        raise ValueError(
            "Loaded basis metadata must be scalar or length-N array of {0,1}. "
            f"Received shape {arr.shape}"
        )

    inferred = infer_basis_from_name(data_path)
    if inferred is not None:
        return torch.full((num_samples,), inferred, dtype=torch.int8)

    raise ValueError(
        "Unable to determine measurement basis. Pass --basis (x/z) explicitly."
    )


def _basis_str_to_id(text: str) -> int:
    lowered = text.strip().lower()
    if lowered in {"x", "0", "bx"}:
        return 0
    if lowered in {"z", "1", "bz"}:
        return 1
    raise ValueError(f"Unsupported basis specifier '{text}'. Use 'x' or 'z'.")


def _validate_basis_id(value: int) -> int:
    if value in (0, 1):
        return int(value)
    raise ValueError(f"Basis IDs must be 0 (X) or 1 (Z); received {value}")


def infer_basis_from_name(path: Path) -> Optional[int]:
    name = path.name.lower()
    if any(tag in name for tag in ("_bx_", "-bx-", "_basisx", "_x_basis")):
        return 0
    if any(tag in name for tag in ("_bz_", "-bz-", "_basisz", "_z_basis")):
        return 1

    tokens = [token for token in re.split(r"[^a-z0-9]+", name) if token]
    for token in tokens:
        if token in {"bx", "x", "basisx", "xbasis"}:
            return 0
        if token in {"bz", "z", "basisz", "zbasis"}:
            return 1
    return None


def prepare_inputs(syndromes: np.ndarray, basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Convert raw syndromes to tensors expected by the decoder."""

    x = np.asarray(syndromes, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :, None]
    elif x.ndim == 3:
        x = x[:, :, :, None]
    elif x.ndim != 4:
        raise ValueError(
            "Syndrome array must have 2, 3 or 4 dimensions; "
            f"received shape {x.shape}"
        )

    N, R, S, F = x.shape
    basis_feat = basis.float().numpy().reshape(N, 1, 1, 1)
    basis_feat = np.broadcast_to(basis_feat, (N, R, S, 1))
    x = np.concatenate([x, basis_feat.astype(np.float32)], axis=-1)

    d = math.isqrt(S + 1)
    if d * d != S + 1:
        d += 1
        pad = d * d - 1 - S
        x = np.concatenate([x, np.zeros((N, R, pad, x.shape[-1]), dtype=np.float32)], axis=2)
        S = x.shape[2]

    grid_size = d - 1
    final_mask = torch.tensor(
        [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
        dtype=torch.int8,
    )
    if final_mask.numel() != S:
        raise RuntimeError(
            "Constructed final_mask has incorrect size: "
            f"expected {S}, got {final_mask.numel()}"
        )

    return torch.from_numpy(x), final_mask, grid_size


def select_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    try:
        torch.empty(1, device=device)
    except (RuntimeError, AssertionError):
        raise RuntimeError(f"Failed to initialise torch device '{requested}'")
    return device


def load_model(
    model_path: Path,
    num_features: int,
    num_stabilizers: int,
    grid_size: int,
    hidden_dim: int,
    heads: int,
    layers: int,
    device: torch.device,
    use_mla: bool = False,
) -> torch.nn.Module:
    def _unwrap_state(obj):
        if isinstance(obj, torch.nn.Module):
            return obj.state_dict()
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state", "model", "module"):
                if key in obj:
                    unwrapped = _unwrap_state(obj[key])
                    if isinstance(unwrapped, dict):
                        return unwrapped
            return obj
        return obj

    def _strip_module_prefix(state_dict: dict) -> dict:
        if not isinstance(state_dict, dict):
            return state_dict
        if not state_dict:
            return state_dict

        prefix = "module."
        if any(key.startswith(prefix) for key in state_dict):
            factory = state_dict.__class__
            stripped = factory()
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    stripped[key[len(prefix):]] = value
                else:
                    stripped[key] = value
            return stripped
        return state_dict

    def _infer_model_params_from_checkpoint(state_dict: dict):
        """Infer model architecture parameters from checkpoint weights."""
        params = {}
        
        # Infer num_stabilizers from index_embedding
        if "embedder.index_embedding.weight" in state_dict:
            params["num_stabilizers"] = state_dict["embedder.index_embedding.weight"].shape[0]
        
        # Infer num_features from feature_projs
        feature_keys = [k for k in state_dict.keys() if k.startswith("embedder.feature_projs.")]
        if feature_keys:
            # Find the maximum index
            indices = []
            for key in feature_keys:
                match = re.match(r"embedder\.feature_projs\.(\d+)\.", key)
                if match:
                    indices.append(int(match.group(1)))
            if indices:
                params["num_features"] = max(indices) + 1
        
        # Infer grid_size from dx_emb/dy_emb/manh_emb
        if "transformer.layers.0.dx_emb.weight" in state_dict:
            dx_size = state_dict["transformer.layers.0.dx_emb.weight"].shape[0]
            # dx_emb has size 2*grid_size + 1
            params["grid_size"] = (dx_size - 1) // 2
        
        # Infer hidden_dim from embedder norm
        if "embedder.norm.weight" in state_dict:
            params["hidden_dim"] = state_dict["embedder.norm.weight"].shape[0]
        
        # Infer num_heads from attention bias projection
        if "transformer.layers.0.bias_proj.weight" in state_dict:
            params["num_heads"] = state_dict["transformer.layers.0.bias_proj.weight"].shape[0]
        
        # Infer num_layers by counting transformer layers
        layer_indices = []
        for key in state_dict.keys():
            match = re.match(r"transformer\.layers\.(\d+)\.", key)
            if match:
                layer_indices.append(int(match.group(1)))
        if layer_indices:
            params["num_layers"] = max(layer_indices) + 1
        
        return params

    state = torch.load(model_path, map_location=device)
    state = _unwrap_state(state)
    state = _strip_module_prefix(state)
    
    # Infer model parameters from checkpoint
    checkpoint_params = _infer_model_params_from_checkpoint(state)
    # checkpoint_params = None
    
    # Override parameters with checkpoint values if they differ
    if checkpoint_params:
        print(f"Checkpoint parameters detected:")
        for key, value in checkpoint_params.items():
            print(f"  {key}: {value}")
        
        # Use checkpoint parameters, overriding provided values
        num_features = checkpoint_params.get("num_features", num_features)
        num_stabilizers = checkpoint_params.get("num_stabilizers", num_stabilizers)
        grid_size = checkpoint_params.get("grid_size", grid_size)
        hidden_dim = checkpoint_params.get("hidden_dim", hidden_dim)
        heads = checkpoint_params.get("num_heads", heads)
        layers = checkpoint_params.get("num_layers", layers)
        
        print(f"\nUsing model configuration:")
        print(f"  num_features: {num_features}")
        print(f"  num_stabilizers: {num_stabilizers}")
        print(f"  grid_size: {grid_size}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  num_heads: {heads}")
        print(f"  num_layers: {layers}")

    # Select model architecture based on use_mla flag
    if use_mla:
        model = AlphaQubitDecoderMLA(
            num_features,
            hidden_dim,
            num_stabilizers,
            grid_size,
            num_heads=heads,
            num_layers=layers,
        )
    else:
        model = AlphaQubitDecoderTransformer(
            num_features,
            hidden_dim,
            num_stabilizers,
            grid_size,
            num_heads=heads,
            num_layers=layers,
        )

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        patched = _maybe_patch_feature_projections(state, num_features)
        if patched is None:
            raise RuntimeError(
                "Failed to load checkpoint and no compatible feature projection "
                "upgrade path was found."
            ) from exc
        model.load_state_dict(patched)

    model.to(device)
    model.eval()
    return model


def configure_cudaq(target: Optional[str]) -> str:
    """Initialise CUDA-Q with the requested target and NVQLink support."""

    try:
        import cudaq
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "CUDA-Q support requested via --cudaq, but the 'cudaq' package is not installed."
        ) from exc

    resolved_target = target or "nvq-link"
    try:
        cudaq.set_target(resolved_target)
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise RuntimeError(f"Failed to configure CUDA-Q target '{resolved_target}'") from exc

    if not _enable_nvqlink(cudaq):
        warnings.warn(
            "CUDA-Q runtime does not expose an NVQLink enablement hook; proceeding without explicit activation.",
            RuntimeWarning,
        )

    return resolved_target


def _enable_nvqlink(cudaq_module) -> bool:
    """Attempt to enable NVQLink within a CUDA-Q runtime module."""

    # Known entry points may vary between CUDA-Q versions; try a few options.
    candidates = (
        getattr(cudaq_module, "enable_nvqlink", None),
        getattr(cudaq_module, "enableNVQLink", None),
    )
    for fn in candidates:
        if callable(fn):
            fn()
            return True

    runtime = getattr(cudaq_module, "runtime", None)
    if runtime is not None:
        for name in ("enable_nvqlink", "enableNVQLink"):
            fn = getattr(runtime, name, None)
            if callable(fn):
                fn()
                return True

    # As a last resort, set a well-named environment toggle in case the runtime
    # inspects it on start-up.
    os.environ.setdefault("CUDAQ_ENABLE_NVQLINK", "1")
    return False


def _maybe_patch_feature_projections(state: dict, expected_features: int) -> Optional[dict]:
    """Back-fill missing ``embedder.feature_projs`` weights when possible."""

    feature_keys = []
    for key in state:
        if not key.startswith("embedder.feature_projs."):
            continue
        parts = key.split(".")
        if len(parts) < 4 or parts[3] not in {"weight", "bias"}:
            continue
        feature_keys.append(key)
    if not feature_keys:
        return None

    present_indices = sorted({int(key.split(".")[2]) for key in feature_keys})
    if present_indices == list(range(expected_features)):
        return None

    if present_indices == list(range(expected_features - 1)):
        template_idx = present_indices[-1]
        missing_idx = expected_features - 1

        patched_state = state.__class__(state)
        for suffix in ("weight", "bias"):
            template_key = f"embedder.feature_projs.{template_idx}.{suffix}"
            missing_key = f"embedder.feature_projs.{missing_idx}.{suffix}"
            if template_key not in state:
                return None
            patched_state[missing_key] = state[template_key].clone()
        return patched_state

    return None


def compute_metrics(probabilities: torch.Tensor, labels: Optional[torch.Tensor]) -> dict:
    probs = probabilities.detach().cpu()
    metrics = {
        "shots": int(probs.numel()),
        "mean_logical_probability": float(probs.mean().item()),
        "predicted_logical_error_rate": float((probs >= 0.5).float().mean().item()),
    }

    if labels is not None:
        labels = labels.detach().cpu().float()
        metrics["observed_logical_error_rate"] = float(labels.mean().item())
        preds = (probs >= 0.5).float()
        metrics["decoder_accuracy"] = float((preds == labels).float().mean().item())
        eps = 1e-7
        bce = -(labels * torch.log(probs + eps) + (1 - labels) * torch.log(1 - probs + eps)).mean()
        metrics["binary_cross_entropy"] = float(bce.item())

    return metrics


def main() -> None:
    args = parse_args()

    loaded = load_syndrome_file(args.data)
    num_samples = int(loaded.syndromes.shape[0])
    basis_vector = infer_basis(args.basis, loaded.basis, args.data, num_samples)
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'loaded.syndromes.shape: {loaded.syndromes.shape}')
    print(f'basis_vector.shape: {basis_vector.shape}')
    inputs, final_mask, grid_size = prepare_inputs(loaded.syndromes, basis_vector)
    print(f'inputs.shape: {inputs.shape}')
    print(f'final_mask.shape: {final_mask.shape}')
    print(f'grid_size: {grid_size}')

    cudaq_target = None
    if args.cudaq:
        cudaq_target = configure_cudaq(args.cudaq_target)
        if args.device is None:
            args.device = "cuda"

    device = select_device(args.device)

    dataset = InferenceDataset(inputs, basis_vector, final_mask)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=(device.type == "cuda"))

    sample_input = inputs[0]
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'sample_input.shape: {sample_input.shape}')
    R, S, F = sample_input.shape

    # Select model architecture based on --mla flag
    if args.mla:
        print("Using MLA (Multi-head Latent Attention) model architecture")
    else:
        print("Using standard transformer model architecture")

    model = load_model(
        args.model,
        num_features=F,
        num_stabilizers=S,
        grid_size=grid_size,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        device=device,
        use_mla=args.mla,
    )

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(model)

    # ------------------------------------------------------------------
    # Inference loop (with optional profiling)
    # ------------------------------------------------------------------
    hook_dict: dict = {}
    operator_counter: defaultdict = defaultdict(int)
    hooks = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        print(f"\nProfiling enabled – capturing first 3 inference batches (output to {args.profile_dir})")
        hooks = register_hooks(model, hook_dict, operator_counter)

    profiler = None
    if args.profile:
        activities = (
            [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [ProfilerActivity.CPU]
        )
        profiler = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.__enter__()

    probs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (xb, basis, mask) in enumerate(loader):
            import sys
            print(f"{__file__}:{sys._getframe().f_lineno}")
            print(f'xb.shape: {xb.shape}')
            print(f'basis.shape: {basis.shape}')
            print(f'mask.shape: {mask.shape}')
            xb = xb.to(device)
            basis = basis.to(device)
            mask = mask.to(device)
            with record_function("model_inference"):
                logits = model(xb, basis, mask)
            import sys
            print(f"{__file__}:{sys._getframe().f_lineno}")
            print(f'logits.shape: {logits.shape}')
            probs.append(torch.sigmoid(logits).cpu())

            if args.profile and batch_idx >= 2:
                break

    if args.profile and profiler is not None:
        profiler.__exit__(None, None, None)

        print_hook_dict_summary(hook_dict, operator_counter)

        print_profiler_summary(profiler)

        if hook_dict:
            hook_file = os.path.join(args.profile_dir, "layer_shapes_decode.json")
            with open(hook_file, 'w') as f:
                json.dump(hook_dict, f, indent=2)
            print(f"Layer shapes saved to: {hook_file}\n")

        trace_file = os.path.join(args.profile_dir, "decode_trace.json")
        profiler.export_chrome_trace(trace_file)
        print(f"\nProfiler trace saved to: {trace_file}")

    if hooks:
        for h in hooks:
            h.remove()

    probabilities = torch.cat(probs)
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'probabilities.shape: {probabilities.shape}')

    labels_tensor = None
    # First try to load labels from the data file itself
    if loaded.labels is not None:
        labels = np.asarray(loaded.labels)
        if labels.ndim > 1:
            labels = labels[:, 0]
        if labels.shape[0] != num_samples:
            raise ValueError(
                "Label array shape mismatch: "
                f"expected length {num_samples}, got {labels.shape}"
            )
        labels_tensor = torch.from_numpy(labels.astype(np.float32))
    # If not found in data file, check for external labels file
    elif args.labels is not None:
        print(f"Loading external labels from {args.labels}")
        external_labels = load_syndrome_file(args.labels)
        if external_labels.syndromes is not None:
            labels = np.asarray(external_labels.syndromes)
            if labels.ndim > 1:
                labels = labels[:, 0] if labels.shape[1] == 1 else labels.flatten()
            if labels.shape[0] != num_samples:
                raise ValueError(
                    f"External label array size mismatch: "
                    f"expected {num_samples}, got {labels.shape[0]}"
                )
            labels_tensor = torch.from_numpy(labels.astype(np.float32))
        elif external_labels.labels is not None:
            labels = np.asarray(external_labels.labels)
            if labels.ndim > 1:
                labels = labels[:, 0]
            if labels.shape[0] != num_samples:
                raise ValueError(
                    f"External label array size mismatch: "
                    f"expected {num_samples}, got {labels.shape[0]}"
                )
            labels_tensor = torch.from_numpy(labels.astype(np.float32))

    metrics = compute_metrics(probabilities, labels_tensor)

    output_path = args.output
    if output_path is None:
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.data.stem}_metrics.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "basis": "X" if basis_vector[0].item() == 0 else "Z",
        "rounds": int(R),
        "stabilizers": int(S),
        "features": int(F),
        **({"cudaq_target": cudaq_target} if cudaq_target is not None else {}),
        **metrics,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Decoded {num_samples} shots. Metrics written to {output_path}.")
    if labels_tensor is None:
        print("(No labels provided – reported rates are model predictions only.)")

    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.predictions, probabilities.numpy())
        print(f"Saved per-shot probabilities to {args.predictions}")


if __name__ == "__main__":
    main()

