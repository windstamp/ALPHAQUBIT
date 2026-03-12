"""Unified training entry point for AlphaQubit models.

This module bridges the gap between the configuration-driven workflow
advertised in the README and the lower level training utilities already
available in :mod:`ai_models.model_mla`.  Given a YAML configuration it
optionally synthesises a dataset with the appropriate noise model and
trains the MLA decoder on it.

Example
-------
    python ai_models/train.py --config configs/dem.yaml

The configuration may also provide a ``training`` section to override
hyperparameters.  Command line flags take precedence, followed by values
in the YAML file, with built-in defaults used as the final fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import psutil  # For memory checking
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

# Try to import torch_npu for Huawei Ascend NPU support
try:
    import torch_npu
    HAS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
except ImportError:
    HAS_NPU = False

from simulator.dem_generator import generate_dem_data
from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator

# Import both model implementations - standard transformer is default
from ai_models.model import AlphaQubitDecoder as AlphaQubitDecoderTransformer
from ai_models.model_mla import AlphaQubitDecoder as AlphaQubitDecoderMLA, train as train_mla
from ai_models.profiling_utils import register_hooks, print_profiler_summary


MODEL_TYPES = {"dem", "si1000", "pauli_plus", "paper_aligned"}


class GeneratedSyndromeDataset(Dataset):
    """Dataset backed by in-memory syndrome arrays.

    Parameters
    ----------
    syndromes:
        Array of shape ``(N, S)`` or ``(N, R, S)`` or ``(N, R, S, F)``
        containing detection event measurements.
    logicals:
        Array of shape ``(N,)`` or ``(N, K)`` with logical observables.
    basis_id:
        Integer identifier for the measurement basis (``0`` for X,
        ``1`` for Z, ``-1`` for unknown/mixed).
    """

    def __init__(self, syndromes: np.ndarray, logicals: np.ndarray, basis_id: int) -> None:
        import sys
        print(f"{__file__}:{sys._getframe().f_lineno}")
        print(f'syndromes.shape: {syndromes.shape}')
        print(f'logicals.shape: {logicals.shape}')
        print(f'basis_id: {basis_id}')

        super().__init__()
        x = np.asarray(syndromes, dtype=np.float32)
        if x.ndim == 2:
            # (N, S) -> (N, R=1, S, F=1)
            x = x[:, None, :, None]
        elif x.ndim == 3:
            # (N, R, S) -> (N, R, S, F=1)
            x = x[:, :, :, None]
        elif x.ndim != 4:
            raise ValueError(
                "Syndrome array must have 2, 3 or 4 dimensions; "
                f"received shape {x.shape}"
            )

        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'x.shape: {x.shape}')

        N, R, S, F = x.shape
        # Embed the basis as an additional feature channel
        basis_feat = np.full((N, R, S, 1), float(basis_id), dtype=np.float32)
        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'basis_feat.shape: {basis_feat.shape}')
        x = np.concatenate([x, basis_feat], axis=-1)
        F = x.shape[-1]

        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'x.shape: {x.shape}')

        # Ensure that S+1 is a perfect square to respect the layout expected by
        # :class:`AlphaQubitDecoder`.  Pad dummy stabilisers if necessary.
        d = math.isqrt(S + 1)
        if d * d != S + 1:
            d += 1
            pad = d * d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=np.float32)], axis=2)
            S = x.shape[2]
        else:
            pad = 0

        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'x.shape: {x.shape}')

        # Final round mask: checkerboard pattern excluding the dummy stabiliser.
        final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
            dtype=torch.int8,
        )
        if final_mask.numel() != S:
            raise RuntimeError(
                "Final mask construction mismatch: expected size "
                f"{S}, obtained {final_mask.numel()}"
            )

        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'final_mask.shape: {final_mask.shape}')

        y = np.asarray(logicals, dtype=np.float32)
        if y.ndim > 1:
            y = y[:, 0]

        # import sys
        # print(f"{__file__}:{sys._getframe().f_lineno}")
        # print(f'y.shape: {y.shape}')

        self.inputs = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)
        self.final_mask = final_mask
        self.basis_tensor = torch.tensor(int(basis_id), dtype=torch.int8)
        self.grid_size = d - 1
        self.pad = pad

        import sys
        print(f"{__file__}:{sys._getframe().f_lineno}")
        print(f'self.inputs.shape: {self.inputs.shape}')
        print(f'self.labels.shape: {self.labels.shape}')
        print(f'self.final_mask.shape: {self.final_mask.shape}')
        print(f'self.basis_tensor.shape: {self.basis_tensor.shape}')
        print(f'self.grid_size: {self.grid_size}')
        print(f'self.pad: {self.pad}')

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        xb = self.inputs[idx]
        return (xb, self.basis_tensor, self.final_mask), self.labels[idx]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def infer_model_type(path: Path, config: Dict[str, Any]) -> str:
    model_type = str(config.get("model", "")).strip().lower()
    if model_type:
        return model_type
    return path.stem.lower()


def resolve_basis(args_basis: str | None, config: Dict[str, Any], model_type: str) -> Tuple[str, int]:
    basis = (args_basis or config.get("basis") or config.get("default_basis") or "z").lower()
    if basis not in {"x", "z"}:
        raise ValueError(f"Unsupported basis '{basis}'. Expected 'x' or 'z'.")
    basis_id = 0 if basis == "x" else 1
    if model_type == "paper_aligned" and basis not in {"x", "z"}:
        raise ValueError("Paper-aligned mode requires basis 'x' or 'z'.")
    return basis.upper(), basis_id


def generate_samples(model_type: str, config: Dict[str, Any], samples: int, basis: str) -> Tuple[np.ndarray, np.ndarray]:
    if model_type == "dem":
        syndromes, logicals = generate_dem_data(samples, config)
    elif model_type == "si1000":
        circuit = si1000_noise_model(config)
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
    elif model_type in {"pauli_plus", "paper_aligned"}:
        simulator = PauliPlusSimulator(config, basis)
        if model_type == "paper_aligned":
            simulator.apply_paper_aligned_noise(config)
        sampler = simulator.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. Expected one of {sorted(MODEL_TYPES)}."
        )
    return syndromes, logicals


def choose(value: Any, config_section: Dict[str, Any], key: str, default: Any) -> Any:
    if value is not None:
        return value
    if config_section and key in config_section:
        return config_section[key]
    return default


def main() -> None:
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")

    parser = argparse.ArgumentParser(description="Train the AlphaQubit decoder using a YAML config")
    parser.add_argument("--config", required=True, type=Path, help="Path to the noise model configuration YAML")
    parser.add_argument("--samples", type=int, default=None, help="Number of Monte-Carlo shots to generate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model-path", type=Path, default=None, help="Where to save the trained weights")
    parser.add_argument("--basis", type=str, default=None, help="Basis override ('x' or 'z')")
    parser.add_argument("--train-split", type=float, default=None, help="Fraction of samples used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splits")
    parser.add_argument("--npu", action="store_true", help="Use NPUs if available")
    parser.add_argument("--mla", action="store_true", help="Use MLA (Multi-head Latent Attention) model instead of standard transformer")
    parser.add_argument("--profile", action="store_true", help="Enable profiling to capture operator execution details")
    parser.add_argument("--profile_dir", type=str, default="./profiling_logs", help="Directory to save profiling results")
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = infer_model_type(args.config, config)
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Configuration '{args.config}' resolves to model '{model_type}', "
            f"which is not supported. Choose from {sorted(MODEL_TYPES)}."
        )

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'model_type: {model_type}')

    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}

    samples = int(choose(args.samples, training_cfg, "samples", 8500000))  # Paper: 8.5M pretraining
    epochs = int(choose(args.epochs, training_cfg, "epochs", 100))  # Paper: 100 epochs
    batch_size = int(choose(args.batch_size, training_cfg, "batch_size", 256))  # Paper: 256
    lr = float(choose(args.lr, training_cfg, "lr", 1e-4))  # Paper: 1e-4 for pretraining
    train_split = float(choose(args.train_split, training_cfg, "train_split", 0.9))
    model_path = choose(args.model_path, training_cfg, "model_path", None)
    if model_path is None:
        model_path = Path(f"alphaqubit_{model_type}.pth")
    else:
        model_path = Path(model_path)

    basis, basis_id = resolve_basis(args.basis, config, model_type)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'basis: {basis}')
    print(f'basis_id: {basis_id}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f"Loading config from {args.config} ({model_type}), generating {samples} samples in {basis}-basis")
    
    # Memory check: warn if samples might exceed available memory
    import psutil
    available_gb = psutil.virtual_memory().available / (1024**3)
    estimated_gb = samples * 2048 / (1024**3)  # ~2KB per sample
    if True and estimated_gb > available_gb * 0.8:
        print(f"⚠️  WARNING: Estimated memory usage ({estimated_gb:.1f}GB) may exceed available memory ({available_gb:.1f}GB)")
        print(f"   Consider reducing --samples (current: {samples}) or use batch generation")
        print(f"   Recommended max samples for your system: {int(available_gb * 0.8 * 1024**3 / 2048)}")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted by user")
            return
    
    syndromes, logicals = generate_samples(model_type, config, samples, basis)
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'syndromes.shape: {syndromes.shape}')
    print(f'logicals.shape: {logicals.shape}')

    dataset = GeneratedSyndromeDataset(syndromes, logicals, basis_id)
    total = len(dataset)
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'total: {total}')
    if total < 2:
        raise RuntimeError("Need at least two samples to perform train/validation split")

    train_size = max(1, int(total * train_split))
    valid_size = total - train_size
    if valid_size == 0:
        valid_size = 1
        train_size = total - valid_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size], generator=generator)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    # print(f'type(train_ds): {type(train_ds)}, type(valid_ds): {type(valid_ds)}')
    print(f'len(train_ds): {len(train_ds)}, len(valid_ds): {len(valid_ds)}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")

    (x0, _, mask0), _ = dataset[0]
    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(f'x0.shape: {x0.shape}, mask0.shape: {mask0.shape}')
    R, S, F = x0.shape
    grid_size = dataset.grid_size
    print(f"Dataset stats: rounds={R}, stabilisers={S}, features={F}, grid={grid_size + 1}x{grid_size + 1}, pad={dataset.pad}")

    # Select model architecture based on --mla flag
    if args.mla:
        print("Using MLA (Multi-head Latent Attention) model architecture")
        model = AlphaQubitDecoderMLA(F, 256, S, grid_size, num_heads=8, num_layers=12)
    else:
        print("Using standard transformer model architecture")
        model = AlphaQubitDecoderTransformer(F, 256, S, grid_size, num_heads=8, num_layers=12)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")
    print(model)

    def resolve_device(preferred: torch.device) -> torch.device:
        """Validate that ``preferred`` can be initialised, falling back if required."""

        def fallback_device() -> torch.device:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        try:
            torch.empty(1, device=preferred)
            return preferred
        except RuntimeError as exc:
            if preferred.type == "npu":
                print(
                    "[warn] Failed to initialise NPU device; falling back to CUDA/CPU.\n"
                    f"        {exc}"
                )
                backup = fallback_device()
                if backup == preferred:
                    raise
                return resolve_device(backup)
            raise

    if args.npu and HAS_NPU:
        print(f"NPU is available, attempting to use NPU device")
        device = resolve_device(torch.device("npu"))
    else:
        if args.npu:
            print("[warn] --npu requested but torch.npu is unavailable; falling back to CUDA/CPU")
        device = resolve_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Using device: {device}")

    model_save_path = str(model_path)
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    import sys
    print(f"{__file__}:{sys._getframe().f_lineno}")

    # ------------------------------------------------------------------
    # Optional profiling pass: run a few forward steps before full training
    # ------------------------------------------------------------------
    if args.profile:
        profile_dir = args.profile_dir
        os.makedirs(profile_dir, exist_ok=True)
        print(f"\nProfiling enabled – capturing first 3 training batches (output to {profile_dir})")

        hook_dict: dict = {}
        hooks = register_hooks(model, hook_dict)
        model.to(device)
        model.train()

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

        import torch.nn as _nn
        _criterion = _nn.BCEWithLogitsLoss()
        _optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for _batch_idx, ((xb, basis, mask), labels) in enumerate(train_loader):
            if _batch_idx >= 3:
                break
            xb, basis, mask, labels = (
                xb.to(device), basis.to(device), mask.to(device), labels.to(device)
            )
            with record_function("forward"):
                logits = model(xb, basis, mask)
            with record_function("backward"):
                loss = _criterion(logits, labels.float())
                _optimizer.zero_grad()
                loss.backward()
                _optimizer.step()
        del _criterion, _optimizer

        profiler.__exit__(None, None, None)

        trace_file = os.path.join(profile_dir, "train_trace.json")
        profiler.export_chrome_trace(trace_file)
        print(f"Profiler trace saved to: {trace_file}")

        print_profiler_summary(profiler)

        if hook_dict:
            hook_file = os.path.join(profile_dir, "layer_shapes_train.json")
            with open(hook_file, 'w') as f:
                json.dump(hook_dict, f, indent=2)
            print(f"Layer shapes saved to: {hook_file}\n")

        for h in hooks:
            h.remove()

    train_mla(model, train_loader, valid_loader, epochs, lr, device, model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
