# Realistic Noise Model Guide

This guide explains how to use the calibration-based realistic noise model to generate training data with spatially-varying noise parameters that match real quantum processors.

## Overview

The realistic noise model system consists of three main components:

1. **Calibration Loader** (`calibration_loader.py`) - Loads and parses per-qubit and per-edge calibration data from JSON files
2. **Realistic Noise Model** (`realistic_noise_model.py`) - Builds Stim circuits with calibration-based noise channels
3. **Data Generator** (`realistic_data_generator.py`) - Generates large-scale training datasets

## Quick Start

### Using Existing Calibration Data

```python
from my_noise_model import (
    DeviceCalibration,
    RealisticNoiseModel,
    RealisticDataGenerator,
)

# Load calibration from JSON file
calibration = DeviceCalibration.from_json_file("configs/realistic_calibration_d5.json")

# Create noise model
noise_model = RealisticNoiseModel(
    calibration=calibration,
    distance=5,
    rounds=25,
)

# Build noisy circuit
circuit = noise_model.build_noisy_circuit(basis="z")

# Sample detection events and observables
det_events, observables = noise_model.sample(num_samples=10000)
```

### Generating Random Calibration Data

```python
from my_noise_model import generate_random_calibration, RealisticNoiseModel

# Generate calibration with statistical properties matching real devices
calibration = generate_random_calibration(
    distance=5,
    seed=42,
    t1_median=73.0,      # Median T1 in microseconds
    t1_std=15.0,         # T1 standard deviation
    cz_error_median=0.0035,  # Median CZ gate error
    cz_error_std=0.0012,     # CZ error standard deviation
    bad_qubit_fraction=0.04, # Fraction of bad qubits
)

# Use with noise model
noise_model = RealisticNoiseModel(calibration=calibration, distance=5, rounds=25)
```

### Using the Data Generator

```python
from my_noise_model import RealisticDataGenerator

# Create generator from calibration file
generator = RealisticDataGenerator.from_calibration_file(
    "configs/realistic_calibration_d5.json",
    distance=5,
    rounds=25,
)

# Or with random calibration
generator = RealisticDataGenerator.with_random_calibration(
    distance=5,
    rounds=25,
    seed=42,
)

# Generate dataset
dataset = generator.generate_samples(num_samples=100000, basis="z")

# Save to file
generator.save_to_npz("realistic_data.npz", dataset)
```

## Calibration Data Format

The calibration JSON file should have the following structure:

```json
{
  "metadata": {
    "distance": 5,
    "seed": 42,
    "stats": {
      "t1_median": 73.0,
      "t1_std": 15.0,
      "cz_error_median": 0.0035,
      ...
    }
  },
  "qubit_calibration": {
    "0": {
      "qubit_id": 0,
      "row": 0,
      "col": 0,
      "t1_us": 77.15,
      "t2_us": 82.99,
      "tphi_us": 179.56,
      "readout_error": 0.0071,
      "reset_error": 0.0012,
      "oneq_error": 0.0008,
      "is_bad_qubit": false
    },
    ...
  },
  "edge_calibration": {
    "(0, 1)": {
      "edge": [0, 1],
      "qubit_a": 0,
      "qubit_b": 1,
      "cz_error": 0.0029,
      "cz_leakage": 0.0003,
      "zz_crosstalk": 0.0004,
      "xeb_fidelity": 0.9971,
      "is_bad_edge": false,
      "has_freq_collision": false
    },
    ...
  },
  "bad_qubits": [37, 38]
}
```

## Physical Parameters

### Per-Qubit Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `t1_us` | Energy relaxation time (T1) | 73 µs |
| `t2_us` | Dephasing time (T2 CPMG) | 80 µs |
| `tphi_us` | Pure dephasing time (Tφ) | 720 µs |
| `readout_error` | Measurement error probability | 0.008 |
| `reset_error` | Reset error probability | 0.0015 |
| `oneq_error` | Single-qubit gate error | 0.0006 |

### Per-Edge Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `cz_error` | Total CZ gate error | 0.0035 |
| `cz_leakage` | CZ-induced leakage probability | 0.0002 |
| `zz_crosstalk` | ZZ crosstalk during CZ | 0.00055 |
| `xeb_fidelity` | Cross-entropy benchmarking fidelity | 0.9965 |

## Command-Line Tool

The `generate_realistic_noise_data.py` script provides a command-line interface:

```bash
# Run demo showing all features
python generate_realistic_noise_data.py --mode demo

# Generate single dataset
python generate_realistic_noise_data.py --mode single \
    --calibration configs/realistic_calibration_d5.json \
    --distance 5 --rounds 25 --samples 100000 \
    --output realistic_d5_r25.npz

# Generate multi-distance pretraining dataset
python generate_realistic_noise_data.py --mode pretrain \
    --output-dir pretrain_data/realistic \
    --samples 100000

# Generate finetuning dataset
python generate_realistic_noise_data.py --mode finetune \
    --calibration configs/realistic_calibration_d5.json \
    --distance 5 --rounds 25 --samples 50000
```

## Data Augmentation

For training diversity, you can perturb calibration data or interpolate between calibrations:

```python
from my_noise_model import perturb_calibration, interpolate_calibrations

# Add random perturbations (simulate drift)
perturbed = perturb_calibration(
    calibration,
    perturbation_scale=0.1,  # 10% variation
    seed=123,
)

# Interpolate between two calibrations
interpolated = interpolate_calibrations(
    calibration1,
    calibration2,
    alpha=0.5,  # 50% blend
)
```

## Integration with AlphaQubit Training

The generated datasets are compatible with the AlphaQubit training pipeline:

```python
import numpy as np
from my_noise_model import GeneratedDataset

# Load generated dataset
dataset = GeneratedDataset.from_npz("realistic_data.npz")

# Use in training
X = dataset.detection_events  # Shape: (N, num_detectors)
y = dataset.observables        # Shape: (N, num_observables)

# Or use with the existing data loading infrastructure
# The .npz format is compatible with numpy's load function
data = np.load("realistic_data.npz")
X = data["detection_events"]
y = data["observables"]
```

## Comparison with Paper Parameters

The default parameters in this implementation align with the AlphaQubit Nature 2024 paper (Table S4):

| Parameter | Paper Value | Default Value |
|-----------|-------------|---------------|
| Cycle time | 1076 ns | 1076 ns |
| T1 | 73 µs | 73 µs |
| Tφ | 720 µs | 720 µs |
| Readout error | 0.8% | 0.8% |
| Reset error | 0.15% | 0.15% |
| CZ leakage | 0.02% | 0.02% |
| ZZ crosstalk | 0.055% | 0.055% |
| 1Q excess | 0.062% | 0.062% |
| CZ excess | 0.275% | 0.275% |

## File Structure

```
my_noise_model/
├── calibration_loader.py      # Load/parse calibration data
├── realistic_noise_model.py   # Build noisy circuits
├── realistic_data_generator.py # Generate training datasets
└── __init__.py                # Module exports

configs/
└── realistic_calibration_d5.json  # Example calibration file

generate_realistic_noise_data.py   # Command-line tool
```

## Tips for Best Results

1. **Use realistic calibration data**: Load from actual device characterization when available
2. **Vary calibration seeds**: Generate diverse training data by using multiple random seeds
3. **Include bad qubits**: Real devices have ~4% bad qubits; include them for realistic training
4. **Match experimental conditions**: Use the same distance/rounds as your target experiment
5. **Augment with perturbations**: Add small random variations to simulate temporal drift
