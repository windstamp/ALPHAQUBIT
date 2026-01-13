"""
Paper-aligned noise builders for ALPHAQUBIT.
Exports:
  - make_si1000_weights
  - IQReadoutModel (full & simplified)
  - soft_xor, soft_detection_sequence
  - twirl_to_pauli_channel (GPTA)
  - build_pauli_plus_channels (leakage, crosstalk, DQLR, etc.)
  - DeviceCalibration, QubitCalibration, EdgeCalibration (calibration loading)
  - RealisticNoiseModel, RealisticNoiseConfig (realistic noise model)
  - RealisticDataGenerator (data generation with realistic noise)
"""
from .si1000 import make_si1000_weights
from .iq_readout import IQReadoutModel
from .softxor import soft_xor, soft_detection_sequence
from .gpta import twirl_to_pauli_channel
from .pauli_plus import build_pauli_plus_channels
from .circuit_builder import build_paper_aligned_circuit

# New: Calibration-based realistic noise model
from .calibration_loader import (
    DeviceCalibration,
    QubitCalibration,
    EdgeCalibration,
    generate_random_calibration,
)
from .realistic_noise_model import (
    RealisticNoiseModel,
    RealisticNoiseConfig,
    create_noise_model_from_calibration_file,
    create_noise_model_with_random_calibration,
)
from .realistic_data_generator import (
    RealisticDataGenerator,
    GeneratedDataset,
    generate_multi_distance_dataset,
    generate_finetuning_dataset,
    perturb_calibration,
    interpolate_calibrations,
)

__all__ = [
    "make_si1000_weights",
    "IQReadoutModel",
    "soft_xor",
    "soft_detection_sequence",
    "twirl_to_pauli_channel",
    "build_pauli_plus_channels",
    "build_paper_aligned_circuit",
    # Calibration loading
    "DeviceCalibration",
    "QubitCalibration", 
    "EdgeCalibration",
    "generate_random_calibration",
    # Realistic noise model
    "RealisticNoiseModel",
    "RealisticNoiseConfig",
    "create_noise_model_from_calibration_file",
    "create_noise_model_with_random_calibration",
    # Data generation
    "RealisticDataGenerator",
    "GeneratedDataset",
    "generate_multi_distance_dataset",
    "generate_finetuning_dataset",
    "perturb_calibration",
    "interpolate_calibrations",
]

