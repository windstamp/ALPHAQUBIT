"""
Experiment Data Loader
======================

Load .dem (Detector Error Model) and .stim (Stim Circuit) files from 
experiment data folders for generating noise samples.

This module supports loading Google's QEC experiment data format where
each experiment folder contains:
- circuit_noisy.stim: The noisy quantum circuit
- circuit.dem or similar: The detector error model (optional)
- Additional metadata files

Usage:
------
    from simulator.experiment_loader import ExperimentLoader, load_experiment_folder
    
    # Load single experiment
    loader = ExperimentLoader("path/to/experiment")
    syndromes, logical_errors = loader.sample(num_shots=10000)
    
    # Load all experiments from a root directory
    datasets = load_all_experiments("path/to/google_qec3v5_experiment_data", shots=1000)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Tuple, List, Dict, Any

import numpy as np

try:
    import stim
    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    stim = None

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata extracted from experiment files and paths."""
    name: str
    path: Path
    distance: Optional[int] = None
    rounds: Optional[int] = None
    basis: Optional[str] = None  # 'x' or 'z'
    num_detectors: Optional[int] = None
    num_observables: Optional[int] = None
    has_dem: bool = False
    has_stim: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SampledData:
    """Container for sampled experiment data."""
    detections: np.ndarray  # Shape: (shots, num_detectors) or (shots, rounds, detectors_per_round)
    observables: np.ndarray  # Shape: (shots, num_observables)
    metadata: ExperimentMetadata
    
    @property
    def syndromes(self) -> np.ndarray:
        """Alias for detections."""
        return self.detections
    
    @property
    def logical_errors(self) -> np.ndarray:
        """Alias for observables (typically logical error indicators)."""
        return self.observables


class ExperimentLoader:
    """
    Load and sample from experiment data files (.dem, .stim).
    
    Supports:
    - Direct .dem file loading and sampling
    - .stim circuit loading with optional .dem override
    - Automatic detection of file format
    - Metadata extraction from filenames and paths
    """
    
    # Regex patterns for extracting metadata from paths
    _RX_ROUNDS = re.compile(r"_r(\d+)|rounds?[_=]?(\d+)", re.IGNORECASE)
    _RX_DISTANCE = re.compile(r"_d(\d+)|distance[_=]?(\d+)", re.IGNORECASE)
    _RX_BASIS = re.compile(r"_(b?[xz])_|basis[_=]?([xz])", re.IGNORECASE)
    
    def __init__(
        self,
        path: Path | str,
        prefer_dem: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize experiment loader.
        
        Parameters
        ----------
        path : Path or str
            Path to experiment folder, .dem file, or .stim file
        prefer_dem : bool
            If True, prefer loading .dem files over generating DEM from .stim
        verbose : bool
            Enable verbose logging
        """
        if not STIM_AVAILABLE:
            raise ImportError("stim package is required. Install with: pip install stim")
        
        self.path = Path(path)
        self.prefer_dem = prefer_dem
        self.verbose = verbose
        
        self._dem = None  # stim.DetectorErrorModel
        self._circuit = None  # stim.Circuit
        self._sampler = None  # stim.CompiledDetectorSampler
        self._metadata: Optional[ExperimentMetadata] = None
        
        self._discover_files()
    
    def _discover_files(self) -> None:
        """Discover .dem and .stim files in the experiment path."""
        if self.path.is_file():
            if self.path.suffix == ".dem":
                self._dem_path = self.path
                self._stim_path = self._find_matching_stim(self.path)
            elif self.path.suffix == ".stim":
                self._stim_path = self.path
                self._dem_path = self._find_matching_dem(self.path)
            else:
                raise ValueError(f"Unsupported file type: {self.path.suffix}")
        elif self.path.is_dir():
            self._stim_path = self._find_stim_in_folder(self.path)
            self._dem_path = self._find_dem_in_folder(self.path)
            
            if self._stim_path is None and self._dem_path is None:
                raise FileNotFoundError(
                    f"No .dem or .stim files found in {self.path}"
                )
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        
        if self.verbose:
            logger.info(f"Found stim: {self._stim_path}, dem: {self._dem_path}")
    
    @staticmethod
    def _find_stim_in_folder(folder: Path) -> Optional[Path]:
        """Find .stim file in folder, preferring circuit_noisy.stim."""
        stim_files = list(folder.glob("*.stim"))
        if not stim_files:
            return None
        
        # Prefer circuit_noisy.stim (Google's naming convention)
        for f in stim_files:
            if "noisy" in f.stem.lower():
                return f
        
        # Otherwise return first found
        return stim_files[0]
    
    @staticmethod
    def _find_dem_in_folder(folder: Path) -> Optional[Path]:
        """Find .dem file in folder."""
        dem_files = list(folder.glob("*.dem")) + list(folder.glob("*.dem.json"))
        if not dem_files:
            return None
        return dem_files[0]
    
    @staticmethod
    def _find_matching_dem(stim_path: Path) -> Optional[Path]:
        """Find .dem file matching a .stim file."""
        stem = stim_path.stem
        for p in stim_path.parent.iterdir():
            if p.is_file() and p.stem.startswith(stem) and ".dem" in p.suffixes:
                return p
        # Also check for generic .dem in same folder
        for p in stim_path.parent.glob("*.dem"):
            return p
        return None
    
    @staticmethod
    def _find_matching_stim(dem_path: Path) -> Optional[Path]:
        """Find .stim file matching a .dem file."""
        stem = dem_path.stem.replace(".dem", "")
        for p in dem_path.parent.iterdir():
            if p.is_file() and p.suffix == ".stim" and stem in p.stem:
                return p
        # Also check for any .stim in same folder
        for p in dem_path.parent.glob("*.stim"):
            return p
        return None
    
    def _parse_metadata_from_path(self) -> ExperimentMetadata:
        """Extract metadata from file/folder path."""
        path_str = str(self.path)
        name = self.path.name if self.path.is_dir() else self.path.stem
        
        # Extract distance
        distance = None
        m = self._RX_DISTANCE.search(path_str)
        if m:
            distance = int(m.group(1) or m.group(2))
        
        # Extract rounds
        rounds = None
        m = self._RX_ROUNDS.search(path_str)
        if m:
            rounds = int(m.group(1) or m.group(2))
        
        # Extract basis
        basis = None
        m = self._RX_BASIS.search(path_str)
        if m:
            basis = (m.group(1) or m.group(2)).lower().replace("b", "")
        
        return ExperimentMetadata(
            name=name,
            path=self.path,
            distance=distance,
            rounds=rounds,
            basis=basis,
            has_dem=self._dem_path is not None,
            has_stim=self._stim_path is not None,
        )
    
    @property
    def metadata(self) -> ExperimentMetadata:
        """Get experiment metadata."""
        if self._metadata is None:
            self._metadata = self._parse_metadata_from_path()
            
            # Enrich with DEM/circuit info if loaded
            if self._dem is not None:
                self._metadata.num_detectors = self._dem.num_detectors
                self._metadata.num_observables = self._dem.num_observables
            elif self._circuit is not None:
                dem = self._circuit.detector_error_model()
                self._metadata.num_detectors = dem.num_detectors
                self._metadata.num_observables = dem.num_observables
        
        return self._metadata
    
    def load_dem(self):
        """Load the detector error model. Returns stim.DetectorErrorModel."""
        if self._dem is not None:
            return self._dem
        
        if self._dem_path is not None and self.prefer_dem:
            if self.verbose:
                logger.info(f"Loading DEM from file: {self._dem_path}")
            self._dem = stim.DetectorErrorModel.from_file(str(self._dem_path))
        elif self._stim_path is not None:
            if self.verbose:
                logger.info(f"Generating DEM from circuit: {self._stim_path}")
            self._circuit = stim.Circuit.from_file(str(self._stim_path))
            self._dem = self._circuit.detector_error_model()
        else:
            raise RuntimeError("No DEM or STIM file available")
        
        return self._dem
    
    def load_circuit(self):
        """Load the stim circuit if available. Returns Optional[stim.Circuit]."""
        if self._circuit is not None:
            return self._circuit
        
        if self._stim_path is not None:
            self._circuit = stim.Circuit.from_file(str(self._stim_path))
        
        return self._circuit
    
    def get_sampler(self):
        """Get or create a compiled detector sampler. Returns stim.CompiledDetectorSampler."""
        if self._sampler is not None:
            return self._sampler
        
        dem = self.load_dem()
        self._sampler = dem.compile_sampler()
        return self._sampler
    
    def sample(
        self,
        num_shots: int,
        reshape_to_rounds: bool = False,
    ) -> SampledData:
        """
        Sample detection events and observables.
        
        Parameters
        ----------
        num_shots : int
            Number of shots to sample
        reshape_to_rounds : bool
            If True, reshape detections to (shots, rounds, detectors_per_round)
        
        Returns
        -------
        SampledData
            Container with detections, observables, and metadata
        """
        # Try circuit sampler first (has separate_observables), then DEM sampler
        if self._stim_path is not None:
            # Use circuit sampler which supports separate_observables
            circuit = self.load_circuit()
            circuit_sampler = circuit.compile_detector_sampler()
            detections, observables = circuit_sampler.sample(
                shots=num_shots,
                separate_observables=True,
            )
        else:
            # Use DEM sampler (different API)
            dem_sampler = self.get_sampler()
            # DEM sampler returns combined detections+observables
            result = dem_sampler.sample(shots=num_shots)
            
            # Split detections and observables
            dem = self.load_dem()
            num_dets = dem.num_detectors
            num_obs = dem.num_observables
            
            detections = result[:, :num_dets]
            observables = result[:, num_dets:num_dets + num_obs]
        
        detections = detections.astype(np.uint8)
        observables = observables.astype(np.uint8)
        
        if reshape_to_rounds and self.metadata.rounds:
            rounds = self.metadata.rounds
            total_dets = detections.shape[1]
            dets_per_round = total_dets // rounds
            
            if total_dets % rounds == 0:
                detections = detections.reshape(num_shots, rounds, dets_per_round)
        
        return SampledData(
            detections=detections,
            observables=observables,
            metadata=self.metadata,
        )
    
    def sample_with_soft_info(
        self,
        num_shots: int,
        snr: float = 10.0,
        leakage_prob: float = 0.01,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample with soft readout information (3-channel output).
        
        Returns detections with soft channels:
        - Channel 0: Hard detection events (0 or 1)
        - Channel 1: Soft probability (computational state posterior)
        - Channel 2: Leakage probability
        
        Parameters
        ----------
        num_shots : int
            Number of shots to sample
        snr : float
            Signal-to-noise ratio for soft readout
        leakage_prob : float
            Probability of leakage events
        device : str
            Device for computation ('cpu', 'cuda', 'npu')
        
        Returns
        -------
        data : np.ndarray
            Shape (shots, rounds, stabilizers, 3) with soft channels
        observables : np.ndarray
            Shape (shots, num_observables)
        """
        sampled = self.sample(num_shots, reshape_to_rounds=True)
        detections = sampled.detections
        observables = sampled.observables
        
        # Generate soft channels
        if detections.ndim == 2:
            # Reshape to (N, R, S)
            N, total = detections.shape
            R = self.metadata.rounds or 1
            S = total // R
            detections = detections.reshape(N, R, S)
        
        N, R, S = detections.shape
        
        # Channel 1: Soft readout probability
        # Simulate IQ readout with Gaussian noise
        soft_prob = self._generate_soft_channel(detections, snr, device)
        
        # Channel 2: Leakage probability
        leakage_prob_arr = self._generate_leakage_channel(N, R, S, leakage_prob, device)
        
        # Stack channels: (N, R, S, 3)
        data = np.stack([
            detections.astype(np.float32),
            soft_prob,
            leakage_prob_arr,
        ], axis=-1)
        
        return data, observables
    
    def _generate_soft_channel(
        self,
        detections: np.ndarray,
        snr: float,
        device: str,
    ) -> np.ndarray:
        """Generate soft readout probabilities."""
        # Simulate IQ readout: |0⟩ centered at (1,0), |1⟩ at (-1,0)
        # Add Gaussian noise based on SNR
        sigma = 1.0 / snr
        
        N, R, S = detections.shape
        
        # Base signal: 1 for |0⟩, -1 for |1⟩
        signal = 1 - 2 * detections.astype(np.float32)
        
        # Add noise
        noise = np.random.normal(0, sigma, size=(N, R, S)).astype(np.float32)
        noisy_signal = signal + noise
        
        # Convert to probability via sigmoid
        soft_prob = 1.0 / (1.0 + np.exp(-snr * noisy_signal))
        
        return soft_prob.astype(np.float32)
    
    def _generate_leakage_channel(
        self,
        N: int, R: int, S: int,
        leakage_prob: float,
        device: str,
    ) -> np.ndarray:
        """Generate leakage probability channel."""
        # Simple model: independent leakage events
        leakage = np.random.uniform(0, 1, size=(N, R, S)) < leakage_prob
        
        # Convert to soft probability
        leakage_soft = np.where(
            leakage,
            np.random.uniform(0.7, 1.0, size=(N, R, S)),
            np.random.uniform(0.0, 0.1, size=(N, R, S)),
        )
        
        return leakage_soft.astype(np.float32)


def load_experiment_folder(
    folder: Path | str,
    num_shots: int = 1000,
    with_soft_info: bool = False,
    **kwargs,
) -> SampledData | Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load and sample from an experiment folder.
    
    Parameters
    ----------
    folder : Path or str
        Path to experiment folder
    num_shots : int
        Number of shots to sample
    with_soft_info : bool
        If True, return data with soft readout channels
    **kwargs
        Additional arguments passed to sampling functions
    
    Returns
    -------
    SampledData or tuple
        Sampled data (with soft info if requested)
    """
    loader = ExperimentLoader(folder)
    
    if with_soft_info:
        return loader.sample_with_soft_info(num_shots, **kwargs)
    else:
        return loader.sample(num_shots, **kwargs)


def discover_experiments(root: Path | str) -> List[Path]:
    """
    Discover all experiment folders under a root directory.
    
    An experiment folder is defined as a directory containing
    at least one .stim or .dem file.
    
    Parameters
    ----------
    root : Path or str
        Root directory to search
    
    Returns
    -------
    List[Path]
        List of experiment folder paths
    """
    root = Path(root)
    if not root.exists():
        return []
    
    experiments = set()
    
    # Find all .stim files
    for stim_file in root.rglob("*.stim"):
        experiments.add(stim_file.parent)
    
    # Find all .dem files
    for dem_file in root.rglob("*.dem"):
        experiments.add(dem_file.parent)
    
    return sorted(experiments)


def load_all_experiments(
    root: Path | str,
    num_shots: int = 1000,
    with_soft_info: bool = False,
    skip_errors: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, SampledData | Tuple]:
    """
    Load and sample from all experiments under a root directory.
    
    Parameters
    ----------
    root : Path or str
        Root directory containing experiment folders
    num_shots : int
        Number of shots per experiment
    with_soft_info : bool
        If True, include soft readout channels
    skip_errors : bool
        If True, skip experiments that fail to load
    verbose : bool
        Enable verbose logging
    **kwargs
        Additional arguments passed to sampling functions
    
    Returns
    -------
    Dict[str, SampledData or tuple]
        Dictionary mapping experiment names to sampled data
    """
    root = Path(root)
    experiments = discover_experiments(root)
    
    if verbose:
        logger.info(f"Found {len(experiments)} experiments under {root}")
    
    results = {}
    
    for exp_path in experiments:
        name = exp_path.relative_to(root).as_posix()
        
        try:
            if verbose:
                logger.info(f"Loading experiment: {name}")
            
            data = load_experiment_folder(
                exp_path,
                num_shots=num_shots,
                with_soft_info=with_soft_info,
                **kwargs,
            )
            results[name] = data
            
        except Exception as e:
            if skip_errors:
                logger.warning(f"Failed to load {name}: {e}")
            else:
                raise
    
    return results


# CLI interface
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Load and sample from experiment data files"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to experiment folder, .dem file, or .stim file",
    )
    parser.add_argument(
        "--shots", "-n",
        type=int,
        default=1000,
        help="Number of shots to sample (default: 1000)",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Include soft readout channels",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (.npz)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.path.is_dir() and not any(args.path.glob("*.stim")) and not any(args.path.glob("*.dem")):
        # Treat as root directory with multiple experiments
        print(f"Discovering experiments under {args.path}...")
        experiments = discover_experiments(args.path)
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments[:20]:
            print(f"  - {exp}")
        if len(experiments) > 20:
            print(f"  ... and {len(experiments) - 20} more")
    else:
        # Single experiment
        loader = ExperimentLoader(args.path, verbose=args.verbose)
        print(f"Experiment: {loader.metadata.name}")
        print(f"  Path: {loader.metadata.path}")
        print(f"  Distance: {loader.metadata.distance}")
        print(f"  Rounds: {loader.metadata.rounds}")
        print(f"  Basis: {loader.metadata.basis}")
        print(f"  Has DEM: {loader.metadata.has_dem}")
        print(f"  Has STIM: {loader.metadata.has_stim}")
        
        print(f"\nSampling {args.shots} shots...")
        
        if args.soft:
            data, obs = loader.sample_with_soft_info(args.shots)
            print(f"  Data shape: {data.shape}")
            print(f"  Observables shape: {obs.shape}")
            
            if args.output:
                np.savez(args.output, data=data, obs=obs)
                print(f"  Saved to {args.output}")
        else:
            sampled = loader.sample(args.shots)
            print(f"  Detections shape: {sampled.detections.shape}")
            print(f"  Observables shape: {sampled.observables.shape}")
            
            if args.output:
                np.savez(
                    args.output,
                    data=sampled.detections,
                    obs=sampled.observables,
                )
                print(f"  Saved to {args.output}")
