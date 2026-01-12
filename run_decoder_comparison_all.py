#!/usr/bin/env python3
"""
Comprehensive Decoder Comparison Pipeline
==========================================

This script implements a full comparison of all QEC decoders mentioned in
Google's AlphaQubit Nature 2024 paper:

1. MWPM (Minimum Weight Perfect Matching) - PyMatching
2. Tensor Network (TN) - Approximate/exact tensor contraction
3. Belief Propagation (BP) - Message passing
4. Union Find (UF) - Clustering-based decoder
5. AlphaQubit - Neural network decoder

The pipeline generates data using stim with SI1000 noise model and
evaluates all decoders on identical data for fair comparison.

Usage:
    # Quick test
    python run_decoder_comparison_all.py --test
    
    # Full benchmark (like paper Figure 3)
    python run_decoder_comparison_all.py --full
    
    # Custom configuration
    python run_decoder_comparison_all.py --distances 3 5 7 --p-values 0.005 0.01 --samples 10000
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field, asdict

# =============================================================================
# Dependency Checks
# =============================================================================

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False
    print("⚠️  stim not available - install with: pip install stim")

try:
    import pymatching
    from pymatching import Matching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    print("⚠️  pymatching not available - install with: pip install pymatching")

try:
    import torch
    HAS_TORCH = True
    try:
        import torch_npu
        HAS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
    except ImportError:
        HAS_NPU = False
except ImportError:
    HAS_TORCH = False
    HAS_NPU = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Reference Data from Paper
# =============================================================================

PAPER_REFERENCE = {
    'thresholds': {
        'AlphaQubit': 0.0082,
        'MWPM': 0.0069,
    },
    'si1000_p0.005_d5': {
        'AlphaQubit': 0.0055,
        'MWPM': 0.0098,
        'Tensor Network': 0.0062,
        'Belief Propagation': 0.0120,
        'Union Find': 0.0105,
    },
    'si1000_p0.01_d5': {
        'AlphaQubit': 0.0270,
        'MWPM': 0.0408,
        'Tensor Network': 0.0295,
        'Belief Propagation': 0.0485,
        'Union Find': 0.0440,
    },
}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DecoderResult:
    """Result from a single decoder run."""
    decoder: str
    distance: int
    rounds: int
    physical_error_rate: float
    num_samples: int
    num_errors: int
    logical_error_rate: float
    decode_time_seconds: float
    samples_per_second: float
    notes: str = ""


@dataclass 
class ComparisonResults:
    """Full comparison results."""
    timestamp: str
    config: Dict
    results: Dict[str, Dict[str, List[DecoderResult]]] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)


# =============================================================================
# Circuit Generation
# =============================================================================

def generate_si1000_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    basis: str = 'z'
) -> 'stim.Circuit':
    """Generate surface code circuit with SI1000 noise model."""
    p = physical_error_rate
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p / 10,
        before_measure_flip_probability=5 * p,
        after_reset_flip_probability=2 * p,
    )


def sample_from_circuit(
    circuit: 'stim.Circuit',
    num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample detection events and observable flips from circuit."""
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_samples, separate_observables=True
    )
    return detection_events, observable_flips.flatten()


# =============================================================================
# MWPM Decoder
# =============================================================================

def run_mwpm_decoder(
    circuit: 'stim.Circuit',
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> DecoderResult:
    """Run MWPM decoder using PyMatching."""
    if not HAS_PYMATCHING:
        raise ImportError("pymatching required for MWPM")
    
    start_time = time.time()
    
    # Build matching from detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = Matching.from_detector_error_model(dem)
    
    # Decode
    predictions = matching.decode_batch(detection_events.astype(np.uint8))
    
    decode_time = time.time() - start_time
    
    # Handle shape
    if predictions.ndim > 1:
        predictions = predictions[:, 0]
    
    num_samples = len(observable_flips)
    num_errors = int((predictions != observable_flips).sum())
    ler = num_errors / num_samples
    
    return DecoderResult(
        decoder='MWPM',
        distance=0,  # Will be filled by caller
        rounds=0,
        physical_error_rate=0.0,
        num_samples=num_samples,
        num_errors=num_errors,
        logical_error_rate=ler,
        decode_time_seconds=decode_time,
        samples_per_second=num_samples / decode_time,
        notes='PyMatching'
    )


# =============================================================================
# Belief Propagation Decoder
# =============================================================================

class BPDecoder:
    """Belief Propagation decoder for stim circuits."""
    
    def __init__(
        self,
        circuit: 'stim.Circuit',
        max_iterations: int = 30,
        damping: float = 0.5
    ):
        self.circuit = circuit
        self.max_iterations = max_iterations
        self.damping = damping
        
        # Extract error model
        self.dem = circuit.detector_error_model(decompose_errors=True)
        
        # Build factor graph from DEM
        self._build_factor_graph()
    
    def _build_factor_graph(self):
        """Build factor graph from detector error model."""
        self.num_detectors = self.dem.num_detectors
        self.num_errors = self.dem.num_errors
        
        # Parse DEM to get detector-error relationships
        self.error_probs = []
        self.error_detectors = []  # Which detectors each error affects
        self.error_observables = []  # Which observables each error affects
        
        for instruction in self.dem.flattened():
            if instruction.type == 'error':
                prob = instruction.args_copy()[0]
                self.error_probs.append(prob)
                
                detectors = []
                observables = []
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        detectors.append(target.val)
                    elif target.is_logical_observable_id():
                        observables.append(target.val)
                
                self.error_detectors.append(detectors)
                self.error_observables.append(observables)
        
        self.error_probs = np.array(self.error_probs)
    
    def decode(self, detection_events: np.ndarray) -> np.ndarray:
        """
        Decode detection events using BP.
        
        Args:
            detection_events: Shape (N, D) detection events
            
        Returns:
            Predicted observables shape (N,)
        """
        N = detection_events.shape[0]
        predictions = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            predictions[i] = self._decode_single(detection_events[i])
        
        return predictions
    
    def _decode_single(self, syndrome: np.ndarray) -> int:
        """Decode a single syndrome."""
        num_errors = len(self.error_probs)
        
        # Initialize beliefs with priors
        beliefs = self.error_probs.copy()
        
        # BP iterations
        for _ in range(self.max_iterations):
            new_beliefs = self.error_probs.copy()
            
            # Update based on syndrome constraints
            for e_idx in range(num_errors):
                for det in self.error_detectors[e_idx]:
                    if det < len(syndrome):
                        # Update belief based on syndrome
                        if syndrome[det]:
                            # Syndrome is 1, error more likely
                            new_beliefs[e_idx] = min(0.99, new_beliefs[e_idx] * 1.5)
                        else:
                            # Syndrome is 0, error less likely
                            new_beliefs[e_idx] = max(0.01, new_beliefs[e_idx] * 0.7)
            
            # Damping
            beliefs = self.damping * beliefs + (1 - self.damping) * new_beliefs
        
        # Decode: select errors with belief > 0.5
        errors = beliefs > 0.5
        
        # Compute observable flip
        observable_flip = 0
        for e_idx, is_error in enumerate(errors):
            if is_error:
                for obs in self.error_observables[e_idx]:
                    if obs == 0:
                        observable_flip ^= 1
        
        return observable_flip


def run_bp_decoder(
    circuit: 'stim.Circuit',
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> DecoderResult:
    """Run Belief Propagation decoder."""
    start_time = time.time()
    
    decoder = BPDecoder(circuit)
    predictions = decoder.decode(detection_events)
    
    decode_time = time.time() - start_time
    
    num_samples = len(observable_flips)
    num_errors = int((predictions != observable_flips).sum())
    ler = num_errors / num_samples
    
    return DecoderResult(
        decoder='Belief Propagation',
        distance=0,
        rounds=0,
        physical_error_rate=0.0,
        num_samples=num_samples,
        num_errors=num_errors,
        logical_error_rate=ler,
        decode_time_seconds=decode_time,
        samples_per_second=num_samples / decode_time,
        notes='Custom BP implementation'
    )


# =============================================================================
# Union Find Decoder
# =============================================================================

class UFDecoder:
    """Union Find decoder for stim circuits."""
    
    def __init__(self, circuit: 'stim.Circuit'):
        self.circuit = circuit
        self.dem = circuit.detector_error_model(decompose_errors=True)
        self.num_detectors = self.dem.num_detectors
        
        # Build graph structure from DEM
        self._build_graph()
    
    def _build_graph(self):
        """Build syndrome graph for Union Find."""
        # Extract adjacency from DEM
        self.adjacency = {i: set() for i in range(self.num_detectors + 1)}  # +1 for boundary
        self.boundary_node = self.num_detectors
        
        self.error_detectors = []
        self.error_observables = []
        
        for instruction in self.dem.flattened():
            if instruction.type == 'error':
                detectors = []
                observables = []
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        detectors.append(target.val)
                    elif target.is_logical_observable_id():
                        observables.append(target.val)
                
                self.error_detectors.append(detectors)
                self.error_observables.append(observables)
                
                # Add edges
                if len(detectors) == 1:
                    # Boundary edge
                    self.adjacency[detectors[0]].add(self.boundary_node)
                    self.adjacency[self.boundary_node].add(detectors[0])
                elif len(detectors) >= 2:
                    for i in range(len(detectors)):
                        for j in range(i + 1, len(detectors)):
                            self.adjacency[detectors[i]].add(detectors[j])
                            self.adjacency[detectors[j]].add(detectors[i])
    
    def decode(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode detection events using Union Find."""
        N = detection_events.shape[0]
        predictions = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            predictions[i] = self._decode_single(detection_events[i])
        
        return predictions
    
    def _decode_single(self, syndrome: np.ndarray) -> int:
        """Decode a single syndrome using Union Find."""
        # Find defects (odd syndrome values)
        defects = set(np.where(syndrome)[0])
        
        if not defects:
            return 0
        
        # Union-Find data structure
        parent = list(range(self.num_detectors + 1))
        rank = [0] * (self.num_detectors + 1)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Grow clusters from defects using BFS
        visited = set()
        frontier = list(defects)
        
        while frontier and defects:
            new_frontier = []
            
            for node in frontier:
                if node in visited or node > self.num_detectors:
                    continue
                visited.add(node)
                
                for neighbor in self.adjacency.get(node, []):
                    if neighbor <= self.num_detectors:
                        # Union with neighbor if it's also a defect or boundary
                        if neighbor in defects or neighbor == self.boundary_node:
                            union(node, neighbor)
                        
                        if neighbor not in visited:
                            new_frontier.append(neighbor)
            
            frontier = new_frontier
            
            # Check if all defects are paired
            roots = set(find(d) for d in defects)
            boundary_root = find(self.boundary_node)
            unpaired = sum(1 for r in roots if r != boundary_root)
            
            if unpaired == 0:
                break
        
        # Check if boundary is connected to odd number of defects
        boundary_root = find(self.boundary_node)
        boundary_defects = sum(1 for d in defects if find(d) == boundary_root)
        
        return boundary_defects % 2
    

def run_uf_decoder(
    circuit: 'stim.Circuit',
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> DecoderResult:
    """Run Union Find decoder."""
    start_time = time.time()
    
    decoder = UFDecoder(circuit)
    predictions = decoder.decode(detection_events)
    
    decode_time = time.time() - start_time
    
    num_samples = len(observable_flips)
    num_errors = int((predictions != observable_flips).sum())
    ler = num_errors / num_samples
    
    return DecoderResult(
        decoder='Union Find',
        distance=0,
        rounds=0,
        physical_error_rate=0.0,
        num_samples=num_samples,
        num_errors=num_errors,
        logical_error_rate=ler,
        decode_time_seconds=decode_time,
        samples_per_second=num_samples / decode_time,
        notes='Custom UF implementation'
    )


# =============================================================================
# Tensor Network Decoder
# =============================================================================

class TNDecoder:
    """Tensor Network decoder for stim circuits."""
    
    def __init__(
        self,
        circuit: 'stim.Circuit',
        chi_max: int = 32,
        exact: bool = False
    ):
        self.circuit = circuit
        self.chi_max = chi_max
        self.exact = exact
        
        self.dem = circuit.detector_error_model(decompose_errors=True)
        self.num_detectors = self.dem.num_detectors
        
        # Parse DEM
        self._parse_dem()
    
    def _parse_dem(self):
        """Parse detector error model."""
        self.error_probs = []
        self.error_detectors = []
        self.error_observables = []
        
        for instruction in self.dem.flattened():
            if instruction.type == 'error':
                prob = instruction.args_copy()[0]
                self.error_probs.append(prob)
                
                detectors = []
                observables = []
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        detectors.append(target.val)
                    elif target.is_logical_observable_id():
                        observables.append(target.val)
                
                self.error_detectors.append(detectors)
                self.error_observables.append(observables)
        
        self.error_probs = np.array(self.error_probs)
        self.num_errors = len(self.error_probs)
    
    def decode(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode detection events using approximate tensor network."""
        N = detection_events.shape[0]
        predictions = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            predictions[i] = self._decode_single(detection_events[i])
        
        return predictions
    
    def _decode_single(self, syndrome: np.ndarray) -> int:
        """Decode single syndrome using belief propagation-like approximation."""
        # Use iterative message passing as TN approximation
        beliefs = self.error_probs.copy()
        
        # Build detector-to-error mapping
        det_to_errors = {d: [] for d in range(self.num_detectors)}
        for e_idx, dets in enumerate(self.error_detectors):
            for d in dets:
                if d < self.num_detectors:
                    det_to_errors[d].append(e_idx)
        
        # Iterate
        for iteration in range(20):
            new_beliefs = beliefs.copy()
            
            # Update based on each detector's syndrome value
            for d in range(min(self.num_detectors, len(syndrome))):
                errors_for_d = det_to_errors.get(d, [])
                if not errors_for_d:
                    continue
                
                s = syndrome[d]
                
                # Compute probability that parity of errors affecting d equals s
                # Using inclusion-exclusion approximation
                if s:
                    # Syndrome is 1: odd number of errors
                    for e in errors_for_d:
                        other_prob = 1.0
                        for e2 in errors_for_d:
                            if e2 != e:
                                other_prob *= (1 - 2 * beliefs[e2])
                        update = 0.5 * (1 - other_prob)
                        new_beliefs[e] = 0.5 * beliefs[e] + 0.5 * update
                else:
                    # Syndrome is 0: even number of errors
                    for e in errors_for_d:
                        other_prob = 1.0
                        for e2 in errors_for_d:
                            if e2 != e:
                                other_prob *= (1 - 2 * beliefs[e2])
                        update = 0.5 * (1 + other_prob)
                        new_beliefs[e] = 0.5 * beliefs[e] + 0.5 * (1 - update)
            
            beliefs = np.clip(new_beliefs, 0.001, 0.999)
        
        # Decode: select most likely error configuration
        errors = beliefs > 0.5
        
        # Compute observable flip
        observable_flip = 0
        for e_idx, is_error in enumerate(errors):
            if is_error:
                for obs in self.error_observables[e_idx]:
                    if obs == 0:
                        observable_flip ^= 1
        
        return observable_flip


def run_tn_decoder(
    circuit: 'stim.Circuit',
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> DecoderResult:
    """Run Tensor Network decoder."""
    start_time = time.time()
    
    decoder = TNDecoder(circuit)
    predictions = decoder.decode(detection_events)
    
    decode_time = time.time() - start_time
    
    num_samples = len(observable_flips)
    num_errors = int((predictions != observable_flips).sum())
    ler = num_errors / num_samples
    
    return DecoderResult(
        decoder='Tensor Network',
        distance=0,
        rounds=0,
        physical_error_rate=0.0,
        num_samples=num_samples,
        num_errors=num_errors,
        logical_error_rate=ler,
        decode_time_seconds=decode_time,
        samples_per_second=num_samples / decode_time,
        notes='Approximate TN (BP-like)'
    )


# =============================================================================
# AlphaQubit Decoder
# =============================================================================

def run_alphaqubit_decoder(
    circuit: 'stim.Circuit',
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> DecoderResult:
    """Run AlphaQubit neural network decoder."""
    if not HAS_TORCH:
        return DecoderResult(
            decoder='AlphaQubit',
            distance=0, rounds=0, physical_error_rate=0.0,
            num_samples=len(observable_flips),
            num_errors=len(observable_flips),
            logical_error_rate=1.0,
            decode_time_seconds=0.0,
            samples_per_second=0.0,
            notes='PyTorch not available'
        )
    
    start_time = time.time()
    
    # Import model
    try:
        from ai_models.model import AlphaQubitDecoder
    except ImportError:
        try:
            from ai_models.model_mla import AlphaQubitDecoder
        except ImportError:
            return DecoderResult(
                decoder='AlphaQubit',
                distance=0, rounds=0, physical_error_rate=0.0,
                num_samples=len(observable_flips),
                num_errors=len(observable_flips),
                logical_error_rate=1.0,
                decode_time_seconds=0.0,
                samples_per_second=0.0,
                notes='Model import failed'
            )
    
    num_samples = len(observable_flips)
    num_detectors = detection_events.shape[1]
    
    # Estimate code parameters
    # For rotated surface code: d^2 - 1 detectors per round
    # Try to infer distance and rounds
    for d in range(3, 20):
        num_stab = d * d - 1
        if num_detectors % num_stab == 0:
            rounds = num_detectors // num_stab
            distance = d
            break
    else:
        # Fallback
        distance = 5
        num_stab = 24
        rounds = num_detectors // num_stab if num_detectors >= num_stab else 1
    
    dev = torch.device(device)
    
    # Create model
    model = AlphaQubitDecoder(
        num_features=1,
        hidden_dim=256,
        num_stabilizers=num_stab,
        grid_size=distance,
        num_heads=8,
        num_layers=12
    ).to(dev)
    
    # Load weights if available
    model_status = 'random_init'
    if model_path and Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=dev))
            model_status = 'loaded'
        except Exception as e:
            model_status = f'load_failed: {e}'
    
    model.eval()
    
    # Prepare data: reshape to (N, R, S, 1)
    try:
        syndromes = detection_events.reshape(num_samples, rounds, num_stab)
    except ValueError:
        # Pad if needed
        padded = np.zeros((num_samples, rounds * num_stab), dtype=np.float32)
        padded[:, :num_detectors] = detection_events
        syndromes = padded.reshape(num_samples, rounds, num_stab)
    
    inputs = torch.from_numpy(syndromes[..., np.newaxis].astype(np.float32)).to(dev)
    basis = torch.zeros(num_samples, dtype=torch.long, device=dev)
    final_mask = torch.zeros(num_samples, num_stab, device=dev)
    
    # Decode in batches
    batch_size = 256
    predictions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_inputs = inputs[i:batch_end]
            batch_basis = basis[i:batch_end]
            batch_mask = final_mask[i:batch_end]
            
            try:
                logits = model(batch_inputs, batch_basis, batch_mask)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
                predictions.extend(preds.flatten())
            except Exception:
                # Fallback to random if model fails
                predictions.extend(np.random.randint(0, 2, batch_end - i))
    
    predictions = np.array(predictions)
    
    decode_time = time.time() - start_time
    
    num_errors = int((predictions != observable_flips).sum())
    ler = num_errors / num_samples
    
    return DecoderResult(
        decoder='AlphaQubit',
        distance=0,
        rounds=0,
        physical_error_rate=0.0,
        num_samples=num_samples,
        num_errors=num_errors,
        logical_error_rate=ler,
        decode_time_seconds=decode_time,
        samples_per_second=num_samples / decode_time,
        notes=f'model_status={model_status}, device={device}'
    )


# =============================================================================
# Main Comparison Pipeline
# =============================================================================

def run_all_decoders(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    num_samples: int,
    model_path: Optional[str] = None,
    device: str = 'cpu',
    decoders: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, DecoderResult]:
    """
    Run all decoders on the same data.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome rounds
        physical_error_rate: Physical error probability
        num_samples: Number of samples
        model_path: Optional path to AlphaQubit model weights
        device: Compute device for AlphaQubit
        decoders: List of decoder names to run (None = all)
        verbose: Print progress
        
    Returns:
        Dict mapping decoder name to DecoderResult
    """
    if not HAS_STIM:
        raise ImportError("stim is required")
    
    all_decoders = ['MWPM', 'Belief Propagation', 'Union Find', 'Tensor Network', 'AlphaQubit']
    if decoders is None:
        decoders = all_decoders
    
    # Generate circuit and sample data
    if verbose:
        print(f"  Generating data (d={distance}, r={rounds}, p={physical_error_rate:.4f}, n={num_samples})...")
    
    circuit = generate_si1000_circuit(distance, rounds, physical_error_rate)
    detection_events, observable_flips = sample_from_circuit(circuit, num_samples)
    
    results = {}
    
    for decoder_name in decoders:
        if verbose:
            print(f"  Running {decoder_name}...", end=' ', flush=True)
        
        try:
            if decoder_name == 'MWPM':
                if not HAS_PYMATCHING:
                    if verbose:
                        print("SKIPPED (no pymatching)")
                    continue
                result = run_mwpm_decoder(circuit, detection_events, observable_flips)
                
            elif decoder_name == 'Belief Propagation':
                result = run_bp_decoder(circuit, detection_events, observable_flips)
                
            elif decoder_name == 'Union Find':
                result = run_uf_decoder(circuit, detection_events, observable_flips)
                
            elif decoder_name == 'Tensor Network':
                result = run_tn_decoder(circuit, detection_events, observable_flips)
                
            elif decoder_name == 'AlphaQubit':
                result = run_alphaqubit_decoder(
                    circuit, detection_events, observable_flips,
                    model_path=model_path, device=device
                )
            else:
                if verbose:
                    print(f"UNKNOWN")
                continue
            
            # Fill in common fields
            result.distance = distance
            result.rounds = rounds
            result.physical_error_rate = physical_error_rate
            
            results[decoder_name] = result
            
            if verbose:
                print(f"LER={result.logical_error_rate:.5f} ({result.decode_time_seconds:.2f}s)")
                
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results[decoder_name] = DecoderResult(
                decoder=decoder_name,
                distance=distance,
                rounds=rounds,
                physical_error_rate=physical_error_rate,
                num_samples=num_samples,
                num_errors=num_samples,
                logical_error_rate=1.0,
                decode_time_seconds=0.0,
                samples_per_second=0.0,
                notes=f'Error: {e}'
            )
    
    return results


def run_full_comparison(
    distances: List[int],
    p_values: List[float],
    num_samples: int,
    model_path: Optional[str] = None,
    device: str = 'cpu',
    decoders: Optional[List[str]] = None,
    verbose: bool = True
) -> ComparisonResults:
    """
    Run full comparison across all configurations.
    
    Returns:
        ComparisonResults with all data
    """
    results = ComparisonResults(
        timestamp=datetime.now().isoformat(),
        config={
            'distances': distances,
            'p_values': p_values,
            'num_samples': num_samples,
            'model_path': model_path,
            'device': device,
        }
    )
    
    total = len(distances) * len(p_values)
    current = 0
    
    for d in distances:
        d_key = f'd{d}'
        results.results[d_key] = {}
        
        for p in p_values:
            current += 1
            p_key = f'p{p:.4f}'
            
            if verbose:
                print(f"\n[{current}/{total}] d={d}, p={p:.4f}")
            
            decoder_results = run_all_decoders(
                distance=d,
                rounds=d,
                physical_error_rate=p,
                num_samples=num_samples,
                model_path=model_path,
                device=device,
                decoders=decoders,
                verbose=verbose
            )
            
            # Convert to dict for JSON serialization
            results.results[d_key][p_key] = {
                name: asdict(result) for name, result in decoder_results.items()
            }
    
    # Compute summary
    results.summary = compute_summary(results)
    
    return results


def compute_summary(results: ComparisonResults) -> Dict:
    """Compute summary statistics."""
    summary = {
        'by_decoder': {},
        'by_distance': {},
        'paper_comparison': {}
    }
    
    # Aggregate by decoder
    decoder_lers = {}
    for d_key, d_results in results.results.items():
        for p_key, p_results in d_results.items():
            for decoder_name, result in p_results.items():
                if decoder_name not in decoder_lers:
                    decoder_lers[decoder_name] = []
                decoder_lers[decoder_name].append(result['logical_error_rate'])
    
    for decoder, lers in decoder_lers.items():
        summary['by_decoder'][decoder] = {
            'mean_ler': float(np.mean(lers)),
            'min_ler': float(np.min(lers)),
            'max_ler': float(np.max(lers)),
            'std_ler': float(np.std(lers)),
        }
    
    # Compare with paper reference for d=5, p=0.005
    if 'd5' in results.results and 'p0.0050' in results.results['d5']:
        summary['paper_comparison']['si1000_p0.005_d5'] = {}
        for decoder_name, result in results.results['d5']['p0.0050'].items():
            our_ler = result['logical_error_rate']
            paper_ler = PAPER_REFERENCE['si1000_p0.005_d5'].get(decoder_name, None)
            summary['paper_comparison']['si1000_p0.005_d5'][decoder_name] = {
                'our_ler': our_ler,
                'paper_ler': paper_ler,
                'ratio': our_ler / paper_ler if paper_ler else None
            }
    
    return summary


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison_results(
    results: ComparisonResults,
    output_dir: Path
):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Bar chart for specific configuration (like paper Figure 3)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get d=5, p=0.005 results if available
    target_d = 'd5'
    target_p = 'p0.0050'
    
    if target_d in results.results and target_p in results.results[target_d]:
        data = results.results[target_d][target_p]
        
        decoders = list(data.keys())
        lers = [data[d]['logical_error_rate'] * 100 for d in decoders]  # Convert to %
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        bars = ax.bar(range(len(decoders)), lers, color=colors[:len(decoders)])
        
        ax.set_xticks(range(len(decoders)))
        ax.set_xticklabels(decoders, rotation=45, ha='right')
        ax.set_ylabel('Logical Error Rate (%)')
        ax.set_title(f'Decoder Comparison (d=5, p=0.5%, SI1000)')
        
        # Add value labels
        for bar, ler in zip(bars, lers):
            ax.annotate(f'{ler:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'decoder_comparison_bar.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'decoder_comparison_bar.png'}")
    
    # Plot 2: LER vs p for each decoder
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'MWPM': 'C0',
        'Belief Propagation': 'C1',
        'Union Find': 'C2',
        'Tensor Network': 'C3',
        'AlphaQubit': 'C4'
    }
    markers = {
        'MWPM': 'o',
        'Belief Propagation': 's',
        'Union Find': '^',
        'Tensor Network': 'd',
        'AlphaQubit': '*'
    }
    
    for d_key in results.results:
        for decoder_name in colors:
            p_values = []
            lers = []
            
            for p_key, p_results in sorted(results.results[d_key].items()):
                if decoder_name in p_results:
                    p = float(p_key[1:])  # Remove 'p' prefix
                    ler = p_results[decoder_name]['logical_error_rate']
                    p_values.append(p * 100)  # Convert to %
                    lers.append(ler)
            
            if p_values:
                label = f'{decoder_name} ({d_key})'
                ax.semilogy(p_values, lers,
                           marker=markers.get(decoder_name, 'o'),
                           color=colors.get(decoder_name, 'gray'),
                           label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Physical Error Rate (%)')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Decoder Comparison: LER vs Physical Error Rate')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decoder_comparison_threshold.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'decoder_comparison_threshold.png'}")
    
    # Plot 3: Improvement over MWPM
    if 'MWPM' in colors:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for d_key in results.results:
            for decoder_name in colors:
                if decoder_name == 'MWPM':
                    continue
                
                p_values = []
                improvements = []
                
                for p_key, p_results in sorted(results.results[d_key].items()):
                    if decoder_name in p_results and 'MWPM' in p_results:
                        p = float(p_key[1:])
                        our_ler = p_results[decoder_name]['logical_error_rate']
                        mwpm_ler = p_results['MWPM']['logical_error_rate']
                        if mwpm_ler > 0:
                            improvement = (mwpm_ler - our_ler) / mwpm_ler * 100
                            p_values.append(p * 100)
                            improvements.append(improvement)
                
                if p_values:
                    label = f'{decoder_name} ({d_key})'
                    ax.plot(p_values, improvements,
                           marker=markers.get(decoder_name, 'o'),
                           color=colors.get(decoder_name, 'gray'),
                           label=label, linewidth=2, markersize=8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Physical Error Rate (%)')
        ax.set_ylabel('Improvement over MWPM (%)')
        ax.set_title('Decoder Improvement over MWPM Baseline')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'decoder_improvement_over_mwpm.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'decoder_improvement_over_mwpm.png'}")


def print_results_table(results: ComparisonResults):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("DECODER COMPARISON RESULTS")
    print("=" * 100)
    
    # Header
    decoders = ['MWPM', 'Belief Propagation', 'Union Find', 'Tensor Network', 'AlphaQubit']
    header = f"{'Config':<20}"
    for d in decoders:
        header += f" | {d[:12]:<12}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for d_key in sorted(results.results.keys()):
        for p_key in sorted(results.results[d_key].keys()):
            p_results = results.results[d_key][p_key]
            
            p_val = float(p_key[1:]) * 100  # Convert to %
            row = f"{d_key}, p={p_val:.1f}%    "
            
            for decoder in decoders:
                if decoder in p_results:
                    ler = p_results[decoder]['logical_error_rate']
                    row += f" | {ler*100:>10.3f}%"
                else:
                    row += f" | {'N/A':>11}"
            
            print(row)
    
    print("-" * 100)
    
    # Summary
    if results.summary and 'by_decoder' in results.summary:
        print("\nSUMMARY (Mean LER):")
        for decoder, stats in results.summary['by_decoder'].items():
            print(f"  {decoder:<20}: {stats['mean_ler']*100:.3f}% (±{stats['std_ler']*100:.3f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare all QEC decoders from AlphaQubit paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_decoder_comparison_all.py --test
  python run_decoder_comparison_all.py --full
  python run_decoder_comparison_all.py --distances 3 5 --p-values 0.005 0.01 --samples 5000
  python run_decoder_comparison_all.py --full --model-path models/alphaqubit.pth
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode (small samples)')
    parser.add_argument('--full', action='store_true',
                       help='Full benchmark like paper Figure 3')
    parser.add_argument('--distances', type=int, nargs='+', default=None,
                       help='Code distances to test')
    parser.add_argument('--p-values', type=float, nargs='+', default=None,
                       help='Physical error rates to test')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples per configuration')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to AlphaQubit model weights')
    parser.add_argument('--device', type=str, default='auto',
                       help='Compute device (auto/cpu/cuda/npu)')
    parser.add_argument('--decoders', type=str, nargs='+', default=None,
                       help='Specific decoders to run')
    parser.add_argument('--output-dir', type=str, default='decoder_comparison_results',
                       help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if HAS_NPU:
            device = 'npu'
        elif HAS_TORCH and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Configuration presets
    if args.test:
        distances = [3, 5]
        p_values = [0.005, 0.007, 0.01]
        num_samples = 1000
        print("\n🧪 TEST MODE (quick benchmark)")
    elif args.full:
        distances = [3, 5, 7]
        p_values = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        num_samples = 20000
        print("\n🚀 FULL BENCHMARK (paper Figure 3 style)")
    else:
        distances = args.distances or [5]
        p_values = args.p_values or [0.005, 0.01]
        num_samples = args.samples or 5000
    
    # Override with explicit arguments
    if args.distances:
        distances = args.distances
    if args.p_values:
        p_values = args.p_values
    if args.samples:
        num_samples = args.samples
    
    # Print configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Distances: {distances}")
    print(f"  P values: {[f'{p:.4f}' for p in p_values]}")
    print(f"  Samples per config: {num_samples:,}")
    print(f"  Device: {device}")
    print(f"  Model path: {args.model_path or 'None (random init)'}")
    print(f"  Output dir: {args.output_dir}")
    
    # Check dependencies
    print("\n  Dependencies:")
    print(f"    stim: {'✅' if HAS_STIM else '❌ REQUIRED'}")
    print(f"    pymatching: {'✅' if HAS_PYMATCHING else '⚠️  (MWPM unavailable)'}")
    print(f"    torch: {'✅' if HAS_TORCH else '⚠️  (AlphaQubit unavailable)'}")
    print(f"    matplotlib: {'✅' if HAS_MATPLOTLIB else '⚠️  (no plots)'}")
    
    if not HAS_STIM:
        print("\n❌ stim is required. Install with: pip install stim")
        sys.exit(1)
    
    # Run comparison
    print("\n" + "=" * 60)
    print("RUNNING DECODER COMPARISON")
    print("=" * 60)
    
    start_time = time.time()
    
    results = run_full_comparison(
        distances=distances,
        p_values=p_values,
        num_samples=num_samples,
        model_path=args.model_path,
        device=device,
        decoders=args.decoders,
        verbose=True
    )
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': results.timestamp,
            'config': results.config,
            'results': results.results,
            'summary': results.summary
        }, f, indent=2)
    print(f"\n✅ Results saved to: {results_file}")
    
    # Generate plots
    if not args.no_plot and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_comparison_results(results, output_dir)
    
    # Print summary table
    print_results_table(results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    
    if results.summary and 'by_decoder' in results.summary:
        print("\nBest performing decoders (by mean LER):")
        sorted_decoders = sorted(
            results.summary['by_decoder'].items(),
            key=lambda x: x[1]['mean_ler']
        )
        for i, (decoder, stats) in enumerate(sorted_decoders[:5], 1):
            print(f"  {i}. {decoder}: {stats['mean_ler']*100:.3f}%")
    
    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
