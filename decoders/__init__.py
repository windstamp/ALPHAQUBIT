"""
Baseline Decoders for Comparison with AlphaQubit
=================================================

This module implements the baseline decoders mentioned in the Nature 2024 paper:
1. MWPM (Minimum Weight Perfect Matching) - using PyMatching
2. Belief Propagation (BP)
3. Union Find (UF)
4. Tensor Network (TN)

These are used to benchmark AlphaQubit's performance improvements.

Usage:
    # Run full comparison pipeline
    python run_decoder_comparison_all.py --full
    
    # Quick test
    python run_decoder_comparison_all.py --test
"""

from .base import BaseDecoder
from .mwpm_decoder import MWPMDecoder
from .belief_propagation import BeliefPropagationDecoder
from .union_find import UnionFindDecoder
from .tensor_network import TensorNetworkDecoder

# Optional stim integration
try:
    from .stim_integration import StimSyndromeGenerator, generate_benchmark_data
    HAS_STIM_INTEGRATION = True
except ImportError:
    HAS_STIM_INTEGRATION = False

__all__ = [
    'BaseDecoder',
    'MWPMDecoder',
    'BeliefPropagationDecoder',
    'UnionFindDecoder',
    'TensorNetworkDecoder',
]

if HAS_STIM_INTEGRATION:
    __all__.extend(['StimSyndromeGenerator', 'generate_benchmark_data'])
