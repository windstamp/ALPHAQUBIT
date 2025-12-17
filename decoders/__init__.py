"""
Baseline Decoders for Comparison with AlphaQubit
=================================================

This module implements the baseline decoders mentioned in the Nature 2024 paper:
1. MWPM (Minimum Weight Perfect Matching) - using PyMatching
2. Belief Propagation (BP)
3. Union Find (UF)
4. Tensor Network (TN)

These are used to benchmark AlphaQubit's performance improvements.
"""

from .mwpm_decoder import MWPMDecoder
from .belief_propagation import BeliefPropagationDecoder
from .union_find import UnionFindDecoder
from .tensor_network import TensorNetworkDecoder
from .base import BaseDecoder

__all__ = [
    'BaseDecoder',
    'MWPMDecoder',
    'BeliefPropagationDecoder',
    'UnionFindDecoder',
    'TensorNetworkDecoder',
]
