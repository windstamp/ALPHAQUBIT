"""
Base Decoder Interface
======================

Abstract base class for all QEC decoders.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class BaseDecoder(ABC):
    """
    Abstract base class for quantum error correction decoders.
    
    All decoder implementations should inherit from this class and
    implement the decode() method.
    """
    
    def __init__(self, distance: int, rounds: Optional[int] = None):
        """
        Initialize the decoder.
        
        Args:
            distance: Code distance d (surface code is d x d)
            rounds: Number of syndrome measurement rounds (default: d)
        """
        self.distance = distance
        self.rounds = rounds if rounds is not None else distance
        self.num_stabilizers = distance * distance - 1
        
    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode a syndrome to predict logical errors.
        
        Args:
            syndrome: Detection events array of shape (N, R, S) or (N, S)
                     N = batch size
                     R = rounds
                     S = number of stabilizers
                     
        Returns:
            Predicted logical errors of shape (N,) with values 0 or 1
        """
        pass
    
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode a batch of syndromes.
        
        Default implementation calls decode() on each sample.
        Override for more efficient batch processing.
        
        Args:
            syndromes: Array of shape (N, R, S) or (N, S)
            
        Returns:
            Predictions of shape (N,)
        """
        return self.decode(syndromes)
    
    def get_info(self) -> Dict[str, Any]:
        """Return decoder information and configuration."""
        return {
            'name': self.__class__.__name__,
            'distance': self.distance,
            'rounds': self.rounds,
            'num_stabilizers': self.num_stabilizers,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.distance}, r={self.rounds})"
