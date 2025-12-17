"""
Belief Propagation (BP) Decoder
===============================

Implementation of Belief Propagation decoder for surface codes.
BP is a message-passing algorithm on factor graphs.

Reference:
- Poulin, D., & Chung, Y. (2008). "On the iterative decoding of sparse 
  quantum codes"
- Paper performance: ~10-20% worse than MWPM on SI1000
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .base import BaseDecoder


class BeliefPropagationDecoder(BaseDecoder):
    """
    Belief Propagation decoder for surface codes.
    
    BP iteratively passes messages between variable nodes (qubits)
    and check nodes (stabilizers) to estimate error probabilities.
    
    Performance is typically 10-20% worse than MWPM but BP is faster
    and more parallelizable.
    """
    
    def __init__(
        self,
        distance: int,
        rounds: Optional[int] = None,
        physical_error_rate: float = 0.01,
        max_iterations: int = 50,
        damping: float = 0.5,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize BP decoder.
        
        Args:
            distance: Code distance d
            rounds: Number of syndrome measurement rounds
            physical_error_rate: Prior error probability
            max_iterations: Maximum BP iterations
            damping: Message damping factor (0-1)
            convergence_threshold: Convergence criterion
        """
        super().__init__(distance, rounds)
        
        self.physical_error_rate = physical_error_rate
        self.max_iterations = max_iterations
        self.damping = damping
        self.convergence_threshold = convergence_threshold
        
        # Build factor graph
        self._build_factor_graph()
        
    def _build_factor_graph(self):
        """
        Build the factor graph for the surface code.
        
        Variable nodes: data qubits (d^2)
        Check nodes: stabilizers ((d^2-1)/2 X-type + (d^2-1)/2 Z-type)
        """
        d = self.distance
        
        # Number of data qubits and stabilizers
        self.num_data_qubits = d * d
        self.num_x_checks = (d - 1) * d // 2 + d * (d - 1) // 2
        self.num_z_checks = self.num_stabilizers - self.num_x_checks
        
        # Build adjacency: which qubits participate in which stabilizers
        # For rotated surface code layout
        self.qubit_to_checks = [[] for _ in range(self.num_data_qubits)]
        self.check_to_qubits = [[] for _ in range(self.num_stabilizers)]
        
        # Simplified connectivity for rotated surface code
        # Each stabilizer involves 4 qubits (or 2 at boundaries)
        self._build_rotated_surface_code_graph()
        
    def _build_rotated_surface_code_graph(self):
        """Build connectivity for rotated surface code."""
        d = self.distance
        
        # Place qubits on a d x d grid
        # Stabilizers are on the dual lattice
        
        check_idx = 0
        
        # X stabilizers (plaquettes)
        for row in range(d - 1):
            for col in range(d - 1):
                # 4 qubits around each plaquette
                qubits = [
                    row * d + col,           # top-left
                    row * d + col + 1,       # top-right
                    (row + 1) * d + col,     # bottom-left
                    (row + 1) * d + col + 1  # bottom-right
                ]
                for q in qubits:
                    if q < self.num_data_qubits:
                        self.qubit_to_checks[q].append(check_idx)
                        self.check_to_qubits[check_idx].append(q)
                check_idx += 1
        
        # Z stabilizers (vertices) - simplified
        while check_idx < self.num_stabilizers:
            # Assign remaining checks
            for q in range(self.num_data_qubits):
                if len(self.qubit_to_checks[q]) < 4 and check_idx < self.num_stabilizers:
                    self.qubit_to_checks[q].append(check_idx)
                    self.check_to_qubits[check_idx].append(q)
            check_idx += 1
            if check_idx >= self.num_stabilizers:
                break
                
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode syndromes using Belief Propagation.
        
        Args:
            syndrome: Detection events of shape (N, R, S) or (N, S)
            
        Returns:
            Predicted logical errors of shape (N,)
        """
        # Ensure 3D format
        if syndrome.ndim == 2:
            syndrome = syndrome[:, np.newaxis, :]
        
        N, R, S = syndrome.shape
        predictions = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            # Combine detection events across rounds
            combined_syndrome = syndrome[i].any(axis=0).astype(np.float32)
            
            # Run BP
            error_probs = self._run_bp(combined_syndrome)
            
            # Decode: find errors and compute logical observable
            errors = (error_probs > 0.5).astype(np.int32)
            predictions[i] = self._compute_logical_observable(errors)
        
        return predictions
    
    def _run_bp(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Run belief propagation algorithm.
        
        Args:
            syndrome: Syndrome vector of shape (S,)
            
        Returns:
            Marginal error probabilities for each qubit
        """
        p = self.physical_error_rate
        
        # Initialize messages (log-likelihood ratios)
        # q_to_c[q][c] = message from qubit q to check c
        # c_to_q[c][q] = message from check c to qubit q
        
        num_q = self.num_data_qubits
        num_c = self.num_stabilizers
        
        # Initialize with prior
        prior_llr = np.log((1 - p) / p) if p > 0 and p < 1 else 10.0
        
        q_to_c = {}
        for q in range(num_q):
            for c in self.qubit_to_checks[q]:
                q_to_c[(q, c)] = prior_llr
        
        c_to_q = {}
        for c in range(num_c):
            for q in self.check_to_qubits[c]:
                c_to_q[(c, q)] = 0.0
        
        # BP iterations
        for iteration in range(self.max_iterations):
            old_q_to_c = q_to_c.copy()
            
            # Update check-to-variable messages
            for c in range(min(num_c, len(syndrome))):
                s = syndrome[c] if c < len(syndrome) else 0
                qubits = self.check_to_qubits[c]
                
                for q in qubits:
                    # Product of tanh of other messages
                    product = 1.0
                    for q2 in qubits:
                        if q2 != q and (q2, c) in q_to_c:
                            product *= np.tanh(q_to_c[(q2, c)] / 2)
                    
                    # Incorporate syndrome
                    if s:
                        product *= -1
                    
                    # Clip to avoid numerical issues
                    product = np.clip(product, -0.9999, 0.9999)
                    new_msg = 2 * np.arctanh(product)
                    
                    # Damping
                    if (c, q) in c_to_q:
                        c_to_q[(c, q)] = self.damping * c_to_q[(c, q)] + (1 - self.damping) * new_msg
                    else:
                        c_to_q[(c, q)] = new_msg
            
            # Update variable-to-check messages
            for q in range(num_q):
                checks = self.qubit_to_checks[q]
                
                for c in checks:
                    # Sum of other messages plus prior
                    total = prior_llr
                    for c2 in checks:
                        if c2 != c and (c2, q) in c_to_q:
                            total += c_to_q[(c2, q)]
                    
                    q_to_c[(q, c)] = total
            
            # Check convergence
            max_diff = 0.0
            for key in q_to_c:
                if key in old_q_to_c:
                    max_diff = max(max_diff, abs(q_to_c[key] - old_q_to_c[key]))
            
            if max_diff < self.convergence_threshold:
                break
        
        # Compute marginals
        marginals = np.zeros(num_q)
        for q in range(num_q):
            total_llr = prior_llr
            for c in self.qubit_to_checks[q]:
                if (c, q) in c_to_q:
                    total_llr += c_to_q[(c, q)]
            
            # Convert LLR to probability
            marginals[q] = 1.0 / (1.0 + np.exp(total_llr))
        
        return marginals
    
    def _compute_logical_observable(self, errors: np.ndarray) -> int:
        """
        Compute the logical observable from error pattern.
        
        For surface code, logical X is a chain crossing the code horizontally,
        logical Z is a chain crossing vertically.
        """
        d = self.distance
        
        # Check if errors form a logical operator
        # Simplified: count errors along a logical chain
        
        # Horizontal chain (logical X)
        middle_row = d // 2
        horizontal_errors = 0
        for col in range(d):
            q = middle_row * d + col
            if q < len(errors):
                horizontal_errors += errors[q]
        
        # Vertical chain (logical Z)
        middle_col = d // 2
        vertical_errors = 0
        for row in range(d):
            q = row * d + middle_col
            if q < len(errors):
                vertical_errors += errors[q]
        
        # Logical error if odd number of errors on logical chain
        return (horizontal_errors + vertical_errors) % 2
    
    def get_info(self) -> Dict[str, Any]:
        """Return decoder information."""
        info = super().get_info()
        info.update({
            'algorithm': 'Belief Propagation',
            'max_iterations': self.max_iterations,
            'damping': self.damping,
            'physical_error_rate': self.physical_error_rate,
        })
        return info


def create_bp_decoder(
    distance: int,
    rounds: Optional[int] = None,
    physical_error_rate: float = 0.01,
    max_iterations: int = 50
) -> BeliefPropagationDecoder:
    """Factory function to create BP decoder."""
    return BeliefPropagationDecoder(
        distance=distance,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
        max_iterations=max_iterations
    )
