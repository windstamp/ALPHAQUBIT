"""
Union Find (UF) Decoder
=======================

Implementation of Union Find decoder for surface codes.
UF is a fast, near-linear time algorithm based on clustering.

Reference:
- Delfosse, N., & Nickerson, N. H. (2021). "Almost-linear time decoding 
  algorithm for topological codes"
- Typically ~5-10% worse than MWPM but much faster O(n α(n))
"""

from typing import Optional, Dict, Any, List, Set
import numpy as np

from .base import BaseDecoder


class UnionFindDecoder(BaseDecoder):
    """
    Union Find decoder for surface codes.
    
    UF grows clusters around detection events and merges them
    using union-find data structure. Very fast O(n α(n)) complexity.
    
    Performance is typically ~5-10% worse than MWPM but much faster,
    making it suitable for real-time decoding.
    """
    
    def __init__(
        self,
        distance: int,
        rounds: Optional[int] = None,
        weighted: bool = True
    ):
        """
        Initialize Union Find decoder.
        
        Args:
            distance: Code distance d
            rounds: Number of syndrome measurement rounds
            weighted: Use weighted union (improves performance)
        """
        super().__init__(distance, rounds)
        self.weighted = weighted
        
        # Build graph structure
        self._build_graph()
        
    def _build_graph(self):
        """Build the syndrome graph for Union Find."""
        d = self.distance
        r = self.rounds
        
        # Nodes: stabilizers at each round + boundary nodes
        self.num_nodes = self.num_stabilizers * r + 2  # +2 for boundaries
        self.boundary_left = self.num_nodes - 2
        self.boundary_right = self.num_nodes - 1
        
        # Build adjacency list
        self.adj = [[] for _ in range(self.num_nodes)]
        
        # Time-like edges (same stabilizer, consecutive rounds)
        for s in range(self.num_stabilizers):
            for t in range(r - 1):
                n1 = s * r + t
                n2 = s * r + t + 1
                self.adj[n1].append(n2)
                self.adj[n2].append(n1)
        
        # Space-like edges (adjacent stabilizers, same round)
        # Simplified: connect stabilizers in a grid pattern
        grid_size = int(np.ceil(np.sqrt(self.num_stabilizers)))
        
        for t in range(r):
            for s in range(self.num_stabilizers):
                row, col = s // grid_size, s % grid_size
                
                # Connect to right neighbor
                if col < grid_size - 1 and s + 1 < self.num_stabilizers:
                    n1 = s * r + t
                    n2 = (s + 1) * r + t
                    self.adj[n1].append(n2)
                    self.adj[n2].append(n1)
                
                # Connect to bottom neighbor
                if row < grid_size - 1 and s + grid_size < self.num_stabilizers:
                    n1 = s * r + t
                    n2 = (s + grid_size) * r + t
                    self.adj[n1].append(n2)
                    self.adj[n2].append(n1)
        
        # Boundary connections (simplified)
        # Left boundary stabilizers
        for s in range(min(grid_size, self.num_stabilizers)):
            for t in range(r):
                n = s * r + t
                self.adj[n].append(self.boundary_left)
                self.adj[self.boundary_left].append(n)
        
        # Right boundary stabilizers
        for s in range(self.num_stabilizers - min(grid_size, self.num_stabilizers), 
                       self.num_stabilizers):
            for t in range(r):
                n = s * r + t
                self.adj[n].append(self.boundary_right)
                self.adj[self.boundary_right].append(n)
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode syndromes using Union Find.
        
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
            # Find detection events
            detection_events = self._find_detection_events(syndrome[i])
            
            # Run Union Find
            clusters = self._union_find(detection_events)
            
            # Compute logical observable from clusters
            predictions[i] = self._compute_logical(clusters)
        
        return predictions
    
    def _find_detection_events(self, syndrome: np.ndarray) -> List[int]:
        """
        Find nodes with detection events.
        
        Args:
            syndrome: Shape (R, S)
            
        Returns:
            List of node indices with detection events
        """
        R, S = syndrome.shape
        events = []
        
        for t in range(R):
            for s in range(S):
                if syndrome[t, s]:
                    node = s * self.rounds + t
                    if node < self.num_nodes - 2:  # Exclude boundary nodes
                        events.append(node)
        
        return events
    
    def _union_find(self, detection_events: List[int]) -> List[Set[int]]:
        """
        Run Union Find algorithm.
        
        Args:
            detection_events: List of nodes with detections
            
        Returns:
            List of clusters (sets of nodes)
        """
        # Initialize union-find structure
        parent = list(range(self.num_nodes))
        rank = [0] * self.num_nodes
        
        def find(x: int) -> int:
            """Find with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int):
            """Union by rank."""
            px, py = find(x), find(y)
            if px == py:
                return
            
            if self.weighted:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
            else:
                parent[py] = px
        
        # Mark odd nodes (detection events)
        odd_nodes = set(detection_events)
        
        # Grow clusters from odd nodes
        # Use BFS-like growth
        active = set(detection_events)
        visited = set()
        
        while active:
            new_active = set()
            
            for node in active:
                if node in visited:
                    continue
                visited.add(node)
                
                # Check neighbors
                for neighbor in self.adj[node]:
                    if neighbor < self.num_nodes:
                        # Union if both are odd or one is boundary
                        if neighbor in odd_nodes or neighbor >= self.num_nodes - 2:
                            union(node, neighbor)
                        
                        if neighbor not in visited:
                            new_active.add(neighbor)
            
            active = new_active
            
            # Stop if all odd nodes are paired
            roots = set(find(n) for n in odd_nodes)
            boundary_roots = {find(self.boundary_left), find(self.boundary_right)}
            
            unpaired = sum(1 for r in roots if r not in boundary_roots)
            if unpaired == 0:
                break
        
        # Extract clusters
        clusters = {}
        for node in range(self.num_nodes):
            root = find(node)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(node)
        
        return list(clusters.values())
    
    def _compute_logical(self, clusters: List[Set[int]]) -> int:
        """
        Compute logical observable from clusters.
        
        A logical error occurs if a cluster connects both boundaries.
        """
        for cluster in clusters:
            if self.boundary_left in cluster and self.boundary_right in cluster:
                return 1
        return 0
    
    def get_info(self) -> Dict[str, Any]:
        """Return decoder information."""
        info = super().get_info()
        info.update({
            'algorithm': 'Union Find',
            'weighted': self.weighted,
            'complexity': 'O(n α(n))',
        })
        return info


def create_uf_decoder(
    distance: int,
    rounds: Optional[int] = None,
    weighted: bool = True
) -> UnionFindDecoder:
    """Factory function to create Union Find decoder."""
    return UnionFindDecoder(
        distance=distance,
        rounds=rounds,
        weighted=weighted
    )
