from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from typing import List, Dict, Tuple
import math

class PurchaseCollaborativeProcessor:
    def __init__(self, use_quantity: bool = True, quantity_transform: str = "log"):
        """
        Initialize the collaborative relationship processor for purchase data.
        
        Args:
            use_quantity: Whether to incorporate purchase quantities into the interaction matrix.
            quantity_transform: How to transform the quantity.
                - "log": uses log1p(quantity)
                - "raw": uses quantity directly
                - "binary": uses 1.0 if quantity > 0
        """
        self.user_item_interactions = {}
        self.item_to_idx = None
        self.interaction_matrix = None
        self.use_quantity = use_quantity
        self.quantity_transform = quantity_transform

    def _transform_quantity(self, quantity: float) -> float:
        """
        Apply chosen transformation to purchase quantity.
        
        Args:
            quantity: Raw purchase quantity value
            
        Returns:
            Transformed quantity value based on selected transform method
        """
        if not self.use_quantity:
            return 1.0  # Binary interaction
        
        if self.quantity_transform == "log":
            return math.log1p(quantity)  # log(1 + quantity) to handle quantity=0
        elif self.quantity_transform == "raw":
            return float(quantity)
        elif self.quantity_transform == "binary":
            return 1.0 if quantity > 0 else 0.0
        else:
            return 1.0  # Default fallback

    def process_interactions(self, 
                           interactions: List[Tuple[str, str, str, float]], 
                           item_mapping: Dict[str, int] = None):
        """
        Process user-item purchase interactions and build interaction matrix
        
        Args:
            interactions: List of (user_id, item_id, timestamp, quantity) tuples
            item_mapping: Dictionary mapping item IDs to indices
        """
        print("\nProcessing user-item purchase interactions...")
        self.item_to_idx = item_mapping
        
        # First pass: Get unique users and build user mapping
        users = sorted(set(user_id for user_id, _, _, _ in interactions))
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.item_to_idx)
        
        print(f"Found {n_users} users and {n_items} items")
        
        # Build sparse interaction matrix
        # Using lil_matrix for efficient matrix construction
        self.interaction_matrix = lil_matrix((n_items, n_users), dtype=np.float32)
        
        # Group purchases by user-item pairs to combine quantities
        user_item_quantities = defaultdict(float)
        for user_id, item_id, _, quantity in interactions:
            if item_id in self.item_to_idx:
                key = (self.item_to_idx[item_id], user_to_idx[user_id])
                # Accumulate transformed quantities for user-item pairs
                user_item_quantities[key] += self._transform_quantity(quantity)
        
        # Fill interaction matrix
        for (item_idx, user_idx), quantity in user_item_quantities.items():
            self.interaction_matrix[item_idx, user_idx] = quantity
        
        # Convert to CSR format for efficient computations
        self.interaction_matrix = self.interaction_matrix.tocsr()
        print(f"Built interaction matrix with {self.interaction_matrix.nnz} non-zero entries")

    def compute_collaborative_relationships(self, matrix_size: int) -> np.ndarray:
        """
        Compute collaborative relationships using normalized co-occurrence
        with quantity-weighted interactions.
        
        Args:
            matrix_size: Size of the output matrix (should match item_to_idx length)
            
        Returns:
            Square matrix of collaborative relationships between items
        """
        print("\nComputing purchase-based collaborative relationships...")
        interaction_matrix = self.interaction_matrix.toarray()
        
        # Normalize by user activity (sum of transformed quantities)
        user_activity = np.sum(interaction_matrix, axis=0)
        user_activity[user_activity == 0] = 1  # Avoid division by zero
        
        normalized = interaction_matrix / np.sqrt(user_activity)
        
        # Compute normalized co-occurrence
        collaborative_matrix = normalized @ normalized.T
        np.fill_diagonal(collaborative_matrix, 0)  # Zero out self-similarities
        
        return collaborative_matrix

    def get_item_co_occurrences(self, item_id: str) -> Dict[str, float]:
        """
        Get weighted co-occurrence patterns for an item
        
        Args:
            item_id: ID of item to get co-occurrences for
            
        Returns:
            Dictionary mapping item IDs to co-occurrence scores
        """
        if self.interaction_matrix is None or item_id not in self.item_to_idx:
            return {}
            
        item_idx = self.item_to_idx[item_id]
        item_vector = self.interaction_matrix[item_idx].toarray().flatten()
        
        co_occurrences = {}
        for other_id, other_idx in self.item_to_idx.items():
            if other_id != item_id:
                other_vector = self.interaction_matrix[other_idx].toarray().flatten()
                # Use element-wise minimum for weighted co-occurrence
                overlap = np.sum(np.minimum(item_vector, other_vector))
                if overlap > 0:
                    co_occurrences[other_id] = float(overlap)
                    
        return co_occurrences

# Example usage
def main():
    # Example purchase data
    # Each tuple is (user_id, item_id, timestamp, quantity)
    purchases = [
        ("user1", "item1", "2024-01-01", 2),
        ("user1", "item2", "2024-01-02", 1),
        ("user2", "item1", "2024-01-01", 5),
        ("user2", "item3", "2024-01-03", 1),
        ("user3", "item2", "2024-01-02", 2),
        ("user3", "item3", "2024-01-03", 1),
        ("user4", "item1", "2024-01-01", 1),
        ("user4", "item2", "2024-01-02", 2),
    ]
    
    # Initialize processor with log transform
    processor = PurchaseCollaborativeProcessor(
        use_quantity=True,
        quantity_transform="log"
    )
    
    # Create item mapping
    items = sorted(set(item_id for _, item_id, _, _ in purchases))
    item_mapping = {item: idx for idx, item in enumerate(items)}
    
    # Process interactions
    processor.process_interactions(purchases, item_mapping)
    
    # Compute collaborative relationships
    collab_matrix = processor.compute_collaborative_relationships(
        matrix_size=len(processor.item_to_idx)
    )
    
    # Example outputs
    print("\nCollaborative Relationship Matrix (purchase-based):")
    print(pd.DataFrame(
        collab_matrix,
        index=list(processor.item_to_idx.keys()),
        columns=list(processor.item_to_idx.keys())
    ))
    
    print("\nCo-occurrence patterns for item1:")
    co_occurrences = processor.get_item_co_occurrences("item1")
    for item_id, score in co_occurrences.items():
        print(f"With {item_id}: {score:.2f} weighted co-purchases")

if __name__ == "__main__":
    main()