from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple

class CollaborativeRelationshipProcessor:
    def __init__(self):
        self.user_item_interactions = {}
        self.item_to_idx = None
        self.interaction_matrix = None

    def process_interactions(self, interactions, item_mapping=None):
        """
        Process user-item interactions and build interaction matrix
        
        Args:
            interactions: List of (user_id, item_id, timestamp, rating) tuples
            item_mapping: Dictionary mapping item IDs to indices
        """
        print("\nProcessing user-item interactions...")
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
        
        # Fill interaction matrix
        for user_id, item_id, _, _ in interactions:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                user_idx = user_to_idx[user_id]
                self.interaction_matrix[item_idx, user_idx] = 1.0
        
        # Convert to CSR format for efficient computations
        self.interaction_matrix = self.interaction_matrix.tocsr()
        print(f"Built interaction matrix with {self.interaction_matrix.nnz} non-zero entries")

    def compute_collaborative_relationships(self, matrix_size):
        """
        Compute collaborative relationships using normalized co-occurrence
        following paper's specification.
        """
        interaction_matrix = self.interaction_matrix.toarray()
        # Normalize only by user activity per paper
        user_activity = np.sum(interaction_matrix, axis=0)
        user_activity[user_activity == 0] = 1
        normalized = interaction_matrix / np.sqrt(user_activity)
        
        # Compute normalized co-occurrence
        collaborative_matrix = normalized @ normalized.T
        np.fill_diagonal(collaborative_matrix, 0)  # Zero out self-similarities
        
        return collaborative_matrix

    def get_item_co_occurrences(self, item_id: str) -> Dict[str, int]:
        """
        Get raw co-occurrence counts for an item
        Useful for debugging and verification
        
        Args:
            item_id: ID of item to get co-occurrences for
            
        Returns:
            Dictionary mapping item IDs to co-occurrence counts
        """
        if self.interaction_matrix is None or item_id not in self.item_to_idx:
            return {}
            
        item_idx = self.item_to_idx[item_id]
        item_vector = self.interaction_matrix[item_idx].toarray().flatten()
        
        co_occurrences = {}
        for other_id, other_idx in self.item_to_idx.items():
            if other_id != item_id:
                other_vector = self.interaction_matrix[other_idx].toarray().flatten()
                count = np.sum(np.logical_and(item_vector > 0, other_vector > 0))
                if count > 0:
                    co_occurrences[other_id] = int(count)
                    
        return co_occurrences
# Example usage
def main():
    # Example interaction data
    # Each tuple is (user_id, item_id, timestamp, rating)
    interactions = [
        ("user1", "item1", "2024-01-01", 5),
        ("user1", "item2", "2024-01-02", 4),
        ("user2", "item1", "2024-01-01", 3),
        ("user2", "item3", "2024-01-03", 5),
        ("user3", "item2", "2024-01-02", 4),
        ("user3", "item3", "2024-01-03", 5),
        ("user4", "item1", "2024-01-01", 4),
        ("user4", "item2", "2024-01-02", 5),
    ]
    
    # Initialize processor
    processor = CollaborativeRelationshipProcessor()
    
    # Process interactions
    processor.process_interactions(interactions)
    
    # Compute collaborative relationships
    collab_matrix = processor.compute_collaborative_relationships(
        matrix_size=len(processor.item_to_idx)
    )
    
    # Example outputs
    print("\nCollaborative Relationship Matrix:")
    print(pd.DataFrame(
        collab_matrix,
        index=list(processor.item_to_idx.keys()),
        columns=list(processor.item_to_idx.keys())
    ))
    
    print("\nCo-occurrence counts for item1:")
    co_occurrences = processor.get_item_co_occurrences("item1")
    for item_id, count in co_occurrences.items():
        print(f"With {item_id}: {count} users")

if __name__ == "__main__":
    main()