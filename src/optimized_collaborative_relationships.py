from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import itertools

class OptimizedCollaborativeProcessor:
    def __init__(self, chunk_size=1000, n_threads=4):
        self.user_item_interactions = defaultdict(list)
        self.item_to_idx = None
        self.chunk_size = chunk_size
        self.n_threads = n_threads
        
    def process_interactions(self, interactions, item_mapping=None):
        """Process user-item interactions with optimized storage"""
        self.item_to_idx = item_mapping or {}
        self.user_item_interactions.clear()
        
        # Use numpy arrays for faster processing
        interaction_data = np.array([(user_id, item_id, timestamp, rating) 
                                   for user_id, item_id, timestamp, rating in interactions
                                   if item_id in self.item_to_idx], 
                                  dtype=[('user_id', 'U50'), ('item_id', 'U50'), 
                                       ('timestamp', 'U50'), ('rating', 'f4')])
        
        # Group by user_id efficiently using numpy operations
        user_ids, indices = np.unique(interaction_data['user_id'], return_index=True)
        sorted_indices = np.argsort(interaction_data['user_id'])
        
        for i in range(len(user_ids)):
            start_idx = indices[i]
            end_idx = indices[i + 1] if i < len(user_ids) - 1 else len(interaction_data)
            user_data = interaction_data[start_idx:end_idx]
            self.user_item_interactions[user_ids[i]] = list(zip(user_data['item_id'], 
                                                              user_data['timestamp'],
                                                              user_data['rating']))

    def _process_user_chunk(self, user_chunk: List[str]) -> lil_matrix:
        """Process a chunk of users to compute partial relationship matrix"""
        matrix_size = len(self.item_to_idx)
        chunk_matrix = lil_matrix((matrix_size, matrix_size))
        
        for user_id in user_chunk:
            items = [item_id for item_id, _, _ in self.user_item_interactions[user_id]]
            if len(items) > 1:
                # Use numpy operations for faster computation
                item_indices = np.array([self.item_to_idx[item] for item in items])
                combinations = np.array(list(itertools.combinations(item_indices, 2)))
                
                if len(combinations) > 0:
                    # Batch update the matrix
                    chunk_matrix[combinations[:, 0], combinations[:, 1]] += 1
                    chunk_matrix[combinations[:, 1], combinations[:, 0]] += 1
        
        return chunk_matrix

    def compute_collaborative_relationships(self, matrix_size):
        """Compute collaborative relationships with parallel processing"""
        # Split users into chunks for parallel processing
        users = list(self.user_item_interactions.keys())
        user_chunks = [users[i:i + self.chunk_size] 
                      for i in range(0, len(users), self.chunk_size)]
        
        # Process chunks in parallel
        partial_matrices = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            partial_matrices = list(executor.map(self._process_user_chunk, user_chunks))
        
        # Combine partial matrices efficiently
        final_matrix = sum(partial_matrices)
        
        # Convert to CSR format and normalize
        final_matrix = final_matrix.tocsr()
        
        # Vectorized normalization
        row_sums = np.asarray(final_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        diag = 1 / row_sums
        final_matrix = final_matrix.multiply(diag[:, np.newaxis])
        
        return final_matrix

    def get_item_co_occurrences(self, item_id: str, min_count: int = 1) -> Dict[str, int]:
        """
        Get number of users who interacted with both items, with minimum threshold
        
        Args:
            item_id: ID of item to get co-occurrences for
            min_count: Minimum co-occurrence count to include
            
        Returns:
            Dictionary mapping item IDs to co-occurrence counts
        """
        if item_id not in self.item_to_idx:
            return {}
            
        item_idx = self.item_to_idx[item_id]
        
        # Use CSR matrix properties for efficient row/column operations
        if not hasattr(self, 'interaction_matrix'):
            return {}
            
        item_vector = self.interaction_matrix[item_idx].toarray().flatten()
        
        # Vectorized computation of co-occurrences
        co_occurrence_vector = self.interaction_matrix.multiply(item_vector).sum(axis=1).A1
        
        # Filter and convert to dictionary
        co_occurrences = {}
        for other_id, other_idx in self.item_to_idx.items():
            if other_id != item_id and co_occurrence_vector[other_idx] >= min_count:
                co_occurrences[other_id] = int(co_occurrence_vector[other_idx])
                    
        return co_occurrences

# Example usage with performance monitoring
def main():
    import time
    
    # Generate larger test dataset
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    n_interactions = 10000
    
    users = [f"user{i}" for i in range(n_users)]
    items = [f"item{i}" for i in range(n_items)]
    item_mapping = {item: idx for idx, item in enumerate(items)}
    
    interactions = [
        (np.random.choice(users),
         np.random.choice(items),
         f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
         np.random.randint(1, 6))
        for _ in range(n_interactions)
    ]
    
    # Initialize processor with optimized parameters
    processor = OptimizedCollaborativeProcessor(chunk_size=100, n_threads=4)
    
    # Measure processing time
    start_time = time.time()
    processor.process_interactions(interactions, item_mapping)
    process_time = time.time() - start_time
    print(f"Processing time: {process_time:.2f} seconds")
    
    # Measure relationship computation time
    start_time = time.time()
    collab_matrix = processor.compute_collaborative_relationships(
        matrix_size=len(processor.item_to_idx)
    )
    compute_time = time.time() - start_time
    print(f"Computation time: {compute_time:.2f} seconds")
    
    # Example co-occurrence lookup
    sample_item = items[0]
    start_time = time.time()
    co_occurrences = processor.get_item_co_occurrences(sample_item, min_count=2)
    lookup_time = time.time() - start_time
    print(f"Lookup time: {lookup_time:.2f} seconds")
    print(f"Number of co-occurrences found: {len(co_occurrences)}")

if __name__ == "__main__":
    main()