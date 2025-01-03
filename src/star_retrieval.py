from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from tqdm import tqdm

class STARRetrieval:
    def __init__(self, 
                 semantic_weight: float = 0.5,    # a parameter from paper
                 temporal_decay: float = 0.7,     # λ parameter 
                 history_length: int = 3):        # l parameter
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length
        
        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}

    def compute_semantic_relationships(self, 
                                    item_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute semantic similarity matrix from item embeddings
        Uses proper normalization as specified in paper
        """
        print("\nComputing semantic relationships...")
        
        # Create item index mapping
        sorted_items = sorted(item_embeddings.keys())
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_items = len(self.item_to_idx)
        semantic_matrix = np.zeros((n_items, n_items))
        
        # Convert embeddings to array and normalize
        embeddings_array = np.zeros((n_items, next(iter(item_embeddings.values())).shape[0]))
        for item_id, embedding in item_embeddings.items():
            embeddings_array[self.item_to_idx[item_id]] = embedding
            
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_array = embeddings_array / norms
        
        # Compute similarities using matrix operations for efficiency
        print("Computing semantic similarities...")
        semantic_matrix = np.dot(embeddings_array, embeddings_array.T)
        
        # Ensure values are between 0 and 1
        semantic_matrix = np.maximum(0, semantic_matrix)
        np.fill_diagonal(semantic_matrix, 0)  # Set self-similarity to 0
        
        self.semantic_matrix = semantic_matrix
        return semantic_matrix
        
    def compute_collaborative_relationships(self, 
                                       user_item_interactions: List[Tuple[str, str, float, str]]) -> np.ndarray:
        """
        Compute collaborative relationship matrix using normalized co-occurrence
        """
        print("\nComputing collaborative relationships...")
        n_items = len(self.item_to_idx)
        
        # Create user-item interaction matrix
        users = sorted(set(u for u, _, _, _ in user_item_interactions))
        user_to_idx = {u: i for i, u in enumerate(users)}
        n_users = len(users)
        
        # Create sparse interaction matrix for memory efficiency
        interaction_matrix = csr_matrix((n_items, n_users), dtype=np.float32)
        
        # Fill interaction matrix
        rows, cols = [], []
        for user_id, item_id, _, _ in user_item_interactions:
            if item_id in self.item_to_idx:
                rows.append(self.item_to_idx[item_id])
                cols.append(user_to_idx[user_id])
        
        interaction_matrix = csr_matrix(
            ([1.0] * len(rows), (rows, cols)),
            shape=(n_items, n_users)
        )
        
        # Convert to dense for normalization
        interaction_matrix = interaction_matrix.toarray()
        
        # Normalize by user activity as specified in paper
        user_activity = np.sum(interaction_matrix, axis=0, keepdims=True)
        user_activity[user_activity == 0] = 1  # Avoid division by zero
        normalized_interactions = interaction_matrix / np.sqrt(user_activity)
        
        # Compute normalized co-occurrence similarities
        print("Computing collaborative similarities...")
        collaborative_matrix = np.dot(normalized_interactions, normalized_interactions.T)
        
        # Ensure values are between 0 and 1
        collaborative_matrix = np.maximum(0, collaborative_matrix)
        np.fill_diagonal(collaborative_matrix, 0)  # Set self-similarity to 0
        
        self.collaborative_matrix = collaborative_matrix
        return collaborative_matrix

    def score_candidates(self,
                        user_history: List[str],
                        ratings: List[float],
                        candidate_items: List[str] = None,
                        top_k: int = None) -> List[Tuple[str, float]]:
        """
        Score candidate items based on user history following paper's formula:
        score(x) = (1/n) * sum(rj * λ^tj * [a * Rs_xj + (1-a) * Rc_xj])
        
        Args:
            user_history: List of item IDs in user's history
            ratings: List of ratings corresponding to history items
            candidate_items: List of candidate items to score (optional)
            top_k: Number of top items to return (optional)
            
        Returns:
            List of (item_id, score) tuples, sorted by score
        """
        # Use only last l items from history
        if len(user_history) > self.history_length:
            user_history = user_history[-self.history_length:]
            ratings = ratings[-self.history_length:]
        
        # If no candidates provided, use all items except history
        if candidate_items is None:
            candidate_items = [
                item for item in self.item_to_idx.keys()
                if item not in set(user_history)
            ]
        
        scores = {}
        n = len(user_history)  # Length for normalization
        
        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue
                
            cand_idx = self.item_to_idx[candidate]
            score = 0.0
            
            # Score against each history item with temporal decay
            for t, (hist_item, rating) in enumerate(zip(reversed(user_history), reversed(ratings))):
                if hist_item not in self.item_to_idx:
                    continue
                    
                hist_idx = self.item_to_idx[hist_item]
                
                # Get similarities
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]
                
                # Combine similarities with weighting
                combined_sim = (
                    self.semantic_weight * sem_sim + 
                    (1 - self.semantic_weight) * collab_sim
                )
                
                # Paper's formula: score(x) = (1/n) * sum(rj * λ^tj * [a * Rs_xj + (1-a) * Rc_xj])
                score += (1/n) * rating * (self.temporal_decay ** t) * combined_sim
            
            scores[candidate] = score
        
        # Sort and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_items = sorted_items[:top_k]
            
        return sorted_items
    def get_similar_items(self, item_id: str, k: int = 10, use_collaborative: bool = True) -> List[Tuple[str, float]]:
        """Get most similar items using combined similarity"""
        if item_id not in self.item_to_idx:
            return []
            
        idx = self.item_to_idx[item_id]
        similarities = np.zeros(len(self.item_to_idx))
        
        for j in range(len(self.item_to_idx)):
            if j != idx:
                sem_sim = self.semantic_matrix[idx, j]
                collab_sim = self.collaborative_matrix[idx, j] if use_collaborative else 0
                similarities[j] = (
                    self.semantic_weight * sem_sim + 
                    (1 - self.semantic_weight) * collab_sim
                )
        
        # Get top-k similar items
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            (self.idx_to_item[j], float(similarities[j]))
            for j in top_indices
        ]


