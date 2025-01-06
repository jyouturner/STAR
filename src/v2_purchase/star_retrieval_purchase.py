from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from datetime import datetime
import math

class PurchaseSTARRetrieval:
    def __init__(self, 
                 semantic_weight: float = 0.5,
                 temporal_decay: float = 0.7,
                 time_scale_days: float = 30.0,
                 history_length: int = 3,
                 use_quantity: bool = True,
                 quantity_transform: str = "log"):
        """
        Initialize STAR retrieval for purchase data with date-based decay
        
        Args:
            semantic_weight: Weight factor α balancing semantic vs collaborative signals
            temporal_decay: Base decay factor λ (e.g. 0.7 means 30% decay per time_scale)
            time_scale_days: Number of days representing one decay step (e.g. 30 for monthly)
            history_length: Maximum number of history items to consider
            use_quantity: Whether to incorporate purchase quantities
            quantity_transform: Method to transform quantities ("log", "raw", "binary")
        """
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.time_scale_days = time_scale_days
        self.history_length = history_length
        self.use_quantity = use_quantity
        self.quantity_transform = quantity_transform
        
        # These will be set from the relationship processor
        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = None
        
    def _parse_timestamp(self, timestamp: Union[str, float, int]) -> float:
        """Convert timestamp to Unix timestamp (seconds since epoch)"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d")
            return dt.timestamp()
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
            except ValueError:
                raise ValueError(f"Unsupported timestamp format: {timestamp}")

    def _transform_quantity(self, quantity: float, clip_max: float = 100.0) -> float:
        """
        Transform purchase quantity with safeguards against extreme values
        
        Args:
            quantity: Raw purchase quantity
            clip_max: Maximum value to clip quantity to before transform
            
        Returns:
            Transformed quantity value
        """
        if not self.use_quantity:
            return 1.0
            
        # Clip extreme quantities
        quantity = min(float(quantity), clip_max)
        
        if self.quantity_transform == "log":
            # log1p for smooth handling of small quantities
            return math.log1p(quantity)
        elif self.quantity_transform == "raw":
            # Linear scaling but clipped
            return quantity
        elif self.quantity_transform == "binary":
            # Simple presence/absence
            return 1.0 if quantity > 0 else 0.0
        else:
            return 1.0

    def _compute_time_decay(self, 
                           hist_time: Union[str, float, int],
                           current_time: Union[str, float, int],
                           min_decay: float = 1e-6) -> float:
        """
        Compute time-based decay factor between timestamps
        
        Args:
            hist_time: Historical purchase timestamp
            current_time: Reference timestamp (usually current time or test item time)
            min_decay: Minimum decay factor to prevent complete zeroing
            
        Returns:
            Decay factor between 0 and 1
        """
        # Convert timestamps to Unix time
        t_hist = self._parse_timestamp(hist_time)
        t_curr = self._parse_timestamp(current_time)
        
        # Get time difference in days
        gap_days = max(0.0, (t_curr - t_hist) / (24 * 3600))
        
        # Compute decay exponent (gap_days / scale)
        exponent = gap_days / self.time_scale_days
        
        # Apply decay with floor
        decay = max(min_decay, self.temporal_decay ** exponent)
        
        return decay

    def score_candidates(self,
                        user_history: List[Tuple[str, Union[str, float, int], float]],
                        current_time: Optional[Union[str, float, int]] = None,
                        candidate_items: Optional[List[str]] = None,
                        top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Score candidate items using purchase history with time decay
        
        Args:
            user_history: List of (item_id, timestamp, quantity) tuples
            current_time: Reference time for decay (if None, use latest purchase)
            candidate_items: Items to score (if None, use all except history)
            top_k: Number of top items to return
            
        Returns:
            List of (item_id, score) tuples sorted by score
        """
        if self.collaborative_matrix is None or self.semantic_matrix is None:
            raise ValueError("Matrices not set. Run compute_relationships first.")
        
        if not user_history:
            return []
            
        # If no current_time provided, use latest purchase time
        if current_time is None:
            current_time = max(t for _, t, _ in user_history)
        
        # Sort history by timestamp (most recent first) and trim
        sorted_history = sorted(
            user_history,
            key=lambda x: self._parse_timestamp(x[1]),
            reverse=True
        )[:self.history_length]
        
        # Get candidate items if not provided
        if candidate_items is None:
            history_items = {item for item, _, _ in sorted_history}
            candidate_items = [item for item in self.item_to_idx.keys() 
                             if item not in history_items]
        
        scores: Dict[str, float] = {}
        n = len(sorted_history)  # For averaging
        
        # Score each candidate
        for candidate in candidate_items:
            if candidate not in self.item_to_idx:
                continue
                
            cand_idx = self.item_to_idx[candidate]
            score = 0.0
            
            for hist_item, hist_time, hist_quantity in sorted_history:
                if hist_item not in self.item_to_idx:
                    continue
                    
                hist_idx = self.item_to_idx[hist_item]
                
                # Get similarities from matrices
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]
                
                # Weighted combination
                combined_sim = (
                    self.semantic_weight * sem_sim + 
                    (1 - self.semantic_weight) * collab_sim
                )
                
                # Get temporal decay
                decay = self._compute_time_decay(hist_time, current_time)
                
                # Get quantity weight
                quantity_weight = self._transform_quantity(hist_quantity)
                
                # Add to score with average factor
                score += (1 / n) * quantity_weight * decay * combined_sim
            
            if score > 0:
                scores[candidate] = score
        
        # Sort by score and optionally trim to top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_items = sorted_items[:top_k]
            
        return sorted_items

# Example usage
def main():
    # Example purchase history data (item_id, timestamp, quantity)
    user_history = [
        ("item1", "2024-01-01", 2),
        ("item2", "2024-02-15", 5),
        ("item3", "2024-03-01", 1),
    ]
    
    # Initialize retrieval
    retrieval = PurchaseSTARRetrieval(
        semantic_weight=0.5,
        temporal_decay=0.7,
        time_scale_days=30.0,
        history_length=3,
        use_quantity=True,
        quantity_transform="log"
    )
    
    # Set up dummy matrices for testing
    n_items = 5
    retrieval.item_to_idx = {f"item{i}": i for i in range(1, n_items + 1)}
    retrieval.semantic_matrix = np.random.random((n_items, n_items))
    retrieval.collaborative_matrix = np.random.random((n_items, n_items))
    
    # Get recommendations
    recommendations = retrieval.score_candidates(
        user_history=user_history,
        current_time="2024-03-15",
        top_k=3
    )
    
    # Print results
    print("\nTop recommendations:")
    for item_id, score in recommendations:
        print(f"{item_id}: {score:.4f}")

if __name__ == "__main__":
    main()