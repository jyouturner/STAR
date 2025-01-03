import numpy as np
from typing import Any, List, Dict, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

class RecommendationEvaluator:
    def __init__(self):
        """Initialize evaluation metrics"""
        self.all_items = set()
        
    def calculate_hits_at_k(self, recommended_items: List[str], 
                          ground_truth: str, 
                          k: int) -> float:
        """Calculate Hits@K metric"""
        return 1.0 if ground_truth in recommended_items[:k] else 0.0

    def calculate_ndcg_at_k(self, recommended_items: List[str], 
                           ground_truth: str, 
                           k: int) -> float:
        """Calculate NDCG@K metric"""
        if ground_truth not in recommended_items[:k]:
            return 0.0
        
        rank = recommended_items[:k].index(ground_truth)
        return 1.0 / np.log2(rank + 2)  # +2 because index starts at 0

    def evaluate_recommendations(
        self,
        test_sequences,
        recommender,
        k_values: List[int],
        n_negative_samples: int = 99  # Paper uses 99 negative samples
    ) -> Dict[str, float]:
        """
        Evaluate recommendations using the paper's protocol
        """
        print("\n=== Starting Evaluation ===")
        metrics = {f"hit@{k}": 0.0 for k in k_values}
        metrics.update({f"ndcg@{k}": 0.0 for k in k_values})
        
        # Get all valid items for negative sampling
        valid_items = list(recommender.item_to_idx.keys())
        total_sequences = len(test_sequences)
        
        if total_sequences == 0:
            print("Warning: No test sequences to evaluate!")
            return metrics

        successful_preds = 0
        
        # For each test sequence
        for idx, (user_id, history, next_item) in enumerate(tqdm(test_sequences, desc="Evaluating")):
            if not history or next_item not in recommender.item_to_idx:
                continue
                
            # Verify history items are in vocabulary
            valid_history = [item for item in history if item in recommender.item_to_idx]
            if not valid_history:
                continue
            
            # Sample negative items (excluding history and next item)
            negative_candidates = set()
            excluded_items = set(history) | {next_item}
            
            while len(negative_candidates) < n_negative_samples:
                item = random.choice(valid_items)
                if item not in excluded_items:
                    negative_candidates.add(item)
            
            # Create candidate set with negative samples + positive item
            candidate_items = list(negative_candidates) + [next_item]
            
            # Get recommendations
            recommendations = recommender.score_candidates(
                user_history=valid_history[-recommender.history_length:],  # Use last l items
                ratings=[1.0] * len(valid_history),  # As per paper, ignore ratings
                candidate_items=candidate_items,
                top_k=max(k_values)
            )
            
            if not recommendations:
                continue
                
            successful_preds += 1
            recommended_items = [item for item, _ in recommendations]
            
            # Calculate metrics
            for k in k_values:
                metrics[f"hit@{k}"] += self.calculate_hits_at_k(
                    recommended_items, next_item, k)
                metrics[f"ndcg@{k}"] += self.calculate_ndcg_at_k(
                    recommended_items, next_item, k)
        
        # Normalize metrics
        if successful_preds > 0:
            for metric in metrics:
                metrics[metric] /= successful_preds
        
        print(f"\nSuccessfully evaluated {successful_preds}/{total_sequences} sequences")
        return metrics


def prepare_evaluation_data(interactions):
    """
    Prepare evaluation data following paper's protocol:
    - Use last l items as history (where l is history_length)
    - Use next item as target
    - Ensure chronological ordering
    """
    user_sequences = defaultdict(list)
    
    # Sort by user and time
    for user_id, item_id, timestamp, _ in sorted(interactions, key=lambda x: (x[0], x[2])):
        user_sequences[user_id].append((item_id, timestamp))
    
    test_sequences = []
    min_history = 3  # Paper uses l=3 for history length
    
    for user_id, interactions in user_sequences.items():
        if len(interactions) >= min_history + 1:  # Need history + test item
            # Sort by time
            sorted_items = [item for item, _ in sorted(interactions, key=lambda x: x[1])]
            
            # Use last min_history items as history
            history = sorted_items[-(min_history+1):-1]  # Last l items before test
            test_item = sorted_items[-1]  # Last item
            
            test_sequences.append((user_id, history, test_item))
    
    return test_sequences