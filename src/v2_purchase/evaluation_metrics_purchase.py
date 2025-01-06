from collections import defaultdict
from typing import Dict, List, Tuple, Set
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime

def prepare_purchase_evaluation_data(
    interactions: List[Tuple[str, str, str, float]], 
    min_sequence_length: int = 5
) -> List[Tuple[str, List[Tuple[str, str, float]], Tuple[str, str, float]]]:
    """
    Prepare evaluation data for purchase-based recommendations
    
    Args:
        interactions: List of (user_id, item_id, timestamp, quantity) tuples
        min_sequence_length: Minimum number of purchases required
        
    Returns:
        List of (user_id, history, test_item) tuples where:
            - history is list of (item_id, timestamp, quantity)
            - test_item is (item_id, timestamp, quantity)
    """
    # Group interactions by user while maintaining temporal order
    user_sequences = defaultdict(list)
    for user_id, item_id, timestamp, quantity in interactions:
        user_sequences[user_id].append((item_id, timestamp, quantity))
    
    test_sequences = []
    skipped_users = 0
    timestamp_issues = 0
    
    for user_id, interactions in user_sequences.items():
        # Sort user's interactions by timestamp
        # Handle potential timestamp formats
        def parse_time(t):
            if isinstance(t, (int, float)):
                return float(t)
            try:
                return datetime.strptime(t, "%Y-%m-%d").timestamp()
            except ValueError:
                return datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp()
                
        sorted_items = sorted(interactions, key=lambda x: parse_time(x[1]))
        
        # Skip if sequence is too short
        if len(sorted_items) < min_sequence_length:
            skipped_users += 1
            continue
            
        # Check for duplicate timestamps
        timestamps = [parse_time(t) for _, t, _ in sorted_items]
        if len(set(timestamps)) != len(timestamps):
            timestamp_issues += 1
            
        # Last purchase is test item
        test_item = sorted_items[-1]
        # Previous purchases are history
        history = sorted_items[:-1]
        
        test_sequences.append((user_id, history, test_item))
    
    print(f"\nEvaluation data preparation:")
    print(f"Total users processed: {len(user_sequences)}")
    print(f"Users skipped (too short): {skipped_users}")
    print(f"Users with timestamp issues: {timestamp_issues}")
    print(f"Final test sequences: {len(test_sequences)}")
    
    return test_sequences

def build_user_purchase_items(
    interactions: List[Tuple[str, str, str, float]]
) -> Dict[str, Set[str]]:
    """Build dictionary mapping user_id -> set of all purchased item_ids"""
    user_items = defaultdict(set)
    for user_id, item_id, _, _ in interactions:
        user_items[user_id].add(item_id)
    return user_items

class PurchaseRecommendationEvaluator:
    def __init__(self):
        """Initialize evaluation metrics"""
        self.all_items = set()
        
    def calculate_hits_at_k(self, 
                           recommended_items: List[str], 
                           ground_truth: str, 
                           k: int) -> float:
        """Calculate Hits@K metric"""
        return 1.0 if ground_truth in recommended_items[:k] else 0.0

    def calculate_ndcg_at_k(self, 
                           recommended_items: List[str], 
                           ground_truth: str, 
                           k: int) -> float:
        """Calculate NDCG@K metric"""
        if ground_truth not in recommended_items[:k]:
            return 0.0
            
        rank = recommended_items[:k].index(ground_truth)
        return 1.0 / np.log2(rank + 2)

    def evaluate_recommendations(
        self,
        test_sequences: List[Tuple[str, List[Tuple[str, str, float]], Tuple[str, str, float]]],
        recommender,
        k_values: List[int],
        user_all_items: Dict[str, Set[str]] = None,
        current_time: str = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations using full item set
        
        Args:
            test_sequences: List of (user_id, history, test_item) from prepare_purchase_evaluation_data
            recommender: Object with score_candidates method
            k_values: List of K values for metrics
            user_all_items: Dict mapping users to their purchased items
            current_time: Reference time for scoring (if None, use test item time)
            
        Returns:
            Dictionary of metric scores
        """
        print("\n=== Starting Evaluation ===")
        metrics = {f"hit@{k}": 0.0 for k in k_values}
        metrics.update({f"ndcg@{k}": 0.0 for k in k_values})
        
        valid_items = list(recommender.item_to_idx.keys())
        total_sequences = len(test_sequences)
        
        if total_sequences == 0:
            print("Warning: No test sequences to evaluate!")
            return metrics

        successful_preds = 0
        
        for idx, (user_id, history, test_item) in enumerate(tqdm(test_sequences, desc="Evaluating")):
            test_item_id, test_time, _ = test_item
            
            if not history or test_item_id not in recommender.item_to_idx:
                continue
            
            # Get items the user hasn't purchased yet
            user_items = user_all_items.get(user_id, set()) if user_all_items else set(h[0] for h in history)
            candidate_items = [item for item in valid_items if item not in user_items]
            
            # Add ground truth item
            if test_item_id not in candidate_items:
                candidate_items.append(test_item_id)
            
            # Use test time as reference if no current_time provided
            eval_time = current_time or test_time
            
            # Get recommendations
            recommendations = recommender.score_candidates(
                user_history=history,
                current_time=eval_time,
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
                    recommended_items, test_item_id, k)
                metrics[f"ndcg@{k}"] += self.calculate_ndcg_at_k(
                    recommended_items, test_item_id, k)
        
        # Normalize metrics
        if successful_preds > 0:
            for metric in metrics:
                metrics[metric] /= successful_preds
        
        print(f"\nSuccessfully evaluated {successful_preds}/{total_sequences} sequences")
        return metrics
