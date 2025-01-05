from collections import defaultdict
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import numpy as np

def build_user_all_items(interactions: List[Tuple]) -> Dict[str, set]:
    """Build dictionary mapping user_id -> set of all item_ids they've interacted with"""
    user_all_items = defaultdict(set)
    for user_id, item_id, timestamp, rating in interactions:
        user_all_items[user_id].add(item_id)
    return user_all_items

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
        return 1.0 / np.log2(rank + 2)

    def evaluate_recommendations(
        self,
        test_sequences: List[Tuple],
        recommender,
        k_values: List[int],
        user_all_items: Dict[str, set] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations using full item set as candidates
        (matching paper's protocol)
        """
        print("\n=== Starting Evaluation ===")
        metrics = {f"hit@{k}": 0.0 for k in k_values}
        metrics.update({f"ndcg@{k}": 0.0 for k in k_values})
        
        # Get all possible items
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

            # Get all items this user hasn't interacted with
            user_items = user_all_items.get(user_id, set()) if user_all_items else set(history)
            candidate_items = [item for item in valid_items if item not in user_items]
            
            # Add ground truth item
            if next_item not in candidate_items:
                candidate_items.append(next_item)
            
            # Get recommendations using all candidates
            recommendations = recommender.score_candidates(
                user_history=valid_history[-recommender.history_length:],
                ratings=[1.0] * len(valid_history),  # Per paper, ignore ratings
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
            
            # Print some stats every 1000 sequences
            if idx % 1000 == 0:
                print(f"\nIntermediate stats at sequence {idx}:")
                print(f"Average candidates per user: {len(candidate_items)}")
                print(f"Current metrics:")
                for metric, value in metrics.items():
                    normalized_value = value / successful_preds if successful_preds > 0 else 0
                    print(f"{metric}: {normalized_value:.4f}")
        
        # Normalize metrics
        if successful_preds > 0:
            for metric in metrics:
                metrics[metric] /= successful_preds
        
        print(f"\nSuccessfully evaluated {successful_preds}/{total_sequences} sequences")
        return metrics


def prepare_evaluation_data(interactions: List[Tuple], min_sequence_length: int = 5) -> List[Tuple]:
    """
    Prepare evaluation data following paper's protocol:
    - Maintain strict temporal ordering
    - Use chronologically last item as test item
    - Previous items as history
    - Minimum sequence length requirement
    """
    # Group interactions by user while maintaining temporal order
    user_sequences = defaultdict(list)
    for user_id, item_id, timestamp, rating in interactions:
        user_sequences[user_id].append((item_id, timestamp, rating))
    
    test_sequences = []
    skipped_users = 0
    timestamp_issues = 0
    
    for user_id, interactions in user_sequences.items():
        # Sort user's interactions by timestamp
        sorted_items = sorted(interactions, key=lambda x: x[1])
        
        # Skip if sequence is too short
        if len(sorted_items) < min_sequence_length:
            skipped_users += 1
            continue
        
        # Check for duplicate timestamps
        timestamps = [t for _, t, _ in sorted_items]
        if len(set(timestamps)) != len(timestamps):
            timestamp_issues += 1
            
        # Extract items in temporal order
        items = [item for item, _, _ in sorted_items]
        
        # Last item is test item
        test_item = items[-1]
        # Previous items are history
        history = items[:-1]
        
        test_sequences.append((user_id, history, test_item))
    
    print(f"\nEvaluation data preparation:")
    print(f"Total users processed: {len(user_sequences)}")
    print(f"Users skipped (too short): {skipped_users}")
    print(f"Users with timestamp issues: {timestamp_issues}")
    print(f"Final test sequences: {len(test_sequences)}")
    
    return test_sequences

def prepare_validation_data(interactions: List[Tuple], min_sequence_length: int = 5) -> List[Tuple]:
    """Prepare validation data using second-to-last item"""
    user_sequences = defaultdict(list)
    for user_id, item_id, timestamp, rating in interactions:
        user_sequences[user_id].append((item_id, timestamp, rating))
    
    validation_sequences = []
    skipped_users = 0
    
    for user_id, interactions in user_sequences.items():
        # Sort by timestamp
        sorted_items = sorted(interactions, key=lambda x: x[1])
        
        # Skip if too short
        if len(sorted_items) < min_sequence_length:
            skipped_users += 1
            continue
        
        # Extract items in temporal order
        items = [item for item, _, _ in sorted_items]
        
        # Second-to-last item is validation item
        validation_item = items[-2]
        # Previous items are history (excluding last and validation items)
        history = items[:-2]
        
        validation_sequences.append((user_id, history, validation_item))
    
    print(f"\nValidation data preparation:")
    print(f"Total users processed: {len(user_sequences)}")
    print(f"Users skipped (too short): {skipped_users}")
    print(f"Final validation sequences: {len(validation_sequences)}")
    
    return validation_sequences