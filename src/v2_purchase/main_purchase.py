from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

from data_loading import get_items_from_purchases, load_item_metadata, load_purchase_dataset
from ..item_embeddings_vertex_ai import ItemEmbeddingGenerator
from star_retrieval_purchase import PurchaseSTARRetrieval
from collaborative_relationships_purchase import PurchaseCollaborativeProcessor
from evaluation_metrics_purchase import (
    PurchaseRecommendationEvaluator,
    prepare_purchase_evaluation_data,
    build_user_purchase_items
)
from ..data_debug import DataDebugger
from ..temporal_utils import TemporalProcessor
from datetime import datetime


def get_training_interactions(purchases: List[Dict]) -> List[Tuple[str, str, str, float]]:
    """
    Extract training interactions from purchase data
    Returns list of (user_id, item_id, timestamp, quantity) tuples
    """
    # Sort purchases by timestamp
    sorted_purchases = sorted(
        purchases,
        key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d")
    )
    
    user_purchases = {}
    for purchase in sorted_purchases:
        user_purchases.setdefault(purchase['user_id'], []).append(purchase)
    
    training_interactions = []
    for user_id, user_history in user_purchases.items():
        if len(user_history) >= 5:  # Min sequence length check
            test_time = user_history[-1]['timestamp']
            train_history = user_history[:-2]  # Exclude test and validation items
            
            for purchase in train_history:
                if purchase['timestamp'] < test_time:
                    training_interactions.append((
                        purchase['user_id'],
                        purchase['item_id'],
                        purchase['timestamp'],
                        purchase['quantity']
                    ))
    
    return training_interactions

def main():
    # Load purchase data
    print("Loading purchase data...")
    purchases = load_purchase_dataset(
        filepath="data/purchases.csv",
        min_interactions=5
    )

    metadata = load_item_metadata("data/item_metadata.csv")
    items = get_items_from_purchases(purchases, metadata)
    metadata = {}  # Load your item metadata if available
    
    # Initialize temporal processor
    temporal_processor = TemporalProcessor()
    
    # Check purchase chronology
    temporal_processor.print_chronology_check(purchases)
    
    # Sort purchases chronologically
    purchases = temporal_processor.sort_reviews_chronologically(purchases)
    
    # Initialize debugger
    debugger = DataDebugger()
    
    # Debug purchase histories
    debugger.debug_print_user_history(purchases)
    
    # Check for duplicates
    debugger.check_for_duplicates(purchases)
    
    # Process items
    items = get_items_from_purchases(purchases, metadata)
    
    # Generate embeddings for items
    print("\nGenerating item embeddings...")
    embedding_generator = ItemEmbeddingGenerator(
        output_dimension=768,
        include_fields={'title', 'description', 'category', 'brand', 'price'}
    )
    
    # Debug prompt for a few items to verify structure
    embedding_generator.debug_prompt(items, num_samples=3)
    
    # Generate embeddings for all items
    embeddings = embedding_generator.generate_item_embeddings(items)
    print(f"Generated embeddings for {len(embeddings)} items")
    
    # Initialize retrieval
    retrieval = PurchaseSTARRetrieval(
        semantic_weight=0.5,
        temporal_decay=0.7,
        time_scale_days=30.0,
        history_length=3,
        use_quantity=True,
        quantity_transform="log"
    )
    
    # Set item mapping and compute semantic relationships
    item_to_idx = {item: idx for idx, item in enumerate(sorted(items.keys()))}
    retrieval.item_to_idx = item_to_idx
    
    # Compute semantic relationships
    print("\nComputing semantic relationships...")
    semantic_matrix = retrieval.compute_semantic_relationships(embeddings)
    retrieval.semantic_matrix = semantic_matrix
    
    # Process purchase interactions
    purchase_interactions = [(
        purchase['user_id'],
        purchase['item_id'],
        purchase['timestamp'],
        purchase['quantity']
    ) for purchase in purchases]
    
    # Build user_all_items for negative sampling
    user_all_items = build_user_purchase_items(purchase_interactions)
    
    # Split data
    train_interactions = get_training_interactions(purchases)
    test_sequences = prepare_purchase_evaluation_data(purchase_interactions)
    
    # Process collaborative relationships
    collab_processor = PurchaseCollaborativeProcessor(
        use_quantity=True,
        quantity_transform="log"
    )
    
    collab_processor.process_interactions(
        train_interactions,
        item_mapping=retrieval.item_to_idx
    )
    
    collaborative_matrix = collab_processor.compute_collaborative_relationships(
        matrix_size=len(retrieval.item_to_idx)
    )
    
    retrieval.collaborative_matrix = collaborative_matrix
    
    # Debug negative sampling
    debugger.debug_negative_sampling(
        test_sequences=test_sequences,
        recommender=retrieval,
        user_all_items=user_all_items
    )
    
    # Debug collaborative matrix
    debugger.analyze_collaborative_matrix(collaborative_matrix)
    
    # Run evaluation
    print("\n=== Running Evaluation ===")
    evaluator = PurchaseRecommendationEvaluator()
    
    # Get current time for evaluation
    current_time = max(purchase['timestamp'] for purchase in purchases)
    
    test_metrics = evaluator.evaluate_recommendations(
        test_sequences=test_sequences,
        recommender=retrieval,
        k_values=[5, 10],
        user_all_items=user_all_items,
        current_time=current_time
    )
    
    # Print results
    print("\nFinal Results:")
    print_metrics_table(test_metrics)

def print_metrics_table(metrics: Dict[str, float]):
    """Print evaluation metrics in a formatted table"""
    print("-" * 30)
    print(f"{'Metric':<15} {'Score':<10}")
    print("-" * 30)
    
    for metric_name in sorted(metrics.keys()):
        score = metrics[metric_name]
        print(f"{metric_name:<15} {score:.4f}")
    
    print("-" * 30)

if __name__ == "__main__":
    main()