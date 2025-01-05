from ast import Dict, Set
import os
from typing import List, Dict, Set
import numpy as np
from pathlib import Path
#from item_embeddings_huggingface import ItemEmbeddingGenerator
from item_embeddings_vertex_ai import ItemEmbeddingGenerator
from star_retrieval import STARRetrieval
from collaborative_relationships import CollaborativeRelationshipProcessor
from evaluation_metrics import RecommendationEvaluator, prepare_evaluation_data, prepare_validation_data, build_user_all_items
from model_analysis import analyze_semantic_matrix, run_full_analysis
from utils import load_amazon_dataset, load_amazon_metadata, get_items_from_data, get_training_interactions, print_metrics_table
from data_quality import DataQualityChecker, verify_item_coverage
from data_debug import DataDebugger
from temporal_utils import TemporalProcessor
from utils import (
    load_amazon_dataset,
    load_amazon_metadata,
    get_items_from_data,
    get_training_interactions,
    print_metrics_table
)

def save_embeddings(embeddings, save_dir='data/embeddings'):
    """Save embeddings to disk"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    items = sorted(embeddings.keys())
    embedding_array = np.stack([embeddings[item] for item in items])
    np.save(f'{save_dir}/embeddings.npy', embedding_array)
    np.save(f'{save_dir}/items.npy', np.array(items))
    print(f"Saved embeddings to {save_dir}")

def load_embeddings(load_dir='data/embeddings'):
    # only for testing and debugging
    return None, None
    """Load embeddings and create item mapping"""
    try:
        embedding_array = np.load(f'{load_dir}/embeddings.npy')
        items = np.load(f'{load_dir}/items.npy')
        embeddings = {item: emb for item, emb in zip(items, embedding_array)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        print(f"Loaded embeddings for {len(embeddings)} items")
        return embeddings, item_to_idx
    except FileNotFoundError:
        print("No saved embeddings found")
        return None, None

def main():
    # Load data
    print("Loading data...")
    reviews = load_amazon_dataset("beauty", min_interactions=5)
    
    # Initialize temporal processor
    temporal_processor = TemporalProcessor()
    
    # Check chronological ordering
    temporal_processor.print_chronology_check(reviews)
    
    # Sort reviews chronologically
    reviews = temporal_processor.sort_reviews_chronologically(reviews)
    metadata = load_amazon_metadata("beauty", min_interactions=5)
    
    # Initialize debugger
    debugger = DataDebugger()
    
    # 1. Check user histories
    debugger.debug_print_user_history(reviews)
    
    # 2. Check for duplicates
    debugger.check_for_duplicates(reviews)
    
    # 3. Analyze ratings
    debugger.analyze_ratings(reviews)
    
    # Process items and prepare data
    items = get_items_from_data(reviews, metadata)
    embeddings, item_to_idx = load_embeddings()
    if embeddings is None:
        # follow the A.1 appendix from the STAR paper
        embedding_generator = ItemEmbeddingGenerator(output_dimension=768, include_fields={'title', 'description', 'category', 'brand', 'price', 'sales_rank'})
        embeddings = embedding_generator.generate_item_embeddings(items)
        #save_embeddings(embeddings)
        item_to_idx = {item: idx for idx, item in enumerate(sorted(embeddings.keys()))}
    
    # Initialize retrieval
    retrieval = STARRetrieval(
        semantic_weight=0.5,
        temporal_decay=0.7,
        history_length=3
    )
    retrieval.item_to_idx = item_to_idx
    
    # Compute relationships
    semantic_matrix = retrieval.compute_semantic_relationships(embeddings)
    retrieval.semantic_matrix = semantic_matrix
    
    # Process collaborative relationships
    interactions = [(review['reviewerID'], review['asin'], 
                    review['unixReviewTime'], review['overall']) 
                   for review in reviews]
    
    # Build user_all_items for negative sampling
    user_all_items = build_user_all_items(interactions)
    
    # Split data
    train_interactions = get_training_interactions(reviews)
    validation_sequences = prepare_validation_data(interactions)
    test_sequences = prepare_evaluation_data(interactions)
    
    # 4. Verify evaluation splits
    debugger.verify_evaluation_splits(test_sequences, reviews)
    
    # Process collaborative relationships
    collab_processor = CollaborativeRelationshipProcessor()
    collab_processor.process_interactions(train_interactions, item_mapping=retrieval.item_to_idx)
    collaborative_matrix = collab_processor.compute_collaborative_relationships(
        matrix_size=len(retrieval.item_to_idx)
    )
    retrieval.collaborative_matrix = collaborative_matrix
    
    # 5. Debug negative sampling
    debugger.debug_negative_sampling(
        test_sequences=test_sequences,
        recommender=retrieval,
        user_all_items=user_all_items
    )
    
    # 6. Analyze collaborative matrix
    debugger.analyze_collaborative_matrix(collaborative_matrix)
    
    # Run evaluation
    print("\n=== Running Evaluation ===")
    evaluator = RecommendationEvaluator()
    test_metrics = evaluator.evaluate_recommendations(
        test_sequences=test_sequences,
        recommender=retrieval,
        k_values=[5, 10],
        #n_negative_samples=99,
        user_all_items=user_all_items
    )
    
    print("\nFinal Results:")
    print_metrics_table(test_metrics, dataset="Beauty")

if __name__ == "__main__":
    main()