import os
import numpy as np
from pathlib import Path
from item_embeddings_huggingface import ItemEmbeddingGenerator
from star_retrieval import STARRetrieval
from collaborative_relationships import CollaborativeRelationshipProcessor
from evaluation_metrics import RecommendationEvaluator, prepare_evaluation_data, prepare_validation_data
from model_analysis import run_full_analysis

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
    metadata = load_amazon_metadata("beauty", min_interactions=5)

    # Process items and embeddings
    print("Processing items and generating embeddings...")
    items = get_items_from_data(reviews, metadata)
    
    # Try loading saved embeddings first
    embeddings, item_to_idx = load_embeddings()
    if embeddings is None:
        embedding_generator = ItemEmbeddingGenerator()
        embeddings = embedding_generator.generate_item_embeddings(items)
        save_embeddings(embeddings)
        item_to_idx = {item: idx for idx, item in enumerate(sorted(embeddings.keys()))}

    # Initialize retrieval with paper's parameters
    retrieval = STARRetrieval(
        semantic_weight=0.5,
        temporal_decay=0.7,
        history_length=3
    )

    # Log parameters
    print(f"\nParameters:")
    print(f"Semantic weight: {retrieval.semantic_weight}")
    print(f"Temporal decay: {retrieval.temporal_decay}")
    print(f"History length: {retrieval.history_length}")

    # Set item mapping and compute relationships
    retrieval.item_to_idx = item_to_idx
    
    print("Computing semantic relationships...")
    semantic_matrix = retrieval.compute_semantic_relationships(embeddings)
    retrieval.semantic_matrix = semantic_matrix

    # Process collaborative relationships
    print("Processing collaborative relationships...")
    interactions = [(review['reviewerID'], review['asin'], 
                    review['unixReviewTime'], review['overall']) 
                   for review in reviews]

    # Split data and prepare sequences
    train_interactions = get_training_interactions(reviews)
    print("Preparing evaluation data...")
    validation_sequences = prepare_validation_data(interactions)
    test_sequences = prepare_evaluation_data(interactions)

    # Compute collaborative relationships
    collab_processor = CollaborativeRelationshipProcessor()
    collab_processor.process_interactions(train_interactions, item_mapping=retrieval.item_to_idx)
    collaborative_matrix = collab_processor.compute_collaborative_relationships(
        matrix_size=len(retrieval.item_to_idx)
    )
    
    if collaborative_matrix is None:
        raise ValueError("Failed to compute collaborative relationships")
    
    retrieval.collaborative_matrix = collaborative_matrix

    # After computing relationships but before evaluation
    analysis_results = run_full_analysis(reviews, items, retrieval)
    print(analysis_results)

    # Run validation
    print("\n=== Running Validation ===")
    evaluator = RecommendationEvaluator()
    validation_metrics = evaluator.evaluate_recommendations(
        test_sequences=validation_sequences,
        recommender=retrieval,
        k_values=[5, 10],
        n_negative_samples=99
    )
    print("\nValidation Results:")
    print_metrics_table(validation_metrics, dataset="Beauty")

    # Run final evaluation
    print("\n=== Running Test Evaluation ===")
    test_metrics = evaluator.evaluate_recommendations(
        test_sequences=test_sequences,
        recommender=retrieval,
        k_values=[5, 10],
        n_negative_samples=99
    )
    print("\nTest Results:")
    print_metrics_table(test_metrics, dataset="Beauty")

if __name__ == "__main__":
    main()