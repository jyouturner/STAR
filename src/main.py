import os
import numpy as np
from pathlib import Path
from item_embeddings_huggingface import ItemEmbeddingGenerator
from optimized_collaborative_relationships import OptimizedCollaborativeProcessor
from star_retrieval import STARRetrieval
from collaborative_relationships import CollaborativeRelationshipProcessor
from evaluation_metrics import RecommendationEvaluator, prepare_evaluation_data
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
    
    # Convert embeddings dict to numpy array
    items = sorted(embeddings.keys())
    embedding_array = np.stack([embeddings[item] for item in items])
    
    # Save embeddings and item mapping
    np.save(f'{save_dir}/embeddings.npy', embedding_array)
    np.save(f'{save_dir}/items.npy', np.array(items))
    
    print(f"Saved embeddings to {save_dir}")

def load_embeddings(load_dir='data/embeddings'):
    """Load embeddings and create item mapping"""
    try:
        embedding_array = np.load(f'{load_dir}/embeddings.npy')
        items = np.load(f'{load_dir}/items.npy')
        
        # Reconstruct embeddings dictionary and item_to_idx mapping
        embeddings = {item: emb for item, emb in zip(items, embedding_array)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        print(f"Loaded embeddings for {len(embeddings)} items")
        return embeddings, item_to_idx
    except FileNotFoundError:
        print("No saved embeddings found")
        return None, None

def main():
    # Load limited data for testing
    print("Loading data...")
    reviews = load_amazon_dataset("beauty", min_interactions=5)
    metadata = load_amazon_metadata("beauty", min_interactions=5)

    # Process items and get embeddings
    print("Processing items and generating embeddings...")
    items = get_items_from_data(reviews, metadata)
    
    # Try to load saved embeddings first
    embeddings, item_to_idx = load_embeddings()
    
    if embeddings is None:
        print("Generating new embeddings...")
        embedding_generator = ItemEmbeddingGenerator()
        embeddings = embedding_generator.generate_item_embeddings(items)
        # Save embeddings for future use
        save_embeddings(embeddings)
    
    # Initialize retrieval with paper's parameters
    retrieval = STARRetrieval(
        semantic_weight=0.5,  # Balance between semantic and collaborative
        temporal_decay=0.7,   # Temporal decay factor
        history_length=3      # Number of recent items to consider
    )
    
    # Create item_to_idx mapping here if needed
    if item_to_idx is None:
        item_to_idx = {item: idx for idx, item in enumerate(sorted(embeddings.keys()))}
    
    # Set the item_to_idx mapping
    retrieval.item_to_idx = item_to_idx
    
    # Compute semantic relationships
    print("Computing semantic relationships...")
    semantic_matrix = retrieval.compute_semantic_relationships(embeddings)
    retrieval.semantic_matrix = semantic_matrix
    
    # Process user interactions and compute collaborative relationships
    print("Processing collaborative relationships...")
    train_interactions = get_training_interactions(reviews)

    collab_processor = CollaborativeRelationshipProcessor()
    collab_processor.process_interactions(train_interactions, item_mapping=retrieval.item_to_idx)
    collaborative_matrix = collab_processor.compute_collaborative_relationships(
        matrix_size=len(retrieval.item_to_idx)
    )

    # Set the collaborative matrix in the retrieval object
    retrieval.collaborative_matrix = collaborative_matrix

    # Prepare evaluation data
    print("Preparing evaluation data...")
    test_sequences = prepare_evaluation_data([
        (review['reviewerID'], review['asin'], 
         review['unixReviewTime'], review['overall']) 
        for review in reviews
    ])
    
    # Run evaluation
    print("Running evaluation...")
    evaluator = RecommendationEvaluator()
    metrics = evaluator.evaluate_recommendations(
        test_sequences=test_sequences,
        recommender=retrieval,
        k_values=[5, 10],
        n_negative_samples=99
    )

    # Print results
    print("\nEvaluation Results:")
    print_metrics_table(metrics, dataset="Beauty")

if __name__ == "__main__":
    main()