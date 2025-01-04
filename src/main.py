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

def run_experiment(
    items: Dict,
    reviews: List[Dict],
    field_sets: List[set],
    output_path: str = 'results.txt'
):
    """Run experiments with different field combinations"""
    results = []
    
    for fields in field_sets:
        print(f"\n=== Testing fields: {sorted(fields)} ===")
        
        # Initialize embedding generator with current fields
        embedding_generator = ItemEmbeddingGenerator(
            output_dimension=768,
            include_fields=fields
        )
        
        # Debug first
        embedding_generator.debug_prompt(items, num_samples=2)
        
        # Generate embeddings
        embeddings = embedding_generator.generate_item_embeddings(items)
        
        # Initialize retrieval with paper's parameters
        retrieval = STARRetrieval(
            semantic_weight=0.5,
            temporal_decay=0.7,
            history_length=3
        )
        retrieval.item_to_idx = {
            item: idx for idx, item in enumerate(sorted(embeddings.keys()))
        }
        
        # Compute relationships
        semantic_matrix = retrieval.compute_semantic_relationships(embeddings)
        retrieval.semantic_matrix = semantic_matrix
        
        # Process collaborative relationships
        interactions = [
            (review['reviewerID'], review['asin'], 
             review['unixReviewTime'], review['overall']) 
            for review in reviews
        ]
        user_all_items = build_user_all_items(interactions)
        
        # Split and prepare data
        train_interactions = get_training_interactions(reviews)
        validation_sequences = prepare_validation_data(interactions)
        test_sequences = prepare_evaluation_data(interactions)
        
        # Compute collaborative relationships
        collab_processor = CollaborativeRelationshipProcessor()
        collab_processor.process_interactions(
            train_interactions, 
            item_mapping=retrieval.item_to_idx
        )
        collaborative_matrix = collab_processor.compute_collaborative_relationships(
            matrix_size=len(retrieval.item_to_idx)
        )
        retrieval.collaborative_matrix = collaborative_matrix
        
        # Run evaluation
        evaluator = RecommendationEvaluator()
        test_metrics = evaluator.evaluate_recommendations(
            test_sequences=test_sequences,
            recommender=retrieval,
            k_values=[5, 10],
            n_negative_samples=99,
            user_all_items=user_all_items
        )
        
        # Store results
        results.append({
            'fields': sorted(fields),
            'metrics': test_metrics,
            'semantic_stats': analyze_semantic_matrix(semantic_matrix)
        })
        
        # Save results
        with open(output_path, 'a') as f:
            f.write(f"\n\nFields: {sorted(fields)}\n")
            f.write("Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\nSemantic Statistics:\n")
            stats = analyze_semantic_matrix(semantic_matrix)
            for stat, value in stats.items():
                f.write(f"{stat}: {value}\n")
                
    return results

def main():
    # Load data
    print("Loading data...")
    reviews = load_amazon_dataset("beauty", min_interactions=5)
    metadata = load_amazon_metadata("beauty", min_interactions=5)
    items = get_items_from_data(reviews, metadata)
    
    # Define field combinations to test
    field_experiments = [
        {'title', 'description', 'category'},  # Minimal
        {'title', 'description', 'category', 'brand'},  # Add brand
        {'title', 'description', 'category', 'price'},  # Add price
        {'title', 'description', 'category', 'sales_rank'},  # Add rank
        {'title', 'description', 'category', 'brand', 'price', 'sales_rank'}  # All
    ]
    
    # Run experiments
    results = run_experiment(
        items=items,
        reviews=reviews,
        field_sets=field_experiments,
        output_path='field_experiments.txt'
    )
    
    # Print summary
    print("\nExperiment Summary:")
    print("=" * 80)
    for result in results:
        print(f"\nFields: {result['fields']}")
        print(f"Mean Similarity: {result['semantic_stats']['mean_sim']:.4f}")
        print("Metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
if __name__ == "__main__":
    main()