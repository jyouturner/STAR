import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModel

def analyze_dataset_stats(reviews):
    user_item_pairs = {(r['reviewerID'], r['asin']) for r in reviews}
    unique_users = {r['reviewerID'] for r in reviews}
    unique_items = {r['asin'] for r in reviews}
    
    stats = {
        "unique_interactions": len(user_item_pairs),
        "total_reviews": len(reviews),
        "unique_users": len(unique_users),
        "unique_items": len(unique_items),
        "avg_reviews_per_interaction": len(reviews)/len(user_item_pairs),
        "density": len(user_item_pairs)/(len(unique_users)*len(unique_items))
    }
    return stats

def compare_embeddings(items: Dict, model1, model2, sample_size: int = 100):
    """Compare embeddings from different models"""
    sample_items = list(items.keys())[:sample_size]
    embeddings1 = model1.generate_item_embeddings({k: items[k] for k in sample_items})
    embeddings2 = model2.generate_item_embeddings({k: items[k] for k in sample_items})
    
    similarities = []
    for item_id in sample_items:
        sim = 1 - cosine(embeddings1[item_id], embeddings2[item_id])
        similarities.append(sim)
        
    print("\nEmbedding Analysis:")
    print(f"Mean similarity: {np.mean(similarities):.3f}")
    print(f"Distribution: {np.percentile(similarities, [25,50,75])}")
    return similarities

def analyze_collaborative_matrix(matrix):
    sparsity = 1 - (matrix > 0).sum()/(matrix.shape[0]*matrix.shape[1])
    nonzero_values = matrix[matrix > 0]
    
    stats = {
        "sparsity": sparsity,
        "mean_nonzero": nonzero_values.mean(),
        "median_nonzero": np.median(nonzero_values),
        "percentiles": np.percentile(nonzero_values, [25,50,75]),
        "max_value": nonzero_values.max(),
        "min_nonzero": nonzero_values.min()
    }
    return stats

def analyze_semantic_matrix(matrix):
    nonzero_values = matrix[matrix > 0]
    stats = {
        "sparsity": 1 - len(nonzero_values)/(matrix.shape[0]*matrix.shape[1]),
        "mean_sim": nonzero_values.mean(),
        "median_sim": np.median(nonzero_values),
        "percentiles": np.percentile(nonzero_values, [25,50,75]),
        "max_sim": nonzero_values.max(),
        "min_nonzero": nonzero_values.min()
    }
    return stats

class GeckoModel:
    """For comparison with MiniLM"""
    def __init__(self):
        self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
        
    def get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.model.get_embeddings([text])
        return np.array(embeddings[0].values)
        
    def generate_item_embeddings(self, items: Dict) -> Dict[str, np.ndarray]:
        embeddings = {}
        for item_id, item_data in items.items():
            prompt = self.create_item_prompt(item_data)
            embedding = self.get_embedding(prompt)
            embeddings[item_id] = embedding
        return embeddings

def run_full_analysis(reviews, items, retrieval_model):
    """Run comprehensive analysis"""
    dataset_stats = analyze_dataset_stats(reviews)
    print("\nDataset Statistics:")
    for k, v in dataset_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    semantic_stats = analyze_semantic_matrix(retrieval_model.semantic_matrix)
    print("\nSemantic Matrix Statistics:")
    for k, v in semantic_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, np.ndarray):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    collab_stats = analyze_collaborative_matrix(retrieval_model.collaborative_matrix)
    print("\nCollaborative Matrix Statistics:")
    for k, v in collab_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, np.ndarray):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    return {
        "dataset": dataset_stats,
        "semantic": semantic_stats,
        "collaborative": collab_stats
    }