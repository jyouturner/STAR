import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple
import faiss
import numpy as np
from scipy.sparse import csr_matrix

class AmazonDataProcessor:
    def __init__(self, min_interactions: int = 5):
        self.min_interactions = min_interactions

    def load_and_filter_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and filter Amazon review data
        """
        # Load JSON data
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['unixReviewTime'], unit='s')
        
        # Filter users and items with minimum interactions
        user_counts = df['reviewerID'].value_counts()
        item_counts = df['asin'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index
        
        df = df[
            df['reviewerID'].isin(valid_users) &
            df['asin'].isin(valid_items)
        ]
        
        return df.sort_values('timestamp')

    def create_chronological_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for each user
        """
        train_data = []
        val_data = []
        test_data = []
        
        for user, group in df.groupby('reviewerID'):
            user_hist = group.sort_values('timestamp')
            
            if len(user_hist) >= 3:  # Need at least 3 interactions
                train_data.extend(user_hist.iloc[:-2].to_dict('records'))
                val_data.append(user_hist.iloc[-2])
                test_data.append(user_hist.iloc[-1])
        
        return (
            pd.DataFrame(train_data),
            pd.DataFrame(val_data),
            pd.DataFrame(test_data)
        )

class EfficientRetrieval:
    def __init__(self, dimension: int):
        """
        Initialize FAISS index for efficient similarity search
        """
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        self.item_ids = []
        
    def add_items(self, item_embeddings: Dict[str, np.ndarray]):
        """
        Add item embeddings to the index
        """
        embeddings = []
        self.item_ids = []
        
        for item_id, embedding in item_embeddings.items():
            # Normalize for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(normalized_embedding)
            self.item_ids.append(item_id)
            
        embeddings_matrix = np.vstack(embeddings)
        self.index.add(embeddings_matrix)
        
    def find_similar_items(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Find k most similar items using FAISS
        """
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            k
        )
        
        # Return (item_id, similarity) pairs
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for padding if fewer than k results
                results.append((self.item_ids[idx], float(sim)))
                
        return results

class SparseCollaborativeProcessor:
    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.interaction_matrix = None
        
    def process_interactions(self, interactions: List[Tuple[str, str, float, str]]):
        """
        Process interactions using sparse matrices
        """
        # Create mappings
        users = sorted(set(u for u, _, _, _ in interactions))
        items = sorted(set(i for _, i, _, _ in interactions))
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: j for j, i in enumerate(items)}
        
        # Create sparse matrix
        rows = []
        cols = []
        data = []
        
        for user_id, item_id, _, _ in interactions:
            rows.append(self.user_to_idx[user_id])
            cols.append(self.item_to_idx[item_id])
            data.append(1.0)
            
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items))
        )
        
    def compute_similarities(self, item_id: str, k: int) -> List[Tuple[str, float]]:
        """
        Compute top-k similar items based on collaborative patterns
        """
        if item_id not in self.item_to_idx:
            return []
            
        item_idx = self.item_to_idx[item_id]
        item_vector = self.interaction_matrix.T[item_idx]
        
        # Compute similarities using sparse operations
        similarities = self.interaction_matrix.T.dot(item_vector.T)
        similarities = similarities.toarray().flatten()
        
        # Get top-k items
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        # Convert back to item IDs
        idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        return [
            (idx_to_item[idx], float(similarities[idx]))
            for idx in top_k_idx
            if idx != item_idx and similarities[idx] > 0
        ]

# Example usage
def main():
    # Initialize processors
    data_processor = AmazonDataProcessor(min_interactions=5)
    efficient_retrieval = EfficientRetrieval(dimension=768)  # For text-embedding-gecko
    sparse_collab = SparseCollaborativeProcessor()
    
    # Load and process data
    df = data_processor.load_and_filter_data('path/to/amazon_reviews.json')
    train_df, val_df, test_df = data_processor.create_chronological_split(df)
    
    # Process items for efficient retrieval
    item_embeddings = {}  # Get these from ItemEmbeddingGenerator
    efficient_retrieval.add_items(item_embeddings)
    
    # Process collaborative information
    interactions = [
        (row['reviewerID'], row['asin'], row['timestamp'], row['overall'])
        for _, row in train_df.iterrows()
    ]
    sparse_collab.process_interactions(interactions)
    
    print("Data processing and initialization complete")
    
if __name__ == "__main__":
    main()