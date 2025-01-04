# Consolidated Poetry Project Files

## Project Structure

```
README.md
check_data.py
download_data.py
pyproject.toml
src/collaborative_relationships.py
src/data_processing.py
src/evaluation_metrics.py
src/item_embeddings_huggingface.py
src/item_embeddings_vertex_ai.py
src/main.py
src/model_analysis.py
src/optimized_collaborative_relationships.py
src/star_retrieval.py
src/utils.py
```

## File Contents

### README.md

```
# STAR: A Simple Training-free Approach for Recommendations using Large Language Models

This repository implements the STAR (Simple Training-free Approach for Recommendation) as described in the paper ["STAR: A Simple Training-free Approach for Recommendations using Large Language Models"](https://arxiv.org/abs/2410.16458). The implementation focuses on the *retrieval pipeline*, which the paper shows achieves competitive performance even without the additional ranking stage.

## Overview

STAR is a training-free recommendation framework that combines:

1. Semantic relationships between items (using LLM embeddings)
2. Collaborative signals from user interactions
3. Temporal decay to prioritize recent interactions

The key insight is that by properly combining these signals, we can achieve competitive recommendation performance without any training.

## Architecture

The implementation consists of several key components:

### 1. Item Embedding Generation (`item_embeddings_huggingface.py`)

- Uses text-embedding-gecko-004 model for generating item embeddings
- Creates prompts combining item metadata (title, description, category, brand, etc.)
- Excludes fields like Item ID and URL to avoid spurious lexical similarities

### 2. Retrieval Pipeline (`star_retrieval.py`)

- Computes semantic relationships using normalized cosine similarity
- Processes collaborative relationships using normalized co-occurrence
- Implements the paper's scoring formula:

  ```
  score(x) = (1/n) * sum(rj * λ^tj * [a * Rs_xj + (1-a) * Rc_xj])
  ```

  where:
  - λ is the temporal decay factor (0.7)
  - a is the semantic weight (0.5)
  - Rs is the semantic relationship matrix
  - Rc is the collaborative relationship matrix

### 3. Collaborative Processing (`collaborative_relationships.py`)

- Computes normalized co-occurrence between items
- Handles user-item interaction processing
- Implements proper normalization for collaborative signals


### 4. Evaluation Metrics (`evaluation_metrics.py`)

- Implements proper evaluation protocol with 99 negative samples
- Calculates Hits@K and NDCG@K metrics
- Maintains temporal ordering in evaluation sequences

## Key Implementation Details

### Data Filtering

- Filters users and items with fewer than 5 interactions
- Two-pass loading process to ensure proper filtering
- Maintains dataset statistics matching the paper

### Normalization

1. Semantic Relationships:

```python
# L2 normalize embeddings
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
normalized_embeddings = embeddings_array / (norms + 1e-8)

# Compute max(0, 1 - cosine) similarity
sim = 1 - cosine(normalized_embeddings[i], normalized_embeddings[j])
semantic_matrix[i,j] = max(0, sim)
```

2. Collaborative Relationships:

```python
# Normalize by user activity
user_activity = np.sum(interaction_matrix, axis=0, keepdims=True)
user_activity[user_activity == 0] = 1
normalized_interactions = interaction_matrix / np.sqrt(user_activity)
```

### Evaluation Protocol

- Last item in sequence used for testing
- 99 negative samples for each positive item
- Proper temporal ordering preserved
- Metrics calculated over successful predictions only

## Usage

1. Install requirements:

```bash
poetry install
```

2. Download Amazon Review dataset:

```python
poetry run python download_data.py --category beauty
```

3. Run the implementation:

```python
poetry run python src/main.py
```

## Key Parameters

The key parameters following the paper:

- `semantic_weight`: 0.5 (balance between semantic and collaborative)
- `temporal_decay`: 0.7 (temporal decay factor)
- `history_length`: 3 (number of recent items to consider)
- `min_interactions`: 5 (minimum interactions filter)
- `n_negative_samples`: 99 (for evaluation)

## Implementation Insights

1. **Proper Filtering is Crucial**
   - Both users and items must meet minimum interaction threshold
   - Filtering must be applied consistently across reviews and metadata

2. **Normalization Matters**
   - L2 normalization of embeddings before similarity computation
   - Proper normalization of user-item interaction matrix
   - Using max(0, sim) to ensure non-negative similarities

3. **Evaluation Protocol Details**
   - Negative sampling must exclude history items
   - Temporal ordering must be preserved
   - Proper normalization of metrics over successful predictions

4. **Embedding Generation**
   - Consistent prompt format is important
   - Excluding ID and URL fields helps avoid spurious similarities
   - Using proper embedding model (text-embedding-gecko-004)

## Results

On the Beauty dataset, output

```
Loading data...
First pass: Counting interactions...

Before filtering:
Total users: 22363
Total items: 12101

After filtering (>= 5 interactions):
Valid users: 22363
Valid items: 12101

Second pass: Loading filtered data...

Final filtered reviews: 198502
First pass: Counting interactions...

Before filtering:
Total users: 22363
Total items: 12101

After filtering (>= 5 interactions):
Valid users: 22363
Valid items: 12101

Second pass: Loading filtered data...

Final filtered reviews: 198502

Loading filtered metadata...
Loaded metadata for 12101 items with 0 errors
Processing items and generating embeddings...
Loaded embeddings for 12101 items
Computing semantic relationships...

Computing semantic relationships...
Computing semantic similarities...
Processing collaborative relationships...

Processing user-item interactions...
Found 20484 users and 12101 items
Built interaction matrix with 137119 non-zero entries

Computing collaborative relationships...
Computing normalized co-occurrences using matrix operations...
Collaborative relationship computation complete
Preparing evaluation data...
Running evaluation...

=== Starting Evaluation ===
Evaluating: 100%|███████████████████████████| 22363/22363 [00:06<00:00, 3345.74it/s]

Successfully evaluated 22363/22363 sequences

Evaluation Results:

Results for Beauty dataset:
------------------------------
Metric          Score     
------------------------------
hit@10          0.4184
hit@5           0.3315
ndcg@10         0.2895
ndcg@5          0.2615
------------------------------
```


## Issues

The results are not aligned with the paper's results.
## Citations

```bibtex
@article{lee2024star,
  title={STAR: A Simple Training-free Approach for Recommendations using Large Language Models},
  author={Lee, Dong-Ho and Kraft, Adam and Jin, Long and Mehta, Nikhil and Xu, Taibai and Hong, Lichan and Chi, Ed H. and Yi, Xinyang},
  journal={arXiv preprint arXiv:2410.16458},
  year={2024}
}
```

```

### check_data.py

```python
import gzip
import json
import os

def check_gzip_file(filepath: str, num_lines: int = 5):
    """
    Check contents of a gzipped JSON file
    
    Args:
        filepath: Path to the gzipped file
        num_lines: Number of lines to check
    """
    print(f"\nChecking file: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            print("\nFirst few lines:")
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    # Try to parse JSON
                    data = json.loads(line.strip())
                    print(f"\nLine {i+1} (parsed JSON):")
                    print(json.dumps(data, indent=2)[:500] + "...")  # Print first 500 chars
                except json.JSONDecodeError as e:
                    print(f"\nLine {i+1} (raw, failed to parse):")
                    print(line.strip()[:500] + "...")
                    print(f"JSON Error: {str(e)}")
                except Exception as e:
                    print(f"Error on line {i+1}: {str(e)}")
                    
    except Exception as e:
        print(f"Error opening file: {str(e)}")

def main():
    # Check both review and metadata files
    files_to_check = [
        'data/meta_Beauty.json.gz',
        'data/reviews_Beauty_5.json.gz'
    ]
    
    for filepath in files_to_check:
        check_gzip_file(filepath)

if __name__ == "__main__":
    main()
```

### download_data.py

```python
#!/usr/bin/env python3
import os
import sys
import urllib.request
import hashlib
import gzip
import shutil
from typing import Dict, Tuple
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: str):
    """
    Download a file with progress bar
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def verify_file(filepath: str, expected_md5: str = None) -> bool:
    """
    Verify file exists and optionally check MD5
    """
    if not os.path.exists(filepath):
        return False
    
    if expected_md5:
        with open(filepath, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            return md5 == expected_md5
            
    return True

def get_dataset_info() -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Get dataset URLs and their MD5 hashes
    """
    base_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    
    datasets = {
        "beauty": {
            "reviews": (
                f"{base_url}reviews_Beauty_5.json.gz",
                None  # Add MD5 if available
            ),
            "metadata": (
                f"{base_url}meta_Beauty.json.gz",
                None
            )
        },
        "toys_and_games": {
            "reviews": (
                f"{base_url}reviews_Toys_and_Games_5.json.gz",
                None
            ),
            "metadata": (
                f"{base_url}meta_Toys_and_Games.json.gz",
                None
            )
        },
        "sports_and_outdoors": {
            "reviews": (
                f"{base_url}reviews_Sports_and_Outdoors_5.json.gz",
                None
            ),
            "metadata": (
                f"{base_url}meta_Sports_and_Outdoors.json.gz",
                None
            )
        }
    }
    
    return datasets

def setup_data_directory(base_dir: str = "data"):
    """
    Create data directory if it doesn't exist
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def download_datasets(data_dir: str = "data", categories: list = None):
    """
    Download specified Amazon Review datasets
    """
    datasets = get_dataset_info()
    
    # If no categories specified, download all
    if categories is None:
        categories = list(datasets.keys())
    
    # Validate categories
    for category in categories:
        if category not in datasets:
            print(f"Invalid category: {category}")
            print(f"Available categories: {list(datasets.keys())}")
            return
    
    # Create data directory
    data_dir = setup_data_directory(data_dir)
    
    # Download each dataset
    for category in categories:
        print(f"\nDownloading {category} datasets...")
        
        # Download reviews
        review_url, review_md5 = datasets[category]["reviews"]
        review_file = os.path.join(data_dir, os.path.basename(review_url))
        
        if not verify_file(review_file, review_md5):
            print(f"Downloading reviews for {category}...")
            download_url(review_url, review_file)
        else:
            print(f"Reviews file for {category} already exists")
            
        # Download metadata
        meta_url, meta_md5 = datasets[category]["metadata"]
        meta_file = os.path.join(data_dir, os.path.basename(meta_url))
        
        if not verify_file(meta_file, meta_md5):
            print(f"Downloading metadata for {category}...")
            download_url(meta_url, meta_file)
        else:
            print(f"Metadata file for {category} already exists")

def verify_downloads(data_dir: str = "data"):
    """
    Verify all required files exist and can be read
    """
    datasets = get_dataset_info()
    all_good = True
    
    for category, files in datasets.items():
        for file_type, (url, _) in files.items():
            filepath = os.path.join(data_dir, os.path.basename(url))
            
            if not os.path.exists(filepath):
                print(f"Missing file: {filepath}")
                all_good = False
                continue
                
            # Try reading the gzipped file
            try:
                with gzip.open(filepath, 'rb') as f:
                    # Try reading first line
                    f.readline()
                print(f"✓ Verified {category} {file_type}")
            except Exception as e:
                print(f"Error reading {filepath}: {str(e)}")
                all_good = False
    
    return all_good

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        categories = sys.argv[1:]
    else:
        categories = None
    
    # Download datasets
    print("Starting download of Amazon Review datasets...")
    download_datasets(categories=categories)
    
    # Verify downloads
    print("\nVerifying downloads...")
    if verify_downloads():
        print("\nAll datasets downloaded and verified successfully!")
    else:
        print("\nSome files are missing or corrupted. Please try downloading again.")

if __name__ == "__main__":
    main()
```

### pyproject.toml

```toml
[tool.poetry]
name = "star"
version = "0.1.0"
description = ""
authors = ["jyouturner <jy2947@gmail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
numpy = "^2.2.1"
pandas = "^2.2.3"
scipy = "^1.14.1"
google-cloud-aiplatform = "^1.75.0"
faiss-cpu = "^1.9.0.post1"
tqdm = "^4.67.1"
transformers = "^4.47.1"
torch = "^2.5.1"


[tool.poetry.group.dev.dependencies]
ruff = "^0.8.5"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "N",  # PEP8 naming conventions
]
ignore = [
    "E501",  # line too long, handled by black
]

[tool.ruff.lint.pydocstyle]
convention = "google"

```

### src/collaborative_relationships.py

```python
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple

class CollaborativeRelationshipProcessor:
    def __init__(self):
        self.user_item_interactions = {}
        self.item_to_idx = None
        self.interaction_matrix = None

    def process_interactions(self, interactions, item_mapping=None):
        """
        Process user-item interactions and build interaction matrix
        
        Args:
            interactions: List of (user_id, item_id, timestamp, rating) tuples
            item_mapping: Dictionary mapping item IDs to indices
        """
        print("\nProcessing user-item interactions...")
        self.item_to_idx = item_mapping
        
        # First pass: Get unique users and build user mapping
        users = sorted(set(user_id for user_id, _, _, _ in interactions))
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.item_to_idx)
        
        print(f"Found {n_users} users and {n_items} items")
        
        # Build sparse interaction matrix
        # Using lil_matrix for efficient matrix construction
        self.interaction_matrix = lil_matrix((n_items, n_users), dtype=np.float32)
        
        # Fill interaction matrix
        for user_id, item_id, _, _ in interactions:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                user_idx = user_to_idx[user_id]
                self.interaction_matrix[item_idx, user_idx] = 1.0
        
        # Convert to CSR format for efficient computations
        self.interaction_matrix = self.interaction_matrix.tocsr()
        print(f"Built interaction matrix with {self.interaction_matrix.nnz} non-zero entries")

    def compute_collaborative_relationships(self, matrix_size):
        """
        Compute collaborative relationships using normalized co-occurrence
        following paper's specification.
        """
        interaction_matrix = self.interaction_matrix.toarray()
        # Normalize only by user activity per paper
        user_activity = np.sum(interaction_matrix, axis=0)
        user_activity[user_activity == 0] = 1
        normalized = interaction_matrix / np.sqrt(user_activity)
        
        # Compute normalized co-occurrence
        collaborative_matrix = normalized @ normalized.T
        np.fill_diagonal(collaborative_matrix, 0)  # Zero out self-similarities
        
        return collaborative_matrix

    def get_item_co_occurrences(self, item_id: str) -> Dict[str, int]:
        """
        Get raw co-occurrence counts for an item
        Useful for debugging and verification
        
        Args:
            item_id: ID of item to get co-occurrences for
            
        Returns:
            Dictionary mapping item IDs to co-occurrence counts
        """
        if self.interaction_matrix is None or item_id not in self.item_to_idx:
            return {}
            
        item_idx = self.item_to_idx[item_id]
        item_vector = self.interaction_matrix[item_idx].toarray().flatten()
        
        co_occurrences = {}
        for other_id, other_idx in self.item_to_idx.items():
            if other_id != item_id:
                other_vector = self.interaction_matrix[other_idx].toarray().flatten()
                count = np.sum(np.logical_and(item_vector > 0, other_vector > 0))
                if count > 0:
                    co_occurrences[other_id] = int(count)
                    
        return co_occurrences
# Example usage
def main():
    # Example interaction data
    # Each tuple is (user_id, item_id, timestamp, rating)
    interactions = [
        ("user1", "item1", "2024-01-01", 5),
        ("user1", "item2", "2024-01-02", 4),
        ("user2", "item1", "2024-01-01", 3),
        ("user2", "item3", "2024-01-03", 5),
        ("user3", "item2", "2024-01-02", 4),
        ("user3", "item3", "2024-01-03", 5),
        ("user4", "item1", "2024-01-01", 4),
        ("user4", "item2", "2024-01-02", 5),
    ]
    
    # Initialize processor
    processor = CollaborativeRelationshipProcessor()
    
    # Process interactions
    processor.process_interactions(interactions)
    
    # Compute collaborative relationships
    collab_matrix = processor.compute_collaborative_relationships(
        matrix_size=len(processor.item_to_idx)
    )
    
    # Example outputs
    print("\nCollaborative Relationship Matrix:")
    print(pd.DataFrame(
        collab_matrix,
        index=list(processor.item_to_idx.keys()),
        columns=list(processor.item_to_idx.keys())
    ))
    
    print("\nCo-occurrence counts for item1:")
    co_occurrences = processor.get_item_co_occurrences("item1")
    for item_id, count in co_occurrences.items():
        print(f"With {item_id}: {count} users")

if __name__ == "__main__":
    main()
```

### src/data_processing.py

```python
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
```

### src/evaluation_metrics.py

```python
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
            
            # Sample exactly n_negative_samples items
            while len(negative_candidates) < n_negative_samples:
                item = random.choice(valid_items)
                if item not in excluded_items:
                    negative_candidates.add(item)
            
            # Create candidate set with negative samples + positive item
            candidate_items = list(negative_candidates) + [next_item]  # Critical: Add positive item last
            
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


def prepare_evaluation_data(interactions, min_sequence_length=5):
    """
    Prepare evaluation data following paper's protocol:
    - Maintain temporal ordering
    - Minimum sequence length of 5
    - Use last item as test item
    - Use previous items as history
    
    Args:
        interactions: List of (user_id, item_id, timestamp, rating) tuples
        min_sequence_length: Minimum required sequence length (default: 5)
        
    Returns:
        List of (user_id, history, test_item) tuples
    """
    # Sort all interactions by user and timestamp
    sorted_interactions = sorted(interactions, key=lambda x: (x[0], x[2]))
    
    # Group by user while maintaining temporal order
    user_sequences = {}
    for user_id, item_id, timestamp, rating in sorted_interactions:
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append((item_id, timestamp, rating))
    
    test_sequences = []
    
    for user_id, interactions in user_sequences.items():
        # Skip if sequence is too short
        if len(interactions) < min_sequence_length:
            continue
        
        # Get items in temporal order
        items = [item for item, _, _ in interactions]
        
        # Last item is test item
        test_item = items[-1]
        # Previous items are history
        history = items[:-1]
        
        test_sequences.append((user_id, history, test_item))
    
    return test_sequences

def prepare_validation_data(interactions, min_sequence_length=5):
    """
    Prepare validation data similar to test data but using second-to-last item
    
    Args:
        interactions: List of (user_id, item_id, timestamp, rating) tuples
        min_sequence_length: Minimum required sequence length (default: 5)
        
    Returns:
        List of (user_id, history, validation_item) tuples
    """
    # Sort all interactions by user and timestamp
    sorted_interactions = sorted(interactions, key=lambda x: (x[0], x[2]))
    
    # Group by user while maintaining temporal order
    user_sequences = {}
    for user_id, item_id, timestamp, rating in sorted_interactions:
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append((item_id, timestamp, rating))
    
    validation_sequences = []
    
    for user_id, interactions in user_sequences.items():
        # Skip if sequence is too short
        if len(interactions) < min_sequence_length:
            continue
        
        # Get items in temporal order
        items = [item for item, _, _ in interactions]
        
        # Second-to-last item is validation item
        validation_item = items[-2]
        # Previous items are history
        history = items[:-2]
        
        validation_sequences.append((user_id, history, validation_item))
    
    return validation_sequences
```

### src/item_embeddings_huggingface.py

```python
from typing import Dict, Any
import numpy as np
import json
from star_retrieval import STARRetrieval
from transformers import AutoTokenizer, AutoModel
import torch

class ItemEmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def create_item_prompt(self, item: Dict) -> str:
        """Create prompt following paper's approach, excluding ID/URL fields"""
        prompt_parts = []
        
        if 'title' in item:
            prompt_parts.append(f"Title: {item['title']}")
            
        if 'description' in item:
            prompt_parts.append(f"Description: {item['description']}")
            
        if 'categories' in item:
            cats = ' > '.join(item['categories'][-1]) if isinstance(item['categories'], list) else str(item['categories'])
            prompt_parts.append(f"Category: {cats}")
            
        if 'brand' in item:
            prompt_parts.append(f"Brand: {item['brand']}")
            
        if 'salesRank' in item:
            if isinstance(item['salesRank'], dict):
                ranks = [f"{cat}: {rank}" for cat, rank in item['salesRank'].items()]
                prompt_parts.append(f"Sales Rank: {', '.join(ranks)}")
            else:
                prompt_parts.append(f"Sales Rank: {item['salesRank']}")
                
        if 'price' in item:
            prompt_parts.append(f"Price: ${item['price']}")
            
        return "\n".join(prompt_parts)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string using the Hugging Face model
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding values
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    def generate_item_embeddings(self, items: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple items
        
        Args:
            items: Dictionary mapping item IDs to their metadata
            
        Returns:
            Dictionary mapping item IDs to their embeddings
        """
        embeddings = {}
        for item_id, item_data in items.items():
            prompt = self.create_item_prompt(item_data)
            embedding = self.get_embedding(prompt)
            embeddings[item_id] = embedding
        return embeddings

# Example usage:
def main():
    # Example items (similar to Amazon product data used in the paper)
    items = {
        "B001": {
            "title": "Professional Makeup Brush Set",
            "description": "Set of 12 professional makeup brushes with synthetic bristles",
            "categories": ["Beauty", "Makeup", "Brushes & Tools"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 1250},
            "price": 24.99
        },
        "B002": {
            "title": "Eyeshadow Palette - Natural Colors",
            "description": "15 highly pigmented natural eyeshadow colors",
            "categories": ["Beauty", "Makeup", "Eyes", "Eyeshadow"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 2100},
            "price": 19.99
        }
    }
    
    # Generate embeddings
    generator = ItemEmbeddingGenerator()
    item_embeddings = generator.generate_item_embeddings(items)
    
    # Use with STAR retrieval
    retrieval = STARRetrieval(semantic_weight=0.5, temporal_decay=0.7, history_length=3)
    retrieval.compute_semantic_relationships(item_embeddings)
    
    # Example output of embeddings and similarity
    print(f"Embedding dimension: {len(item_embeddings['B001'])}")
    semantic_sim = retrieval.semantic_matrix[0,1]
    print(f"Semantic similarity between items: {semantic_sim:.3f}")

if __name__ == "__main__":
    main()

```

### src/item_embeddings_vertex_ai.py

```python
from typing import Dict, Any
import numpy as np
import json
from star_retrieval import STARRetrieval
from vertexai.language_models import TextEmbeddingModel  # for Google's text-embedding-gecko

class ItemEmbeddingGenerator:
    def __init__(self):
        """Initialize the embedding model"""
        self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
        
    def create_item_prompt(self, item: Dict[str, Any]) -> str:
        """
        Create a text prompt for an item following the paper's approach.
        The paper mentions using title, description, category, brand, sales rank, and price,
        but omitting ID and URL fields.
        
        Args:
            item: Dictionary containing item metadata
            
        Returns:
            Formatted text prompt for the item
        """
        prompt_parts = []
        
        # Add title
        if 'title' in item:
            prompt_parts.append(f"Title: {item['title']}")
            
        # Add description
        if 'description' in item:
            prompt_parts.append(f"Description: {item['description']}")
            
        # Add categories
        if 'categories' in item:
            if isinstance(item['categories'], list):
                cats = ' > '.join(item['categories'])
            else:
                cats = str(item['categories'])
            prompt_parts.append(f"Categories: {cats}")
            
        # Add brand
        if 'brand' in item:
            prompt_parts.append(f"Brand: {item['brand']}")
            
        # Add sales rank
        if 'salesRank' in item:
            if isinstance(item['salesRank'], dict):
                # Handle case where salesRank is category-specific
                ranks = [f"{cat}: {rank}" for cat, rank in item['salesRank'].items()]
                prompt_parts.append(f"Sales Rank: {', '.join(ranks)}")
            else:
                prompt_parts.append(f"Sales Rank: {item['salesRank']}")
                
        # Add price
        if 'price' in item:
            prompt_parts.append(f"Price: ${item['price']}")
            
        return "\n".join(prompt_parts)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string using the LLM
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding values
        """
        embeddings = self.model.get_embeddings([text])
        return np.array(embeddings[0].values)

    def generate_item_embeddings(self, items: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple items
        
        Args:
            items: Dictionary mapping item IDs to their metadata
            
        Returns:
            Dictionary mapping item IDs to their embeddings
        """
        embeddings = {}
        for item_id, item_data in items.items():
            prompt = self.create_item_prompt(item_data)
            embedding = self.get_embedding(prompt)
            embeddings[item_id] = embedding
        return embeddings

# Example usage:
def main():
    # Example items (similar to Amazon product data used in the paper)
    items = {
        "B001": {
            "title": "Professional Makeup Brush Set",
            "description": "Set of 12 professional makeup brushes with synthetic bristles",
            "categories": ["Beauty", "Makeup", "Brushes & Tools"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 1250},
            "price": 24.99
        },
        "B002": {
            "title": "Eyeshadow Palette - Natural Colors",
            "description": "15 highly pigmented natural eyeshadow colors",
            "categories": ["Beauty", "Makeup", "Eyes", "Eyeshadow"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 2100},
            "price": 19.99
        }
    }
    
    # Generate embeddings
    generator = ItemEmbeddingGenerator()
    item_embeddings = generator.generate_item_embeddings(items)
    
    # Use with STAR retrieval
    retrieval = STARRetrieval(semantic_weight=0.5, temporal_decay=0.7, history_length=3)
    retrieval.compute_semantic_relationships(item_embeddings)
    
    # Example output of embeddings and similarity
    print(f"Embedding dimension: {len(item_embeddings['B001'])}")
    semantic_sim = retrieval.semantic_matrix[0,1]
    print(f"Semantic similarity between items: {semantic_sim:.3f}")

if __name__ == "__main__":
    main()
```

### src/main.py

```python
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
```

### src/model_analysis.py

```python
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
```

### src/optimized_collaborative_relationships.py

```python
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import itertools

class OptimizedCollaborativeProcessor:
    def __init__(self, chunk_size=1000, n_threads=4):
        self.user_item_interactions = defaultdict(list)
        self.item_to_idx = None
        self.chunk_size = chunk_size
        self.n_threads = n_threads
        
    def process_interactions(self, interactions, item_mapping=None):
        """Process user-item interactions with optimized storage"""
        self.item_to_idx = item_mapping or {}
        self.user_item_interactions.clear()
        
        # Use numpy arrays for faster processing
        interaction_data = np.array([(user_id, item_id, timestamp, rating) 
                                   for user_id, item_id, timestamp, rating in interactions
                                   if item_id in self.item_to_idx], 
                                  dtype=[('user_id', 'U50'), ('item_id', 'U50'), 
                                       ('timestamp', 'U50'), ('rating', 'f4')])
        
        # Group by user_id efficiently using numpy operations
        user_ids, indices = np.unique(interaction_data['user_id'], return_index=True)
        sorted_indices = np.argsort(interaction_data['user_id'])
        
        for i in range(len(user_ids)):
            start_idx = indices[i]
            end_idx = indices[i + 1] if i < len(user_ids) - 1 else len(interaction_data)
            user_data = interaction_data[start_idx:end_idx]
            self.user_item_interactions[user_ids[i]] = list(zip(user_data['item_id'], 
                                                              user_data['timestamp'],
                                                              user_data['rating']))

    def _process_user_chunk(self, user_chunk: List[str]) -> lil_matrix:
        """Process a chunk of users to compute partial relationship matrix"""
        matrix_size = len(self.item_to_idx)
        chunk_matrix = lil_matrix((matrix_size, matrix_size))
        
        for user_id in user_chunk:
            items = [item_id for item_id, _, _ in self.user_item_interactions[user_id]]
            if len(items) > 1:
                # Use numpy operations for faster computation
                item_indices = np.array([self.item_to_idx[item] for item in items])
                combinations = np.array(list(itertools.combinations(item_indices, 2)))
                
                if len(combinations) > 0:
                    # Batch update the matrix
                    chunk_matrix[combinations[:, 0], combinations[:, 1]] += 1
                    chunk_matrix[combinations[:, 1], combinations[:, 0]] += 1
        
        return chunk_matrix

    def compute_collaborative_relationships(self, matrix_size):
        """Compute collaborative relationships with parallel processing"""
        # Split users into chunks for parallel processing
        users = list(self.user_item_interactions.keys())
        user_chunks = [users[i:i + self.chunk_size] 
                      for i in range(0, len(users), self.chunk_size)]
        
        # Process chunks in parallel
        partial_matrices = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            partial_matrices = list(executor.map(self._process_user_chunk, user_chunks))
        
        # Combine partial matrices efficiently
        final_matrix = sum(partial_matrices)
        
        # Convert to CSR format and normalize
        final_matrix = final_matrix.tocsr()
        
        # Vectorized normalization
        row_sums = np.asarray(final_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        diag = 1 / row_sums
        final_matrix = final_matrix.multiply(diag[:, np.newaxis])
        
        return final_matrix

    def get_item_co_occurrences(self, item_id: str, min_count: int = 1) -> Dict[str, int]:
        """
        Get number of users who interacted with both items, with minimum threshold
        
        Args:
            item_id: ID of item to get co-occurrences for
            min_count: Minimum co-occurrence count to include
            
        Returns:
            Dictionary mapping item IDs to co-occurrence counts
        """
        if item_id not in self.item_to_idx:
            return {}
            
        item_idx = self.item_to_idx[item_id]
        
        # Use CSR matrix properties for efficient row/column operations
        if not hasattr(self, 'interaction_matrix'):
            return {}
            
        item_vector = self.interaction_matrix[item_idx].toarray().flatten()
        
        # Vectorized computation of co-occurrences
        co_occurrence_vector = self.interaction_matrix.multiply(item_vector).sum(axis=1).A1
        
        # Filter and convert to dictionary
        co_occurrences = {}
        for other_id, other_idx in self.item_to_idx.items():
            if other_id != item_id and co_occurrence_vector[other_idx] >= min_count:
                co_occurrences[other_id] = int(co_occurrence_vector[other_idx])
                    
        return co_occurrences

# Example usage with performance monitoring
def main():
    import time
    
    # Generate larger test dataset
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    n_interactions = 10000
    
    users = [f"user{i}" for i in range(n_users)]
    items = [f"item{i}" for i in range(n_items)]
    item_mapping = {item: idx for idx, item in enumerate(items)}
    
    interactions = [
        (np.random.choice(users),
         np.random.choice(items),
         f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
         np.random.randint(1, 6))
        for _ in range(n_interactions)
    ]
    
    # Initialize processor with optimized parameters
    processor = OptimizedCollaborativeProcessor(chunk_size=100, n_threads=4)
    
    # Measure processing time
    start_time = time.time()
    processor.process_interactions(interactions, item_mapping)
    process_time = time.time() - start_time
    print(f"Processing time: {process_time:.2f} seconds")
    
    # Measure relationship computation time
    start_time = time.time()
    collab_matrix = processor.compute_collaborative_relationships(
        matrix_size=len(processor.item_to_idx)
    )
    compute_time = time.time() - start_time
    print(f"Computation time: {compute_time:.2f} seconds")
    
    # Example co-occurrence lookup
    sample_item = items[0]
    start_time = time.time()
    co_occurrences = processor.get_item_co_occurrences(sample_item, min_count=2)
    lookup_time = time.time() - start_time
    print(f"Lookup time: {lookup_time:.2f} seconds")
    print(f"Number of co-occurrences found: {len(co_occurrences)}")

if __name__ == "__main__":
    main()
```

### src/star_retrieval.py

```python
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.sparse import csr_matrix
from tqdm import tqdm

class STARRetrieval:
    def __init__(self, 
                 semantic_weight: float = 0.5,    
                 temporal_decay: float = 0.7,     
                 history_length: int = 3):        
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length
        
        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}

    def compute_semantic_relationships(self, 
                                    item_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute semantic similarity matrix from item embeddings"""
        print("\nComputing semantic relationships...")
        
        sorted_items = sorted(item_embeddings.keys())
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_items = len(self.item_to_idx)
        
        # Convert embeddings to array and normalize
        embeddings_array = np.zeros((n_items, next(iter(item_embeddings.values())).shape[0]))
        for item_id, embedding in item_embeddings.items():
            embeddings_array[self.item_to_idx[item_id]] = embedding
            
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        embeddings_array = embeddings_array / norms
        
        # Compute similarities using cosine distance
        semantic_matrix = 1 - cdist(embeddings_array, embeddings_array, metric='cosine')
        np.fill_diagonal(semantic_matrix, 0)
        
        self.semantic_matrix = np.maximum(0, semantic_matrix)
        return self.semantic_matrix

    def score_candidates(self,
                        user_history: List[str],
                        ratings: List[float],
                        candidate_items: List[str] = None,
                        top_k: int = None) -> List[Tuple[str, float]]:
        if self.collaborative_matrix is None:
            raise ValueError("Collaborative matrix not set. Run compute_collaborative_relationships first.")

        """Score candidate items based on user history"""
        if len(user_history) > self.history_length:
            user_history = user_history[-self.history_length:]
            ratings = ratings[-self.history_length:]
        
        if candidate_items is None:
            candidate_items = [item for item in self.item_to_idx.keys() 
                             if item not in set(user_history)]
        
        scores = {}
        n = len(user_history)
        
        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue
                
            cand_idx = self.item_to_idx[candidate]
            score = 0.0
            
            for t, (hist_item, rating) in enumerate(zip(reversed(user_history), 
                                                      reversed(ratings))):
                if hist_item not in self.item_to_idx:
                    continue
                    
                hist_idx = self.item_to_idx[hist_item]
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]
                
                combined_sim = (self.semantic_weight * sem_sim + 
                              (1 - self.semantic_weight) * collab_sim)
                
                score += (1/n) * rating * (self.temporal_decay ** t) * combined_sim
            
            scores[candidate] = score
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_items = sorted_items[:top_k]
            
        return sorted_items
        

    def get_similar_items(self, item_id: str, k: int = 10, use_collaborative: bool = True) -> List[Tuple[str, float]]:
        """Get most similar items using combined similarity"""
        if item_id not in self.item_to_idx:
            return []
            
        idx = self.item_to_idx[item_id]
        similarities = np.zeros(len(self.item_to_idx))
        
        for j in range(len(self.item_to_idx)):
            if j != idx:
                sem_sim = self.semantic_matrix[idx, j]
                collab_sim = self.collaborative_matrix[idx, j] if use_collaborative else 0
                similarities[j] = (
                    self.semantic_weight * sem_sim + 
                    (1 - self.semantic_weight) * collab_sim
                )
        
        # Get top-k similar items
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            (self.idx_to_item[j], float(similarities[j]))
            for j in top_indices
        ]



```

### src/utils.py

```python
import os
import json
import gzip
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import ast

def load_amazon_dataset(
    category: str, 
    data_dir: str = "data", 
    min_interactions: int = 5,
    max_reviews: int = None
) -> List[Dict[str, Any]]:
    """
    Load Amazon review dataset with proper filtering
    
    Args:
        category: Category name ('beauty', 'toys_and_games', or 'sports_and_outdoors')
        data_dir: Directory containing the data files
        min_interactions: Minimum number of interactions required for users and items
        max_reviews: Maximum number of reviews to load (None for all)
        
    Returns:
        List of filtered review dictionaries
    """
    # Map category names to file names
    category_files = {
        'beauty': 'reviews_Beauty_5.json.gz',
        'toys_and_games': 'reviews_Toys_and_Games_5.json.gz',
        'sports_and_outdoors': 'reviews_Sports_and_Outdoors_5.json.gz'
    }
    
    if category.lower() not in category_files:
        raise ValueError(f"Category must be one of: {list(category_files.keys())}")
        
    file_path = os.path.join(data_dir, category_files[category.lower()])
    
    # First pass: Count interactions
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    
    print("First pass: Counting interactions...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_reviews and i >= max_reviews:
                break
            review = json.loads(line.strip())
            user_counts[review['reviewerID']] += 1
            item_counts[review['asin']] += 1
    
    # Filter users and items with minimum interactions
    valid_users = {user for user, count in user_counts.items() if count >= min_interactions}
    valid_items = {item for item, count in item_counts.items() if count >= min_interactions}
    
    print(f"\nBefore filtering:")
    print(f"Total users: {len(user_counts)}")
    print(f"Total items: {len(item_counts)}")
    
    print(f"\nAfter filtering (>= {min_interactions} interactions):")
    print(f"Valid users: {len(valid_users)}")
    print(f"Valid items: {len(valid_items)}")
    
    # Second pass: Load filtered data
    filtered_data = []
    print("\nSecond pass: Loading filtered data...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_reviews and i >= max_reviews:
                break
            review = json.loads(line.strip())
            if (review['reviewerID'] in valid_users and 
                review['asin'] in valid_items):
                filtered_data.append(review)
    
    print(f"\nFinal filtered reviews: {len(filtered_data)}")
    return filtered_data

def load_amazon_metadata(
    category: str, 
    data_dir: str = "data", 
    min_interactions: int = 5,
    max_items: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load Amazon product metadata with proper filtering
    
    Args:
        category: Category name
        data_dir: Directory containing the data files
        min_interactions: Only include items with this many interactions
        max_items: Maximum number of items to load (None for all)
    """
    metadata_files = {
        'beauty': 'meta_Beauty.json.gz',
        'toys_and_games': 'meta_Toys_and_Games.json.gz',
        'sports_and_outdoors': 'meta_Sports_and_Outdoors.json.gz'
    }
    
    file_path = os.path.join(data_dir, metadata_files[category.lower()])
    
    # First load review data to get valid items
    reviews = load_amazon_dataset(category, data_dir, min_interactions)
    valid_items = {review['asin'] for review in reviews}
    
    metadata = {}
    error_count = 0
    
    print("\nLoading filtered metadata...")
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line_number, line in enumerate(f, 1):
            if max_items and len(metadata) >= max_items:
                break
                
            try:
                # Clean the line
                line = line.strip()
                if not line:
                    continue
                
                # Try parsing as Python literal first
                try:
                    item = ast.literal_eval(line)
                except:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        line = line.replace("'", '"')
                        item = json.loads(line)
                
                # Only include items that passed interaction filter
                if isinstance(item, dict) and 'asin' in item and item['asin'] in valid_items:
                    metadata[item['asin']] = item
                    
            except Exception as e:
                error_count += 1
                if error_count < 10:
                    print(f"Warning: Error processing line {line_number}: {str(e)}")
                continue
                
    print(f"Loaded metadata for {len(metadata)} items with {error_count} errors")
    return metadata

def get_items_from_data(reviews: List[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract unique items from review data and combine with metadata
    Only includes items that passed the interaction filter
    """
    items = {}
    
    # Get unique items from reviews
    unique_asins = set(review['asin'] for review in reviews)
    
    # Combine review and metadata information
    for asin in unique_asins:
        item_info = {
            'asin': asin,
            'title': '',
            'description': '',
            'categories': [],
            'brand': '',
            'price': None,
            'salesRank': {}
        }
        
        # Add metadata if available
        if asin in metadata:
            meta = metadata[asin]
            item_info.update({
                'title': meta.get('title', ''),
                'description': meta.get('description', ''),
                'categories': meta.get('categories', []),
                'brand': meta.get('brand', ''),
                'price': meta.get('price', None),
                'salesRank': meta.get('salesRank', {})
            })
            
        items[asin] = item_info
        
    return items


def get_training_interactions(reviews):
    sorted_reviews = sorted(reviews, key=lambda x: (x['reviewerID'], x['unixReviewTime']))
    user_reviews = defaultdict(list)
    
    # Group reviews by user
    for review in sorted_reviews:
        user_reviews[review['reviewerID']].append(review)
        
    training_interactions = []
    for user_id, group in user_reviews.items():
        if len(group) >= 5:  # Match min_sequence_length
            test_time = group[-1]['unixReviewTime']
            train_history = group[:-2]
            
            for review in train_history:
                if review['unixReviewTime'] < test_time:
                    training_interactions.append((
                        review['reviewerID'], 
                        review['asin'],
                        review['unixReviewTime'], 
                        1.0
                    ))
    return training_interactions

def print_metrics_table(metrics: Dict[str, float], dataset: str) -> None:
    """
    Print evaluation metrics in a formatted table
    
    Args:
        metrics: Dictionary of metric names and values
        dataset: Name of the dataset
    """
    print(f"\nResults for {dataset} dataset:")
    print("-" * 30)
    print(f"{'Metric':<15} {'Score':<10}")
    print("-" * 30)
    
    # Sort metrics by name for consistent display
    for metric_name in sorted(metrics.keys()):
        score = metrics[metric_name]
        print(f"{metric_name:<15} {score:.4f}")
    
    print("-" * 30)

def create_requirements_txt():
    """
    Create requirements.txt file with necessary dependencies
    """
    requirements = """
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
faiss-cpu>=1.7.0
google-cloud-aiplatform>=1.0.0
scikit-learn>=0.24.0
tqdm>=4.62.0
    """.strip()
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)

# Example usage
def main():
    # Example of loading and processing data
    category = 'beauty'
    
    # Load data
    print(f"Loading {category} dataset...")
    reviews = load_amazon_dataset(category)
    metadata = load_amazon_metadata(category)
    
    # Process items
    items = get_items_from_data(reviews, metadata)
    print(f"Found {len(items)} unique items")
    
    # Get training interactions
    interactions = get_training_interactions(reviews)
    print(f"Extracted {len(interactions)} training interactions")
    
    # Example metrics
    example_metrics = {
        'H@5': 0.068,
        'N@5': 0.048,
        'H@10': 0.098,
        'N@10': 0.057
    }
    
    # Print results
    print_metrics_table(example_metrics, category)
    
    # Create requirements file
    create_requirements_txt()
    print("\nCreated requirements.txt")

if __name__ == "__main__":
    main()
```

