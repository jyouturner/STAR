# Consolidated Poetry Project Files

## Project Structure

```
check_data.py
download_data.py
pyproject.toml
src/collaborative_relationships.py
src/data_debug.py
src/data_quality.py
src/evaluation_metrics.py
src/item_embeddings_vertex_ai.py
src/main.py
src/model_analysis.py
src/optimized_collaborative_relationships.py
src/star_retrieval.py
src/temporal_utils.py
src/utils.py
```

## File Contents

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

### src/data_debug.py

```python
from collections import defaultdict, Counter
import random
from typing import List, Dict, Set, Tuple
import numpy as np
from datetime import datetime

class DataDebugger:
    @staticmethod
    def debug_print_user_history(reviews: List[Dict], num_users: int = 5):
        """Print detailed history for random users"""
        user_map = defaultdict(list)
        for r in reviews:
            user_map[r['reviewerID']].append(r)
        
        # Get mix of high/medium/low activity users
        user_counts = [(uid, len(reviews)) for uid, reviews in user_map.items()]
        user_counts.sort(key=lambda x: x[1])
        n = len(user_counts)
        
        selected_users = (
            # Low activity
            [uid for uid, _ in user_counts[:n//3]][0:num_users//3] +
            # Medium activity
            [uid for uid, _ in user_counts[n//3:2*n//3]][0:num_users//3] +
            # High activity
            [uid for uid, _ in user_counts[2*n//3:]][0:num_users//3]
        )
        
        print("\n=== Sample User Histories ===")
        for user_id in selected_users:
            user_reviews = sorted(user_map[user_id], key=lambda x: x['unixReviewTime'])
            print(f"\nUser: {user_id} ({len(user_reviews)} reviews)")
            prev_time = None
            for rv in user_reviews:
                time = rv['unixReviewTime']
                time_str = datetime.fromtimestamp(time).strftime('%Y-%m-%d')
                
                # Check if timestamps are in order
                if prev_time and time < prev_time:
                    print("  ⚠️ OUT OF ORDER!")
                prev_time = time
                
                print(f"  {time_str}: {rv['asin']}, rating={rv['overall']}")
                if 'summary' in rv:
                    print(f"    Summary: {rv['summary'][:100]}...")

    @staticmethod
    def check_for_duplicates(reviews: List[Dict]) -> Tuple[int, List[Tuple]]:
        """Check for exact and partial duplicates"""
        # Exact duplicates (user, item, time)
        exact_seen = set()
        exact_dupes = []
        
        # Partial duplicates (user, item)
        partial_seen = defaultdict(list)
        partial_dupes = []
        
        for r in reviews:
            # Check exact duplicates
            exact_key = (r['reviewerID'], r['asin'], r['unixReviewTime'])
            if exact_key in exact_seen:
                exact_dupes.append(exact_key)
            else:
                exact_seen.add(exact_key)
            
            # Check partial duplicates
            partial_key = (r['reviewerID'], r['asin'])
            if partial_key in partial_seen:
                partial_dupes.append((partial_key, r['unixReviewTime']))
            partial_seen[partial_key].append(r['unixReviewTime'])
        
        print("\n=== Duplicate Analysis ===")
        print(f"Total reviews: {len(reviews)}")
        print(f"Exact duplicates: {len(exact_dupes)}")
        print(f"Users with multiple reviews for same item: {len(partial_dupes)}")
        
        if exact_dupes:
            print("\nSample exact duplicates:")
            for dupe in exact_dupes[:3]:
                print(f"  User {dupe[0]}, Item {dupe[1]}, Time {dupe[2]}")
                
        return len(exact_dupes), exact_dupes

    @staticmethod
    def analyze_ratings(reviews: List[Dict]):
        """Analyze rating distribution and patterns"""
        rating_counts = Counter(r['overall'] for r in reviews)
        user_rating_counts = defaultdict(Counter)
        
        for r in reviews:
            user_rating_counts[r['reviewerID']][r['overall']] += 1
        
        print("\n=== Rating Analysis ===")
        print("Overall rating distribution:")
        total = sum(rating_counts.values())
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            print(f"  {rating}: {count} ({count/total*100:.1f}%)")
            
        # Check for users with unusual patterns
        unusual_users = []
        for user_id, counts in user_rating_counts.items():
            if len(counts) == 1:  # User gives same rating always
                unusual_users.append((user_id, list(counts.keys())[0]))
                
        if unusual_users:
            print("\nUsers with single rating value:")
            for user_id, rating in unusual_users[:5]:
                print(f"  User {user_id}: all {rating}s ({user_rating_counts[user_id][rating]} reviews)")

    @staticmethod
    def verify_evaluation_splits(
        test_sequences: List[Tuple],
        reviews: List[Dict]
    ):
        """Verify test sequences are properly created"""
        print("\n=== Evaluation Split Verification ===")
        
        # Build user timeline
        user_reviews = defaultdict(list)
        for r in reviews:
            user_reviews[r['reviewerID']].append((r['asin'], r['unixReviewTime']))
            
        # Check each test sequence
        issues = 0
        for user_id, history, next_item in test_sequences:
            # Get user's reviews sorted by time
            timeline = sorted(user_reviews[user_id], key=lambda x: x[1])
            
            # Verify next_item is truly last
            if timeline[-1][0] != next_item:
                issues += 1
                print(f"\nIssue with user {user_id}:")
                print(f"  Test item: {next_item}")
                print(f"  Actually last: {timeline[-1][0]}")
                if issues >= 5:  # Show first 5 issues only
                    break
                    
        print(f"\nFound {issues} sequences where test item isn't truly last")
        
        # Verify history lengths
        history_lengths = Counter(len(h) for _, h, _ in test_sequences)
        print("\nHistory length distribution:")
        for length in sorted(history_lengths.keys()):
            count = history_lengths[length]
            print(f"  Length {length}: {count} sequences")

    @staticmethod
    def debug_negative_sampling(
        test_sequences: List[Tuple],
        recommender,
        user_all_items: Dict[str, Set[str]],
        n_checks: int = 5
    ):
        """Verify negative sampling process"""
        print("\n=== Negative Sampling Debug ===")
        
        seq_sample = random.sample(test_sequences, min(n_checks, len(test_sequences)))
        for user_id, history, next_item in seq_sample:
            # Get excluded items
            excluded = user_all_items.get(user_id, set(history) | {next_item})
            
            # Sample negatives
            valid_items = set(recommender.item_to_idx.keys())
            negatives = valid_items - excluded
            
            print(f"\nUser: {user_id}")
            print(f"History length: {len(history)}")
            print(f"Total valid items: {len(valid_items)}")
            print(f"Excluded items: {len(excluded)}")
            print(f"Available negatives: {len(negatives)}")
            
            if len(negatives) < 99:
                print("⚠️ WARNING: Less than 99 possible negative samples!")

    @staticmethod
    def analyze_collaborative_matrix(matrix: np.ndarray):
        """Analyze collaborative relationship matrix"""
        print("\n=== Collaborative Matrix Analysis ===")
        
        # Basic statistics
        nonzero = matrix[matrix > 0]
        print(f"Shape: {matrix.shape}")
        print(f"Density: {len(nonzero)/(matrix.shape[0]*matrix.shape[1]):.4f}")
        print(f"Mean nonzero value: {nonzero.mean():.4f}")
        print(f"Max value: {matrix.max():.4f}")
        
        # Check diagonal
        diag = np.diag(matrix)
        if np.any(diag != 0):
            print("⚠️ WARNING: Non-zero diagonal elements found!")
            print(f"Non-zero diagonals: {np.count_nonzero(diag)}")
            
        # Distribution of values
        percentiles = np.percentile(nonzero, [25, 50, 75])
        print("\nValue distribution (nonzero):")
        print(f"  25th percentile: {percentiles[0]:.4f}")
        print(f"  Median: {percentiles[1]:.4f}")
        print(f"  75th percentile: {percentiles[2]:.4f}")

```

### src/data_quality.py

```python
from collections import defaultdict
from typing import Dict, List, Tuple
import re

class DataQualityChecker:
    def __init__(self, min_description_length: int = 100,
                 min_title_length: int = 10,
                 required_fields: List[str] = None):
        """
        Initialize data quality checker
        
        Args:
            min_description_length: Minimum characters in description
            min_title_length: Minimum characters in title
            required_fields: List of required metadata fields
        """
        self.min_description_length = min_description_length
        self.min_title_length = min_title_length
        self.required_fields = required_fields or ['title', 'description', 'categories', 'brand', 'price']

    def check_text_quality(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text has sufficient information content"""
        issues = []
        
        if not text:
            return False, ["Empty text"]
            
        # Check if text is just placeholder content
        placeholder_patterns = [
            r'n/a', r'not available', r'no description',
            r'^\s*$',  # only whitespace
            r'^\W+$'   # only special characters
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, text.lower()):
                issues.append(f"Contains placeholder content: {pattern}")
                
        # Check for very repetitive content
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                issues.append("Text is highly repetitive")
                
        return len(issues) == 0, issues

    def check_item_metadata(self, item_data: Dict) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check if item has sufficient metadata quality
        
        Returns:
            Tuple of (passed_check, issues_dict)
        """
        issues = {}
        
        # Check required fields presence
        for field in self.required_fields:
            if field not in item_data or not item_data[field]:
                issues[field] = [f"Missing {field}"]
                
        # Check title quality
        if 'title' in item_data:
            title = str(item_data['title'])
            if len(title) < self.min_title_length:
                issues['title'] = [f"Title too short ({len(title)} chars)"]
            title_quality, title_issues = self.check_text_quality(title)
            if not title_quality:
                issues.setdefault('title', []).extend(title_issues)
                
        # Check description quality
        if 'description' in item_data:
            desc = str(item_data['description'])
            if len(desc) < self.min_description_length:
                issues['description'] = [f"Description too short ({len(desc)} chars)"]
            desc_quality, desc_issues = self.check_text_quality(desc)
            if not desc_quality:
                issues.setdefault('description', []).extend(desc_issues)
                
        # Check categories
        if 'categories' in item_data:
            cats = item_data['categories']
            if not cats or (isinstance(cats, list) and not cats[0]):
                issues['categories'] = ["Empty categories"]
            elif isinstance(cats[0], list) and len(cats[0]) < 2:
                issues['categories'] = ["Insufficient category hierarchy depth"]
                
        # Check numerical fields
        if 'price' in item_data:
            try:
                price = float(item_data['price'])
                if price <= 0:
                    issues['price'] = ["Invalid price value"]
            except (ValueError, TypeError):
                issues['price'] = ["Price not a valid number"]
                
        # Check sales rank if present
        if 'salesRank' in item_data:
            if not isinstance(item_data['salesRank'], dict):
                issues['salesRank'] = ["Sales rank not in proper format"]
                
        return len(issues) == 0, issues

    def filter_items(self, items: Dict) -> Tuple[Dict, Dict[str, Dict[str, List[str]]]]:
        """
        Filter items based on metadata quality
        
        Returns:
            Tuple of (filtered_items, rejected_items_with_reasons)
        """
        filtered_items = {}
        rejected_items = {}
        
        print("\nChecking data quality for items...")
        total_items = len(items)
        
        for idx, (item_id, item_data) in enumerate(items.items()):
            if idx % 1000 == 0:
                print(f"Checking item {idx}/{total_items}")
                
            passed, issues = self.check_item_metadata(item_data)
            
            if passed:
                filtered_items[item_id] = item_data
            else:
                rejected_items[item_id] = issues
        
        print(f"\nData quality check complete:")
        print(f"Passed: {len(filtered_items)} items")
        print(f"Rejected: {len(rejected_items)} items")
        
        # Print sample rejection reasons
        if rejected_items:
            print("\nSample rejection reasons:")
            for item_id, issues in list(rejected_items.items())[:3]:
                print(f"\nItem {item_id}:")
                for field, field_issues in issues.items():
                    print(f"  {field}: {', '.join(field_issues)}")
        
        return filtered_items, rejected_items

def verify_item_coverage(items: Dict) -> Dict[str, float]:
    """Analyze metadata coverage across items"""
    total_items = len(items)
    if total_items == 0:
        return {}
        
    coverage = {}
    field_lengths = defaultdict(list)
    
    # Check all possible fields
    all_fields = set()
    for item_data in items.values():
        all_fields.update(item_data.keys())
    
    # Calculate coverage for each field
    for field in all_fields:
        present_count = sum(1 for item in items.values() if field in item and item[field])
        coverage[f"{field}_present"] = present_count / total_items
        
        # Calculate length statistics for text fields
        if field in ['title', 'description']:
            lengths = [len(str(item[field])) for item in items.values() 
                      if field in item and item[field]]
            if lengths:
                coverage[f"{field}_avg_length"] = sum(lengths) / len(lengths)
                coverage[f"{field}_min_length"] = min(lengths)
                coverage[f"{field}_max_length"] = max(lengths)
    
    return coverage

```

### src/evaluation_metrics.py

```python
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
```

### src/item_embeddings_vertex_ai.py

```python
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingInput
from typing import Dict, List, Set
import numpy as np

class ItemEmbeddingGenerator:
    def __init__(self, 
                output_dimension: int = 768,
                include_fields: Set[str] = None):
        """
        Initialize generator with configurable fields
        
        Args:
            output_dimension: Embedding dimension (default matches paper)
            include_fields: Set of fields to include in prompt
                          (title, description, category, brand, price, sales_rank)
        """
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        self.output_dimension = output_dimension
        # Default to minimal set of fields if none specified
        self.include_fields = include_fields or {'title', 'description', 'category'}
        
    def create_embedding_input(self, item_data: Dict) -> TextEmbeddingInput:
        """Create simplified prompt following paper's structure"""
        prompt_parts = []
        
        # Handle description first
        if 'description' in self.include_fields:
            desc = str(item_data.get('description', '')).strip()
            if desc:
                prompt_parts.append("description:")
                prompt_parts.append(desc)
        
        # Add basic fields with minimal formatting
        if 'title' in self.include_fields and (title := item_data.get('title')):
            prompt_parts.append(f"title: {title}")
            
        if 'category' in self.include_fields and (cats := item_data.get('categories')):
            if isinstance(cats[0], list):
                category_str = " > ".join(cats[0])
            else:
                category_str = " > ".join(cats)
            if category_str:
                prompt_parts.append(f"category: {category_str}")
                
        # Optional fields based on configuration
        if 'price' in self.include_fields and (price := item_data.get('price')):
            prompt_parts.append(f"price: {price}")
            
        if 'brand' in self.include_fields and (brand := item_data.get('brand')):
            # Skip ASIN-like brands
            if not (brand.startswith('B0') and len(brand) >= 10):
                prompt_parts.append(f"brand: {brand}")
                
        if 'sales_rank' in self.include_fields and (rank := item_data.get('salesRank')):
            prompt_parts.append(f"salesRank: {str(rank)}")
            
        return TextEmbeddingInput(
            task_type="RETRIEVAL_DOCUMENT",
            title=item_data.get('title', ''),
            text='\n'.join(prompt_parts)
        )

    def generate_item_embeddings(self, items: Dict) -> Dict[str, np.ndarray]:
        """Generate embeddings with simplified prompts"""
        embeddings = {}
        total_items = len(items)
        batch_size = 5
        
        print(f"\nGenerating embeddings for {total_items} items...")
        print(f"Including fields: {sorted(self.include_fields)}")
        
        # Process items in batches
        item_list = list(items.items())
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch = item_list[batch_start:batch_end]
            
            if batch_start % 1000 == 0:
                print(f"Processing items {batch_start}-{batch_end}/{total_items}")
            
            try:
                embedding_inputs = [
                    self.create_embedding_input(item_data)
                    for _, item_data in batch
                ]
                
                kwargs = {}
                if self.output_dimension:
                    kwargs['output_dimensionality'] = self.output_dimension
                    
                predictions = self.model.get_embeddings(embedding_inputs, **kwargs)
                
                for (item_id, _), embedding in zip(batch, predictions):
                    embeddings[item_id] = np.array(embedding.values)
                    
                    if len(embeddings) == 1:
                        print(f"Embedding dimension: {len(embedding.values)}")
                
            except Exception as e:
                print(f"Error in batch {batch_start}-{batch_end}: {str(e)}")
                continue
        
        return embeddings

    def debug_prompt(self, items: Dict, num_samples: int = 3):
        """Debug utility to verify prompt structure"""
        print("\nSample prompts with current field selection:")
        print("=" * 80)
        print(f"Including fields: {sorted(self.include_fields)}")
        print("=" * 80)
        
        for item_id in list(items.keys())[:num_samples]:
            print(f"\nItem ID: {item_id}")
            print("-" * 40)
            embedding_input = self.create_embedding_input(items[item_id])
            print("Title:", embedding_input.title)
            print("\nText:", embedding_input.text)
            print("=" * 80)

if __name__ == "__main__":
    # Example usage
    items = {
        "item1": {"title": "Product 1", "description": "Description 1", "categories": ["Category 1", "Subcategory 1"]},
        "item2": {"title": "Product 2", "description": "Description 2", "categories": ["Category 2", "Subcategory 2"]},
        "item3": {"title": "Product 3", "description": "Description 3", "categories": ["Category 3", "Subcategory 3"]}
    }
    generator = ItemEmbeddingGenerator()
    generator.debug_prompt(items, num_samples=3)

```

### src/main.py

```python
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

### src/temporal_utils.py

```python
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

class TemporalProcessor:
    @staticmethod
    def sort_reviews_chronologically(reviews: List[Dict]) -> List[Dict]:
        """Sort all reviews by user ID and timestamp"""
        return sorted(reviews, key=lambda x: (x['reviewerID'], x['unixReviewTime']))

    @staticmethod
    def check_temporal_ordering(reviews: List[Dict]) -> List[Dict]:
        """
        Check and fix temporal ordering issues
        Returns list of problematic sequences
        """
        issues = []
        user_sequences = defaultdict(list)
        
        # Group by user
        for review in reviews:
            user_sequences[review['reviewerID']].append(review)
            
        # Check each user's sequence
        for user_id, sequence in user_sequences.items():
            # Sort by timestamp
            sorted_sequence = sorted(sequence, key=lambda x: x['unixReviewTime'])
            
            # Check if original sequence was out of order
            if sequence != sorted_sequence:
                # Record the issue
                issues.append({
                    'user_id': user_id,
                    'original_sequence': [(r['asin'], r['unixReviewTime']) for r in sequence],
                    'sorted_sequence': [(r['asin'], r['unixReviewTime']) for r in sorted_sequence]
                })
                
        return issues

    @staticmethod
    def verify_train_test_chronology(
        train_interactions: List[Tuple],
        test_sequences: List[Tuple],
        validation_sequences: List[Tuple] = None
    ) -> Dict:
        """
        Verify chronological integrity of train/test split
        Returns dictionary of issues found
        """
        issues = {
            'train_after_test': [],
            'train_after_val': [],
            'val_after_test': []
        }
        
        # Build timeline for each user
        user_timelines = defaultdict(dict)
        
        # Add training interactions to timeline
        for user_id, item_id, timestamp, _ in train_interactions:
            if user_id not in user_timelines:
                user_timelines[user_id] = {'train': [], 'val': None, 'test': None}
            user_timelines[user_id]['train'].append((item_id, timestamp))
            
        # Add test sequences
        for user_id, history, test_item in test_sequences:
            if user_id in user_timelines:
                # Find test item timestamp (should be in original reviews)
                test_timestamp = max(t for _, t in user_timelines[user_id]['train'] 
                                  if _ == test_item)
                user_timelines[user_id]['test'] = (test_item, test_timestamp)
                
        # Add validation sequences if provided
        if validation_sequences:
            for user_id, history, val_item in validation_sequences:
                if user_id in user_timelines:
                    # Find validation item timestamp
                    val_timestamp = max(t for _, t in user_timelines[user_id]['train'] 
                                     if _ == val_item)
                    user_timelines[user_id]['val'] = (val_item, val_timestamp)
        
        # Check chronological integrity
        for user_id, timeline in user_timelines.items():
            train_times = [t for _, t in timeline['train']]
            test_time = timeline['test'][1] if timeline['test'] else None
            val_time = timeline['val'][1] if timeline['val'] else None
            
            # Check if any training interaction is after test
            if test_time and any(t > test_time for t in train_times):
                issues['train_after_test'].append({
                    'user_id': user_id,
                    'test_time': test_time,
                    'problematic_train': [(item, t) for item, t in timeline['train'] 
                                        if t > test_time]
                })
                
            # Check validation chronology
            if val_time:
                if any(t > val_time for t in train_times):
                    issues['train_after_val'].append({
                        'user_id': user_id,
                        'val_time': val_time,
                        'problematic_train': [(item, t) for item, t in timeline['train'] 
                                            if t > val_time]
                    })
                if test_time and val_time > test_time:
                    issues['val_after_test'].append({
                        'user_id': user_id,
                        'val_time': val_time,
                        'test_time': test_time
                    })
                    
        return issues

    @staticmethod
    def print_chronology_check(reviews: List[Dict]):
        """Print detailed chronological analysis of reviews"""
        print("\n=== Chronological Analysis ===")
        
        # Sort reviews
        sorted_reviews = TemporalProcessor.sort_reviews_chronologically(reviews)
        
        # Check for ordering issues
        issues = TemporalProcessor.check_temporal_ordering(reviews)
        
        if issues:
            print(f"\nFound {len(issues)} users with temporal ordering issues:")
            for i, issue in enumerate(issues[:5], 1):  # Show first 5 issues
                print(f"\nIssue {i}:")
                print(f"User: {issue['user_id']}")
                print("Original sequence:")
                for item, time in issue['original_sequence']:
                    time_str = datetime.fromtimestamp(time).strftime('%Y-%m-%d')
                    print(f"  {time_str}: {item}")
                print("Sorted sequence:")
                for item, time in issue['sorted_sequence']:
                    time_str = datetime.fromtimestamp(time).strftime('%Y-%m-%d')
                    print(f"  {time_str}: {item}")
                    
        # Print overall statistics
        print("\nTemporal Statistics:")
        timestamps = [r['unixReviewTime'] for r in reviews]
        min_time = datetime.fromtimestamp(min(timestamps))
        max_time = datetime.fromtimestamp(max(timestamps))
        print(f"Date range: {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}")
        
        # Check for same-timestamp reviews
        user_day_counts = defaultdict(lambda: defaultdict(int))
        for r in reviews:
            day = datetime.fromtimestamp(r['unixReviewTime']).strftime('%Y-%m-%d')
            user_day_counts[r['reviewerID']][day] += 1
            
        multiple_reviews = sum(1 for user in user_day_counts.values() 
                             for count in user.values() if count > 1)
        print(f"\nUsers with multiple reviews on same day: {multiple_reviews}")
        
        if multiple_reviews > 0:
            print("\nSample cases of multiple reviews per day:")
            shown = 0
            for user_id, day_counts in user_day_counts.items():
                for day, count in day_counts.items():
                    if count > 1 and shown < 5:
                        print(f"User {user_id}: {count} reviews on {day}")
                        shown += 1
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

