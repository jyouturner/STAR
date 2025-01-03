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