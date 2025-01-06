import pandas as pd
import json
from typing import List, Dict, Union
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict

def load_purchase_dataset(
    filepath: Union[str, Path],
    min_interactions: int = 5,
    date_format: str = "%Y-%m-%d"
) -> List[Dict]:
    """
    Load purchase data from CSV or JSON file.
    
    Expected CSV format:
    user_id,item_id,timestamp,quantity
    user1,item123,2024-01-01,2
    user1,item456,2024-01-02,1
    
    Expected JSON format:
    [
        {
            "user_id": "user1",
            "item_id": "item123",
            "timestamp": "2024-01-01",
            "quantity": 2
        },
        ...
    ]
    
    Args:
        filepath: Path to purchase data file (CSV or JSON)
        min_interactions: Minimum number of interactions per user/item to keep
        date_format: Expected format of timestamp in data
        
    Returns:
        List of purchase records with standardized format
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Purchase data file not found: {filepath}")
    
    purchases = []
    
    try:
        if filepath.suffix.lower() == '.csv':
            # Load CSV
            df = pd.read_csv(filepath)
            required_cols = {'user_id', 'item_id', 'timestamp', 'quantity'}
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Convert to list of dicts
            purchases = df.to_dict('records')
            
        elif filepath.suffix.lower() == '.json':
            # Load JSON
            with open(filepath, 'r') as f:
                purchases = json.load(f)
                
            # Verify required fields
            required_fields = {'user_id', 'item_id', 'timestamp', 'quantity'}
            if not all(all(field in purchase for field in required_fields) 
                      for purchase in purchases):
                raise ValueError(f"JSON records must contain fields: {required_fields}")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Validate and clean data
        cleaned_purchases = []
        for purchase in purchases:
            try:
                # Parse and standardize timestamp
                timestamp = datetime.strptime(purchase['timestamp'], date_format)
                
                # Convert quantity to float
                quantity = float(purchase['quantity'])
                if quantity <= 0:
                    continue
                
                cleaned_purchases.append({
                    'user_id': str(purchase['user_id']),
                    'item_id': str(purchase['item_id']),
                    'timestamp': timestamp.strftime(date_format),
                    'quantity': quantity
                })
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping invalid purchase record: {purchase}. Error: {e}")
                continue
        
        # Filter by minimum interactions
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for purchase in cleaned_purchases:
            user_counts[purchase['user_id']] += 1
            item_counts[purchase['item_id']] += 1
        
        valid_users = {user for user, count in user_counts.items() 
                      if count >= min_interactions}
        valid_items = {item for item, count in item_counts.items() 
                      if count >= min_interactions}
        
        filtered_purchases = [
            purchase for purchase in cleaned_purchases
            if (purchase['user_id'] in valid_users and 
                purchase['item_id'] in valid_items)
        ]
        
        print(f"\nPurchase data loading summary:")
        print(f"Total records: {len(purchases)}")
        print(f"Valid records: {len(cleaned_purchases)}")
        print(f"Filtered records: {len(filtered_purchases)}")
        print(f"Unique users: {len(valid_users)}")
        print(f"Unique items: {len(valid_items)}")
        
        return filtered_purchases
        
    except Exception as e:
        raise ValueError(f"Error loading purchase data: {e}")

def load_item_metadata(
    filepath: Union[str, Path]
) -> Dict[str, Dict]:
    """
    Load item metadata from CSV or JSON file.
    
    Expected CSV format:
    item_id,title,description,categories,brand,price
    item123,"Product 1","Description 1","['Category1', 'Category2']",Brand1,19.99
    
    Expected JSON format:
    {
        "item123": {
            "title": "Product 1",
            "description": "Description 1",
            "categories": ["Category1", "Category2"],
            "brand": "Brand1",
            "price": 19.99
        },
        ...
    }
    
    Args:
        filepath: Path to metadata file (CSV or JSON)
        
    Returns:
        Dictionary mapping item_id to metadata dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    try:
        if filepath.suffix.lower() == '.csv':
            # Load CSV
            df = pd.read_csv(filepath)
            required_cols = {'item_id', 'title', 'description', 'categories', 'brand', 'price'}
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Convert categories from string to list if needed
            if 'categories' in df.columns:
                df['categories'] = df['categories'].apply(eval)
            
            # Convert to dictionary
            metadata = df.set_index('item_id').to_dict('index')
            
        elif filepath.suffix.lower() == '.json':
            # Load JSON
            with open(filepath, 'r') as f:
                metadata = json.load(f)
                
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Validate and clean metadata
        cleaned_metadata = {}
        for item_id, meta in metadata.items():
            try:
                cleaned_meta = {
                    'title': str(meta.get('title', '')),
                    'description': str(meta.get('description', '')),
                    'categories': list(meta.get('categories', [])),
                    'brand': str(meta.get('brand', '')),
                    'price': float(meta.get('price', 0.0)) if meta.get('price') else None
                }
                cleaned_metadata[str(item_id)] = cleaned_meta
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping invalid metadata for item {item_id}. Error: {e}")
                continue
        
        print(f"\nMetadata loading summary:")
        print(f"Total items: {len(metadata)}")
        print(f"Valid items: {len(cleaned_metadata)}")
        
        return cleaned_metadata
        
    except Exception as e:
        raise ValueError(f"Error loading item metadata: {e}")

def get_items_from_purchases(
    purchases: List[Dict],
    metadata: Dict[str, Dict] = None
) -> Dict[str, Dict]:
    """
    Extract unique items from purchase data and combine with metadata.
    
    Args:
        purchases: List of purchase records
        metadata: Dictionary of item metadata (optional)
        
    Returns:
        Dictionary mapping item_id to combined item information
    """
    # Get unique items from purchases
    unique_items = {purchase['item_id'] for purchase in purchases}
    
    # Initialize items dictionary
    items = {}
    
    # Process each unique item
    for item_id in unique_items:
        # Create base item info
        item_info = {
            'item_id': item_id,
            'title': '',
            'description': '',
            'categories': [],
            'brand': '',
            'price': None
        }
        
        # Add metadata if available
        if metadata and item_id in metadata:
            meta = metadata[item_id]
            item_info.update({
                'title': meta.get('title', item_info['title']),
                'description': meta.get('description', item_info['description']),
                'categories': meta.get('categories', item_info['categories']),
                'brand': meta.get('brand', item_info['brand']),
                'price': meta.get('price', item_info['price'])
            })
        
        items[item_id] = item_info
    
    print(f"\nItem processing summary:")
    print(f"Total items: {len(items)}")
    print(f"Items with metadata: {sum(1 for item_id in items if metadata and item_id in metadata)}")
    
    return items

# Example usage
def main():
    # Load purchase data
    purchases = load_purchase_dataset(
        filepath="purchases.csv",
        min_interactions=5,
        date_format="%Y-%m-%d"
    )
    
    # Load metadata
    metadata = load_item_metadata("item_metadata.csv")
    
    # Process items
    items = get_items_from_purchases(purchases, metadata)
    
    # Print sample data
    print("\nSample purchase record:")
    print(json.dumps(purchases[0], indent=2))
    
    print("\nSample item data:")
    sample_item_id = next(iter(items))
    print(json.dumps(items[sample_item_id], indent=2))

if __name__ == "__main__":
    main()