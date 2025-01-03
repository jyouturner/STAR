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