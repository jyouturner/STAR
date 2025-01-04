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
