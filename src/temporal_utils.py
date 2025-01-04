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