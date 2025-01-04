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
