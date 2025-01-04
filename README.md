# STAR: A Simple Training-free Approach for Recommendations using Large Language Models

This repository implements the retrieval pipeline from the paper [STAR: A Simple Training-free Approach for Recommendations using Large Language Models](https://arxiv.org/abs/2410.16458). It aims to help understand how a training-free recommendation system can be built using:

1. LLM embeddings for semantic similarity
2. User interaction patterns for collaborative signals 
3. Temporal decay for recent history weighting

---

## Key Components & Implementation Details

### 1. Item Embeddings (`item_embeddings_vertex_ai.py`)

The embeddings are the foundation of semantic similarity:

```python
class ItemEmbeddingGenerator:
    def create_embedding_input(self, item_data: Dict) -> TextEmbeddingInput:
        # Creates rich text prompts including:
        # - Full item description
        # - Title
        # - Category hierarchy
        # - Brand (if not ASIN-like)
        # - Price and sales rank
```

Key implementation details:
- Uses Vertex AI's `text-embedding-005` model (768 dimensions)
- Excludes IDs/URLs to avoid trivial matching
- Preserves complete metadata structure

---

### 2. STAR Retrieval (`star_retrieval.py`)

The core scoring logic combines three components:

1. **Semantic Matrix** (R_s):
```python
# Compute cosine similarities between normalized embeddings
semantic_matrix = 1 - cdist(embeddings_array, embeddings_array, metric='cosine')
np.fill_diagonal(semantic_matrix, 0)  # Zero out self-similarities
```

2. **Collaborative Matrix** (R_c) (`collaborative_relationships.py`):
```python
# Normalize by user activity sqrt
user_activity = np.sum(interaction_matrix, axis=0)
normalized = interaction_matrix / np.sqrt(user_activity)
collaborative_matrix = normalized @ normalized.T
```

3. **Scoring Formula**:
```python
score = 0.0
for t, (hist_item, rating) in enumerate(zip(reversed(user_history), reversed(ratings))):
    sem_sim = semantic_matrix[cand_idx, hist_idx]
    collab_sim = collaborative_matrix[cand_idx, hist_idx]
    combined_sim = (semantic_weight * sem_sim + (1 - semantic_weight) * collab_sim)
    score += (1/n) * rating * (temporal_decay ** t) * combined_sim
```

### 3. Critical Implementation Details

#### Chronological Ordering
The code strictly maintains temporal order (`temporal_utils.py`):
- Sorts reviews by timestamp
- Handles duplicate timestamps
- Ensures test items are truly last in sequence

#### Negative Sampling
Proper negative sampling is crucial (`evaluation_metrics.py`):
```python
# Exclude ALL items user has interacted with (past AND future)
excluded_items = set(user_all_items[user_id])
negative_candidates = set()
while len(negative_candidates) < n_negative_samples:
    item = random.choice(valid_items)
    if item not in excluded_items:
        negative_candidates.add(item)
```

#### Evaluation Protocol
The evaluation matches the paper's setup:
- Leave-last-out evaluation
- 99 negative samples per test item
- Metrics: Hits@5/10, NDCG@5/10

## Dataset and Download Instructions

1. **Download** the [Stanford SNAP 5-core Amazon datasets](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) using `download_data.py`.  
   For example:
   ```bash
   poetry run python download_data.py --category beauty
   ```
   This downloads `reviews_Beauty_5.json.gz` and `meta_Beauty.json.gz` into the `data/` folder.

2. **Check data** with `check_data.py`:
   ```bash
   poetry run python check_data.py
   ```
   This prints the first few lines and verifies the JSON parse.

> **Note**: These files named `reviews_Beauty_5.json.gz` etc. are already **5-core** datasets. The code still enforces ≥5 interactions, but typically no users/items are removed since the data is already filtered.

---

## Running the Project

1. **Install** Python dependencies via Poetry:
   ```bash
   poetry install
   ```
2. **Run** the main pipeline:
   ```bash
   poetry run python src/main.py
   ```
   This:
   - Loads reviews and metadata,
   - Sorts each user’s reviews by timestamp (fixing potential out-of-order entries),
   - Creates or loads item embeddings,
   - Computes the semantic and collaborative matrices,
   - Splits data into train/val/test in a leave-last-out manner,
   - Runs evaluation with 99 negative samples for each user’s test item,
   - Prints final Hits@K, NDCG@K metrics.

---

## Implementation Tips & Pitfalls

1. **Data Quality Matters**
   - Use `DataQualityChecker` to verify metadata richness
   - Check for duplicate timestamps
   - Verify chronological ordering

2. **Embedding Generation**
   - Include all relevant metadata for rich embeddings
   - Avoid ID/URL information that could leak
   - Use consistent field ordering in prompts

3. **Matrix Computation**
   - Normalize embeddings before similarity
   - Proper user activity normalization for collaborative
   - Zero out diagonal elements

4. **Common Issues**
   - Future item leakage in negative sampling
   - Timestamp ordering issues
   - Inadequate metadata in prompts

---

## Key Parameters

```python
# Retrieval parameters (star_retrieval.py)
semantic_weight = 0.5    # Weight between semantic/collaborative
temporal_decay = 0.7    # Decay factor for older items
history_length = 3      # Number of recent items to use

# Evaluation parameters (evaluation_metrics.py)
n_negative_samples = 99  # Number of negative samples
k_values = [5, 10]      # Top-k for metrics
```

## Understanding the Output

The code provides detailed statistics:
```
Semantic Matrix Statistics:
- mean_sim: Average semantic similarity
- sparsity: Fraction of zero elements
- min/max_sim: Similarity range

Collaborative Matrix Statistics:
- mean_nonzero: Average co-occurrence strength
- sparsity: Interaction density
```

These help diagnose if the embeddings or collaborative signals are working as expected.

## Typical Results

We typically see metrics around **hit@5 ≈ 0.40**–**0.43** and **hit@10 ≈ 0.48**–**0.52** on the 5-core **Beauty** data, which is somewhat **higher** than the baseline numbers from the STAR paper (which sometimes report ~0.33–0.42). Possible reasons:

- The paper may have used a slightly different (raw) dataset or combined multiple categories differently.
- We are using a **more advanced LLM** or more complete metadata fields (brand, price, sales rank).  
- Our dataset is purely 5-core and might be more homogeneous and easier to predict.

Despite these differences, our pipeline still **follows** the STAR retrieval logic faithfully, and higher results are not necessarily a bug—just a reflection of differences in data distribution or stronger embeddings.

---

## Caveats and Known Differences

1. **5-Core Data** vs. raw 2014 data. We start from the 5-core snapshots, but the STAR paper might have had a different or earlier version.  
2. **Metadata coverage**: Some items in `meta_Beauty.json.gz` have minimal text fields, while others have paragraphs. This can result in high average item–item similarity if many items share the same brand or repetitive bullet points.  
3. **Single test item**: We only hold out the user’s final item as test. Some papers average performance across multiple test items per user, which typically lowers the reported metrics.

---

## Example Output

```
$ poetry run python src/main.py 
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

=== Chronological Analysis ===

Found 19424 users with temporal ordering issues:

Issue 1:
User: A1YJEY40YUW4SE
Original sequence:
  2014-01-29: 7806397051
  2012-02-01: B0020YLEYK
  2014-01-29: B002WLWX82
  2011-10-17: B004756YJA
  2011-10-17: B004ZT0SSG
Sorted sequence:
  2011-10-17: B004756YJA
  2011-10-17: B004ZT0SSG
  2012-02-01: B0020YLEYK
  2014-01-29: 7806397051
  2014-01-29: B002WLWX82

Issue 2:
User: A60XNB876KYML
Original sequence:
  2014-04-17: 7806397051
  2014-04-17: B0000YUX4O
  2014-02-14: B0009P4PZC
  2014-04-16: B00812ZWOS
  2014-03-30: B009HULFLW
  2014-03-30: B00BZ1QN2C
  2014-04-06: B00G2TQNZ4
Sorted sequence:
  2014-02-14: B0009P4PZC
  2014-03-30: B009HULFLW
  2014-03-30: B00BZ1QN2C
  2014-04-06: B00G2TQNZ4
  2014-04-16: B00812ZWOS
  2014-04-17: 7806397051
  2014-04-17: B0000YUX4O

Issue 3:
User: A3G6XNM240RMWA
Original sequence:
  2013-09-05: 7806397051
  2014-02-27: B00011JI88
  2013-11-18: B001MP471K
  2013-09-08: B002S8TOYU
  2014-03-15: B003ATNYJC
  2013-09-05: B003H8180I
  2014-05-16: B003ZS6ONQ
  2013-09-05: B00538TSMU
  2014-03-02: B00C1F13CQ
Sorted sequence:
  2013-09-05: 7806397051
  2013-09-05: B003H8180I
  2013-09-05: B00538TSMU
  2013-09-08: B002S8TOYU
  2013-11-18: B001MP471K
  2014-02-27: B00011JI88
  2014-03-02: B00C1F13CQ
  2014-03-15: B003ATNYJC
  2014-05-16: B003ZS6ONQ

Issue 4:
User: A1PQFP6SAJ6D80
Original sequence:
  2013-12-07: 7806397051
  2013-10-04: B00027D8IC
  2013-10-04: B002PMLGOU
  2013-09-13: B0030HKJ8I
  2013-12-07: B004Z40048
  2013-10-04: B00BN1MPPS
Sorted sequence:
  2013-09-13: B0030HKJ8I
  2013-10-04: B00027D8IC
  2013-10-04: B002PMLGOU
  2013-10-04: B00BN1MPPS
  2013-12-07: 7806397051
  2013-12-07: B004Z40048

Issue 5:
User: A38FVHZTNQ271F
Original sequence:
  2013-10-18: 7806397051
  2013-11-20: B002BGDLDO
  2013-11-20: B003VWZCMK
  2013-11-19: B007EHWDTS
  2013-11-04: B008LQX8J0
  2013-11-19: B009DDGHFC
  2013-11-03: B009PZVOF6
  2014-01-05: B00DAYGJVW
  2013-11-20: B00DQ2ILQY
Sorted sequence:
  2013-10-18: 7806397051
  2013-11-03: B009PZVOF6
  2013-11-04: B008LQX8J0
  2013-11-19: B007EHWDTS
  2013-11-19: B009DDGHFC
  2013-11-20: B002BGDLDO
  2013-11-20: B003VWZCMK
  2013-11-20: B00DQ2ILQY
  2014-01-05: B00DAYGJVW

Temporal Statistics:
Date range: 2002-06-11 to 2014-07-22

Users with multiple reviews on same day: 36802

Sample cases of multiple reviews per day:
User A1YJEY40YUW4SE: 2 reviews on 2014-01-29
User A1YJEY40YUW4SE: 2 reviews on 2011-10-17
User A60XNB876KYML: 2 reviews on 2014-04-17
User A60XNB876KYML: 2 reviews on 2014-03-30
User A3G6XNM240RMWA: 3 reviews on 2013-09-05
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

=== Sample User Histories ===

User: A02849582PREZYNEI31CV (5 reviews)
  2014-03-30: B007RT19V6, rating=5.0
    Summary: great tools !...
  2014-03-30: B007TL60IE, rating=5.0
    Summary: very cute!!!...
  2014-03-30: B0083QNBCM, rating=5.0
    Summary: cute cheap brushes...
  2014-03-30: B0087O4XKE, rating=1.0
    Summary: comes off easy!!!...
  2014-03-30: B00AZBSPQA, rating=2.0
    Summary: pop off easy!...

User: A1AN74S2DBATOJ (6 reviews)
  2014-02-24: B00A19WNC8, rating=5.0
    Summary: These are one of my Favorite designs!!...
  2014-03-09: B001169LHK, rating=4.0
    Summary: Nail Glue......
  2014-03-09: B0084BPHY6, rating=4.0
    Summary: Buff away!...
  2014-03-22: B005JD4NZQ, rating=4.0
    Summary: FUN!!...
  2014-03-22: B006U95N34, rating=4.0
    Summary: They are doing just what they're suppose to do!...
  2014-03-28: B007USPWS0, rating=4.0
    Summary: Cute!...

User: A2Q9EK9WKGFGCG (8 reviews)
  2011-05-09: B0043TURKC, rating=4.0
    Summary: So silky!...
  2013-10-06: B00D6EDGYE, rating=4.0
    Summary: Lightweight coverage and great scent!...
  2014-01-20: B00G2TQNZ4, rating=5.0
    Summary: I love this stuff!...
  2014-04-21: B00IBS9QC6, rating=5.0
    Summary: Loved it, great stuff!...
  2014-05-16: B00CGN9LQ8, rating=5.0
    Summary: I love this brush...
  2014-05-20: B00JJVG6HC, rating=5.0
    Summary: Great for dry itchy scalp!...
  2014-05-20: B00JL2TURM, rating=5.0
    Summary: Great for acne prone skin...
  2014-07-15: B00EYSNWXG, rating=5.0
    Summary: It's a good curler!...

=== Duplicate Analysis ===
Total reviews: 198502
Exact duplicates: 0
Users with multiple reviews for same item: 0

=== Rating Analysis ===
Overall rating distribution:
  1.0: 10526 (5.3%)
  2.0: 11456 (5.8%)
  3.0: 22248 (11.2%)
  4.0: 39741 (20.0%)
  5.0: 114531 (57.7%)

Users with single rating value:
  User A00700212KB3K0MVESPIY: all 5.0s (9 reviews)
  User A0508779FEO1DUNOSQNX: all 5.0s (5 reviews)
  User A05306962T0DL4FS2RA7L: all 5.0s (5 reviews)
  User A100VQNP6I54HS: all 5.0s (8 reviews)
  User A1010QRG4BH51B: all 5.0s (11 reviews)
Loaded embeddings for 12101 items

Computing semantic relationships...

Validation data preparation:
Total users processed: 22363
Users skipped (too short): 0
Final validation sequences: 22363

Evaluation data preparation:
Total users processed: 22363
Users skipped (too short): 0
Users with timestamp issues: 19586
Final test sequences: 22363

=== Evaluation Split Verification ===

Found 0 sequences where test item isn't truly last

History length distribution:
  Length 4: 7162 sequences
  Length 5: 4221 sequences
  Length 6: 2680 sequences
  Length 7: 1811 sequences
  Length 8: 1366 sequences
  Length 9: 881 sequences
  Length 10: 695 sequences
  Length 11: 558 sequences
  Length 12: 439 sequences
  Length 13: 361 sequences
  Length 14: 262 sequences
  Length 15: 207 sequences
  Length 16: 213 sequences
  Length 17: 141 sequences
  Length 18: 118 sequences
  Length 19: 112 sequences
  Length 20: 117 sequences
  Length 21: 84 sequences
  Length 22: 66 sequences
  Length 23: 73 sequences
  Length 24: 74 sequences
  Length 25: 61 sequences
  Length 26: 54 sequences
  Length 27: 44 sequences
  Length 28: 44 sequences
  Length 29: 38 sequences
  Length 30: 30 sequences
  Length 31: 38 sequences
  Length 32: 20 sequences
  Length 33: 19 sequences
  Length 34: 21 sequences
  Length 35: 19 sequences
  Length 36: 23 sequences
  Length 37: 24 sequences
  Length 38: 22 sequences
  Length 39: 15 sequences
  Length 40: 17 sequences
  Length 41: 7 sequences
  Length 42: 10 sequences
  Length 43: 10 sequences
  Length 44: 15 sequences
  Length 45: 7 sequences
  Length 46: 11 sequences
  Length 47: 8 sequences
  Length 48: 4 sequences
  Length 49: 8 sequences
  Length 50: 9 sequences
  Length 51: 8 sequences
  Length 52: 9 sequences
  Length 53: 6 sequences
  Length 54: 8 sequences
  Length 55: 2 sequences
  Length 56: 6 sequences
  Length 57: 4 sequences
  Length 58: 7 sequences
  Length 59: 3 sequences
  Length 60: 1 sequences
  Length 62: 4 sequences
  Length 63: 2 sequences
  Length 64: 2 sequences
  Length 65: 4 sequences
  Length 66: 1 sequences
  Length 67: 3 sequences
  Length 68: 4 sequences
  Length 69: 2 sequences
  Length 70: 2 sequences
  Length 71: 2 sequences
  Length 72: 4 sequences
  Length 73: 3 sequences
  Length 74: 3 sequences
  Length 75: 1 sequences
  Length 76: 5 sequences
  Length 77: 2 sequences
  Length 78: 3 sequences
  Length 79: 1 sequences
  Length 80: 1 sequences
  Length 81: 2 sequences
  Length 82: 4 sequences
  Length 83: 1 sequences
  Length 84: 3 sequences
  Length 86: 3 sequences
  Length 92: 2 sequences
  Length 93: 1 sequences
  Length 95: 1 sequences
  Length 96: 1 sequences
  Length 98: 2 sequences
  Length 99: 4 sequences
  Length 106: 1 sequences
  Length 107: 1 sequences
  Length 109: 1 sequences
  Length 114: 2 sequences
  Length 115: 1 sequences
  Length 116: 1 sequences
  Length 118: 1 sequences
  Length 122: 1 sequences
  Length 130: 1 sequences
  Length 148: 2 sequences
  Length 149: 1 sequences
  Length 153: 1 sequences
  Length 181: 1 sequences
  Length 191: 1 sequences
  Length 203: 1 sequences

Processing user-item interactions...
Found 20484 users and 12101 items
Built interaction matrix with 137119 non-zero entries

=== Negative Sampling Debug ===

User: A2YNJPT91EB0HQ
History length: 5
Total valid items: 12101
Excluded items: 6
Available negatives: 12095

User: A65UWTN0W4Z1E
History length: 5
Total valid items: 12101
Excluded items: 6
Available negatives: 12095

User: A2AOBR9Z97NWF3
History length: 9
Total valid items: 12101
Excluded items: 10
Available negatives: 12091

User: A15JENG42JCP0I
History length: 6
Total valid items: 12101
Excluded items: 7
Available negatives: 12094

User: A60E8FTIA3XW8
History length: 14
Total valid items: 12101
Excluded items: 15
Available negatives: 12086

=== Collaborative Matrix Analysis ===
Shape: (12101, 12101)
Density: 0.0107
Mean nonzero value: 0.0743
Max value: 8.4437

Value distribution (nonzero):
  25th percentile: 0.0154
  Median: 0.0385
  75th percentile: 0.0909

=== Running Evaluation ===

=== Starting Evaluation ===
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████| 22363/22363 [00:06<00:00, 3303.81it/s]

Successfully evaluated 22363/22363 sequences

Final Results:

Results for Beauty dataset:
------------------------------
Metric          Score     
------------------------------
hit@10          0.4921
hit@5           0.4026
ndcg@10         0.3460
ndcg@5          0.3172
------------------------------
```

---

## Apply to Your Own Data

See [Application Data Specification](application_data_spec.md) for how to prepare your own data.

---

## References

```bibtex
@article{lee2024star,
  title={STAR: A Simple Training-free Approach for Recommendations using Large Language Models},
  author={Lee, Dong-Ho and Kraft, Adam and Jin, Long and Mehta, Nikhil and Xu, Taibai and Hong, Lichan and Chi, Ed H. and Yi, Xinyang},
  journal={arXiv preprint arXiv:2410.16458},
  year={2024}
}
```