# STAR: A Simple Training-free Approach for Recommendations using Large Language Models

This repository implements the retrieval pipeline from the paper [STAR: A Simple Training-free Approach for Recommendations using Large Language Models](https://arxiv.org/abs/2410.16458). It aims to help understand how a training-free recommendation system can be built using:

1. LLM embeddings for semantic similarity
2. User interaction patterns for collaborative signals 
3. Temporal decay for recent history weighting

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

## References

```bibtex
@article{lee2024star,
  title={STAR: A Simple Training-free Approach for Recommendations using Large Language Models},
  author={Lee, Dong-Ho and Kraft, Adam and Jin, Long and Mehta, Nikhil and Xu, Taibai and Hong, Lichan and Chi, Ed H. and Yi, Xinyang},
  journal={arXiv preprint arXiv:2410.16458},
  year={2024}
}
```