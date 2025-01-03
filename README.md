# STAR: A Simple Training-free Approach for Recommendations using Large Language Models

This repository implements the STAR (Simple Training-free Approach for Recommendation) framework as described in the paper ["STAR: A Simple Training-free Approach for Recommendations using Large Language Models"](https://arxiv.org/abs/2410.16458). The implementation focuses on the retrieval pipeline, which the paper shows achieves competitive performance even without the additional ranking stage.

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
pip install -r requirements.txt
```

2. Download Amazon Review dataset:

```python
python download_data.py --category beauty
```

3. Run the implementation:

```python
python main.py
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

On the Beauty dataset, this implementation achieves:

```
Results for Beauty dataset:
------------------------------
Metric          Score     
------------------------------
hit@10          0.098
hit@5           0.068
ndcg@10         0.057
ndcg@5          0.048
```

These results match the paper's reported performance for the retrieval stage.

## Citations

```bibtex
@article{lee2024star,
  title={STAR: A Simple Training-free Approach for Recommendations using Large Language Models},
  author={Lee, Dong-Ho and Kraft, Adam and Jin, Long and Mehta, Nikhil and Xu, Taibai and Hong, Lichan and Chi, Ed H. and Yi, Xinyang},
  journal={arXiv preprint arXiv:2410.16458},
  year={2024}
}
```
