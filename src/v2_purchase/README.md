# STAR-v2: Purchase-Based Adaptation of “A Simple Training-free Approach for Recommendations”

This folder contains a **modified** version of the [STAR retrieval pipeline](https://arxiv.org/abs/2410.16458) that uses **purchase history** (user–item–date–quantity) data, rather than user reviews. The core ideas—combining semantic item embeddings with collaborative co-occurrence and a temporal decay factor—remain the same, but we adapt the pipeline to better align with **production eCommerce** usage.

## Key Changes from v1 Original STAR

1. **Purchase Data Instead of Reviews**  
   - We replaced the “review-based” approach (`(reviewerID, asin, unixReviewTime, overall)`) with **purchase records**:  

     ```python
     (user_id, item_id, purchase_date, quantity)
     ```

   - The filtering logic (≥5 interactions) now applies to “≥5 purchases per user/item.”

2. **Quantity Handling**  
   - The collaborative relationship matrix can incorporate **quantity** signals. For example, we use `log(1 + quantity)` in place of the old binary `1.0` for each interaction.  
   - This means heavy/bulk purchases contribute more strongly to co-occurrence, while single purchases are a smaller signal.

3. **Temporal Decay in Scoring**  
   - We now factor **purchase_date** into the final scoring function. Older purchases get **lower** weight, more recent purchases get higher weight.  
   - We do **not** degrade old events inside the co-occurrence matrix—only in the final summation when we multiply each item’s similarity by a decay term.

4. **Data Loading**  
   - In `data_loading.py` (or whichever file you’re using), we define a `load_purchase_data()` function that returns the list of purchase events, sorted by time.  
   - `main_purchase.py` (or similar) ensures the train/val/test split is based on each user’s last purchase in chronological order.

5. **Evaluation**  
   - The evaluation logic (Hits@K, NDCG@K) is effectively the same—**but** we treat the user’s last purchase (by date) as the “test item.”  
   - Negative sampling or full ranking can still be applied. The pipeline sees `(user_id, purchase_history, last_purchased_item)` and attempts to rank that item among the candidate set.

## File Structure

```
v2_purchase/
  ├─ README.md                             # This file
  ├─ main_purchase.py                           # Main entry point for purchase-based pipeline
  ├─ collaborative_relationships_purchase.py    # Modified to incorporate quantity weighting
  ├─ star_retrieval_purchase.py                 # Scoring function with date-based decay
  └─ evaluation_metrics_purchase.py             # Same logic, except "last purchase" is test item
```


## How It Works

1. **Data Loading**:  
   - `data_loading.py` loads purchase records in the format:

     ```python
     (user_id, item_id, purchase_timestamp, quantity)
     ```

   - It filters out users and items with <5 total purchases (optional).  
   - It sorts them by `purchase_timestamp`.

2. **Building Collaborative Matrix** (`collaborative_relationships_purchase.py`):  
   - Instead of marking an entry `(item_idx, user_idx) = 1.0`, we do something like `log1p(quantity)`.  
   - We then do the usual normalization and multiply to get the item–item co-occurrence matrix \(R_c\).

3. **Item Embeddings** (`item_embeddings_vertex_ai.py`):  
   - The same LLM embedding approach as before, using textual metadata.  
   - No changes except references to “purchase dataset” in logs.  

4. **Scoring** (`star_retrieval_purchase.py`):  
   - For each user’s **history** (the last \(l\) purchases), we compute a time-based exponent to weigh older purchases less.  
   - The final formula looks like: $\mathrm{score}(x) = \frac{1}{n}\sum_{j=1}^{n} \Bigl( \lambda^{\mathrm{timeDecay}(j)} \cdot [\alpha R_s(x, j) + (1 - \alpha)R_c(x, j)]\cdot \mathrm{QuantityFactor}\_j \Bigr)$

   - `timeDecay(j)` might be `(currentTime - purchaseTime_j)/30` if we measure months, for instance.  
   - `QuantityFactor_j` can be `log1p(quantity_j)` if desired.

5. **Evaluation** (`evaluation_metrics_purchase.py`):  
   - We identify each user’s final purchase as the “test item.”  
   - We rank that test item among all (or sampled) items not purchased by the user.  
   - We measure if the ground-truth item is in the top-k (hits@k) or use the standard NDCG formula.


## Data Specification

[data_spec.md](data_spec.md)

## Usage

1. **Install** dependencies (same as v1):

   ```bash
   poetry install
   ```

2. **Run** the pipeline:

   ```bash
   poetry run python src/v2_purchase/main_purchase.py
   ```

   or whichever script name you’re using. This will:
   - Load the purchase data (CSV, JSON, or DB queries).  
   - Build or load item embeddings.  
   - Compute the co-occurrence matrix.  
   - Split data into train/val/test by purchase date.  
   - Perform retrieval and evaluation, printing final hits@k/ndcg@k scores.

