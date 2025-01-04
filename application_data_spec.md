# Application Data Specification

Below is a **generic data specification** that you can use as a reference when applying this STAR retrieval pipeline to **other eCommerce datasets** (beyond Amazon). The key is that you’ll need **two main data files**:

1. **Interaction Data** (user–item reviews or clicks)  
2. **Item Metadata** (titles, descriptions, brand, etc.)

Below we describe each in detail, along with notes on how to integrate them into the pipeline.

---

## 1. Interaction Data Specification

This file (or set of files) contains *one record per user–item interaction*, typically in JSON-lines format. Each record should minimally have:

| Field            | Type       | Description                                                                                       |
|------------------|------------|---------------------------------------------------------------------------------------------------|
| `reviewerID`     | `string`   | A unique user identifier (e.g., `user123`). Often called `userID` in other contexts.              |
| `asin`           | `string`   | A unique item identifier (e.g., `itemX`).                                                         |
| `unixReviewTime` | `int`      | A Unix timestamp (seconds since epoch) or similar numeric time. Ensures we can sort chronologically.  |
| `overall`        | `float` or `int` | (Optional) The user’s rating for the item (1–5 stars, or 1–10). You can ignore if not relevant. |
| `reviewText`     | `string`   | (Optional) The textual review content or short comment, if available.                            |
| `summary`        | `string`   | (Optional) A short summary or heading for the review.                                            |

### Notes / Requirements

- **Minimal Fields**: For the pipeline to work, you absolutely need (1) `reviewerID`, (2) `asin` (or itemID), (3) a **timestamp**. If you have no numeric rating, you can assign a default rating (like 1.0) for all interactions.
- **Chronology**: `unixReviewTime` is critical because we do a *strictly time-ordered* train/test split. Ensure each row has a valid timestamp (e.g., no placeholder like `0`).
- **One Row per Interaction**: If a user has multiple interactions with the same item, it’s best to keep them all as separate rows (unless you specifically want to combine them into one row with a cumulative rating).
- **Filtering**: The code expects you to pick a threshold (like ≥5 interactions per user and item) to filter out extremely sparse entities.

---

## 2. Item Metadata Specification

This file (or set of files) contains **product-level** or **item-level** metadata that you feed into the LLM to generate embeddings. Typically, each record is also in JSON-lines format or easily parsed JSON. Each record should have:

| Field          | Type            | Description                                                                                                    |
|----------------|-----------------|----------------------------------------------------------------------------------------------------------------|
| `asin` (itemID)| `string`        | Unique item identifier that **matches** the `asin` field in the interaction data.                              |
| `title`        | `string`        | A short textual title.                                                                                         |
| `description`  | `string`        | A longer textual description, bullet points, specs, etc.                                                      |
| `categories`   | `list[str]` or `list[list[str]]` | One or more category paths, e.g. `[["Beauty","Hair Care","Shampoo"]]`. (Can also be a single path list.)   |
| `brand`        | `string`        | (Optional) Brand or manufacturer name.                                                                         |
| `price`        | `float`         | (Optional) Price in your local currency.                                                                       |
| `salesRank`    | `dict` or `int` | (Optional) Sales rank or popularity measure. Often a dict: `{"CategoryName": rankValue}`.                      |

### Notes / Requirements

- **Key**: The `asin` (or itemID) here **must match** the one in the Interaction Data. Otherwise, those items won’t be recognized in the pipeline.
- **Optional Fields**:  
  - If you don’t have brand, price, or sales rank, you can leave them out.  
  - If you want to replicate the STAR approach exactly, it’s helpful to have a few fields like brand/category to give the LLM more context.
- **Textual Fields**: The pipeline creates a textual prompt from fields (e.g. `title`, `description`, `brand`, etc.), so they should be as **human-readable** as possible. (e.g., “Title: <title>\nDescription: <desc>”)

---

## 3. Integrating with This Codebase

1. **Adjust Data Paths**:  
   - By default, the code looks for Amazon `reviews_{category}_5.json.gz` in `data/`.  
   - You’ll want to rename or point to your custom files (e.g., `reviews_custom.json.gz`, `meta_custom.json.gz`) in `src/utils.py` or wherever you do `load_amazon_dataset` and `load_amazon_metadata`.

2. **Filtering**:  
   - If your dataset is not already “5-core,” you’ll see lines in the log like:  
     ```
     Before filtering:
       Total users: 100000
       Total items: 50000
     After filtering (>=5):
       Valid users: 25000
       Valid items: 22000
     ```
     That’s normal. The pipeline will only keep user/item pairs above the threshold.  
   - Adjust `min_interactions` to suit your dataset density.

3. **Timestamp**:  
   - Make sure your `unixReviewTime` is correct (if your data has a different time format, you’ll need to convert it to an integer timestamp).  
   - The pipeline’s “strict chronological splitting” depends on sorting by that field.

4. **Metadata Mappings**:  
   - If your item metadata uses slightly different field names (`product_name` instead of `title`), either rename them in your JSON or adapt the code in `get_items_from_data` or your embedding generator.

5. **LLM Prompt**:  
   - In `item_embeddings_huggingface.py` or `item_embeddings_vertex_ai.py`, you’ll see how we construct a text prompt from each item’s metadata. Adjust or remove fields as needed. For instance, if you don’t have `brand`, just skip that.

---

## 4. Running the Pipeline

Once your files match these specifications (or you’ve adapted the code to parse them correctly):

1. **Set Up**:  
   - Place your `reviews_custom.json.gz` and `meta_custom.json.gz` (or similarly named) in the `data/` folder.
2. **Update Code**:  
   - In `src/utils.py` (or where you load data), point the file paths to your new data.  
   - E.g., replace references to `reviews_Beauty_5.json.gz` with `reviews_custom.json.gz`.
3. **Run**:  
   - `poetry install`  
   - `poetry run python src/main.py`  
4. **Debug**:  
   - If you see mismatches (e.g., “0 items found in metadata”), confirm your `asin` field is consistent between the interaction data and metadata.

---

## 5. Potential Differences for Non-Amazon Data

- **Ratings**: If you have clickstream or add-to-cart logs without explicit ratings, just treat each interaction as `overall=1.0` to keep the pipeline uniform.  
- **Large Variation in Timestamps**: Some eCommerce logs might only have daily or monthly aggregated timestamps. This is still workable, but you lose fine-grained time. The pipeline simply needs consistent ordering (the actual granularity matters less).  
- **Metadata Completeness**: Some eCommerce data might have shorter item descriptions or fewer categories. That can affect embedding quality. The more item text you have, the better the LLM-based similarity typically is.

---

## Example Minimal JSON Record

**Interaction Data** (per line in `reviews_custom.json.gz`):

```json
{
  "reviewerID": "user123",
  "asin": "itemXYZ",
  "overall": 4.0,
  "unixReviewTime": 1588291200,
  "reviewText": "Really liked this product!",
  "summary": "Great value"
}
```

**Metadata** (per line in `meta_custom.json.gz`):

```json
{
  "asin": "itemXYZ",
  "title": "XYZ Brand Wireless Earbuds",
  "description": "Noise-cancelling earbuds with 5 hours battery life...",
  "categories": [
    ["Electronics","Audio","Headphones"]
  ],
  "brand": "XYZ",
  "price": 29.99,
  "salesRank": { "Electronics": 2000 }
}
```

---

## Key Takeaways

- You must supply (1) **interaction** data with consistent user IDs, item IDs, timestamps, and (2) **metadata** that references the same item IDs.  
- The pipeline then applies **filtering**, **embedding generation**, **co-occurrence** computation, and **evaluation** with negative sampling.  
- If you see unexpectedly high or low results, verify that your data is time-sorted, your negative sampling is correct, and your metadata is representative of your actual eCommerce items.

---

**That’s it!** With these guidelines, you can adapt the STAR retrieval pipeline to practically any eCommerce or product-centric dataset you have, just by aligning your interaction logs and item metadata to the code’s expectations.