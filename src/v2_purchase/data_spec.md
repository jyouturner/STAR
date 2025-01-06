# Data Specification for Purchase-based STAR Pipeline

## Overview

This document specifies the required data formats for the purchase-based STAR recommendation pipeline. The system requires two main data sources:

1. Purchase transaction data
2. Item metadata

## 1. Purchase Transaction Data

### Required Fields

| Field       | Type    | Description                                      | Example           |
|-------------|---------|--------------------------------------------------|-------------------|
| user_id     | string  | Unique identifier for each user                  | "user123"         |
| item_id     | string  | Unique identifier for each item                  | "item456"         |
| timestamp   | string  | Purchase timestamp in YYYY-MM-DD format          | "2024-01-15"      |
| quantity    | float   | Purchase quantity (must be positive)             | 2.0               |

### Format Options

#### CSV Format

```csv
user_id,item_id,timestamp,quantity
user123,item456,2024-01-15,2.0
user123,item789,2024-01-16,1.0
user456,item123,2024-01-15,3.0
```

#### JSON Format

```json
[
    {
        "user_id": "user123",
        "item_id": "item456",
        "timestamp": "2024-01-15",
        "quantity": 2.0
    },
    {
        "user_id": "user123",
        "item_id": "item789",
        "timestamp": "2024-01-16",
        "quantity": 1.0
    }
]
```

### Data Requirements

- All fields are required
- `quantity` must be positive
- `timestamp` must be in YYYY-MM-DD format
- Both `user_id` and `item_id` will be converted to strings internally
- Purchase records should be sorted by timestamp (the pipeline will sort them if they aren't)

## 2. Item Metadata

### Required Fields

| Field       | Type           | Description                               | Example                    |
|-------------|----------------|-------------------------------------------|-----------------------------|
| item_id     | string        | Unique identifier matching purchase data   | "item456"                  |
| title       | string        | Item title/name                           | "Blue Cotton T-Shirt"       |
| description | string        | Item description                          | "Comfortable cotton t-shirt"|
| categories  | list[string]  | List of category hierarchies              | ["Clothing", "T-Shirts"]    |
| brand       | string        | Brand name                                | "BrandName"                 |
| price       | float         | Item price (can be null)                  | 19.99                      |

### Format Options

#### CSV Format

```csv
item_id,title,description,categories,brand,price
item456,"Blue T-Shirt","Comfortable cotton t-shirt","['Clothing', 'T-Shirts']",BrandName,19.99
item789,"Red Shoes","Running shoes","['Footwear', 'Athletic']",ShoeBrand,89.99
```

#### JSON Format

```json
{
    "item456": {
        "title": "Blue T-Shirt",
        "description": "Comfortable cotton t-shirt",
        "categories": ["Clothing", "T-Shirts"],
        "brand": "BrandName",
        "price": 19.99
    },
    "item789": {
        "title": "Red Shoes",
        "description": "Running shoes",
        "categories": ["Footwear", "Athletic"],
        "brand": "ShoeBrand",
        "price": 89.99
    }
}
```

### Data Requirements

- `item_id` must match IDs in purchase data
- All fields except `price` must be non-null
- `categories` should be a list of strings, even if only one category
- Empty strings are allowed for `title`, `description`, and `brand`
- `price` can be null but must be numeric if present

## Data Filtering

The pipeline applies the following filters:

1. Minimum interactions per user (default: 5)
2. Minimum interactions per item (default: 5)
3. Removal of invalid records (wrong format, negative quantities, etc.)
