from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingInput
from typing import Dict, List, Set
import numpy as np

class ItemEmbeddingGenerator:
    def __init__(self, 
                output_dimension: int = 768,
                include_fields: Set[str] = None):
        """
        Initialize generator with configurable fields
        
        Args:
            output_dimension: Embedding dimension (default matches paper)
            include_fields: Set of fields to include in prompt
                          (title, description, category, brand, price, sales_rank)
        """
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        self.output_dimension = output_dimension
        # Default to minimal set of fields if none specified
        self.include_fields = include_fields or {'title', 'description', 'category'}
        
    def create_embedding_input(self, item_data: Dict) -> TextEmbeddingInput:
        """Create simplified prompt following paper's structure"""
        prompt_parts = []
        
        # Handle description first
        if 'description' in self.include_fields:
            desc = str(item_data.get('description', '')).strip()
            if desc:
                prompt_parts.append("description:")
                prompt_parts.append(desc)
        
        # Add basic fields with minimal formatting
        if 'title' in self.include_fields and (title := item_data.get('title')):
            prompt_parts.append(f"title: {title}")
            
        if 'category' in self.include_fields and (cats := item_data.get('categories')):
            if isinstance(cats[0], list):
                category_str = " > ".join(cats[0])
            else:
                category_str = " > ".join(cats)
            if category_str:
                prompt_parts.append(f"category: {category_str}")
                
        # Optional fields based on configuration
        if 'price' in self.include_fields and (price := item_data.get('price')):
            prompt_parts.append(f"price: {price}")
            
        if 'brand' in self.include_fields and (brand := item_data.get('brand')):
            # Skip ASIN-like brands
            if not (brand.startswith('B0') and len(brand) >= 10):
                prompt_parts.append(f"brand: {brand}")
                
        if 'sales_rank' in self.include_fields and (rank := item_data.get('salesRank')):
            prompt_parts.append(f"salesRank: {str(rank)}")
            
        return TextEmbeddingInput(
            task_type="RETRIEVAL_DOCUMENT",
            title=item_data.get('title', ''),
            text='\n'.join(prompt_parts)
        )

    def generate_item_embeddings(self, items: Dict) -> Dict[str, np.ndarray]:
        """Generate embeddings with simplified prompts"""
        embeddings = {}
        total_items = len(items)
        batch_size = 5
        
        print(f"\nGenerating embeddings for {total_items} items...")
        print(f"Including fields: {sorted(self.include_fields)}")
        
        # Process items in batches
        item_list = list(items.items())
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch = item_list[batch_start:batch_end]
            
            if batch_start % 1000 == 0:
                print(f"Processing items {batch_start}-{batch_end}/{total_items}")
            
            try:
                embedding_inputs = [
                    self.create_embedding_input(item_data)
                    for _, item_data in batch
                ]
                
                kwargs = {}
                if self.output_dimension:
                    kwargs['output_dimensionality'] = self.output_dimension
                    
                predictions = self.model.get_embeddings(embedding_inputs, **kwargs)
                
                for (item_id, _), embedding in zip(batch, predictions):
                    embeddings[item_id] = np.array(embedding.values)
                    
                    if len(embeddings) == 1:
                        print(f"Embedding dimension: {len(embedding.values)}")
                
            except Exception as e:
                print(f"Error in batch {batch_start}-{batch_end}: {str(e)}")
                continue
        
        return embeddings

    def debug_prompt(self, items: Dict, num_samples: int = 3):
        """Debug utility to verify prompt structure"""
        print("\nSample prompts with current field selection:")
        print("=" * 80)
        print(f"Including fields: {sorted(self.include_fields)}")
        print("=" * 80)
        
        for item_id in list(items.keys())[:num_samples]:
            print(f"\nItem ID: {item_id}")
            print("-" * 40)
            embedding_input = self.create_embedding_input(items[item_id])
            print("Title:", embedding_input.title)
            print("\nText:", embedding_input.text)
            print("=" * 80)

if __name__ == "__main__":
    # Example usage
    items = {
        "item1": {"title": "Product 1", "description": "Description 1", "categories": ["Category 1", "Subcategory 1"]},
        "item2": {"title": "Product 2", "description": "Description 2", "categories": ["Category 2", "Subcategory 2"]},
        "item3": {"title": "Product 3", "description": "Description 3", "categories": ["Category 3", "Subcategory 3"]}
    }
    generator = ItemEmbeddingGenerator()
    generator.debug_prompt(items, num_samples=3)
