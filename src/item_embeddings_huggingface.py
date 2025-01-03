from typing import Dict, Any
import numpy as np
import json
from star_retrieval import STARRetrieval
from transformers import AutoTokenizer, AutoModel
import torch

class ItemEmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def create_item_prompt(self, item: Dict) -> str:
        """Create prompt following paper's approach, excluding ID/URL fields"""
        prompt_parts = []
        
        if 'title' in item:
            prompt_parts.append(f"Title: {item['title']}")
            
        if 'description' in item:
            prompt_parts.append(f"Description: {item['description']}")
            
        if 'categories' in item:
            cats = ' > '.join(item['categories'][-1]) if isinstance(item['categories'], list) else str(item['categories'])
            prompt_parts.append(f"Category: {cats}")
            
        if 'brand' in item:
            prompt_parts.append(f"Brand: {item['brand']}")
            
        if 'salesRank' in item:
            if isinstance(item['salesRank'], dict):
                ranks = [f"{cat}: {rank}" for cat, rank in item['salesRank'].items()]
                prompt_parts.append(f"Sales Rank: {', '.join(ranks)}")
            else:
                prompt_parts.append(f"Sales Rank: {item['salesRank']}")
                
        if 'price' in item:
            prompt_parts.append(f"Price: ${item['price']}")
            
        return "\n".join(prompt_parts)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string using the Hugging Face model
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding values
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    def generate_item_embeddings(self, items: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple items
        
        Args:
            items: Dictionary mapping item IDs to their metadata
            
        Returns:
            Dictionary mapping item IDs to their embeddings
        """
        embeddings = {}
        for item_id, item_data in items.items():
            prompt = self.create_item_prompt(item_data)
            embedding = self.get_embedding(prompt)
            embeddings[item_id] = embedding
        return embeddings

# Example usage:
def main():
    # Example items (similar to Amazon product data used in the paper)
    items = {
        "B001": {
            "title": "Professional Makeup Brush Set",
            "description": "Set of 12 professional makeup brushes with synthetic bristles",
            "categories": ["Beauty", "Makeup", "Brushes & Tools"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 1250},
            "price": 24.99
        },
        "B002": {
            "title": "Eyeshadow Palette - Natural Colors",
            "description": "15 highly pigmented natural eyeshadow colors",
            "categories": ["Beauty", "Makeup", "Eyes", "Eyeshadow"],
            "brand": "BeautyPro",
            "salesRank": {"Beauty": 2100},
            "price": 19.99
        }
    }
    
    # Generate embeddings
    generator = ItemEmbeddingGenerator()
    item_embeddings = generator.generate_item_embeddings(items)
    
    # Use with STAR retrieval
    retrieval = STARRetrieval(semantic_weight=0.5, temporal_decay=0.7, history_length=3)
    retrieval.compute_semantic_relationships(item_embeddings)
    
    # Example output of embeddings and similarity
    print(f"Embedding dimension: {len(item_embeddings['B001'])}")
    semantic_sim = retrieval.semantic_matrix[0,1]
    print(f"Semantic similarity between items: {semantic_sim:.3f}")

if __name__ == "__main__":
    main()
