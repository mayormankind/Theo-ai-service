# src/ai-service/app/services/embeddings.py
import os
import numpy as np
from openai import OpenAI
from typing import List

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache for embeddings
_embedding_cache = {}

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text using OpenAI API.
    Returns zero-vector fallback if API fails.
    """
    if not text or not text.strip():
        return [0.0] * 1536  # text-embedding-3-small dimension
    
    # Check cache first
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        _embedding_cache[text] = embedding
        return embedding
        
    except Exception as e:
        print(f"OpenAI API error for embedding: {e}")
        # Return zero-vector fallback
        return [0.0] * 1536

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts using OpenAI batch API.
    Returns zero-vector fallbacks if API fails.
    """
    if not texts:
        return []
    
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    # Check cache first
    for i, text in enumerate(texts):
        if not text or not text.strip():
            embeddings.append([0.0] * 1536)
        elif text in _embedding_cache:
            embeddings.append(_embedding_cache[text])
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder
    
    # Batch API call for uncached texts
    if uncached_texts:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=uncached_texts
            )
            
            # Update cache and results
            for i, embedding_data in enumerate(response.data):
                text = uncached_texts[i]
                embedding = embedding_data.embedding
                original_index = uncached_indices[i]
                
                _embedding_cache[text] = embedding
                embeddings[original_index] = embedding
                
        except Exception as e:
            print(f"OpenAI batch API error: {e}")
            # Fill with zero-vectors for failed batch
            for i in uncached_indices:
                embeddings[i] = [0.0] * 1536
    
    return embeddings

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors using numpy.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    
    a_array = np.array(a)
    b_array = np.array(b)
    
    # Handle zero vectors
    a_norm = np.linalg.norm(a_array)
    b_norm = np.linalg.norm(b_array)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    return np.dot(a_array, b_array) / (a_norm * b_norm)
