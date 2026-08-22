# src/ai-service/app/services/embeddings.py
import os
import json
import atexit
import numpy as np
from openai import OpenAI
from typing import List

class EmbeddingError(Exception):
    """
    Raised when the embedding provider fails.

    Callers MUST treat this as "not gradeable" and flag the script for
    re-grading. It must never be swallowed into a zero-vector, because a
    zero-vector yields cosine similarity 0.0 and would silently score a
    student 0 for an infrastructure blip rather than a wrong answer.
    """
    pass


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Persistent file-backed embedding cache ---
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "embedding_cache.json")
_CACHE_PATH = os.path.abspath(_CACHE_PATH)

def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"[Embeddings] Warning: could not persist cache: {e}")

# Load once at module import; in-memory dict is the hot layer
_embedding_cache: dict = _load_cache()

# Flush to disk on clean shutdown
atexit.register(_save_cache, _embedding_cache)


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text using OpenAI API.

    An empty/blank input legitimately returns a zero-vector (blank answer).
    A provider failure raises EmbeddingError so the caller can flag the
    script as ungraded instead of silently scoring it 0.
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
        # Do NOT fall back to a zero-vector — that would score the student 0
        # for an API failure. Surface the error so grading can be aborted.
        print(f"OpenAI API error for embedding: {e}")
        raise EmbeddingError(str(e)) from e

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts using OpenAI batch API.

    Blank inputs legitimately map to zero-vectors. A provider failure
    raises EmbeddingError so the caller can flag the script as ungraded
    instead of silently scoring it 0.
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
            # Do NOT fill with zero-vectors — that would silently score every
            # answer in this batch 0 on an API failure. Abort so the caller
            # can flag the script for re-grading.
            print(f"OpenAI batch API error: {e}")
            raise EmbeddingError(str(e)) from e
    
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
