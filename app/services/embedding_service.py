from sentence_transformers import SentenceTransformer

# We load the Sentence-BERT model once at application startup.
# This prevents reloading the model for every single API request,
# ensuring the API responds reasonably fast.
MODEL_NAME = 'all-MiniLM-L6-v2'
print(f"Loading Sentence-BERT model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")

def get_embeddings(texts: list) -> list:
    """
    Converts a list of text strings into semantic embeddings.
    """
    if not texts:
        return []
    embeddings = model.encode(texts)
    return embeddings.tolist()