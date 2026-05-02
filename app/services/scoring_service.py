from app.services.embeddings import cosine_similarity
from app.config.constants import SIMILARITY_FULL, SIMILARITY_PARTIAL

def calculate_similarity(student_answer_emb: list, rubric_embs: list) -> list:
    """
    Calculates cosine similarity between a student answer embedding
    and a list of rubric point embeddings.
    Returns a list of similarities [0.0 to 1.0].
    """
    if not rubric_embs:
        return []
        
    similarities = []
    for rubric_emb in rubric_embs:
        similarity = cosine_similarity(student_answer_emb, rubric_emb)
        similarities.append(similarity)
    
    return similarities

def threshold_score(similarity: float) -> float:
    """
    Applies the threshold function to convert a raw cosine similarity to a discrete score.
    Returns:
     - 1.0 if similarity >= SIMILARITY_FULL
     - 0.5 if similarity >= SIMILARITY_PARTIAL
     - 0.0 otherwise
    """
    if similarity >= SIMILARITY_FULL:
        return 1.0
    elif similarity >= SIMILARITY_PARTIAL:
        return 0.5
    else:
        return 0.0

def calculate_final_score(similarities: list, weights: list) -> tuple:
    """
    Given similarities for rubric points and their respective weights,
    calculates the final accumulated score and the confidence level.
    
    Score = Σ (w_i × f(similarity_i))
    Confidence = Average of raw similarity values
    """
    if not similarities or not weights or len(similarities) != len(weights):
        return 0.0, 0.0
        
    # Apply score thresholding function f() for each similarity to calculate the final sum
    score = sum(w * threshold_score(s) for w, s in zip(weights, similarities))
    
    # Confidence is average of raw similarity values
    confidence = sum(similarities) / len(similarities)
    
    return score, confidence
