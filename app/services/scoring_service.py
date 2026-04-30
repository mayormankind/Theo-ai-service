from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(student_answer_emb: list, rubric_embs: list) -> list:
    """
    Calculates cosine similarity between a student answer embedding
    and a list of rubric point embeddings.
    Returns a list of similarities [0.0 to 1.0].
    """
    if not rubric_embs:
        return []
        
    student_emb_array = np.array(student_answer_emb).reshape(1, -1)
    rubric_embs_array = np.array(rubric_embs)
    
    similarities = cosine_similarity(student_emb_array, rubric_embs_array)[0]
    return similarities.tolist()

def threshold_score(similarity: float) -> float:
    """
    Applies the threshold function to convert a raw cosine similarity to a discrete score.
    Returns:
     - 1.0 if similarity >= 0.85
     - 0.7 if similarity >= 0.70
     - 0.4 if similarity >= 0.50
     - 0.0 otherwise
    """
    if similarity >= 0.85:
        return 1.0
    elif similarity >= 0.70:
        return 0.7
    elif similarity >= 0.50:
        return 0.4
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
