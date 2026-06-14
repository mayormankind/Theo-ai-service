# src/ai-service/app/services/scoring_service.py
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
    Converts a raw cosine similarity to a continuous score in [0.0, 1.0].

    Scoring bands:
     - similarity >= SIMILARITY_FULL  → 1.0 (full credit)
     - SIMILARITY_PARTIAL <= s < FULL → linearly interpolated between 0.5 and 1.0
     - similarity < SIMILARITY_PARTIAL → 0.0 (no credit)

    Linear interpolation prevents a cliff-edge where a student scoring
    0.74 receives the same mark as one scoring 0.51.
    """
    if similarity >= SIMILARITY_FULL:
        return 1.0
    elif similarity >= SIMILARITY_PARTIAL:
        # Linear interpolation: 0.5 at PARTIAL, 1.0 at FULL
        ratio = (similarity - SIMILARITY_PARTIAL) / (SIMILARITY_FULL - SIMILARITY_PARTIAL)
        return 0.5 + 0.5 * ratio
    else:
        return 0.0


def calculate_final_score(
    similarities: list,
    weights: list,
    question_max_score: float = 0.0
) -> tuple:
    """
    Calculates actual marks earned for a question.
    
    For single-entity rubric (one expected answer per 
    question):
      score = threshold_score(similarity) × max_marks
    
    For multi-point rubric (multiple weighted points):
      score = sum(weight_ratio × max_marks × threshold)
    
    Returns (actual_marks_earned, confidence)
    """
    if not similarities or not weights:
        return 0.0, 0.0

    confidence = sum(similarities) / len(similarities)

    if question_max_score <= 0:
        # Fallback: return threshold value only
        # (legacy behaviour, should not happen in practice)
        score = sum(
            w * threshold_score(s)
            for w, s in zip(weights, similarities)
        )
        return round(score, 2), round(confidence, 4)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0, round(confidence, 4)

    # Distribute max marks proportionally by weight
    # For single entity: one weight of 1.0, so full 
    # max_marks × threshold_score
    score = sum(
        (w / total_weight) * question_max_score 
        * threshold_score(s)
        for w, s in zip(weights, similarities)
    )

    return round(score, 2), round(confidence, 4)
