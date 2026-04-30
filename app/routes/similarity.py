from fastapi import APIRouter
from app.models.request_models import SimilarityRequest
from app.services.embedding_service import get_embeddings
from app.services.scoring_service import calculate_similarity
from app.utils.text_preprocessing import preprocess_text

router = APIRouter()

@router.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest):
    """
    Calculates the cosine similarity between a student's answer and a list of rubric points.
    """
    # Preprocess texts before embedding
    clean_student_answer = preprocess_text(req.student_answer)
    clean_rubrics = [preprocess_text(rp) for rp in req.rubric]
    
    if not clean_rubrics:
        return {"similarities": []}
    
    # Get vector embeddings
    student_emb = get_embeddings([clean_student_answer])[0]
    rubric_embs = get_embeddings(clean_rubrics)
    
    # Calculate similarities
    similarities = calculate_similarity(student_emb, rubric_embs)
    
    return {"similarities": similarities}
