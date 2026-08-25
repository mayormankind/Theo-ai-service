# src/ai-service/app/routes/similarity.py
from fastapi import APIRouter
from app.models.request_models import SimilarityRequest
from app.services.embedding_service import get_embeddings
from app.services.scoring_service import calculate_similarity

router = APIRouter()

@router.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest):
    """
    Calculates the cosine similarity between a student's answer and a list of rubric points.

    NOTE: embeddings are computed on the raw (only whitespace-trimmed) text —
    identically to the /grade pipeline — so the two endpoints always agree.
    Heavy preprocessing (lemmatisation / stop-word removal) is intentionally NOT
    applied: text-embedding-3-small performs better on natural language and it
    previously made /similarity and /grade disagree for the same inputs.
    """
    student_answer = (req.student_answer or "").strip()
    clean_rubrics = [rp.strip() for rp in req.rubric]

    if not clean_rubrics:
        return {"similarities": []}

    # Get vector embeddings
    student_emb = get_embeddings([student_answer])[0]
    rubric_embs = get_embeddings(clean_rubrics)

    # Calculate similarities
    similarities = calculate_similarity(student_emb, rubric_embs)

    return {"similarities": similarities}
